"""
benchmarks.py — Standard benchmark evaluation for HalluDet-Lite.

Runs the full detection system on:
  1. HaluEval test set (held-out) — primary detection benchmark
  2. TruthfulQA — factuality quality of Phi-2 DPO model
  3. Custom domain-specific test queries

Usage:
    python scripts/run_all_eval.py
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

from src.utils import get_logger, get_project_root, save_json
from evaluation.metrics import compute_all_metrics, run_ablation_study

log = get_logger("evaluation.benchmarks")

RESULTS_DIR = get_project_root() / "evaluation" / "results"


# ── HaluEval Benchmark ────────────────────────────────────────────────────────

def evaluate_on_halueval(
    nli_predictor,
    ensemble,
    rav,
    meta_predictor,
    max_samples: int = 1_000,
    save_results: bool = True,
) -> Dict:
    """
    Evaluate the full pipeline on the HaluEval QA test split.

    Since HaluEval has no official test split, we use a held-out portion
    (the last 10% of samples, which we excluded during feature dataset building).

    Args:
        nli_predictor:   Phase 1 NLIPredictor
        ensemble:        Phase 2 EnsembleDisagreement
        rav:             Phase 3 LightweightRAV
        meta_predictor:  Phase 4 MetaLearnerPredictor
        max_samples:     Max samples to evaluate (for speed)
        save_results:    Save results to disk

    Returns:
        Dict of evaluation metrics
    """
    from src.phase1_nli.claim_splitter import split_claims
    from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
    import spacy

    log.info(f"Loading HaluEval for benchmark (max {max_samples} samples)...")
    halueval = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # Use last 10% as test (held out during training)
    n_total = len(halueval)
    start_idx = int(n_total * 0.9)
    test_samples = list(halueval)[start_idx:start_idx + max_samples]

    nlp_sm = spacy.load("en_core_web_sm")
    ppl    = TinyLLaMaPerplexity()

    y_true  = []
    y_scores = []

    log.info(f"Evaluating on {len(test_samples)} HaluEval samples...")

    for i, sample in enumerate(test_samples):
        question  = sample["question"]
        knowledge = sample.get("knowledge", "")
        if not knowledge:
            continue

        label = 1 if sample["hallucination"] == "yes" else 0
        response = sample["answer"]
        claims = split_claims(response)
        if not claims:
            continue

        # Score each claim and take the max (worst-case aggregation)
        claim_scores = []
        for claim in claims:
            try:
                nli_r = nli_predictor.score_claim(knowledge, claim)
                ens_r = ensemble.score(knowledge, claim)
                rav_r = rav.verify_claim(claim)
                ppl_r = ppl.score_claim(claim)
                doc   = nlp_sm(claim)

                feat = np.array([
                    nli_r["probs"]["entailment"],
                    nli_r["probs"]["contradiction"],
                    ens_r["mean_entailment"],
                    ens_r["disagreement"],
                    min(ppl_r["log_perplexity"], 10.0),
                    rav_r["max_ent"],
                    max((e["retrieval_score"] for e in rav_r.get("evidence_used", [])), default=0.5),
                    float(np.log1p(len(doc.ents))),
                ], dtype=np.float32)

                score = meta_predictor.predict(feat)
                claim_scores.append(score)
            except Exception as e:
                log.debug(f"Error scoring claim: {e}")
                continue

        if claim_scores:
            # Response-level score = max claim score
            response_score = max(claim_scores)
            y_true.append(label)
            y_scores.append(response_score)

        if i % 100 == 0:
            log.info(f"  Evaluated {i}/{len(test_samples)}")

    results = compute_all_metrics(y_true, y_scores, name="HaluEval Full System")

    if save_results:
        os.makedirs(str(RESULTS_DIR), exist_ok=True)
        save_json(results, str(RESULTS_DIR / "halueval_results.json"))

    return results


# ── TruthfulQA Benchmark ──────────────────────────────────────────────────────

def evaluate_phi2_truthfulqa(
    phi2_model=None,
    phi2_tokenizer=None,
    max_samples: int = 200,
    save_results: bool = True,
) -> Dict:
    """
    Evaluate Phi-2 DPO model on TruthfulQA multiple-choice.

    Compares DPO fine-tuned model vs. base Phi-2 accuracy.
    Requires the DPO adapter from Phase 5.

    Args:
        phi2_model:     Fine-tuned Phi-2 model (PeftModel). If None, loads from disk.
        phi2_tokenizer: Phi-2 tokenizer. If None, loads from disk.
        max_samples:    Number of TruthfulQA questions to evaluate.
        save_results:   Save results to disk.

    Returns:
        Dict with accuracy and per-question results.
    """
    import torch

    log.info("Loading TruthfulQA benchmark...")
    truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation")

    if phi2_model is None:
        adapter_path = str(get_project_root() / "models" / "phi2_dpo_lora" / "final")
        if not os.path.exists(adapter_path):
            log.error(
                "Phi-2 DPO adapter not found. "
                "Complete Phase 5 (Colab training) first."
            )
            return {"error": "adapter_not_found"}

        log.info(f"Loading Phi-2 DPO model from {adapter_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        phi2_model     = PeftModel.from_pretrained(base, adapter_path)
        phi2_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    phi2_model.eval()
    device = next(phi2_model.parameters()).device

    correct = 0
    total   = 0
    per_question = []

    samples = list(truthfulqa)[:max_samples]
    log.info(f"Evaluating on {len(samples)} TruthfulQA questions...")

    for sample in samples:
        question = sample["question"]
        mc1      = sample["mc1_targets"]
        choices  = mc1["choices"]
        labels   = mc1["labels"]  # 1 = correct, 0 = incorrect

        # Score each choice as the log-prob of the answer given the question
        choice_scores = []
        for choice in choices:
            prompt = f"Question: {question}\nAnswer: {choice}"
            tokens = phi2_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = phi2_model(**tokens, labels=tokens.input_ids)
            # Lower loss = higher probability = model prefers this answer
            choice_scores.append(-out.loss.item())

        # Predicted answer = highest log-prob
        predicted_idx = choice_scores.index(max(choice_scores))
        correct_idx   = labels.index(1)

        is_correct = (predicted_idx == correct_idx)
        if is_correct:
            correct += 1
        total += 1

        per_question.append({
            "question":      question,
            "correct":       is_correct,
            "predicted":     choices[predicted_idx],
            "ground_truth":  choices[correct_idx],
        })

    accuracy = correct / total if total > 0 else 0.0
    results  = {
        "benchmark":       "TruthfulQA MC1",
        "accuracy":        float(accuracy),
        "correct":         correct,
        "total":           total,
        "per_question":    per_question[:20],  # save first 20 for inspection
    }

    log.info(f"\nTruthfulQA MC1 Accuracy: {accuracy:.4f} ({correct}/{total})")
    log.info("(Base Phi-2 baseline: ~0.52 — target after DPO: ~0.57–0.60)")

    if save_results:
        os.makedirs(str(RESULTS_DIR), exist_ok=True)
        save_json(results, str(RESULTS_DIR / "truthfulqa_results.json"))

    return results


# ── Demo test queries ─────────────────────────────────────────────────────────

DEMO_TEST_CASES = [
    {
        "question": "When did Albert Einstein win the Nobel Prize?",
        "correct_response": "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        "hallucinated_response": "Albert Einstein won the Nobel Prize in Physics in 1922 for his theory of relativity.",
        "label_correct": 0,
        "label_hallucinated": 1,
    },
    {
        "question": "Who wrote the play Hamlet?",
        "correct_response": "Hamlet was written by William Shakespeare, most likely around 1600–1601.",
        "hallucinated_response": "Hamlet was written by Christopher Marlowe in 1589.",
        "label_correct": 0,
        "label_hallucinated": 1,
    },
    {
        "question": "What is the capital of Australia?",
        "correct_response": "The capital of Australia is Canberra.",
        "hallucinated_response": "The capital of Australia is Sydney, which is also the largest city.",
        "label_correct": 0,
        "label_hallucinated": 1,
    },
    {
        "question": "What does DNA stand for?",
        "correct_response": "DNA stands for Deoxyribonucleic Acid.",
        "hallucinated_response": "DNA stands for Dynamic Nucleic Assembly.",
        "label_correct": 0,
        "label_hallucinated": 1,
    },
]


def run_demo_tests(
    nli_predictor,
    ensemble,
    rav,
    meta_predictor,
) -> List[Dict]:
    """
    Run the pipeline on a small set of known test cases.
    Useful for a quick sanity check and for the demo section.

    Returns list of result dicts showing per-case scores.
    """
    from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
    import spacy

    nlp_sm = spacy.load("en_core_web_sm")
    ppl    = TinyLLaMaPerplexity()

    from src.phase1_nli.claim_splitter import split_claims

    results = []
    log.info("Running demo test cases...")

    for case in DEMO_TEST_CASES:
        for (response, expected_label) in [
            (case["correct_response"],     case["label_correct"]),
            (case["hallucinated_response"], case["label_hallucinated"]),
        ]:
            claims = split_claims(response)
            claim_scores = []

            for claim in claims:
                nli_r = nli_predictor.score_claim(case["question"], claim)
                ens_r = ensemble.score(case["question"], claim)
                rav_r = rav.verify_claim(claim)
                ppl_r = ppl.score_claim(claim)
                doc   = nlp_sm(claim)

                feat = np.array([
                    nli_r["probs"]["entailment"],
                    nli_r["probs"]["contradiction"],
                    ens_r["mean_entailment"],
                    ens_r["disagreement"],
                    min(ppl_r["log_perplexity"], 10.0),
                    rav_r["max_ent"],
                    max((e["retrieval_score"] for e in rav_r.get("evidence_used", [])), default=0.5),
                    float(np.log1p(len(doc.ents))),
                ], dtype=np.float32)

                claim_scores.append(meta_predictor.predict(feat))

            overall_score = max(claim_scores) if claim_scores else 0.5

            result = {
                "question":        case["question"],
                "response":        response[:100] + "...",
                "expected":        "HALLUCINATED" if expected_label == 1 else "FACTUAL",
                "predicted":       "HALLUCINATED" if overall_score >= 0.5 else "FACTUAL",
                "hal_score":       float(overall_score),
                "correct":         (overall_score >= 0.5) == (expected_label == 1),
            }
            results.append(result)
            status = "✓" if result["correct"] else "✗"
            log.info(
                f"{status} Q: {case['question'][:40]}... "
                f"Score={overall_score:.3f} Expected={result['expected']}"
            )

    correct = sum(1 for r in results if r["correct"])
    log.info(f"\nDemo accuracy: {correct}/{len(results)}")

    return results
