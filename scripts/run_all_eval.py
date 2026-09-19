"""
run_all_eval.py — Full evaluation suite for the final report and paper.

Runs:
  1. Demo test cases (quick sanity check)
  2. Ablation study (NLI-only → +ensemble → +RAV → full)
  3. HaluEval benchmark (held-out test set)
  4. TruthfulQA evaluation of Phi-2 DPO model (if available)
  5. Prints final results table

Usage:
    python scripts/run_all_eval.py
    python scripts/run_all_eval.py --demo-only    # just demo cases
    python scripts/run_all_eval.py --no-phi2      # skip TruthfulQA
"""

import argparse
import sys
import os
# TIP: Set HF_TOKEN in your environment to avoid unauthenticated HuggingFace
# requests and their lower rate limits:
#   $env:HF_TOKEN = "hf_your_token_here"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from src.utils import get_logger, get_project_root, free_gpu_memory

log = get_logger("script.eval")


def print_results_table(all_results: dict):
    """Print a formatted results table for the paper."""
    print("\n" + "=" * 65)
    print("HALLUDET-LITE — FINAL RESULTS TABLE")
    print("=" * 65)
    print(f"{'Component':<28} {'AUROC':>8} {'F1':>8} {'ECE':>8}")
    print("-" * 65)

    phase_results = all_results.get("ablation", {})
    for name, metrics in phase_results.items():
        display = {
            "nli_only":      "Phase 1 — NLI only",
            "nli_ensemble":  "Phase 1+2 — + Ensemble",
            "nli_rav":       "Phase 1+3 — + RAV",
            "full_ensemble": "Phase 1–4 — Full ensemble",
        }.get(name, name)
        print(
            f"{display:<28} {metrics.get('auroc',0):>8.4f} "
            f"{metrics.get('f1',0):>8.4f} {metrics.get('ece',0):>8.4f}"
        )

    print("-" * 65)

    if "halueval" in all_results:
        m = all_results["halueval"]
        print(f"{'HaluEval Test Set':<28} {m.get('auroc',0):>8.4f} "
              f"{m.get('f1',0):>8.4f} {m.get('ece',0):>8.4f}")

    if "truthfulqa" in all_results:
        t = all_results["truthfulqa"]
        print(f"\n{'TruthfulQA MC1 (Phi-2 DPO)':<28} Accuracy: {t.get('accuracy',0):.4f}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Full evaluation suite")
    parser.add_argument("--demo-only",  action="store_true", help="Only run demo cases")
    parser.add_argument("--no-phi2",    action="store_true", help="Skip TruthfulQA eval")
    parser.add_argument("--max-samples", type=int, default=500, help="Max eval samples")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HalluDet-Lite — Full Evaluation Suite")
    log.info("=" * 60)

    root = get_project_root()
    all_results = {}

    # ── Load all models ───────────────────────────────────────────────────────
    log.info("Loading pipeline models...")
    from src.phase1_nli.predict            import NLIPredictor
    from src.phase2_uncertainty.ensemble   import EnsembleDisagreement
    from src.phase3_rav.retriever          import LightweightRAV
    from src.phase4_ensemble.meta_learner  import MetaLearnerPredictor

    nli   = NLIPredictor()
    ens   = EnsembleDisagreement(nli_predictor=nli, load_small=True, load_bart=True)
    rav   = LightweightRAV(nli_predictor=nli)
    meta  = MetaLearnerPredictor()

    # ── Load TinyLLaMA once (shared across all benchmarks) ───────────────────
    log.info("Loading TinyLLaMA perplexity model (shared instance)...")
    from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
    ppl = TinyLLaMaPerplexity()

    # ── Demo cases ────────────────────────────────────────────────────────────
    log.info("\n--- Running Demo Test Cases ---")
    from evaluation.benchmarks import run_demo_tests
    demo_results = run_demo_tests(nli, ens, rav, meta, ppl=ppl)
    all_results["demo"] = demo_results
    demo_acc = sum(1 for r in demo_results if r["correct"]) / len(demo_results)
    log.info(f"Demo accuracy: {demo_acc:.1%}")

    if args.demo_only:
        print_results_table(all_results)
        return

    # ── Build test records for ablation ──────────────────────────────────────
    log.info("\n--- Building ablation test set ---")
    from datasets import load_dataset
    from src.phase1_nli.claim_splitter import split_claims

    halueval  = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    n_total   = len(halueval)
    test_slice = list(halueval)[int(n_total * 0.9): int(n_total * 0.9) + args.max_samples]

    test_records = []
    for sample in test_slice:
        knowledge = sample.get("knowledge", "")
        if not knowledge:
            continue
        label = 1 if sample["hallucination"] == "yes" else 0
        response = sample["answer"]
        for claim in split_claims(response):
            test_records.append({
                "premise": knowledge,
                "claim":   claim,
                "label":   label,
            })

    log.info(f"Ablation test records: {len(test_records):,}")

    # ── Ablation study ────────────────────────────────────────────────────────
    log.info("\n--- Running Ablation Study ---")
    from evaluation.metrics import run_ablation_study
    ablation_results = run_ablation_study(
        test_records=test_records,
        nli_predictor=nli,
        ensemble=ens,
        rav=rav,
        meta_predictor=meta,
        save_path=str(root / "evaluation" / "results" / "ablation_results.json"),
        ppl=ppl,
    )
    all_results["ablation"] = ablation_results

    # ── HaluEval benchmark ────────────────────────────────────────────────────
    log.info("\n--- HaluEval Benchmark ---")
    from evaluation.benchmarks import evaluate_on_halueval
    halueval_results = evaluate_on_halueval(
        nli_predictor=nli,
        ensemble=ens,
        rav=rav,
        meta_predictor=meta,
        max_samples=args.max_samples,
        ppl=ppl,
    )
    all_results["halueval"] = halueval_results

    # ── TruthfulQA (Phi-2 DPO) ────────────────────────────────────────────────
    if not args.no_phi2:
        adapter_path = str(root / "models" / "phi2_dpo_lora" / "final")
        if os.path.exists(adapter_path):
            log.info("\n--- TruthfulQA Evaluation (Phi-2 DPO) ---")
            # Free GPU memory before loading Phi-2
            nli.to_cpu(); free_gpu_memory()
            from evaluation.benchmarks import evaluate_phi2_truthfulqa
            tqa_results = evaluate_phi2_truthfulqa(max_samples=200)
            all_results["truthfulqa"] = tqa_results
        else:
            log.info("Phi-2 DPO adapter not found — skipping TruthfulQA.")
            log.info("Complete Phase 5 (Colab) first: open colab/phase5_phi2_dpo.ipynb")

    # ── Save all results ──────────────────────────────────────────────────────
    from src.utils import save_json
    os.makedirs(str(root / "evaluation" / "results"), exist_ok=True)
    save_json(all_results, str(root / "evaluation" / "results" / "full_eval.json"))
    log.info("All results saved to evaluation/results/full_eval.json")

    # ── Print final table ─────────────────────────────────────────────────────
    print_results_table(all_results)
    log.info("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
