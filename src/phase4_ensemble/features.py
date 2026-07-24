"""
features.py — Feature extraction for the meta-learner ensemble.

Combines signals from all 3 phases into an 8-dimensional feature vector
for each claim. These features are then fed to the meta-learner MLP.

Feature vector (8 dimensions):
  F0: NLI entailment probability     (Phase 1 DeBERTa-base)
  F1: NLI contradiction probability  (Phase 1)
  F2: Ensemble mean entailment       (Phase 2 — 3-model average)
  F3: Ensemble disagreement          (Phase 2 — variance)
  F4: Claim perplexity (log-norm)    (Phase 2 TinyLLaMA)
  F5: RAV max entailment             (Phase 3 FAISS + NLI)
  F6: RAV top retrieval similarity   (Phase 3 FAISS cosine score)
  F7: Named entity count (log-norm)  (spaCy)
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy

from src.utils import get_logger, get_project_root, save_json

log = get_logger("ensemble.features")

FEATURE_DIM   = 8
FEATURE_NAMES = [
    "nli_entailment",
    "nli_contradiction",
    "ensemble_mean_entailment",
    "ensemble_disagreement",
    "claim_log_perplexity",
    "rav_max_entailment",
    "rav_retrieval_score",
    "entity_count_log",
]


def extract_features(
    claim:    str,
    premise:  str,
    nli,               # NLIPredictor
    ensemble,          # EnsembleDisagreement
    perplexity_scorer, # TinyLLaMaPerplexity
    rav,               # LightweightRAV
    nlp,               # spacy model
) -> np.ndarray:
    """
    Extract all 8 features for one (premise, claim) pair.

    Args:
        claim:             Atomic claim string.
        premise:           Evidence context string.
        nli:               Phase 1 NLIPredictor instance.
        ensemble:          Phase 2 EnsembleDisagreement instance.
        perplexity_scorer: Phase 2 TinyLLaMaPerplexity instance.
        rav:               Phase 3 LightweightRAV instance.
        nlp:               spaCy model (en_core_web_sm).

    Returns:
        np.ndarray of shape (8,) with float32 values.
    """
    # F0, F1 — NLI signals (GPU)
    nli_result = nli.score_claim(premise, claim)
    f0 = nli_result["probs"]["entailment"]
    f1 = nli_result["probs"]["contradiction"]

    # F2, F3 — Ensemble disagreement (mixed CPU/GPU)
    ens_result = ensemble.score(premise, claim)
    f2 = ens_result["mean_entailment"]
    f3 = ens_result["disagreement"]

    # F4 — Perplexity (GPU with TinyLLaMA)
    ppl_result = perplexity_scorer.score_claim(claim)
    f4 = min(ppl_result["log_perplexity"], 10.0)  # cap at 10

    # F5, F6 — RAV signals (CPU FAISS + GPU NLI)
    rav_result = rav.verify_claim(claim)
    f5 = rav_result["max_ent"]
    f6 = max(
        (e["retrieval_score"] for e in rav_result.get("evidence_used", [])),
        default=0.0,
    )

    # F7 — Named entity count (CPU spaCy)
    doc = nlp(claim)
    f7 = float(np.log1p(len(doc.ents)))

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)


def build_feature_dataset(
    save_path: str,
    max_samples: int = 10_000,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the full feature dataset from HaluEval for meta-learner training.
    This is the most time-consuming step — cache the results to disk.

    Args:
        save_path:   Where to save the final features array.
        max_samples: Max HaluEval samples to process (each gives 2+ claims).
        cache_path:  If given, load from here if it exists (avoids recomputation).

    Returns:
        (X, y) — feature matrix and binary labels.
    """
    if cache_path and os.path.exists(cache_path):
        log.info(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["X"], data["y"]

    log.info("Building feature dataset from HaluEval...")
    log.info("Running sequentially per-model to save VRAM/RAM...")

    from datasets import load_dataset
    from src.phase1_nli.claim_splitter import split_claims
    import gc
    import torch

    halueval = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # 1. Pre-process into a flat list of items
    items = []
    for i, sample in enumerate(halueval):
        if i >= max_samples:
            break
        knowledge = sample.get("knowledge", "")
        if not knowledge:
            continue
        response = sample.get("answer", "")
        label_str = str(sample.get("hallucination", "no")).strip().lower()
        label = 1 if label_str == "yes" else 0
        
        claims = split_claims(response)
        for claim in claims:
            items.append({"claim": claim, "premise": knowledge, "label": label})

    N = len(items)
    log.info(f"Extracted {N} atomic claims for feature processing.")
    X = np.zeros((N, 8), dtype=np.float32)
    y = np.array([item["label"] for item in items], dtype=np.float32)

    nli_for_rav = None

    # 1. Phase 3: RAV (Put this first so the 400MB JSON can load without fragmentation)
    try:
        from src.phase1_nli.predict import NLIPredictor
        from src.phase3_rav.retriever import LightweightRAV
        log.info("Loading NLI Phase 1 Model...")
        nli_for_rav = NLIPredictor()

        log.info("Loading Lightweight RAV and Sentence Transformers...")
        rav = LightweightRAV(nli_predictor=nli_for_rav)
        log.info("Processing F5, F6 (Retrieval-Augmented Verification)...")
        for idx, item in enumerate(items):
            try:
                res = rav.verify_claim(item["claim"])
                X[idx, 5] = res["max_ent"]
                X[idx, 6] = max((e["retrieval_score"] for e in res.get("evidence_used", [])), default=0.0)
            except Exception:
                pass
            if idx > 0 and idx % 1000 == 0: log.info(f"  RAV progress: {idx}/{N}")
        
        # Cleanup RAV entirely, including the massive JSON
        log.info("Cleaning up RAV models and massive JSON dictionaries...")
        rav.chunks = None
        rav.index = None
        rav.retriever = None
        del rav
        
        # Pre-cleanup NLI too just to give TinyLLaMA pure clean RAM
        nli_for_rav.model = None
        del nli_for_rav
        nli_for_rav = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log.error(f"RAV processing failed: {e}")

    # 2. Phase 2: Perplexity (TinyLLaMA)
    try:
        from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
        log.info("Loading TinyLLaMA model (requires ~4.4GB RAM peak)...")
        ppl = TinyLLaMaPerplexity()
        log.info("Processing F4 (Perplexity)...")
        for idx, item in enumerate(items):
            try:
                res = ppl.score_claim(item["claim"])
                X[idx, 4] = min(res["log_perplexity"], 10.0)
            except Exception:
                X[idx, 4] = 10.0
            if idx > 0 and idx % 1000 == 0: log.info(f"  Perplexity progress: {idx}/{N}")

        # Cleanup perplexity
        log.info("Cleaning up TinyLLaMA model from RAM/VRAM...")
        ppl.model = None
        del ppl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log.error(f"Perplexity processing failed: {e}")

    # 3. Phase 1: NLI
    nli = None
    try:
        from src.phase1_nli.predict import NLIPredictor
        log.info("Reloading NLI Phase 1 Model...")
        nli = NLIPredictor()
        log.info("Processing F0, F1 (NLI)...")
        for idx, item in enumerate(items):
            try:
                res = nli.score_claim(item["premise"], item["claim"])
                X[idx, 0] = res["probs"]["entailment"]
                X[idx, 1] = res["probs"]["contradiction"]
            except Exception:
                pass
            if idx > 0 and idx % 1000 == 0: log.info(f"  NLI progress: {idx}/{N}")
    except Exception as e:
        log.error(f"NLI processing failed: {e}")

    # 4. Phase 2: Ensemble
    try:
        if nli is not None:
            from src.phase2_uncertainty.ensemble import EnsembleDisagreement
            log.info("Loading Ensemble Models (BART-large & DeBERTa-small)...")
            ens = EnsembleDisagreement(nli_predictor=nli)
            log.info("Processing F2, F3 (Ensemble Disagreement)...")
            for idx, item in enumerate(items):
                try:
                    res = ens.score(item["premise"], item["claim"])
                    X[idx, 2] = res["mean_entailment"]
                    X[idx, 3] = res["disagreement"]
                except Exception:
                    pass
                if idx > 0 and idx % 1000 == 0: log.info(f"  Ensemble progress: {idx}/{N}")
            
            # Cleanup ensemble
            log.info("Cleaning up Ensemble models from RAM/VRAM...")
            ens.m2 = None
            ens.m3 = None
            del ens
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        log.error(f"Ensemble processing failed: {e}")

    # 6. spaCy Entity Count
    try:
        log.info("Processing F7 (spaCy Entity Count)...")
        nlp_sm = spacy.load("en_core_web_sm")
        for idx, item in enumerate(items):
            try:
                doc = nlp_sm(item["claim"])
                X[idx, 7] = float(np.log1p(len(doc.ents)))
            except Exception:
                pass
            if idx > 0 and idx % 1000 == 0: log.info(f"  spaCy progress: {idx}/{N}")
            
        del nlp_sm
        gc.collect()
    except Exception as e:
        log.error(f"spaCy processing failed: {e}")

    # Cleanup NLI
    if nli is not None:
        log.info("Cleaning up initial NLI model...")
        nli.model = None
        del nli
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info(f"Feature dataset: X={X.shape}, y={y.shape}")
    log.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path + "_X.npy", X)
    np.save(save_path + "_y.npy", y)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"X": X, "y": y}, f)
        log.info(f"Features cached to {cache_path}")

    return X, y
