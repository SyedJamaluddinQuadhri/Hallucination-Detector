"""
metrics.py — Evaluation metrics for HalluDet-Lite.

Computes:
  - AUROC (primary ranking metric)
  - F1, Precision, Recall at threshold 0.5
  - ECE (Expected Calibration Error) — are scores true probabilities?
  - Ablation comparison across phases
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    average_precision_score,
)

from src.utils import get_logger, get_project_root, save_json

log = get_logger("evaluation.metrics")


# ── Core Metrics ─────────────────────────────────────────────────────────────

def compute_auroc(y_true: List[int], y_scores: List[float]) -> float:
    """Area Under the ROC Curve — primary detection metric."""
    return float(roc_auc_score(y_true, y_scores))


def compute_auprc(y_true: List[int], y_scores: List[float]) -> float:
    """Area Under the Precision-Recall Curve — useful for imbalanced data."""
    return float(average_precision_score(y_true, y_scores))


def compute_ece(
    y_true: List[int],
    y_scores: List[float],
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error.
    Measures whether predicted probabilities match true frequencies.
    ECE < 0.05 = well-calibrated. ECE < 0.10 = acceptable.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (np.array(y_scores) >= lo) & (np.array(y_scores) < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = np.array(y_true)[mask].mean()
        bin_conf = np.array(y_scores)[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5,
) -> Dict:
    """Compute F1, Precision, Recall at a fixed threshold."""
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    return {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "report":    classification_report(
            y_true, y_pred,
            target_names=["Factual", "Hallucinated"],
            zero_division=0,
        ),
    }


def compute_all_metrics(
    y_true: List[int],
    y_scores: List[float],
    name: str = "model",
) -> Dict:
    """Compute the full evaluation suite for one model."""
    auroc   = compute_auroc(y_true, y_scores)
    auprc   = compute_auprc(y_true, y_scores)
    ece     = compute_ece(y_true, y_scores)
    clf     = compute_classification_metrics(y_true, y_scores)

    results = {
        "name":       name,
        "n_samples":  len(y_true),
        "auroc":      auroc,
        "auprc":      auprc,
        "ece":        ece,
        "f1":         clf["f1"],
        "precision":  clf["precision"],
        "recall":     clf["recall"],
    }

    log.info(f"\n{'='*55}")
    log.info(f"Evaluation: {name}")
    log.info(f"  AUROC:     {auroc:.4f}  (target > 0.83)")
    log.info(f"  AUPRC:     {auprc:.4f}")
    log.info(f"  ECE:       {ece:.4f}   (target < 0.06)")
    log.info(f"  F1:        {clf['f1']:.4f}")
    log.info(f"  Precision: {clf['precision']:.4f}")
    log.info(f"  Recall:    {clf['recall']:.4f}")
    log.info(f"{'='*55}\n")
    log.info(clf["report"])

    return results


# ── Ablation Study ────────────────────────────────────────────────────────────

def run_ablation_study(
    test_records: List[Dict],
    nli_predictor,
    ensemble,
    rav,
    meta_predictor,
    save_path: Optional[str] = None,
    ppl=None,
) -> Dict:
    """
    Run an ablation study comparing each phase's contribution.

    Test configurations:
      1. NLI only (Phase 1 baseline)
      2. NLI + Ensemble disagreement (+ Phase 2)
      3. NLI + RAV (+ Phase 3)
      4. Full ensemble (all phases + meta-learner)

    Args:
        test_records:    List of {premise, claim, label} dicts (label: 0/1)
        nli_predictor:   Phase 1 NLIPredictor
        ensemble:        Phase 2 EnsembleDisagreement
        rav:             Phase 3 LightweightRAV
        meta_predictor:  Phase 4 MetaLearnerPredictor
        save_path:       Where to save results JSON
        ppl:             Optional pre-loaded TinyLLaMaPerplexity instance.
                         Pass the shared instance to avoid loading the model
                         a second time.

    Returns:
        Dict mapping config name -> metrics dict
    """
    import spacy
    from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
    from src.phase4_ensemble.features import extract_features

    nlp_sm = spacy.load("en_core_web_sm")
    ppl    = TinyLLaMaPerplexity() if ppl is None else ppl

    y_true = [r["label"] for r in test_records]
    ablation_scores = {
        "nli_only":       [],
        "nli_ensemble":   [],
        "nli_rav":        [],
        "full_ensemble":  [],
    }

    log.info(f"Running ablation on {len(test_records)} test records...")

    for i, rec in enumerate(test_records):
        premise = rec["premise"]
        claim   = rec["claim"]

        # Config 1: NLI only
        nli_r = nli_predictor.score_claim(premise, claim)
        ablation_scores["nli_only"].append(nli_r["hallucination_score"])

        # Config 2: NLI + Ensemble
        ens_r = ensemble.score(premise, claim)
        nli_ens = (nli_r["hallucination_score"] * 0.6 +
                   ens_r["ensemble_hal_score"]  * 0.4)
        ablation_scores["nli_ensemble"].append(nli_ens)

        # Config 3: NLI + RAV
        rav_r = rav.verify_claim(claim)
        nli_rav = (nli_r["hallucination_score"] * 0.6 +
                   rav_r["rav_hal_score"]        * 0.4)
        ablation_scores["nli_rav"].append(nli_rav)

        # Config 4: Full ensemble (extract all features, run meta-learner)
        ppl_r = ppl.score_claim(claim)
        doc   = nlp_sm(claim)
        feat  = np.array([
            nli_r["probs"]["entailment"],
            nli_r["probs"]["contradiction"],
            ens_r["mean_entailment"],
            ens_r["disagreement"],
            min(ppl_r["log_perplexity"], 10.0),
            rav_r["max_ent"],
            max((e["retrieval_score"] for e in rav_r.get("evidence_used", [])), default=0.0),
            float(np.log1p(len(doc.ents))),
        ], dtype=np.float32)
        full_score = meta_predictor.predict(feat)
        ablation_scores["full_ensemble"].append(full_score)

        if i % 100 == 0:
            log.info(f"  Ablation progress: {i}/{len(test_records)}")

    # Compute metrics for each config
    results = {}
    for config_name, scores in ablation_scores.items():
        results[config_name] = compute_all_metrics(y_true, scores, name=config_name)

    # Print summary table
    log.info("\n" + "=" * 60)
    log.info(f"{'Config':<22} {'AUROC':>8} {'F1':>8} {'ECE':>8}")
    log.info("-" * 60)
    for name, metrics in results.items():
        log.info(
            f"{name:<22} {metrics['auroc']:>8.4f} "
            f"{metrics['f1']:>8.4f} {metrics['ece']:>8.4f}"
        )
    log.info("=" * 60)

    if save_path:
        save_json(results, save_path)
        log.info(f"Ablation results saved to {save_path}")

    return results
