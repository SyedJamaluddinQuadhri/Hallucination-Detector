"""
run_phase2.py — Test and validate the uncertainty signal modules.

Tests:
  1. Multi-model ensemble disagreement on sample claims
  2. TinyLLaMA perplexity scoring
  3. Caches scored examples for Phase 4 feature building

Runtime: ~10 minutes for smoke test
VRAM:    ~1.5GB peak (models loaded sequentially)
Output:  data/processed/phase2_test_results.json

Usage:
    python scripts/run_phase2.py
    python scripts/run_phase2.py --full    # score all HaluEval samples (slow)
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from src.utils import get_logger, free_gpu_memory, get_project_root

log = get_logger("script.phase2")

TEST_CLAIMS = [
    # (premise, claim, expected_label)
    (
        "Albert Einstein was born on March 14, 1879. He won the Nobel Prize in Physics in 1921.",
        "Einstein won the Nobel Prize in 1921.",
        0,  # not hallucinated
    ),
    (
        "Albert Einstein was born on March 14, 1879. He won the Nobel Prize in Physics in 1921.",
        "Einstein won the Nobel Prize in 1935.",
        1,  # hallucinated
    ),
    (
        "The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889.",
        "The Eiffel Tower is in London.",
        1,  # hallucinated
    ),
    (
        "The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889.",
        "The Eiffel Tower was constructed in the late 1880s.",
        0,  # not hallucinated
    ),
]


def run_ensemble_test():
    """Test the ensemble disagreement module."""
    log.info("Testing ensemble disagreement...")
    from src.phase1_nli.predict import NLIPredictor
    from src.phase2_uncertainty.ensemble import EnsembleDisagreement

    nli = NLIPredictor()
    ens = EnsembleDisagreement(nli_predictor=nli, load_small=True, load_bart=False)

    results = []
    for premise, claim, label in TEST_CLAIMS:
        r = ens.score(premise, claim)
        correct = (r["ensemble_hal_score"] >= 0.5) == (label == 1)
        status  = "✓" if correct else "✗"
        log.info(
            f"  {status} [{label}] Score={r['ensemble_hal_score']:.3f} "
            f"Disagree={r['disagreement']:.4f} | {claim[:50]}..."
        )
        results.append({"claim": claim, "label": label, **r})

    accuracy = sum(1 for r in results if (r["ensemble_hal_score"] >= 0.5) == (r["label"] == 1)) / len(results)
    log.info(f"  Ensemble test accuracy: {accuracy:.1%}")
    return results


def run_perplexity_test():
    """Test the TinyLLaMA perplexity module."""
    log.info("Testing TinyLLaMA perplexity...")
    import torch
    from src.phase1_nli.predict import NLIPredictor
    from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity

    # Free NLI VRAM before loading TinyLLaMA
    free_gpu_memory()

    ppl = TinyLLaMaPerplexity(use_4bit=True)

    claims_to_test = [
        ("Einstein won the Nobel Prize in 1921.", 0),  # factual - expect low ppl
        ("Einstein won the Nobel Prize in 1953.", 1),  # wrong date - expect higher ppl
        ("Water is composed of hydrogen and oxygen.", 0),  # factual
        ("Water is composed of nitrogen and carbon.", 1),  # wrong
    ]

    for claim, label in claims_to_test:
        r = ppl.score_claim(claim)
        log.info(
            f"  Label={label} PPL={r['perplexity']:.1f} "
            f"LogPPL={r['log_perplexity']:.2f} | {claim}"
        )

    ppl.to_cpu()
    free_gpu_memory()
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Test uncertainty signals")
    parser.add_argument("--full",     action="store_true", help="Run full HaluEval scoring")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip perplexity test (saves time)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HalluDet-Lite — Phase 2: Uncertainty Signals")
    log.info("=" * 60)

    # Test ensemble
    ensemble_results = run_ensemble_test()

    # Test perplexity
    if not args.skip_ppl:
        run_perplexity_test()
    else:
        log.info("Skipping perplexity test (--skip-ppl set)")

    # Save test results
    out_path = str(get_project_root() / "data" / "processed" / "phase2_test_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ensemble_results, f, indent=2)
    log.info(f"Test results saved to {out_path}")

    log.info("\n✓ Phase 2 complete!")
    log.info("Next step: python scripts/run_phase3.py")


if __name__ == "__main__":
    main()
