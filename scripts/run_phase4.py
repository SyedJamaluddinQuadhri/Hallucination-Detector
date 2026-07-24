"""
run_phase4.py — Build feature dataset and train the meta-learner.

Also builds the DPO dataset for Phase 5 if --build-dpo flag is set.

Runtime:  Feature building: ~2–4 hours (slow — uses all models)
          Meta-learner training: ~2 minutes (CPU)
Output:   models/meta_learner/
          data/processed/dpo_dataset.json (if --build-dpo)

Usage:
    python scripts/run_phase4.py
    python scripts/run_phase4.py --samples 500   # quick run with 500 samples
    python scripts/run_phase4.py --build-dpo     # also build DPO dataset
    python scripts/run_phase4.py --train-only    # skip feature build, train from cache
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils import get_logger, get_project_root

log = get_logger("script.phase4")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train meta-learner ensemble")
    parser.add_argument("--samples",    type=int,  default=5_000,
                        help="HaluEval samples for feature building")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip feature extraction, use cached features")
    parser.add_argument("--build-dpo",  action="store_true",
                        help="Also build the DPO preference dataset for Phase 5")
    parser.add_argument("--epochs",     type=int,  default=100,
                        help="Meta-learner training epochs")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HalluDet-Lite — Phase 4: Ensemble Meta-Learner")
    log.info("=" * 60)

    root      = get_project_root()
    feat_path = str(root / "data" / "processed" / "ensemble_features")
    cache_path = str(root / "data" / "processed" / "features_cache.pkl")

    # ── Build or load feature dataset ────────────────────────────────────────
    if args.train_only and os.path.exists(feat_path + "_X.npy"):
        log.info("Loading cached features...")
        X = np.load(feat_path + "_X.npy")
        y = np.load(feat_path + "_y.npy")
        log.info(f"Loaded: X={X.shape}, y={y.shape}")
    else:
        log.info(f"Building feature dataset from {args.samples} HaluEval samples...")
        log.info("This uses all pipeline models sequentially — takes 2–4 hours.")
        log.info("TIP: Run with nohup and check logs/phase4.log for progress")

        from src.phase4_ensemble.features import build_feature_dataset
        X, y = build_feature_dataset(
            save_path=feat_path,
            max_samples=args.samples,
            cache_path=cache_path,
        )

    # ── Train meta-learner ────────────────────────────────────────────────────
    log.info(f"\nTraining meta-learner on {len(X):,} feature vectors...")
    from src.phase4_ensemble.meta_learner import train_meta_learner
    result = train_meta_learner(X, y, n_epochs=args.epochs)

    metrics = result["results"]
    log.info(f"\nFinal Results:")
    log.info(f"  AUROC:     {metrics['test_auroc']:.4f}")
    log.info(f"  F1:        {metrics['test_f1']:.4f}")
    log.info(f"  ECE:       ...")

    # ── Optionally build DPO dataset ──────────────────────────────────────────
    if args.build_dpo:
        log.info("\nBuilding DPO preference dataset for Phase 5...")
        from src.phase5_dpo.build_dataset import build_dpo_dataset
        dataset = build_dpo_dataset()
        log.info(f"DPO dataset: {len(dataset['train']):,} train, {len(dataset['val']):,} val")
        log.info("Upload data/processed/dpo_dataset.json to Google Colab for Phase 5")

    log.info("\n✓ Phase 4 complete!")
    log.info("Next steps:")
    log.info("  Phase 5 (Colab): open colab/phase5_phi2_dpo.ipynb")
    log.info("  Skip to demo:    python demo/app.py")
    log.info("  Run evaluation:  python scripts/run_all_eval.py")


if __name__ == "__main__":
    main()
