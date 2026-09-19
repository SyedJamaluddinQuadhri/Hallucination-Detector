"""
fix_platt_scaler.py — Refit meta-learner on 4 strong features from cached data.

This script:
  1. Loads cached features from data/processed/ensemble_features_{X,y}.npy
  2. Applies the ACTIVE_FEATURES=[0,1,2,5] mask to drop 4 near-useless features
  3. Retrains the MLP meta-learner (in_dim=4) on CPU — takes ~1-2 minutes
  4. Refits StandardScaler on the 4-feature training set
  5. Refits Platt logistic regression with class_weight='balanced'
  6. Overwrites models/meta_learner/{meta_learner.pt, feature_scaler.pkl, platt_scaler.pkl}
  7. Prints before/after stats so you can verify the fix

Run with:
    .\\halludet_env\\Scripts\\python.exe scripts/fix_platt_scaler.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, get_project_root
from src.phase4_ensemble.meta_learner import (
    train_meta_learner,
    ACTIVE_FEATURES,
    ACTIVE_FEATURE_DIM,
    MODEL_PATH,
    SCALER_PATH,
    PLATT_PATH,
)

log = get_logger("fix_platt_scaler")


def main():
    root = get_project_root()
    X_path = str(root / "data" / "processed" / "ensemble_features_X.npy")
    y_path = str(root / "data" / "processed" / "ensemble_features_y.npy")

    if not os.path.exists(X_path):
        log.error(
            f"Feature cache not found at '{X_path}'.\n"
            "Run Phase 4 first: python scripts/run_phase4.py --train-only"
        )
        sys.exit(1)

    # ── Load cached features ────────────────────────────────────────────────
    log.info(f"Loading cached feature matrix from {X_path} ...")
    X = np.load(X_path)
    y = np.load(y_path)
    log.info(f"Loaded X={X.shape}, y={y.shape}")
    log.info(
        f"Label balance: {(y==0).sum()} factual / {(y==1).sum()} hallucinated "
        f"({y.mean()*100:.1f}% hallucinated)"
    )

    # ── Diagnose BEFORE fix ─────────────────────────────────────────────────
    log.info("\n--- BEFORE FIX ---")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(PLATT_PATH):
        try:
            from src.phase4_ensemble.meta_learner import MetaLearnerNet, ACTIVE_FEATURES
            import pickle, warnings
            old_model = MetaLearnerNet(in_dim=8)  # old 8-dim model
            old_state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            # Only load if shapes match (old model may be 8-dim)
            old_model.load_state_dict(old_state)
            old_model.eval()
            with open(SCALER_PATH, "rb") as f:
                old_scaler = pickle.load(f)
            with open(PLATT_PATH, "rb") as f:
                old_platt = pickle.load(f)

            # Use all 8 features for old model
            from sklearn.model_selection import train_test_split
            _, X_test_old, _, y_test_old = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )
            X_test_s_old = old_scaler.transform(X_test_old).astype("float32")
            with torch.no_grad():
                raw_old = old_model(torch.tensor(X_test_s_old)).numpy()
            cal_old = old_platt.predict_proba(raw_old.reshape(-1, 1))[:, 1]
            log.info(
                f"Old calibrated scores — mean={cal_old.mean():.4f}  "
                f"std={cal_old.std():.4f}  min={cal_old.min():.4f}  max={cal_old.max():.4f}"
            )
            log.info(f"Old Platt coef={old_platt.coef_}  intercept={old_platt.intercept_}")
            try:
                log.info(f"Old AUROC on test set: {roc_auc_score(y_test_old, cal_old):.4f}")
            except Exception:
                pass
        except Exception as e:
            log.warning(f"Could not load old model for comparison: {e}")
    else:
        log.info("(No existing model found — skipping before-fix comparison)")

    # ── Refit with correct settings ─────────────────────────────────────────
    log.info(
        f"\n--- REFITTING META-LEARNER ---\n"
        f"  Feature mask : {ACTIVE_FEATURES} (4 of 8 features)\n"
        f"  Platt        : class_weight='balanced'\n"
        f"  Epochs       : 100 (CPU, ~1-2 min)\n"
    )

    result = train_meta_learner(
        X=X,
        y=y,
        n_epochs=100,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        val_fraction=0.15,
        test_fraction=0.15,
        random_state=42,
    )

    # ── Diagnose AFTER fix ──────────────────────────────────────────────────
    log.info("\n--- AFTER FIX ---")
    with open(PLATT_PATH, "rb") as f:
        new_platt = pickle.load(f)
    log.info(
        f"New Platt coef={new_platt.coef_}  intercept={new_platt.intercept_}"
    )

    # Quick sanity check on the 4-feature test slice
    from sklearn.model_selection import train_test_split
    from src.phase4_ensemble.meta_learner import MetaLearnerNet
    X4 = X[:, ACTIVE_FEATURES]
    _, X4_test, _, y4_test = train_test_split(
        X4, y, test_size=0.15, stratify=y, random_state=42
    )
    with open(SCALER_PATH, "rb") as f:
        new_scaler = pickle.load(f)
    new_model = MetaLearnerNet(in_dim=ACTIVE_FEATURE_DIM)
    new_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    new_model.eval()
    X4_s = new_scaler.transform(X4_test).astype("float32")
    with torch.no_grad():
        raw_new = new_model(torch.tensor(X4_s)).numpy()
    cal_new = new_platt.predict_proba(raw_new.reshape(-1, 1))[:, 1]

    log.info(
        f"New calibrated scores — mean={cal_new.mean():.4f}  "
        f"std={cal_new.std():.4f}  min={cal_new.min():.4f}  max={cal_new.max():.4f}"
    )
    try:
        new_auroc = roc_auc_score(y4_test, cal_new)
        log.info(f"New AUROC on test set: {new_auroc:.4f}")
    except Exception as e:
        log.warning(f"Could not compute AUROC: {e}")

    log.info(
        "\n✓ Platt scaler, StandardScaler, and MLP re-saved to models/meta_learner/\n"
        "  Next: run scripts/diagnose_pipeline.py to verify demo cases"
    )


if __name__ == "__main__":
    main()
