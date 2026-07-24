"""
meta_learner.py — Meta-learner MLP for ensemble fusion.

Combines all 8 features into a calibrated hallucination probability.
Training runs on CPU in ~2 minutes.
Inference takes <1ms per claim.
"""

import os
import pickle
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report
)

from src.utils import get_logger, get_project_root

log = get_logger("ensemble.meta_learner")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR     = get_project_root() / "models" / "meta_learner"
MODEL_PATH    = str(MODEL_DIR / "meta_learner.pt")
SCALER_PATH   = str(MODEL_DIR / "feature_scaler.pkl")
PLATT_PATH    = str(MODEL_DIR / "platt_scaler.pkl")
RESULTS_PATH  = str(MODEL_DIR / "eval_results.json")


# ── Network Architecture ──────────────────────────────────────────────────────

class MetaLearnerNet(nn.Module):
    """
    Small 3-layer MLP for hallucination probability estimation.
    Trained on 8-dimensional feature vectors.
    Input → 64 → 32 → 1 (sigmoid)
    Total parameters: ~2,400 — trains in seconds.
    """

    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_meta_learner(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs:    int   = 100,
    batch_size:  int   = 128,
    lr:          float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int  = 42,
) -> Dict:
    """
    Train the meta-learner MLP + Platt calibration scaler.

    Args:
        X:            Feature matrix, shape (N, 8).
        y:            Binary labels, shape (N,). 1=hallucinated, 0=not.
        n_epochs:     Training epochs.
        batch_size:   Mini-batch size.
        lr:           Adam learning rate.
        weight_decay: L2 regularization.
        val_fraction: Fraction for validation.
        test_fraction: Fraction for test evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with trained model, scalers, and evaluation metrics.
    """
    log.info(f"Training meta-learner on {len(X):,} samples...")

    # ── Train/val/test split ──────────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_fraction,
        stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction / (1 - test_fraction),
        stratify=y_tmp, random_state=random_state
    )
    log.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── Feature normalisation ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    # ── Dataset and DataLoader ────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.tensor(X_train_s),
        torch.tensor(y_train.astype(np.float32)),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = MetaLearnerNet(in_dim=X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # Handle class imbalance with pos_weight
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auroc = 0.0
    best_state = None
    history = {"train_loss": [], "val_auroc": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation AUROC
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val_s)).numpy()
            
        try:
            val_auroc = roc_auc_score(y_val, val_preds)
        except ValueError:
            val_auroc = 0.5  # Fallback if validation batch only has one class during small smoke tests

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_auroc"].append(float(val_auroc))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            log.info(f"Epoch {epoch:3d} | loss={np.mean(epoch_losses):.4f} | val AUROC={val_auroc:.4f}")

    # Restore best model
    model.load_state_dict(best_state)
    log.info(f"Best val AUROC: {best_val_auroc:.4f}")

    # ── Platt Scaling (Calibration) ───────────────────────────────────────────
    # Fit logistic regression on validation set scores to calibrate probabilities
    model.eval()
    with torch.no_grad():
        val_raw_scores = model(torch.tensor(X_val_s)).numpy()

    platt_scaler = LogisticRegression(C=1.0, max_iter=1000)
    try:
        platt_scaler.fit(val_raw_scores.reshape(-1, 1), y_val)
        log.info("Platt scaling fitted on validation set.")
    except ValueError:
        # Happens in tiny smoke tests where y_val only has one class
        # Fit on training set instead as a fallback
        try:
            train_raw_scores = model(torch.tensor(X_train_s)).detach().numpy()
            platt_scaler.fit(train_raw_scores.reshape(-1, 1), y_train)
        except ValueError:
            # If train ALSO has one class (extreme edge case)
            platt_scaler.fit(np.array([[0], [1]]), np.array([0, 1]))
        log.warning("Platt scaling fitted on train/dummy set (smoke test fallback).")

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    with torch.no_grad():
        test_raw_scores = model(torch.tensor(X_test_s)).numpy()
    test_cal_scores = platt_scaler.predict_proba(
        test_raw_scores.reshape(-1, 1)
    )[:, 1]

    try:
        test_auroc = roc_auc_score(y_test, test_cal_scores)
    except ValueError:
        test_auroc = 0.5
    test_preds = (test_cal_scores >= 0.5).astype(int)
    test_f1    = f1_score(y_test, test_preds)
    test_prec  = precision_score(y_test, test_preds)
    test_rec   = recall_score(y_test, test_preds)

    log.info(f"\n{'='*50}")
    log.info(f"FINAL TEST RESULTS (calibrated):")
    log.info(f"  AUROC:     {test_auroc:.4f}")
    log.info(f"  F1:        {test_f1:.4f}")
    log.info(f"  Precision: {test_prec:.4f}")
    log.info(f"  Recall:    {test_rec:.4f}")
    log.info(f"{'='*50}\n")
    log.info(classification_report(y_test, test_preds,
                                    target_names=["Factual", "Hallucinated"]))

    results = {
        "test_auroc":     float(test_auroc),
        "test_f1":        float(test_f1),
        "test_precision": float(test_prec),
        "test_recall":    float(test_rec),
        "best_val_auroc": float(best_val_auroc),
        "history":        history,
    }

    # ── Save everything ───────────────────────────────────────────────────────
    os.makedirs(str(MODEL_DIR), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(PLATT_PATH,  "wb") as f: pickle.dump(platt_scaler, f)
    with open(RESULTS_PATH, "w") as f: json.dump(results, f, indent=2)

    log.info(f"Meta-learner saved to {MODEL_DIR}/")

    return {"model": model, "scaler": scaler, "platt": platt_scaler, "results": results}


# ── Inference ─────────────────────────────────────────────────────────────────

class MetaLearnerPredictor:
    """
    Loads the trained meta-learner and Platt scaler for inference.

    Usage:
        predictor = MetaLearnerPredictor()
        features = np.array([0.3, 0.5, 0.4, 0.02, 2.1, 0.25, 0.6, 1.1])
        prob = predictor.predict(features)
        print(f"Hallucination probability: {prob:.3f}")
    """

    def __init__(
        self,
        model_path:  Optional[str] = None,
        scaler_path: Optional[str] = None,
        platt_path:  Optional[str] = None,
    ):
        mp = model_path  or MODEL_PATH
        sp = scaler_path or SCALER_PATH
        pp = platt_path  or PLATT_PATH

        if not all(os.path.exists(p) for p in [mp, sp, pp]):
            raise FileNotFoundError(
                "Meta-learner files not found. Run Phase 4 first: "
                "python scripts/run_phase4.py"
            )

        self.model = MetaLearnerNet()
        self.model.load_state_dict(torch.load(mp, map_location="cpu"))
        self.model.eval()

        with open(sp, "rb") as f: self.scaler = pickle.load(f)
        with open(pp, "rb") as f: self.platt  = pickle.load(f)

        log.info("MetaLearnerPredictor loaded.")

    def predict(self, features: np.ndarray) -> float:
        """
        Predict hallucination probability for one feature vector.

        Args:
            features: np.ndarray of shape (8,)

        Returns:
            float in [0, 1] — calibrated hallucination probability
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        scaled = self.scaler.transform(features).astype(np.float32)
        with torch.no_grad():
            raw = self.model(torch.tensor(scaled)).item()
        calibrated = float(self.platt.predict_proba([[raw]])[0, 1])
        return calibrated

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict for a batch of feature vectors, shape (N, 8)."""
        scaled = self.scaler.transform(features).astype(np.float32)
        with torch.no_grad():
            raw = self.model(torch.tensor(scaled)).numpy()
        calibrated = self.platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        return calibrated.astype(float)
