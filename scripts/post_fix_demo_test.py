"""
post_fix_demo_test.py -- Verify the fix actually produces real separation
on demo examples by checking raw feature vectors and MLP outputs.

Run from project root:
    python scripts/post_fix_demo_test.py
"""
import sys, pickle, warnings
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
warnings.filterwarnings('ignore')
import numpy as np
import torch

print('=' * 70)
print('POST-FIX VERIFICATION: Feature vectors and MLP output per demo claim')
print('=' * 70)

# Load new scaler + MLP + Platt (just saved)
from src.phase4_ensemble.meta_learner import (
    MetaLearnerNet, ACTIVE_FEATURES, ACTIVE_FEATURE_DIM, MODEL_DIR
)
with open(str(MODEL_DIR / 'feature_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(str(MODEL_DIR / 'platt_scaler.pkl'), 'rb') as f:
    platt = pickle.load(f)

net = MetaLearnerNet(in_dim=ACTIVE_FEATURE_DIM)
net.load_state_dict(torch.load(str(MODEL_DIR / 'meta_learner.pt'), map_location='cpu', weights_only=True))
net.eval()

print(f'\nACTIVE_FEATURES = {ACTIVE_FEATURES}  (dim={ACTIVE_FEATURE_DIM})')
print(f'Scaler mean: {scaler.mean_.round(4)}')
print()

# === Scenario: What if F2 collapses to 0.5 (BART disabled) but F0 is correct? ===
print('SCENARIO A: If NLI (F0) has correct signal, F2 collapsed (BART off)')
print('  Factual claim: F0=0.95, F1=0.02, F2=0.52 (BART off)')
print('  Hallucin claim: F0=0.01, F1=0.85, F2=0.48 (BART off)')
print()

test_cases = [
    ('Factual  (correct NLI)', np.array([0.95, 0.02, 0.52, 0.01, 4.5, 0.70, 0.55, 0.69], dtype=np.float32), 0),
    ('Hallucin (correct NLI)', np.array([0.01, 0.85, 0.48, 0.04, 4.2, 0.10, 0.55, 1.10], dtype=np.float32), 1),
]

for name, feat, label in test_cases:
    scaled = scaler.transform(feat.reshape(1, -1)).astype(np.float32)
    with torch.no_grad():
        raw = net(torch.tensor(scaled)).item()
    cal = platt.predict_proba([[raw]])[0][1]
    fn = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7']
    feat_str = ' '.join(f'{fn[i]}={feat[i]:.2f}' for i in range(8))
    print(f'  {name}  raw={raw:.4f}  cal={cal:.4f}  (expected: {"HALL" if label else "FACT"})')
    print(f'    features: {feat_str}')

# === Scenario: Both F0 AND F2 collapsed (worst case) ===
print()
print('SCENARIO B: Both NLI (F0) AND ensemble (F2) collapsed OOD')
print('  Both factual and hallucinated score near 0.5 on F0 and F2')

test_cases_b = [
    ('Factual  (both OOD)', np.array([0.50, 0.30, 0.52, 0.01, 4.5, 0.70, 0.55, 0.69], dtype=np.float32), 0),
    ('Hallucin (both OOD)', np.array([0.48, 0.32, 0.50, 0.02, 4.3, 0.12, 0.55, 1.10], dtype=np.float32), 1),
]

for name, feat, label in test_cases_b:
    scaled = scaler.transform(feat.reshape(1, -1)).astype(np.float32)
    with torch.no_grad():
        raw = net(torch.tensor(scaled)).item()
    cal = platt.predict_proba([[raw]])[0][1]
    print(f'  {name}  raw={raw:.4f}  cal={cal:.4f}  (expected: {"HALL" if label else "FACT"})')

print()
print('=' * 70)
print('PLATT SCALER INTERNALS (new model):')
print(f'  coef_={platt.coef_}  intercept_={platt.intercept_}')
coef = platt.coef_[0][0]
ic = platt.intercept_[0]
threshold = -ic / coef
print(f'  Decision threshold (raw MLP -> cal=0.5): {threshold:.4f}')
print()
print('Raw -> Calibrated mapping:')
for raw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    cal = platt.predict_proba([[raw]])[0][1]
    print(f'  raw={raw:.1f} -> cal={cal:.4f}')

print()
print('=' * 70)
print('TRAINING DATA VERIFICATION: MLP output distribution')
print('=' * 70)

with open(r'e:\HalluDet\Hallu_Detect\data\processed\features_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
X = cache['X']
y = cache['y']

X_scaled = scaler.transform(X[:, ACTIVE_FEATURES]).astype(np.float32)
with torch.no_grad():
    raw_preds = net(torch.tensor(X_scaled)).numpy()
cal_preds = platt.predict_proba(raw_preds.reshape(-1, 1))[:, 1]

from sklearn.metrics import roc_auc_score, brier_score_loss
print(f'MLP raw: min={raw_preds.min():.4f} max={raw_preds.max():.4f} std={raw_preds.std():.4f}')
print(f'  Factual  raw: mean={raw_preds[y==0].mean():.4f} std={raw_preds[y==0].std():.4f}')
print(f'  Hallucin raw: mean={raw_preds[y==1].mean():.4f} std={raw_preds[y==1].std():.4f}')
print(f'Calibrated: min={cal_preds.min():.4f} max={cal_preds.max():.4f} std={cal_preds.std():.4f}')
print(f'  Factual  cal: mean={cal_preds[y==0].mean():.4f} std={cal_preds[y==0].std():.4f}')
print(f'  Hallucin cal: mean={cal_preds[y==1].mean():.4f} std={cal_preds[y==1].std():.4f}')

# ECE calculation
def compute_ece(y_true, y_scores, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_scores >= lo) & (y_scores < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_scores[mask].mean())
    return float(ece)

auroc = roc_auc_score(y, cal_preds)
ece = compute_ece(y, cal_preds)
print(f'\nAUROC (calibrated): {auroc:.4f}')
print(f'ECE:                {ece:.4f}  (target < 0.06)')
print(f'Brier score:        {brier_score_loss(y, cal_preds):.4f}')
