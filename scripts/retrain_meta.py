"""
retrain_meta.py -- Retrain the meta-learner with all 8 features.

Run from project root:
    python scripts/retrain_meta.py
"""
import sys, pickle, warnings
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
warnings.filterwarnings('ignore')
import numpy as np

print('Loading cached features...')
with open(r'e:\HalluDet\Hallu_Detect\data\processed\features_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
X = cache['X']
y = cache['y'].astype(np.float32)
print(f'X shape: {X.shape}, y shape: {y.shape}')
ld = dict(zip(*np.unique(y, return_counts=True)))
print(f'Label dist: {ld}')

from src.phase4_ensemble.meta_learner import train_meta_learner, ACTIVE_FEATURES, ACTIVE_FEATURE_DIM

print(f'Training with ACTIVE_FEATURES={ACTIVE_FEATURES} (dim={ACTIVE_FEATURE_DIM})')
results = train_meta_learner(X, y, n_epochs=150, batch_size=128, lr=1e-3, weight_decay=1e-4)

r = results['results']
print()
print('=' * 55)
print('RETRAIN COMPLETE')
print(f'  Test AUROC:  {r["test_auroc"]:.4f}')
print(f'  Test F1:     {r["test_f1"]:.4f}')
print(f'  Precision:   {r["test_precision"]:.4f}')
print(f'  Recall:      {r["test_recall"]:.4f}')
print(f'  Val AUROC:   {r["best_val_auroc"]:.4f}')
print('=' * 55)
