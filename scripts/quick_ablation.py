# Quick ablation on cached training features showing each phase contribution
import sys, pickle, warnings
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

with open(r'e:\HalluDet\Hallu_Detect\data\processed\features_cache.pkl','rb') as f:
    cache = pickle.load(f)
X = cache['X']; y = cache['y'].astype(int)

print('=' * 60)
print('ABLATION STUDY (cross-validated AUROC, 5-fold):')
print('=' * 60)

configs = {
    'Phase1 NLI only         ': [0, 1],
    'Phase2 Ensemble only    ': [2, 3],
    'Phase2 Perplexity only  ': [4],
    'Phase3 RAV only         ': [5, 6],
    'Phase1+2 NLI+Ensemble   ': [0, 1, 2, 3],
    'Phase1+3 NLI+RAV        ': [0, 1, 5, 6],
    'Phase1+2+3 (no ppl/ent) ': [0, 1, 2, 3, 5, 6],
    'All 8 features           ': [0, 1, 2, 3, 4, 5, 6, 7],
}

for name, feats in configs.items():
    Xf = X[:, feats]
    sc = StandardScaler()
    Xfs = sc.fit_transform(Xf)
    lr = LogisticRegression(max_iter=500, class_weight='balanced')
    scores = cross_val_score(lr, Xfs, y, cv=5, scoring='roc_auc')
    print(f'  {name}: AUROC={scores.mean():.4f} +/- {scores.std():.4f}')

print()
print('Single-feature discriminability (raw AUROC):')
fn = ['F0_nli_ent','F1_nli_con','F2_ens_mn','F3_disagr','F4_logppl','F5_rav','F6_ret','F7_ent']
for i,nm in enumerate(fn):
    a = roc_auc_score(y, X[:,i]); a = max(a, 1-a)
    print(f'  {nm}: {a:.4f}')
