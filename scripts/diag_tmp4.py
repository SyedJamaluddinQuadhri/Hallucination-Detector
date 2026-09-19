# First inspect what the existing cached features look like for F0/F5 which were abandoned
import sys, pickle, warnings
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score

with open(r'e:\HalluDet\Hallu_Detect\data\processed\features_cache.pkl','rb') as f:
    cache = pickle.load(f)
X = cache['X']; y = cache['y']

fn = ['F0_nli_ent','F1_nli_con','F2_ens_mn','F3_disagr','F4_logppl','F5_rav','F6_ret','F7_ent']
print('ALL 8 features AUROC and separability:')
for i,nm in enumerate(fn):
    try:
        a=roc_auc_score(y, X[:,i]); a=max(a,1-a)
        print(f'  {nm}: AUROC={a:.4f}')
    except: print(f'  {nm}: ERROR')

# Multi-feature combination AUROC using logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Try different feature combinations
combinations = {
    'F2_only': [2],
    'F0+F2': [0,2],
    'F0+F2+F5': [0,2,5],
    'F0+F1+F2+F5': [0,1,2,5],
    'all_8': [0,1,2,3,4,5,6,7],
}
print()
print('Cross-validated AUROC for different feature combos (LogReg):')
for name, feats in combinations.items():
    Xf = X[:, feats]
    sc = StandardScaler()
    Xfs = sc.fit_transform(Xf)
    lr = LogisticRegression(max_iter=500)
    scores = cross_val_score(lr, Xfs, y.astype(int), cv=5, scoring='roc_auc')
    print(f'  {name:<20}: mean={scores.mean():.4f} std={scores.std():.4f}')
