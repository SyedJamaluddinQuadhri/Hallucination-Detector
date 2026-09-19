import sys, pickle, warnings
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
warnings.filterwarnings('ignore')
import numpy as np, torch

print('='*65)
print('STEP 1: Feature cache')
print('='*65)
with open(r'e:\HalluDet\Hallu_Detect\data\processed\features_cache.pkl','rb') as f:
    cache = pickle.load(f)
X = cache['X']; y = cache['y']
print(f'X shape: {X.shape}, y shape: {y.shape}')
ld = dict(zip(*np.unique(y, return_counts=True)))
print(f'Label dist: {ld}')
print(f'% hallucinated: {ld.get(1.0,0)/len(y)*100:.1f}%')
fn = ['F0_nli_ent','F1_nli_con','F2_ens_mn','F3_disagr','F4_logppl','F5_rav','F6_ret','F7_ent']
fac = X[y==0]; hal = X[y==1]
print()
print(f'{"Feature":<14} {"Fact_mean":>10} {"Hall_mean":>10} {"Diff":>9} {"F_std":>9} {"H_std":>9}')
print('-'*65)
for i,nm in enumerate(fn):
    fm=fac[:,i].mean(); hm=hal[:,i].mean(); d=abs(fm-hm)
    fs=fac[:,i].std(); hs=hal[:,i].std()
    print(f'{nm:<14} {fm:>10.4f} {hm:>10.4f} {d:>9.4f} {fs:>9.4f} {hs:>9.4f}')

from sklearn.metrics import roc_auc_score
print()
print('Per-feature AUROC:')
for i,nm in enumerate(fn):
    try:
        a=roc_auc_score(y, X[:,i]); a=max(a,1-a)
        print(f'  {nm:<14}: AUROC={a:.4f}')
    except: print(f'  {nm:<14}: ERROR')

print()
print('='*65)
print('STEP 2: Platt scaler internals')
print('='*65)
with open(r'e:\HalluDet\Hallu_Detect\models\meta_learner\platt_scaler.pkl','rb') as f:
    platt=pickle.load(f)
print(f'coef_={platt.coef_}  intercept_={platt.intercept_}')
coef=platt.coef_[0][0]; ic=platt.intercept_[0]
print(f'Decision threshold (raw=0.5 calibrated): {-ic/coef:.6f}')
print('Raw->Cal sweep:')
for raw in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    cal=platt.predict_proba([[raw]])[0][1]
    print(f'  raw={raw:.1f} -> cal={cal:.6f}')

print()
print('='*65)
print('STEP 3: MLP raw output range on training data')
print('='*65)
from src.phase4_ensemble.meta_learner import MetaLearnerNet, ACTIVE_FEATURES, ACTIVE_FEATURE_DIM, MODEL_DIR
with open(r'e:\HalluDet\Hallu_Detect\models\meta_learner\feature_scaler.pkl','rb') as f:
    scaler=pickle.load(f)
net=MetaLearnerNet(in_dim=ACTIVE_FEATURE_DIM)
net.load_state_dict(torch.load(str(MODEL_DIR/'meta_learner.pt'),map_location='cpu',weights_only=True))
net.eval()
Xa=X[:,ACTIVE_FEATURES]; Xs=scaler.transform(Xa).astype(np.float32)
with torch.no_grad(): rp=net(torch.tensor(Xs)).numpy()
print(f'Active features: {ACTIVE_FEATURES} = {[fn[i] for i in ACTIVE_FEATURES]}')
print(f'MLP raw range: min={rp.min():.6f} max={rp.max():.6f} std={rp.std():.6f}')
print(f'MLP raw factual  mean={rp[y==0].mean():.6f} std={rp[y==0].std():.6f}')
print(f'MLP raw hallucin mean={rp[y==1].mean():.6f} std={rp[y==1].std():.6f}')
cp=platt.predict_proba(rp.reshape(-1,1))[:,1]
print(f'Cal range: min={cp.min():.6f} max={cp.max():.6f} std={cp.std():.6f}')
print(f'Cal factual  mean={cp[y==0].mean():.6f} std={cp[y==0].std():.6f}')
print(f'Cal hallucin mean={cp[y==1].mean():.6f} std={cp[y==1].std():.6f}')
print(f'AUROC raw={roc_auc_score(y,rp):.4f}  cal={roc_auc_score(y,cp):.4f}')
