import sys, pickle, warnings
sys.path.insert(0, r'e:\HalluDet\Hallu_Detect')
warnings.filterwarnings('ignore')
import numpy as np, torch

# Inspect the ensemble feature distribution in training data
with open(r'e:\HalluDet\Hallu_Detect\data\processed\features_cache.pkl','rb') as f:
    cache = pickle.load(f)
X = cache['X']; y = cache['y']

print('='*65)
print('ROOT CAUSE ANALYSIS: Why demo scores all cluster at ~0.98')
print('='*65)

# Show F2 (ens_mean_ent) distribution in training data
f2_fac = X[y==0, 2]; f2_hal = X[y==1, 2]
print(f'Training F2 (ens_mean_ent):')
print(f'  Factual:    min={f2_fac.min():.4f} max={f2_fac.max():.4f} mean={f2_fac.mean():.4f}')
print(f'  Hallucin:   min={f2_hal.min():.4f} max={f2_hal.max():.4f} mean={f2_hal.mean():.4f}')
print()

# If demo ens_mean_ent ~= 0.53 for both, what does MLP output?
from src.phase4_ensemble.meta_learner import MetaLearnerNet, ACTIVE_FEATURES, ACTIVE_FEATURE_DIM, MODEL_DIR

with open(r'e:\HalluDet\Hallu_Detect\models\meta_learner\feature_scaler.pkl','rb') as f:
    scaler=pickle.load(f)
with open(r'e:\HalluDet\Hallu_Detect\models\meta_learner\platt_scaler.pkl','rb') as f:
    platt=pickle.load(f)
net=MetaLearnerNet(in_dim=ACTIVE_FEATURE_DIM)
net.load_state_dict(torch.load(str(MODEL_DIR/'meta_learner.pt'),map_location='cpu',weights_only=True))
net.eval()

print('If ens_mean_ent (F2) is near 0.53 for demo inputs:')
for test_f2 in [0.30, 0.40, 0.50, 0.53, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
    feat = np.array([[test_f2]], dtype=np.float32)
    sc = scaler.transform(feat).astype(np.float32)
    with torch.no_grad(): raw=net(torch.tensor(sc)).item()
    cal=platt.predict_proba([[raw]])[0][1]
    print(f'  ens_mean_ent={test_f2:.2f}  raw_mlp={raw:.6f}  cal={cal:.6f}')

print()
print('='*65)
print('CHECKING: What F2 range the scaler was fit on')
print('='*65)
print(f'Scaler mean: {scaler.mean_}')
print(f'Scaler scale: {scaler.scale_}')
print()

# Check: was ens trained with load_bart=False? That changes F2!
# In training (build_feature_dataset) ALL 3 models loaded
# But in demo: EnsembleDisagreement(load_bart=False)
# With only M1 (fine-tuned DeBERTa) and M2 (cross-encoder):
# M3 contributes 0.5 fallback, dragging mean toward 0.5!
print('CRITICAL: Demo uses load_bart=False (see app.py line 79)')
print('This means M3=BART contributes 0.5 (fallback) instead of real score')
print()
print('Training used ALL 3 models. Demo uses only M1+M2+M3_fallback.')
print()
print('With M3 always=0.5 (fallback):')
print('  ens_mean_ent = (M1_score + M2_score + 0.5) / 3')
print()
print('If M1 high (factual in training domain): ~(0.99+X+0.5)/3')
print('But at demo M1 also saturates near-zero due to OOD! So:')
print('  For FACTUAL demo claim: (~0.01 + M2 + 0.5)/3 = ~0.3-0.5')
print('  For HALLUCIN demo claim: (~0.01 + M2_low + 0.5)/3 = ~0.2-0.4')
print()
print('Both land in the [0.3-0.5] zone where MLP outputs ~0.9+ (hallucinated!)')
