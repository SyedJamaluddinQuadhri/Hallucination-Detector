"""
deep_debug.py -- 4-step root cause debugger for constant hal_score bug.
"""
import sys, pickle, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import numpy as np
import torch

print("\n" + "="*65)
print("STEP 0  Loading models")
print("="*65)

from src.phase1_nli.predict import NLIPredictor
from src.phase2_uncertainty.ensemble import EnsembleDisagreement
from src.phase2_uncertainty.perplexity import TinyLLaMaPerplexity
from src.phase3_rav.retriever import LightweightRAV
from src.phase4_ensemble.meta_learner import MetaLearnerPredictor, MetaLearnerNet
from src.phase4_ensemble.meta_learner import MODEL_DIR, ACTIVE_FEATURES, ACTIVE_FEATURE_DIM
import spacy

nli  = NLIPredictor()
ens  = EnsembleDisagreement(nli_predictor=nli, load_small=True, load_bart=True)
rav  = LightweightRAV(nli_predictor=nli)
ppl  = TinyLLaMaPerplexity()
nlp  = spacy.load("en_core_web_sm")
meta = MetaLearnerPredictor()

print(f"\nACTIVE_FEATURES = {ACTIVE_FEATURES}  (dim={ACTIVE_FEATURE_DIM})")

CASES = [
    ("Einstein FACTUAL", "Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.", "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.", 0),
    ("Einstein HALL.", "Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.", "Albert Einstein won the Nobel Prize in Physics in 1922 for his theory of relativity.", 1),
    ("Hamlet FACTUAL", "Hamlet is a tragedy written by William Shakespeare.", "Hamlet was written by William Shakespeare, most likely around 1600-1601.", 0),
    ("Hamlet HALL.", "Hamlet is a tragedy written by William Shakespeare.", "Hamlet was written by Christopher Marlowe in 1589.", 1),
    ("Australia FACTUAL", "The capital city of Australia is Canberra.", "The capital of Australia is Canberra.", 0),
    ("Australia HALL.", "The capital city of Australia is Canberra.", "The capital of Australia is Sydney, which is also the largest city.", 1),
    ("DNA FACTUAL", "DNA stands for Deoxyribonucleic Acid.", "DNA stands for Deoxyribonucleic Acid.", 0),
    ("DNA HALL.", "DNA stands for Deoxyribonucleic Acid.", "DNA stands for Dynamic Nucleic Assembly.", 1),
]

print("\n" + "="*65)
print("STEP 2  Full 8-feature vectors")
print("="*65)
print(f"{'Case':<22} {'F0_ent':>7} {'F1_con':>7} {'F2_ens':>7} {'F3_dis':>7} {'F4_ppl':>7} {'F5_rav':>7} {'F6_ret':>7} {'F7_ent':>7}")
print("-"*90)

all_feats = []
for label, knowledge, claim, y in CASES:
    nli_r = nli.score_claim(knowledge, claim)
    ens_r = ens.score(knowledge, claim)
    rav_r = rav.verify_claim(claim)
    ppl_r = ppl.score_claim(claim)
    doc   = nlp(claim)
    feat  = np.array([
        nli_r["probs"]["entailment"],
        nli_r["probs"]["contradiction"],
        ens_r["mean_entailment"],
        ens_r["disagreement"],
        min(ppl_r["log_perplexity"], 10.0),
        rav_r["max_ent"],
        max((e["retrieval_score"] for e in rav_r.get("evidence_used", [])), default=0.0),
        float(np.log1p(len(doc.ents))),
    ], dtype=np.float32)
    all_feats.append(feat)
    print(f"{label:<22} {feat[0]:>7.4f} {feat[1]:>7.4f} {feat[2]:>7.4f} {feat[3]:>7.4f} {feat[4]:>7.4f} {feat[5]:>7.4f} {feat[6]:>7.4f} {feat[7]:>7.4f}")

print("\n-- Einstein pair DIFF --")
fn = ["F0_nli_ent","F1_nli_con","F2_ens_mn","F3_disagr","F4_logppl","F5_rav","F6_ret","F7_ent"]
e_fac, e_hal = all_feats[0], all_feats[1]
for i,(nm,fv,hv) in enumerate(zip(fn, e_fac, e_hal)):
    diff = abs(float(fv)-float(hv))
    sig  = "YES" if diff > 0.05 else "no "
    act  = " <<ACTIVE" if i in ACTIVE_FEATURES else ""
    print(f"  {nm:<12} fac={fv:.4f} hal={hv:.4f} diff={diff:.4f}  signal={sig}{act}")

print("\n" + "="*65)
print("STEP 1  Raw MLP output BEFORE Platt scaling")
print("="*65)
net = MetaLearnerNet(in_dim=ACTIVE_FEATURE_DIM)
net.load_state_dict(torch.load(str(MODEL_DIR / "meta_learner.pt"), map_location="cpu"))
net.eval()
with open(str(MODEL_DIR / "feature_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

print(f"\n{'Case':<22} {'Active_feat':>10}  {'Raw_MLP':>9}  {'Calibrated':>10}  {'Label':>5}")
print("-"*65)
for i,(label,_,_,y) in enumerate(CASES):
    feat   = all_feats[i]
    active = feat[ACTIVE_FEATURES]
    scaled = scaler.transform(active.reshape(1,-1))
    with torch.no_grad():
        raw = net(torch.tensor(scaled, dtype=torch.float32)).item()
    cal = meta.predict(feat)
    af  = " ".join(f"{v:.4f}" for v in active)
    print(f"{label:<22} [{af}]  {raw:>9.4f}  {cal:>10.4f}  {'HAL' if y else 'FAC'}")

print("\n" + "="*65)
print("STEP 3  Platt scaler internals")
print("="*65)
with open(str(MODEL_DIR / "platt_scaler.pkl"), "rb") as f:
    platt = pickle.load(f)
print(f"coef_={platt.coef_}  intercept_={platt.intercept_}")
coef = platt.coef_[0][0]; intercept = platt.intercept_[0]
threshold = -intercept / coef
print(f"Decision threshold (raw where cal=0.5): {threshold:.4f}")
print("Raw -> Calibrated mapping:")
for raw in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    cal = platt.predict_proba(np.array([[raw]]))[0][1]
    print(f"  raw={raw:.1f}  cal={cal:.4f}")

print("\n" + "="*65)
print("STEP 4  Stale cache check")
print("="*65)
r1a = nli.score_claim("Einstein received Nobel 1921.", "Einstein won Nobel 1921.")
r1b = nli.score_claim("Einstein received Nobel 1921.", "Einstein won Nobel 1921.")
r2  = nli.score_claim("DNA=Deoxyribonucleic Acid.", "DNA=Dynamic Nucleic Assembly.")
same = abs(r1a["probs"]["entailment"] - r1b["probs"]["entailment"]) < 1e-5
diff = abs(r1a["probs"]["entailment"] - r2["probs"]["entailment"]) > 0.01
print(f"Same claim twice => identical: {same} (ent={r1a['probs']['entailment']:.4f} vs {r1b['probs']['entailment']:.4f})")
print(f"Different claims => different: {diff} (ent1={r1a['probs']['entailment']:.4f} ent2={r2['probs']['entailment']:.4f})")
print("\nNo stale cache bug =", same and diff)

print("\n" + "="*65)
print("ROOT CAUSE VERDICT")
print("="*65)
diff_f0 = abs(float(e_fac[0])-float(e_hal[0]))
diff_f2 = abs(float(e_fac[2])-float(e_hal[2]))
print(f"F0 (nli_ent) diff Einstein pair = {diff_f0:.4f}  {'DEAD - NLI OOD' if diff_f0<0.05 else 'alive'}")
print(f"F2 (ens_mean) diff Einstein pair = {diff_f2:.4f}  {'alive' if diff_f2>0.05 else 'DEAD'}")
print(f"ACTIVE_FEATURES = {ACTIVE_FEATURES}")
