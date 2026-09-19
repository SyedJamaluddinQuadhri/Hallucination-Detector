"""Quick feature diagnostic for demo cases (fixed knowledge)."""
import sys
sys.path.insert(0, '.')

from src.phase1_nli.predict import NLIPredictor
from src.phase2_uncertainty.ensemble import EnsembleDisagreement
from src.phase3_rav.retriever import LightweightRAV
from src.phase4_ensemble.meta_learner import MetaLearnerPredictor
import numpy as np

nli  = NLIPredictor()
ens  = EnsembleDisagreement(nli_predictor=nli, load_small=True, load_bart=False)
rav  = LightweightRAV(nli_predictor=nli)
meta = MetaLearnerPredictor()

test_cases = [
    {
        "label":  "FACTUAL",
        "knowledge": "Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        "claim":     "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        "expected_hal": 0,
    },
    {
        "label":  "HALLUCINATED",
        "knowledge": "Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        "claim":     "Albert Einstein won the Nobel Prize in Physics in 1922 for his theory of relativity.",
        "expected_hal": 1,
    },
    {
        "label":  "FACTUAL",
        "knowledge": "The capital city of Australia is Canberra. Sydney is the largest city in Australia.",
        "claim":     "The capital of Australia is Canberra.",
        "expected_hal": 0,
    },
    {
        "label":  "HALLUCINATED",
        "knowledge": "The capital city of Australia is Canberra. Sydney is the largest city in Australia.",
        "claim":     "The capital of Australia is Sydney, which is also the largest city.",
        "expected_hal": 1,
    },
]

print(f"{'Label':<14} {'NLI_ent':>8} {'NLI_con':>8} {'ENS_mean':>9} {'RAV_ent':>8} {'HAL_score':>10} {'OK':>4}")
print("-" * 70)

for tc in test_cases:
    r_nli = nli.score_claim(tc["knowledge"], tc["claim"])
    r_ens = ens.score(tc["knowledge"], tc["claim"])
    r_rav = rav.verify_claim(tc["claim"])
    feat = np.array([
        r_nli["probs"]["entailment"],
        r_nli["probs"]["contradiction"],
        r_ens["mean_entailment"],
        r_ens["disagreement"],
        5.0,  # dummy ppl
        r_rav["max_ent"],
        0.0,
        0.0,
    ], dtype=np.float32)
    score = meta.predict(feat)
    predicted_hal = 1 if score >= 0.5 else 0
    ok = "PASS" if predicted_hal == tc["expected_hal"] else "FAIL"
    print(
        f"{tc['label']:<14}"
        f" {r_nli['probs']['entailment']:>8.4f}"
        f" {r_nli['probs']['contradiction']:>8.4f}"
        f" {r_ens['mean_entailment']:>9.4f}"
        f" {r_rav['max_ent']:>8.4f}"
        f" {score:>10.4f}"
        f" {ok:>4}"
    )
