"""
01_eda.py — Exploratory Data Analysis for HalluDet-Lite.

Run as a Jupyter notebook or as a standalone script:
    jupyter notebook notebooks/01_eda.py
    python notebooks/01_eda.py

Covers:
  - HaluEval dataset statistics
  - Label distribution
  - Claim length distribution
  - Example hallucinated vs correct responses
  - Baseline NLI score distribution on a sample
"""

# %% Setup
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import numpy as np

print("Loading HaluEval...")
halueval = load_dataset("pminervini/HaluEval", "qa_samples", split="train")
print(f"Total samples: {len(halueval):,}")

# %% Dataset statistics
print("\n=== HaluEval QA Statistics ===")
print(f"Columns: {halueval.column_names}")

# Sample a few examples
for i in range(3):
    s = halueval[i]
    print(f"\n--- Sample {i+1} ---")
    print(f"Question:    {s['question']}")
    print(f"Correct:     {s['right_answer'][:100]}...")
    print(f"Hallucinated:{s['hallucinated_answer'][:100]}...")

# %% Claim length distribution
from src.phase1_nli.claim_splitter import ClaimSplitter

splitter = ClaimSplitter()
correct_lengths = []
hal_lengths = []

for sample in list(halueval)[:500]:
    correct_claims = splitter.split(sample["right_answer"])
    hal_claims     = splitter.split(sample["hallucinated_answer"])
    correct_lengths.append(len(correct_claims))
    hal_lengths.append(len(hal_claims))

print("\n=== Claim Count Distribution ===")
print(f"Correct responses   — mean claims per response: {np.mean(correct_lengths):.1f}")
print(f"Hallucinated responses — mean claims per response: {np.mean(hal_lengths):.1f}")

# %% Sample NLI scores (requires Phase 1 training to be done)
import os
nli_model_path = "models/nli_detector/best"
if os.path.exists(nli_model_path):
    from src.phase1_nli.predict import NLIPredictor
    nli = NLIPredictor()

    print("\n=== Sample NLI Scores ===")
    print(f"{'Label':<12} {'Verdict':<15} {'Entailment':>10} {'Hal_Score':>10}")
    print("-" * 55)

    for sample in list(halueval)[:10]:
        knowledge = sample.get("knowledge", "")
        if not knowledge:
            continue
        for (response, label) in [
            (sample["right_answer"],        "correct"),
            (sample["hallucinated_answer"], "hallucinated"),
        ]:
            claims = splitter.split(response)
            if not claims:
                continue
            result = nli.score_claim(knowledge, claims[0])
            print(
                f"{label:<12} {result['verdict']:<15} "
                f"{result['probs']['entailment']:>10.3f} "
                f"{result['hallucination_score']:>10.3f}"
            )
else:
    print("\nNLI model not trained yet. Run python scripts/run_phase1.py first.")

print("\n=== EDA Complete ===")
