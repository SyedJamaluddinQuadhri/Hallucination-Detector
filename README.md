# HalluDet-Lite 🔍
### Lightweight LLM Hallucination Detection — Runs on RTX 2050 (4GB VRAM)

> **College & Hackathon Level Project**  
> Target Venue: ACL Student Research Workshop / EMNLP Efficient NLP Workshop  
> Novel Claim: *"First systematic study of LLM hallucination detection under a strict 4GB VRAM budget"*

---

## 📋 Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Project Structure](#3-project-structure)
4. [Installation — Step by Step](#4-installation--step-by-step)
5. [How to Run — Phase by Phase](#5-how-to-run--phase-by-phase)
6. [Troubleshooting GPU Issues](#6-troubleshooting-gpu-issues)
7. [How Each Phase Works](#7-how-each-phase-works)
8. [Expected Results](#8-expected-results)
9. [Running the Demo](#9-running-the-demo)
10. [For Hackathon Presentation](#10-for-hackathon-presentation)
11. [Quick Command Reference](#11-quick-command-reference)

---

## 1. What This Project Does

Large Language Models (LLMs) like ChatGPT, Phi-2, and LLaMA sometimes generate text that **sounds confident but is factually wrong**. This is called **hallucination**.

**HalluDet-Lite** is a system that automatically detects when an LLM is hallucinating. Given a question and an LLM response, it:

- Splits the response into individual factual claims
- Checks each claim against a Wikipedia knowledge base
- Measures how uncertain the model is about each claim
- Combines all signals into a single hallucination probability (0 to 100%)
- Highlights which specific claims are likely fabricated

### Example

```
Question:    "When did Einstein win the Nobel Prize?"

LLM Response: "Einstein won the Nobel Prize in 1922 for his theory of relativity."

HalluDet-Lite Output:
  Claim 1: "Einstein won the Nobel Prize" → SUPPORTED   (12% risk)  ✓
  Claim 2: "It was in 1922"              → HALLUCINATED (89% risk)  ✗  ← Real year: 1921
  Claim 3: "It was for relativity"       → HALLUCINATED (94% risk)  ✗  ← Real reason: photoelectric effect

  Overall Risk: 94% — HIGH HALLUCINATION RISK
```

---

## 2. Hardware Requirements

| Component | Your PC | Minimum Needed |
|-----------|---------|----------------|
| GPU | RTX 2050 (4GB VRAM) | Any CUDA GPU with 4GB+ |
| RAM | 16GB | 12GB |
| Storage | 512GB SSD | 20GB free space |
| CPU | Ryzen 7 | Any modern 4-core CPU |
| OS | Windows 10/11 or Linux | — |

> **Note:** Phase 5 (fine-tuning Phi-2) runs on **Google Colab free tier (T4 GPU)**, not your PC.
> Your PC handles all other phases comfortably within 4GB VRAM.

---

## 3. Project Structure

```
halludet_lite/
│
├── README.md                        ← You are here
├── requirements.txt                 ← All Python packages needed
├── setup.py                         ← Package installer
├── configs/
│   └── config.yaml                  ← All settings in one place
│
├── src/                             ← All source code
│   ├── utils.py                     ← Shared helpers (GPU memory, logging, caching)
│   │
│   ├── phase1_nli/                  ← PHASE 1: NLI Hallucination Classifier
│   │   ├── claim_splitter.py        ← Splits LLM response into atomic claims
│   │   ├── dataset.py               ← Downloads and prepares HaluEval + FEVER data
│   │   ├── train.py                 ← Fine-tunes DeBERTa-base (RTX 2050 safe config)
│   │   └── predict.py               ← Runs NLI inference on new claim-evidence pairs
│   │
│   ├── phase2_uncertainty/          ← PHASE 2: Uncertainty Signals
│   │   ├── ensemble.py              ← 3-model ensemble disagreement scorer
│   │   └── perplexity.py            ← TinyLLaMA 4-bit token perplexity scorer
│   │
│   ├── phase3_rav/                  ← PHASE 3: Wikipedia Evidence Retrieval
│   │   ├── build_index.py           ← Streams Wikipedia, builds FAISS search index
│   │   └── retriever.py             ← Retrieves evidence and verifies each claim
│   │
│   ├── phase4_ensemble/             ← PHASE 4: Combine All Signals
│   │   ├── features.py              ← Extracts 8 features per claim from all phases
│   │   └── meta_learner.py          ← MLP that fuses signals into one calibrated score
│   │
│   └── phase5_dpo/                  ← PHASE 5: Fine-tune Phi-2 (Google Colab)
│       ├── build_dataset.py         ← Creates preference pairs for DPO training
│       └── dpo_train.py             ← DPO training script (copy-paste into Colab)
│
├── evaluation/
│   ├── metrics.py                   ← AUROC, F1, ECE metric calculations
│   └── benchmarks.py                ← HaluEval and TruthfulQA evaluation runners
│
├── demo/
│   └── app.py                       ← Gradio web demo (your hackathon showcase)
│
├── scripts/                         ← One-command runners for each phase
│   ├── run_phase1.py
│   ├── run_phase2.py
│   ├── run_phase3.py
│   ├── run_phase4.py
│   └── run_all_eval.py
│
├── colab/
│   └── phase5_phi2_dpo.ipynb        ← Upload this to Google Colab for Phase 5
│
├── notebooks/
│   └── 01_eda.py                    ← Data exploration and analysis
│
├── data/                            ← Created automatically when scripts run
│   ├── raw/                         ← Downloaded datasets
│   ├── processed/                   ← Prepared training files
│   └── knowledge_base/              ← Wikipedia FAISS index files
│
├── models/                          ← Saved model weights (created by training)
│   ├── nli_detector/                ← Phase 1 DeBERTa-base checkpoint
│   ├── meta_learner/                ← Phase 4 MLP + Platt scaler
│   └── phi2_dpo_lora/               ← Phase 5 Phi-2 LoRA adapter (from Colab)
│
└── logs/                            ← Training logs stored here
```

---

## 4. Installation — Step by Step

### Step 1 — Check your Python version

Open Command Prompt or PowerShell:

```bash
python --version
```

You need **Python 3.10 or 3.11**. Python 3.12 may cause package conflicts.
If you need 3.11, download it from [python.org/downloads](https://python.org/downloads).

---

### Step 2 — Create a virtual environment

```bash
# Go to your project folder
cd path\to\halludet_lite

# Create the virtual environment
python -m venv halludet_env
```

---

### Step 3 — Activate the virtual environment

```bash
# Windows — Command Prompt
halludet_env\Scripts\activate.bat

# Windows — PowerShell
halludet_env\Scripts\Activate.ps1

# Linux
source halludet_env/bin/activate
```

After activation you will see **(halludet_env)** at the start of your terminal prompt.

> **PowerShell error "scripts is disabled"?** Run this once, then activate again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 4 — Find your CUDA version

```bash
nvidia-smi
```

Look at the **top-right corner** of the output. It shows `CUDA Version: XX.X`.

---

### Step 5 — Install PyTorch with the correct CUDA version

```bash
# For CUDA 12.1 or higher (most RTX 2050 setups)
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 6 — Verify the GPU is detected

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')"
```

Expected output:
```
GPU: NVIDIA GeForce RTX 2050
VRAM: 4.0 GB
```

If you see an error or `CUDA available: False`, the PyTorch CUDA install failed. Repeat Step 5.

---

### Step 7 — Install all remaining packages

```bash
pip install -r requirements.txt
```

This installs HuggingFace Transformers, FAISS, sentence-transformers, spaCy, Gradio, scikit-learn, and everything else. Takes about 5–10 minutes.

---

### Step 8 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

---

### Step 9 — Set the GPU memory environment variable

This prevents memory fragmentation that causes crashes on 4GB GPUs even when total usage is under the limit.

```bash
# Windows — Command Prompt (run this every session before training)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Windows — PowerShell
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Linux
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

To set it permanently on Windows: search "Environment Variables" in the Start menu → System Properties → Environment Variables → New User Variable → Name: `PYTORCH_CUDA_ALLOC_CONF` → Value: `max_split_size_mb:128`.

---

### Step 10 — Final verification

```bash
python -c "
import torch
from transformers import AutoTokenizer
import spacy, faiss, sentence_transformers, gradio
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
print('All packages: OK')
print('Ready to run!')
"
```

If everything prints without error, installation is complete.

---

## 5. How to Run — Phase by Phase

> Always activate your virtual environment first before running any command:
> ```bash
> halludet_env\Scripts\activate.bat    # Windows CMD
> source halludet_env/bin/activate     # Linux
> ```

---

### Phase 1 — Train the NLI Detector

**What happens:** Downloads HaluEval and FEVER-NLI datasets automatically, then fine-tunes DeBERTa-base to tell supported claims apart from hallucinated ones.

| Info | Value |
|------|-------|
| Runtime | 3–4 hours |
| VRAM used | ~3.2 GB |
| Output | `models/nli_detector/best/` |
| Run overnight? | Yes |

```bash
# Standard run (run overnight)
python scripts/run_phase1.py

# Quick 1-epoch test to verify everything works before overnight run
python scripts/run_phase1.py --epochs 1 

# If training was interrupted, resume from last checkpoint
python scripts/run_phase1.py --resume models/nli_detector/checkpoint-500
```

**To verify it is running on GPU:**
Open a second terminal while training and run:
```bash
nvidia-smi
```
You should see a Python process under "Processes" using ~3.2 GB VRAM.

---

### Phase 2 — Test Uncertainty Signals

**What happens:** Loads three NLI models and TinyLLaMA to verify the ensemble disagreement and perplexity signals work correctly on sample claims.

| Info | Value |
|------|-------|
| Runtime | ~10 minutes |
| VRAM used | ~1.5 GB (models loaded one at a time) |
| Output | `data/processed/phase2_test_results.json` |

```bash
python scripts/run_phase2.py

# Skip perplexity test if you want it faster
python scripts/run_phase2.py --skip-ppl
```

> **Important:** This phase unloads and reloads models between steps to stay within 4GB VRAM. Do not interrupt it midway.

---

### Phase 3 — Build the Wikipedia Knowledge Base

**What happens:** Streams 100,000 Wikipedia articles without downloading the full 20GB dump. Filters for relevant domains (science, history, geography, people). Chunks each article into 150-word passages. Encodes all passages with MiniLM. Saves a searchable FAISS index.

| Info | Value |
|------|-------|
| Runtime | ~90 minutes |
| VRAM used | 0 GB (runs entirely on CPU) |
| RAM used | ~4 GB peak |
| Storage used | ~2 GB |
| Output | `data/knowledge_base/minilm_wiki.faiss` |
| Run overnight? | Yes |

```bash
# Full build (100K articles — run overnight)
python scripts/run_phase3.py

# Quick test with 10K articles (~10 minutes, good for verifying setup)
python scripts/run_phase3.py --articles 10000

# Test the existing index without rebuilding
python scripts/run_phase3.py --test-only

# Force a complete rebuild
python scripts/run_phase3.py --force
```

> **Good news:** This phase uses zero VRAM. You can use your PC normally while it runs.

---

### Phase 4 — Train the Meta-Learner

**What happens:** Runs all prior phase models on each HaluEval sample to extract 8 features per claim. Trains a small MLP to combine these features. Applies Platt scaling for probability calibration.

| Info | Value |
|------|-------|
| Feature extraction | 2–4 hours (runs all phase models, slow) |
| Meta-learner training | ~2 minutes (CPU, very fast) |
| Output | `models/meta_learner/` |

```bash
# Standard run with 5000 samples
python scripts/run_phase4.py --samples 5000

# Also build the DPO dataset for Phase 5 at the same time
python scripts/run_phase4.py --samples 5000 --build-dpo

# Skip feature extraction if features are already cached, train only
python scripts/run_phase4.py --train-only
```

---

### Phase 5 — Fine-tune Phi-2 on Google Colab

**This phase runs on Google Colab, not your PC.**

**Step 1 — Build the DPO dataset on your PC** (if not done in Phase 4):
```bash
python scripts/run_phase4.py --build-dpo
```

**Step 2 — Go to Google Colab:**
Visit [colab.research.google.com](https://colab.research.google.com)

**Step 3 — Upload the notebook:**
Click `File → Upload notebook` → select `colab/phase5_phi2_dpo.ipynb`

**Step 4 — Set GPU runtime:**
Click `Runtime → Change runtime type → T4 GPU → Save`

**Step 5 — Upload your dataset:**
In the left sidebar, click the folder icon → upload `data/processed/dpo_dataset.json`

**Step 6 — Run the notebook:**
Click `Runtime → Run all` and wait. Training takes about 2 hours.

**Step 7 — Download the adapter:**
After training, go to `Google Drive → MyDrive → halludet_phi2 → final` and download the entire `final` folder.

**Step 8 — Put it on your PC:**
Place the downloaded folder at `models/phi2_dpo_lora/final/`

> **Session expired mid-training?** The notebook auto-saves every 50 steps to Google Drive. Just reopen the notebook, re-upload, and run again — it will automatically resume from the latest checkpoint.

---

### Evaluation — Run All Benchmarks

**What happens:** Evaluates the full pipeline on the HaluEval test set, runs an ablation study comparing each phase's contribution, and evaluates Phi-2's TruthfulQA score before and after DPO.

```bash
# Full evaluation (takes ~1–2 hours)
python scripts/run_all_eval.py

# Quick sanity check on 4 hand-crafted test cases (~5 minutes)
python scripts/run_all_eval.py --demo-only

# Skip TruthfulQA if Phase 5 is not done yet
python scripts/run_all_eval.py --no-phi2

# Limit to 200 test samples for faster results
python scripts/run_all_eval.py --max-samples 200
```

The final output prints a table like:
```
=================================================================
HALLUDET-LITE — FINAL RESULTS TABLE
=================================================================
Component                    AUROC       F1      ECE
-----------------------------------------------------------------
Phase 1 — NLI only           0.7821   0.7104   0.0921
Phase 1+2 — + Ensemble       0.8134   0.7398   0.0812
Phase 1+3 — + RAV            0.8302   0.7561   0.0734
Phase 1–4 — Full ensemble    0.8547   0.7823   0.0521
=================================================================
```

---

## 6. Troubleshooting GPU Issues

### Problem: `CUDA available: False`

PyTorch was installed without CUDA. Fix:
```bash
pip uninstall torch torchvision -y
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Problem: `CUDA out of memory`

Two models are loaded at once, exceeding 4GB. Fix:
```bash
# Set this before running (every session)
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   # Windows CMD

# Also try reducing batch size
python scripts/run_phase1.py --batch-size 4
```

---

### Problem: `nvidia-smi` shows no process during training

The script is running on CPU, not GPU. Diagnose:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
If it prints `False`, PyTorch CUDA install is broken. Reinstall from Step 5.

---

### Problem: `ModuleNotFoundError: No module named 'XXX'`

A package failed to install. Fix:
```bash
pip install -r requirements.txt

# Or install the specific missing package
pip install transformers==4.40.0
pip install sentence-transformers==2.7.0
pip install faiss-cpu
```

---

### Problem: Phase 3 crashes with a memory error

Your system RAM is full during FAISS encoding. Fix — use a smaller article count:
```bash
python scripts/run_phase3.py --articles 50000
```

---

### Problem: `Error: NLI model not found`

Phase 1 training has not been completed yet. You must run the phases in order:
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Evaluation
```

---

## 7. How Each Phase Works

### Phase 1 — Natural Language Inference

The core idea: if a claim is factual, a Wikipedia passage about the topic should **entail** (support) it. If the claim is hallucinated, the passage will either **contradict** it or be **neutral** (unrelated).

We fine-tune **DeBERTa-base** on labeled examples from HaluEval (which has explicit hallucinated/correct response pairs) and FEVER (a Wikipedia fact verification dataset). After training, the model outputs three probabilities for any claim-evidence pair: P(entailment), P(neutral), P(contradiction).

Hallucination score = 1 − P(entailment)

We use DeBERTa-**base** not DeBERTa-**large** because the large model requires 8GB+ VRAM. The base model uses 3.2GB. The accuracy gap is only 3–5%, and that gap itself is a publishable finding showing what is achievable on consumer hardware.

---

### Phase 2 — Uncertainty Signals

The key insight: a model that **knows** the answer gives the same answer repeatedly. A model that is **guessing** gives different answers each time.

Instead of sampling one big model 15 times (impossible on 4GB), we use **three different NLI models** and measure how much they disagree with each other. High disagreement = the claim is uncertain = hallucination risk.

We also use **TinyLLaMA-1.1B** (0.8GB in 4-bit) to measure how surprised the model is by each word. If a specific date or name has very high perplexity, it means the model was not confident about generating that fact.

---

### Phase 3 — Retrieval-Augmented Verification

For each claim, we search 500,000 Wikipedia passages for supporting evidence using FAISS (Facebook AI Similarity Search). This is like asking: "Is there any Wikipedia article that backs up this claim?"

The search uses **MiniLM-L6-v2** (22M parameters, CPU-only) to convert both the claim and all Wikipedia passages into vectors. FAISS finds the most similar passages in milliseconds. We then run NLI between the claim and the top-5 retrieved passages. If even the most relevant Wikipedia passage does not support the claim, it is flagged as a hallucination candidate.

The entire retrieval step uses zero VRAM — it runs on CPU and Ryzen 7.

---

### Phase 4 — Ensemble Meta-Learner

We now have 8 numbers per claim:
- F1, F2: NLI entailment and contradiction probabilities (Phase 1)
- F3, F4: Ensemble mean entailment and disagreement (Phase 2)
- F5: Claim perplexity from TinyLLaMA (Phase 2)
- F6, F7: RAV max entailment and FAISS retrieval score (Phase 3)
- F8: Named entity count in the claim (proxy for factual density)

A 3-layer MLP (8→64→32→1) learns the optimal combination of these features. Then Platt scaling makes the output a true probability — if it says 80%, 80% of such claims are really hallucinated in practice.

---

### Phase 5 — DPO Fine-Tuning

Direct Preference Optimization teaches Phi-2 to **prefer factual responses** over hallucinated ones. We show it pairs: (correct response, hallucinated response) for the same question. The training pushes the model to assign higher probability to the correct one while a frozen reference model prevents it from drifting to incoherent outputs.

After training, the model itself generates fewer hallucinations — measured by improvement on the TruthfulQA benchmark.

---

## 8. Expected Results

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 — NLI alone | AUROC | 0.76 – 0.79 |
| Phase 1 + 2 — with ensemble | AUROC | 0.79 – 0.82 |
| Phase 1 + 3 — with RAV | AUROC | 0.81 – 0.84 |
| All phases combined | AUROC | **0.83 – 0.87** |
| Calibration (all phases) | ECE | < 0.06 |
| Phi-2 DPO — TruthfulQA | Accuracy improvement | +5 to +8% |
| Demo latency | Time per response | < 5 seconds |

**What AUROC means:** It ranges from 0.5 (random guessing) to 1.0 (perfect). Our target of 0.85+ is competitive with systems that require A100 GPUs (40GB VRAM), which is the core research contribution.

---

## 9. Running the Demo

```bash
# Local only — opens at http://localhost:7860
python demo/app.py

# With public URL (for hackathon, works for 72 hours)
python demo/app.py --share
```

The terminal prints:
```
Running on local URL:  http://localhost:7860
Running on public URL: https://abc123xyz.gradio.live   ← share this with judges
```

### What the demo shows

- **Overall risk score** with color-coded banner (green / amber / red)
- **Per-claim breakdown** — each claim gets its own card showing the verdict, risk percentage, and the Wikipedia evidence used
- **Signal grid** — NLI entailment score, ensemble disagreement, and perplexity for each claim
- **Entity badges** — named entities (people, dates, places) highlighted with their spaCy types
- **JSON tab** — raw feature scores for technical judges

---

## 10. For Hackathon Presentation

### 30-Second Pitch

*"LLMs like ChatGPT make up facts with total confidence. We built a system that catches them — claim by claim, with Wikipedia evidence shown for every decision. The key thing: it runs on a 4GB gaming laptop GPU that costs under ₹50,000. No cloud API, no subscription, no A100. Just this laptop."*

### Opening Move

Open the demo in a browser before your slot. When it is your turn, type this example immediately:

- **Question:** `When did Einstein win the Nobel Prize?`
- **Response:** `Einstein won the Nobel Prize in 1922 for his theory of relativity.`

Watch it turn red in real time. Explain that 1922 is wrong (1921) and relativity is wrong (photoelectric effect). The demo makes this viscerally obvious.

### What Judges Care About

1. **The ablation table** — phase-by-phase AUROC improvement. Print this on your poster. It shows scientific rigor.
2. **The hardware constraint** — this is your differentiator. Emphasize it repeatedly.
3. **TruthfulQA improvement** — Phi-2 becomes measurably less likely to hallucinate after DPO training. Concrete before/after numbers are compelling.
4. **Real-time demo** — a live system beats any slide.

---

## 11. Quick Command Reference

```bash
# ── Every session — activate environment first ────────────────────────────
halludet_env\Scripts\activate.bat              # Windows CMD
source halludet_env/bin/activate               # Linux

# ── Every session — set GPU memory variable ──────────────────────────────
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128        # Windows CMD
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"     # PowerShell
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128     # Linux

# ── Run phases in order ──────────────────────────────────────────────────
python scripts/run_phase1.py                   # Train NLI — ~4 hours overnight
python scripts/run_phase2.py                   # Test uncertainty signals — ~10 min
python scripts/run_phase3.py                   # Build FAISS index — ~90 min overnight
python scripts/run_phase4.py --build-dpo       # Meta-learner + DPO dataset — ~4 hours

# Phase 5 runs on Google Colab, not your PC
# Upload colab/phase5_phi2_dpo.ipynb to colab.research.google.com

# ── After all phases are done ────────────────────────────────────────────
python scripts/run_all_eval.py                 # Full benchmark evaluation
python demo/app.py --share                     # Launch demo with public URL

# ── Useful one-off commands ──────────────────────────────────────────────
python -c "import torch; print(torch.cuda.get_device_name(0))"   # Check GPU
nvidia-smi                                     # Check VRAM usage live
python scripts/run_phase1.py --epochs 1        # Quick test before overnight run
python scripts/run_phase3.py --articles 10000  # Quick knowledge base test
python scripts/run_all_eval.py --demo-only     # Fast sanity check (5 min)
```

---

*HalluDet-Lite — Designed for RTX 2050 · 16GB RAM · 512GB SSD · Python 3.11*