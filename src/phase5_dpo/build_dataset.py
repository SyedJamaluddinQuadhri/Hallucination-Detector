"""
build_dataset.py — Build DPO preference dataset for Phi-2 fine-tuning.

Creates (prompt, chosen, rejected) triples from:
  1. HaluEval QA — has explicit hallucinated/correct pairs
  2. HaluEval Dialogue — dialogue hallucination pairs
  3. TruthfulQA — known-false questions for rejection

Run this on your PC, then upload the dataset to Google Colab for training.
"""

import os
import json
import random
from typing import Dict, List, Optional

from datasets import load_dataset

from src.utils import get_logger, get_project_root, save_json

log = get_logger("dpo.dataset")

OUTPUT_PATH = str(get_project_root() / "data" / "processed" / "dpo_dataset.json")


def build_dpo_dataset(
    use_dialogue:     bool = True,
    use_summarization: bool = False,  # can be noisy
    max_qa_samples:   int  = 8_000,
    val_fraction:     float = 0.1,
    save_path:        str  = OUTPUT_PATH,
) -> Dict:
    """
    Build the DPO preference dataset.

    Each record has: {prompt, chosen, rejected}
    - prompt:   the question / instruction
    - chosen:   the factually correct response (preferred)
    - rejected: the hallucinated response (dispreferred)

    Args:
        use_dialogue:     Include dialogue hallucination pairs.
        use_summarization: Include summarization pairs (noisy — off by default).
        max_qa_samples:   Max QA pairs (default 8K).
        val_fraction:     Fraction for validation split.
        save_path:        Output path for the dataset JSON.

    Returns:
        Dict with 'train' and 'val' lists of DPO records.
    """
    random.seed(42)
    all_records = []

    # ── Source 1: HaluEval QA ─────────────────────────────────────────────────
    log.info("Loading HaluEval QA...")
    halueval_qa = load_dataset("pminervini/HaluEval", "qa", split="data")
    qa_count = 0
    for sample in halueval_qa:
        if qa_count >= max_qa_samples:
            break
        right_ans = sample.get("right_answer", "").strip()
        hal_ans   = sample.get("hallucinated_answer", "").strip()
        question  = sample.get("question", "").strip()
        if not all([right_ans, hal_ans, question]):
            continue
        # Format as instruction-following prompt
        prompt = f"Answer the following question accurately and concisely:\n\n{question}"
        all_records.append({
            "prompt":   prompt,
            "chosen":   right_ans,
            "rejected": hal_ans,
            "source":   "halueval_qa",
        })
        qa_count += 1
    log.info(f"HaluEval QA: {qa_count:,} records")

    # ── Source 2: HaluEval Dialogue ───────────────────────────────────────────
    if use_dialogue:
        log.info("Loading HaluEval Dialogue...")
        try:
            halueval_d = load_dataset("pminervini/HaluEval", "dialogue", split="data")
            diag_count = 0
            for sample in halueval_d:
                history  = sample.get("dialogue_history", "").strip()
                right_r  = sample.get("right_response", "").strip()
                hal_r    = sample.get("hallucinated_response", "").strip()
                if not all([history, right_r, hal_r]):
                    continue
                prompt = f"Continue the following conversation appropriately:\n\n{history}\n\nResponse:"
                all_records.append({
                    "prompt":   prompt,
                    "chosen":   right_r,
                    "rejected": hal_r,
                    "source":   "halueval_dialogue",
                })
                diag_count += 1
            log.info(f"HaluEval Dialogue: {diag_count:,} records")
        except Exception as e:
            log.warning(f"Could not load dialogue data: {e}")

    # ── Source 3: HaluEval Summarization ──────────────────────────────────────
    if use_summarization:
        log.info("Loading HaluEval Summarization...")
        try:
            halueval_s = load_dataset("pminervini/HaluEval", "summarization", split="data")
            sum_count = 0
            for sample in halueval_s:
                doc      = sample.get("document", "")[:800].strip()
                right_s  = sample.get("right_summary", "").strip()
                hal_s    = sample.get("hallucinated_summary", "").strip()
                if not all([doc, right_s, hal_s]):
                    continue
                prompt = f"Summarize the following document accurately:\n\n{doc}\n\nSummary:"
                all_records.append({
                    "prompt":   prompt,
                    "chosen":   right_s,
                    "rejected": hal_s,
                    "source":   "halueval_summarization",
                })
                sum_count += 1
            log.info(f"HaluEval Summarization: {sum_count:,} records")
        except Exception as e:
            log.warning(f"Could not load summarization data: {e}")

    # ── Shuffle and split ─────────────────────────────────────────────────────
    random.shuffle(all_records)
    n_val = int(len(all_records) * val_fraction)
    val_records   = all_records[:n_val]
    train_records = all_records[n_val:]

    log.info(f"DPO dataset — train: {len(train_records):,}, val: {len(val_records):,}")

    dataset = {"train": train_records, "val": val_records}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_json(dataset, save_path)
    log.info(f"DPO dataset saved to {save_path}")

    return dataset
