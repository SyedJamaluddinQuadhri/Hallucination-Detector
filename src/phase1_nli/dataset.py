"""
dataset.py — Dataset preparation for NLI fine-tuning.

Combines FEVER-NLI (general fact verification) and HaluEval
(LLM hallucination detection) into a unified NLI training set.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, concatenate_datasets

from src.utils import get_logger, save_json, load_json, get_project_root

log = get_logger("nli.dataset")

# Label mapping used throughout the project
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class NLIDataset(Dataset):
    """
    PyTorch Dataset for NLI training.

    Each sample is a (premise, hypothesis, label) triple.
    - premise:    the evidence document / ground-truth context
    - hypothesis: the claim to verify (single atomic sentence)
    - label:      0=entailment, 1=neutral, 2=contradiction
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 256,   # 256 saves ~40% VRAM vs 512 on RTX 2050
    ):
        """
        Args:
            records:    List of dicts with keys: premise, hypothesis, label
            tokenizer:  HuggingFace tokenizer
            max_length: Max token length (256 recommended for 4GB VRAM)
        """
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        encoding = self.tokenizer(
            rec["premise"],
            rec["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(LABEL2ID[rec["label"]], dtype=torch.long),
        }


# ── Dataset Builders ──────────────────────────────────────────────────────────

def load_fever_nli(split: str = "train") -> List[Dict]:
    """
    Load FEVER-NLI dataset (fact verification from Wikipedia claims).
    Source: pietrolesci/nli_fever on HuggingFace Hub.
    Size: ~185K train pairs.
    """
    log.info(f"Loading FEVER-NLI ({split})...")
    ds = load_dataset("pietrolesci/nli_fever", split=split)
    records = []
    for row in ds:
        label = row["label"].lower()
        if label not in LABEL2ID:
            continue  # skip 'not_enough_info' and similar
        records.append({
            "premise":    row["premise"],
            "hypothesis": row["hypothesis"],
            "label":      label,
            "source":     "fever",
        })
    log.info(f"FEVER-NLI {split}: {len(records):,} records")
    return records


def load_halueval_nli(split: str = "train") -> List[Dict]:
    """
    Load HaluEval QA dataset and convert to NLI format.
    Source: pminervini/HaluEval on HuggingFace Hub.

    Actual schema: knowledge, question, answer, hallucination (yes/no)
    Conversion:
    - hallucination=='no'  -> entailment  (correct answer)
    - hallucination=='yes' -> contradiction (hallucinated answer)

    Note: HaluEval has no dev/test splits — we create them manually.
    """
    log.info("Loading HaluEval QA dataset...")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    records = []
    for row in ds:
        knowledge = row.get("knowledge", "").strip()
        answer    = row.get("answer", "").strip()
        is_hal    = str(row.get("hallucination", "no")).strip().lower()
        if not knowledge or not answer:
            continue

        label = "contradiction" if is_hal == "yes" else "entailment"
        records.append({
            "premise":    knowledge,
            "hypothesis": answer,
            "label":      label,
            "source":     f"halueval_{is_hal}",
            "question":   row.get("question", ""),
        })

    log.info(f"HaluEval NLI: {len(records):,} records")
    return records


def load_halueval_summarization() -> List[Dict]:
    """
    Load HaluEval summarization split for faithfulness-specific training.
    Actual schema: document, summary, hallucination (yes/no)
    Converts to NLI format:
    - hallucination=='no'  -> entailment
    - hallucination=='yes' -> contradiction
    """
    log.info("Loading HaluEval Summarization dataset...")
    try:
        ds = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")
    except Exception as e:
        log.warning(f"Could not load summarization split: {e}")
        return []

    records = []
    for row in ds:
        doc     = row.get("document", "").strip()
        summary = row.get("summary", "").strip()
        is_hal  = str(row.get("hallucination", "no")).strip().lower()
        if not doc or not summary:
            continue
        label = "contradiction" if is_hal == "yes" else "entailment"
        records.append({
            "premise":  doc[:800],
            "hypothesis": summary,
            "label":    label,
            "source":   f"halueval_summ_{is_hal}",
        })

    log.info(f"HaluEval Summarization: {len(records):,} records")
    return records


def build_halueval_nli_dataset(
    use_fever: bool = True,
    use_summarization: bool = True,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    save_path: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build the full NLI training dataset by combining all sources.

    Returns:
        (train_records, val_records, test_records)
    """
    import random
    random.seed(42)

    all_records = []

    # Load HaluEval QA (primary)
    all_records.extend(load_halueval_nli())

    # Load FEVER-NLI (general NLI capability)
    if use_fever:
        fever = load_fever_nli()
        # Sample 30K FEVER pairs to balance with HaluEval
        random.shuffle(fever)
        all_records.extend(fever[:30_000])

    # Load summarization (faithfulness)
    if use_summarization:
        all_records.extend(load_halueval_summarization())

    # Shuffle
    random.shuffle(all_records)

    # Split
    n = len(all_records)
    n_test = int(n * test_fraction)
    n_val  = int(n * val_fraction)

    test_records  = all_records[:n_test]
    val_records   = all_records[n_test:n_test + n_val]
    train_records = all_records[n_test + n_val:]

    log.info(
        f"Dataset split — train: {len(train_records):,}, "
        f"val: {len(val_records):,}, test: {len(test_records):,}"
    )

    # Log label distribution
    for split_name, split_data in [("train", train_records),
                                    ("val",   val_records),
                                    ("test",  test_records)]:
        counts = {}
        for r in split_data:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        log.info(f"{split_name} labels: {counts}")

    # Save to disk if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_json({"train": train_records, "val": val_records, "test": test_records},
                  save_path)
        log.info(f"Dataset saved to {save_path}")

    return train_records, val_records, test_records
