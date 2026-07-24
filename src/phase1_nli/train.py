"""
train.py — Fine-tune DeBERTa-base for NLI hallucination detection.

Optimised for RTX 2050 4GB VRAM:
- DeBERTa-base (86M params) instead of large (400M)
- max_length=256 instead of 512
- fp16 mixed precision
- gradient checkpointing
- batch_size=8 with grad_accumulation=4 (effective=32)

Expected training time: 3–4 hours on RTX 2050
"""

import os
from pathlib import Path
from typing import Optional

import torch
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from src.utils import get_logger, get_project_root, free_gpu_memory
from src.phase1_nli.dataset import (
    NLIDataset,
    build_halueval_nli_dataset,
    LABEL2ID,
    ID2LABEL,
)

log = get_logger("nli.train")

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME   = "microsoft/deberta-v3-base"   # BASE not large — 4GB VRAM safe
OUTPUT_DIR   = str(get_project_root() / "models" / "nli_detector")
DATASET_PATH = str(get_project_root() / "data" / "processed" / "nli_dataset.json")


# ── Metrics ───────────────────────────────────────────────────────────────────

f1_metric  = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute macro F1 and accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return {"f1_macro": f1["f1"], "accuracy": acc["accuracy"]}


# ── Training Function ─────────────────────────────────────────────────────────

def train_nli_model(
    model_name:          str   = MODEL_NAME,
    output_dir:          str   = OUTPUT_DIR,
    dataset_path:        str   = DATASET_PATH,
    num_epochs:          int   = 4,
    batch_size:          int   = 8,
    grad_accum_steps:    int   = 4,
    learning_rate:       float = 2e-5,
    max_length:          int   = 256,
    warmup_ratio:        float = 0.06,
    weight_decay:        float = 0.01,
    early_stopping:      int   = 2,
    wandb_project:       Optional[str] = "halludet-lite",
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Fine-tune DeBERTa-base on the NLI hallucination dataset.

    Args:
        model_name:    HuggingFace model ID (use deberta-v3-base for 4GB VRAM)
        output_dir:    Where to save checkpoints
        dataset_path:  Path to pre-built dataset JSON (built by dataset.py)
        num_epochs:    Training epochs (4 is sufficient for convergence)
        batch_size:    Per-device batch size (8 is safe for RTX 2050 4GB)
        grad_accum_steps: Gradient accumulation (effective batch = batch_size × steps)
        learning_rate: AdamW learning rate
        max_length:    Max token length (256 saves 40% VRAM vs 512)
        warmup_ratio:  Fraction of steps for LR warmup
        weight_decay:  L2 regularization
        early_stopping: Patience for early stopping
        wandb_project: W&B project name (set to None to disable)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    log.info("=" * 60)
    log.info("Phase 1 — NLI Model Training")
    log.info(f"Model:  {model_name}")
    log.info(f"Output: {output_dir}")
    log.info("=" * 60)

    # ── Load or build dataset ─────────────────────────────────────────────────
    from src.utils import load_json
    if os.path.exists(dataset_path):
        log.info(f"Loading cached dataset from {dataset_path}")
        data = load_json(dataset_path)
        train_records = data["train"]
        val_records   = data["val"]
        test_records  = data["test"]
    else:
        log.info("Building dataset from HaluEval + FEVER...")
        train_records, val_records, test_records = build_halueval_nli_dataset(
            save_path=dataset_path
        )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = NLIDataset(train_records, tokenizer, max_length)
    val_ds   = NLIDataset(val_records,   tokenizer, max_length)

    log.info(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model parameters: {param_count:.1f}M")

    # ── Training arguments — RTX 2050 OPTIMISED ──────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Epochs + batch
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum_steps,

        # Learning rate schedule
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",

        # RTX 2050 CRITICAL: fp16 (NOT bf16 — RTX 2050 doesn't support bf16)
        fp16=True,
        bf16=False,

        # Memory optimisations
        gradient_checkpointing=True,       # ~30% less VRAM
        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        # Evaluation & saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        # Logging
        logging_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="wandb" if wandb_project else "none",
        run_name="deberta-base-nli-phase1",

        # Save space
        save_total_limit=2,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    callbacks = []
    if early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting training...")
    if resume_from_checkpoint:
        log.info(f"Resuming from: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save best model ───────────────────────────────────────────────────────
    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    log.info(f"Best model saved to {best_dir}")

    # ── Final evaluation on test set ──────────────────────────────────────────
    log.info("Evaluating on test set...")
    test_ds = NLIDataset(test_records, tokenizer, max_length)
    test_results = trainer.evaluate(eval_dataset=test_ds)
    log.info(f"Test results: {test_results}")

    # Save test results
    from src.utils import save_json
    save_json(test_results, os.path.join(output_dir, "test_results.json"))

    return trainer, model, tokenizer


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_nli_model()
