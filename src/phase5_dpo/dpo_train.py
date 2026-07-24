"""
dpo_train.py — Phi-2 DPO fine-tuning script.

THIS FILE RUNS ON GOOGLE COLAB (T4 GPU, 15GB VRAM), NOT YOUR PC.

Steps:
  1. Build dataset on your PC: python scripts/run_phase4.py --build-dpo
  2. Upload data/processed/dpo_dataset.json to Google Drive
  3. Open colab/phase5_phi2_dpo.ipynb and run it
  4. Download models/phi2_dpo_lora/ from Drive back to your PC

Or copy-paste this file directly into a Colab code cell.
"""

# ── This script is designed for Google Colab T4 (15GB VRAM) ─────────────────
# ── Do NOT run on RTX 2050 — Phi-2 base alone needs ~5.5GB ─────────────────

import os
import json
from pathlib import Path
from typing import Optional

# ── Colab-specific: install if not present ───────────────────────────────────
def install_colab_deps():
    """Install required packages in Colab environment."""
    import subprocess
    packages = [
        "transformers==4.40.0",
        "datasets",
        "peft==0.10.0",
        "trl==0.8.6",
        "bitsandbytes==0.43.0",
        "accelerate==0.29.0",
        "wandb",
    ]
    subprocess.run(["pip", "install", "-q"] + packages, check=True)
    print("Dependencies installed.")


def run_dpo_training(
    data_path: str,
    output_dir: str,
    model_name: str = "microsoft/phi-2",
    beta: float = 0.1,
    num_epochs: int = 2,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_length: int = 512,
    max_prompt_length: int = 256,
    wandb_project: Optional[str] = "halludet-phi2-dpo",
):
    """
    Fine-tune Phi-2 with DPO using QLoRA on Google Colab T4.

    Args:
        data_path:         Path to dpo_dataset.json
        output_dir:        Where to save checkpoints (use Google Drive path)
        model_name:        HuggingFace model ID
        beta:              DPO KL penalty (0.1 is a robust default)
        num_epochs:        Training epochs (2 is enough for Phi-2)
        batch_size:        Per-device batch (4 for T4 15GB)
        grad_accum:        Gradient accumulation steps
        lr:                Learning rate
        lora_r:            LoRA rank (8 for Phi-2 2.7B)
        lora_alpha:        LoRA scaling factor
        max_length:        Max sequence length
        max_prompt_length: Max prompt length
        wandb_project:     W&B project name (None to disable)
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path}...")
    with open(data_path, "r") as f:
        raw = json.load(f)

    train_dataset = Dataset.from_list(raw["train"])
    eval_dataset  = Dataset.from_list(raw["val"])
    print(f"Train: {len(train_dataset):,} | Val: {len(eval_dataset):,}")

    # ── 4-bit quantization config (QLoRA) ────────────────────────────────────
    # T4 uses fp16 (NOT bf16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16 for T4
    )

    # ── Load Phi-2 base model ─────────────────────────────────────────────────
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Better for causal LM

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "dense"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Reference model (frozen base) ────────────────────────────────────────
    print("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── DPO Training config ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        # T4 uses fp16 (NOT bf16 — T4 doesn't fully support bf16)
        fp16=True,
        bf16=False,

        # Memory optimisations
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",

        # Sequence lengths
        max_length=max_length,
        max_prompt_length=max_prompt_length,

        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rewards/margins",

        # Logging
        logging_steps=10,
        report_to="wandb" if wandb_project else "none",
        run_name="phi2-dpo-halludet",

        # Remove default data collator issues
        remove_unused_columns=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting DPO training...")
    print("Monitor: rewards/chosen should increase, rewards/rejected should decrease")
    print("If kl_divergence > 5.0, stop and increase beta.")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nLoRA adapter saved to: {final_dir}")
    print("Download this folder to your PC: models/phi2_dpo_lora/")

    return trainer


# ── Standalone Colab execution block ─────────────────────────────────────────
# Copy everything below this line into a Colab cell:

COLAB_SCRIPT = '''
# ============================================================
# HalluDet-Lite Phase 5 — Phi-2 DPO Fine-Tuning
# Run this in Google Colab with T4 GPU
# Runtime > Change runtime type > T4 GPU
# ============================================================

# Step 1: Install dependencies
!pip install transformers==4.40.0 datasets peft==0.10.0 trl==0.8.6 bitsandbytes accelerate wandb -q

# Step 2: Mount Google Drive (checkpoints save here)
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = '/content/drive/MyDrive/halludet_phi2'
os.makedirs(DRIVE_DIR, exist_ok=True)

# Step 3: Upload your dpo_dataset.json to Colab
# (drag-and-drop to Colab file browser, or mount Drive)
DATA_PATH = '/content/dpo_dataset.json'
# Alternatively: DATA_PATH = f'{DRIVE_DIR}/dpo_dataset.json'

# Step 4: Run training
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

with open(DATA_PATH) as f:
    raw = json.load(f)
train_ds = Dataset.from_list(raw["train"])
eval_ds  = Dataset.from_list(raw["val"])

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", quantization_config=bnb,
    device_map="auto", trust_remote_code=True
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","dense"],
    bias="none"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", quantization_config=bnb,
    device_map="auto", trust_remote_code=True
)

cfg = DPOConfig(
    output_dir=DRIVE_DIR, beta=0.1, num_train_epochs=2,
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=1e-4, warmup_ratio=0.05, fp16=True, bf16=False,
    gradient_checkpointing=True, optim="paged_adamw_8bit",
    max_length=512, max_prompt_length=256,
    evaluation_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=3,
    logging_steps=10, report_to="none", remove_unused_columns=False
)

trainer = DPOTrainer(
    model=model, ref_model=ref_model, args=cfg,
    train_dataset=train_ds, eval_dataset=eval_ds, tokenizer=tokenizer
)
trainer.train()

final = f"{DRIVE_DIR}/final"
trainer.save_model(final)
tokenizer.save_pretrained(final)
print(f"Done! Download {final} to your PC as models/phi2_dpo_lora/")
'''
