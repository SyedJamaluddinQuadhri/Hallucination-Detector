"""
Patch script for phase5_phi2_dpo.ipynb
Fixes:
  Cell 5: add tokenizer.truncation_side = 'left'
  Cell 7: replace dataset_num_proc with dataset_kwargs, add warning suppression,
          add tokenizer sanity checks, fix import order
"""
import json, pathlib, sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

NB_PATH = pathlib.Path(__file__).parent / "phase5_phi2_dpo.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ── helpers ──────────────────────────────────────────────────────────────────
def src(cell):
    """Return source as a single string."""
    return "".join(cell["source"])

def set_src(cell, text):
    """Store source back as a list of lines (each ending with \\n except last)."""
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# ── Cell 5: add truncation_side = 'left' ─────────────────────────────────────
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "tokenizer.padding_side = 'left'" in src(cell):
        old = "tokenizer.padding_side = 'left'   # required for DPO"
        new = (
            "tokenizer.padding_side    = 'left'   # required for DPO (causal LM)\n"
            "tokenizer.truncation_side = 'left'   # FIX: suppress DPOTrainer tokenize_row truncation warning"
        )
        updated = src(cell).replace(old, new)
        if updated != src(cell):
            set_src(cell, updated)
            print("OK Cell 5: added truncation_side = 'left'")
        else:
            print("SKIP Cell 5: truncation_side already present or pattern not found")
        break

# ── Cell 7: full replacement ──────────────────────────────────────────────────
NEW_CELL7 = '''\
# -- Cell 7: DPO Training --------------------------------------------------
# Key fixes vs earlier versions:
#   1. evaluation_strategy  -> eval_strategy         (deprecated in trl>=0.12)
#   2. tokenizer=tokenizer  -> processing_class=     (deprecated in trl>=0.12)
#   3. DPOConfig is used (NOT TrainingArguments)      (required since trl>=0.9)
#   4. dataloader_num_workers=0                       (Colab multiprocessing fix)
#   5. Removed DPODataCollatorWithPadding import      (now built-in)
#   6. dataset_num_proc -> dataset_kwargs={'num_proc':1}  (trl>=1.0 API change - FIX)
#   7. tokenizer.truncation_side='left' in Cell 5    (fixes tokenize_row warning - FIX)

import warnings, glob

# Suppress known informational FutureWarnings from trl/transformers.
# These do NOT affect training correctness when using the corrected API below.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*tokenizer.*deprecated.*processing_class.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*`padding_side`.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r".*pad_token.*")

from trl import DPOTrainer, DPOConfig

# -- Verify tokenizer settings before training (sanity check) ----------------
print(f"Tokenizer pad_token:       {tokenizer.pad_token!r}")
print(f"Tokenizer padding_side:    {tokenizer.padding_side}")
print(f"Tokenizer truncation_side: {tokenizer.truncation_side}")
assert tokenizer.padding_side    == "left", "padding_side must be 'left' for DPO - re-run Cell 5"
assert tokenizer.truncation_side == "left", "truncation_side must be 'left' - re-run Cell 5"
assert tokenizer.pad_token is not None,     "pad_token must be set - re-run Cell 5"
print("Tokenizer checks passed.")

# -- Check if resuming from an existing checkpoint ---------------------------
checkpoints = sorted(
    glob.glob(f"{DRIVE_DIR}/checkpoint-*"),
    key=lambda p: int(p.split("-")[-1])
)
resume_from = checkpoints[-1] if checkpoints else None
if resume_from:
    print(f">>> Resuming from checkpoint: {resume_from}")
else:
    print("Starting fresh training run.")

cfg = DPOConfig(
    output_dir=DRIVE_DIR,
    beta=0.1,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,        # effective batch = 16
    learning_rate=1e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=True,
    bf16=False,                            # T4 = fp16 only (A100/H100 use bf16)
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_length=512,
    max_prompt_length=256,
    eval_strategy="steps",                # FIX: was "evaluation_strategy" (deprecated)
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,
    report_to="none",                      # set "wandb" if you have it configured
    remove_unused_columns=False,
    dataloader_num_workers=0,              # REQUIRED: Colab breaks with >0 workers
    # FIX (trl>=1.0): dataset_num_proc is NOT a top-level DPOConfig arg any more.
    # Use dataset_kwargs to pass num_proc for the internal dataset .map() calls.
    dataset_kwargs={"num_proc": 1},        # single-process map - avoids fork errors on Colab
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,                   # None = automatic PEFT adapter-disable mode
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,            # FIX: was tokenizer= (deprecated in trl>=0.12)
)

print("Starting DPO training...")
print("Watch: rewards/margins should increase steadily.")
print("If kl_divergence > 5.0, increase beta in cfg (e.g., beta=0.2).")
print("If OOM error: set USE_EXPLICIT_REF_MODEL=False in Cell 6 and restart.")
trainer.train(resume_from_checkpoint=resume_from)\
'''

patched = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "Cell 7: DPO Training" in src(cell):
        set_src(cell, NEW_CELL7)
        print("OK Cell 7: fully patched (dataset_kwargs, warning suppression, tokenizer checks)")
        patched = True
        break

if not patched:
    print("ERROR: Cell 7 not found!")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print("Done. Upload the updated notebook to Colab and re-run from Cell 5.")
