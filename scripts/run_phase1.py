"""
run_phase1.py — Train the NLI hallucination detector (DeBERTa-base).

Runtime: ~3–4 hours on RTX 2050
VRAM:    ~3.2GB peak
Output:  models/nli_detector/best/

Usage:
    python scripts/run_phase1.py
    python scripts/run_phase1.py --epochs 2      # quick test run
    python scripts/run_phase1.py --no-fever       # skip FEVER dataset
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, setup_environment
from src.phase1_nli.train import train_nli_model

log = get_logger("script.phase1")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train NLI detector")
    parser.add_argument("--epochs",     type=int,   default=4,     help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=8,     help="Batch size (RTX 2050 safe: 8)")
    parser.add_argument("--lr",         type=float, default=2e-5,  help="Learning rate")
    parser.add_argument("--max-len",    type=int,   default=256,   help="Max token length")
    parser.add_argument("--no-fever",   action="store_true",       help="Skip FEVER-NLI")
    parser.add_argument("--no-wandb",   action="store_true",       help="Disable W&B logging")
    parser.add_argument("--resume",     type=str,   default=None,  help="Resume from checkpoint path")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("HalluDet-Lite — Phase 1: NLI Detector Training")
    log.info("=" * 60)
    log.info(f"Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    log.info("Expected time: 3–4 hours on RTX 2050")
    log.info("TIP: Run overnight with: nohup python scripts/run_phase1.py > logs/phase1.log &")
    log.info("=" * 60)

    setup_environment()

    # Build dataset with or without FEVER
    from src.phase1_nli.dataset import build_halueval_nli_dataset
    from src.utils import get_project_root, load_json

    dataset_path = str(get_project_root() / "data" / "processed" / "nli_dataset.json")

    if not os.path.exists(dataset_path):
        log.info("Building NLI dataset...")
        build_halueval_nli_dataset(
            use_fever=not args.no_fever,
            use_summarization=True,
            save_path=dataset_path,
        )

    trainer, model, tokenizer = train_nli_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_len,
        wandb_project=None if args.no_wandb else "halludet-lite",
        resume_from_checkpoint=args.resume,
    )

    log.info("\n✓ Phase 1 complete!")
    log.info("Next step: python scripts/run_phase2.py")


if __name__ == "__main__":
    main()
