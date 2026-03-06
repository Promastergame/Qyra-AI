"""
train_ru.py — Unified pretrain/finetune runner with small CPU presets.

Supports:
  - Model size presets: 12M, 20M, 35M (CPU-friendly)
  - Token-based training limits
  - Flexible data paths
  - Mixed precision, gradient checkpointing
  - Auto-resume from checkpoint

Usage (CPU, e.g. i5-3340):
    python train_ru.py --mode pretrain --model_size 12M --batch_size 2 --grad_accum 4
    python train_ru.py --mode finetune --model_size 12M --batch_size 4
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


# Model size presets: (d_model, n_heads, n_kv_heads, n_layers, d_ff)
# Slimmed down for older CPUs (e.g., Intel Core i5-3340).
MODEL_PRESETS = {
    "12M":  (384, 6, 2, 6, 1536),   # matches tiny-gpt/config.py defaults
    "20M":  (416, 8, 2, 8, 1664),   # slightly larger but still CPU friendly
    "35M":  (448, 10, 2, 8, 2048),  # upper limit for 8–12 GB RAM
    "60M":  (512, 8, 4, 16, 2560),  # larger model for 12–16 GB RAM (d_model/n_heads = 64)
}


def parse_tokens(token_str: str) -> int:
    """Parse token string like '1B', '500M', '60M' to int."""
    token_str = token_str.upper().strip()
    if token_str.endswith("B"):
        return int(float(token_str[:-1]) * 1e9)
    elif token_str.endswith("M"):
        return int(float(token_str[:-1]) * 1e6)
    elif token_str.endswith("K"):
        return int(float(token_str[:-1]) * 1e3)
    else:
        return int(token_str)


def run_pretrain(args):
    """Run pretraining."""
    cmd = ["python", "pretrain.py"]

    # Model size
    if args.model_size in MODEL_PRESETS:
        d_model, n_heads, n_kv_heads, n_layers, d_ff = MODEL_PRESETS[args.model_size]
        cmd.extend([
            "--d_model", str(d_model),
            "--n_heads", str(n_heads),
            "--n_kv_heads", str(n_kv_heads),
            "--n_layers", str(n_layers),
            "--d_ff", str(d_ff),
        ])

    # Training params
    if args.max_tokens:
        # Estimate steps from tokens
        # steps = max_tokens / (batch_size * grad_accum * seq_len)
        seq_len = args.max_seq_len or 512
        batch = args.batch_size or 16
        grad_accum = args.grad_accum or 4
        max_tokens = parse_tokens(args.max_tokens)
        steps = max_tokens // (batch * grad_accum * seq_len)
        cmd.extend(["--max_steps", str(steps)])
        print(f"  Max tokens: {max_tokens:,} -> ~{steps:,} steps")

    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.grad_accum:
        cmd.extend(["--grad_accum", str(args.grad_accum)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.max_seq_len:
        cmd.extend(["--max_seq_len", str(args.max_seq_len)])
    if args.data_dir:
        cmd.extend(["--data_dir", args.data_dir])
    if args.resume:
        cmd.extend(["--resume", args.resume])

    # Speed options
    if args.no_compile:
        cmd.append("--no_compile")
    if args.no_checkpoint:
        cmd.append("--no_checkpoint")

    # Mixed precision
    if args.fp16:
        print("  [WARN] AMP is auto-enabled on GPU")

    print(f"\n>>> Running pretrain: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def run_finetune(args):
    """Run fine-tuning."""
    cmd = ["python", "finetune.py"]

    # Required
    if args.data:
        cmd.extend(["--data", args.data])
    else:
        cmd.extend(["--data", "data/finetune/chat_merged.jsonl"])

    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    else:
        cmd.extend(["--checkpoint", "checkpoints/best_pretrain.pt"])

    # Training params
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.max_steps:
        cmd.extend(["--max_steps", str(args.max_steps)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    if args.grad_accum:
        cmd.extend(["--grad_accum", str(args.grad_accum)])

    # Speed options
    if args.no_compile:
        cmd.append("--no_compile")
    if args.no_checkpoint:
        cmd.append("--no_checkpoint")

    print(f"\n>>> Running finetune: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode


def run_test(args):
    """Run quick test training (debug mode)."""
    print("\n>>> Running TEST mode (quick debug)\n")

    # Test pretrain: 1 epoch, 10 steps
    cmd = ["python", "pretrain.py",
           "--epochs", "1", "--max_steps", "10",
           "--batch_size", "2", "--no_compile"]
    print(f"Test pretrain: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    # Test finetune: 1 epoch, 5 steps
    cmd = ["python", "finetune.py",
           "--data", "data/finetune/chat.jsonl",
           "--epochs", "1", "--max_steps", "5",
           "--batch_size", "2", "--no_compile"]
    print(f"Test finetune: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for Qyra models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune", "test"],
        help="Training mode",
    )

    # Model size
    parser.add_argument(
        "--model_size",
        type=str,
        default="12M",
        choices=list(MODEL_PRESETS.keys()),
        help="Model size preset (12M, 20M, 35M) — CPU friendly",
    )

    # Data
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to training data (JSONL for finetune, dir for pretrain)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Data directory for pretrain",
    )

    # Training limits
    parser.add_argument(
        "--max_tokens",
        type=str,
        help="Max tokens to train on (e.g., 1B, 500M, 60M). Auto-calculates max_steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Max training steps (overrides max_tokens if both specified)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
    )

    # Batch/optimization
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        help="Batch size per device",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Max sequence length",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for finetune/resume",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume pretrain from checkpoint",
    )

    # Speed options
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile()",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision (auto on GPU)",
    )

    # Test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test (debug mode)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("QYRA TRAINING SCRIPT")
    print("=" * 60)
    print(f"Mode:        {args.mode}")
    print(f"Model size:  {args.model_size}")
    if args.max_tokens:
        print(f"Max tokens:  {args.max_tokens}")
    if args.max_steps:
        print(f"Max steps:   {args.max_steps}")
    if args.epochs:
        print(f"Epochs:      {args.epochs}")
    if args.batch_size:
        print(f"Batch size:  {args.batch_size}")
    print("=" * 60)

    # Test mode
    if args.test or args.mode == "test":
        sys.exit(run_test(args))

    # Run training
    if args.mode == "pretrain":
        sys.exit(run_pretrain(args))
    elif args.mode == "finetune":
        sys.exit(run_finetune(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
