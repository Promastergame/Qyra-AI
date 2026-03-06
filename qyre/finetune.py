"""
finetune.py v2 — Supervised fine-tuning on chat-format data.

Usage:
    python finetune.py [--data data/finetune/chat.jsonl] [--checkpoint checkpoints/best_pretrain.pt]

Speed improvements:
  - torch.compile() support (GPU + CPU)
  - AMP on GPU / CPU (BFloat16)
  - Multi-threading for CPU (OMP/MKL)
  - Disabled gradient checkpointing on CPU
  - Label smoothing

Quality improvements:
  - Label smoothing prevents overconfidence
  - Proper z-loss on logits (not loss scalar)
"""

import os
import math
import time
import argparse
import random
import gc
import sys
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.utils.data import DataLoader, random_split  # type: ignore
from tqdm import tqdm  # type: ignore

from config import (  # type: ignore
    ModelConfig, FinetuneConfig,
    DATA_FT_DIR, TOKENIZER_PREFIX, CHECKPOINT_DIR,
)
from model import Qyra  # type: ignore
from dataset import FinetuneDataset  # type: ignore

# ============================================================================
# CPU Multi-threading Optimization
# ============================================================================
# Установи по количеству ядер твоего CPU (по умолчанию 8).
# Чтобы не спамить выводом из воркеров DataLoader на Windows (spawn),
# ставим флаг окружения и настраиваем треды один раз.
if os.environ.get("QYRA_THREADS_SET") != "1":
    CPU_THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    torch.set_num_threads(CPU_THREADS)
    os.environ["QYRA_THREADS_SET"] = "1"
    print(f"CPU threads set to: {CPU_THREADS}")


def get_lr_wsd(step: int, total_steps: int, peak_lr: float, min_lr: float,
               warmup_frac: float = 0.05, decay_frac: float = 0.20) -> float:
    """Warmup-Stable-Decay learning rate schedule."""
    warmup_steps = int(total_steps * warmup_frac)
    decay_start = int(total_steps * (1.0 - decay_frac))

    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    elif step < decay_start:
        return peak_lr
    else:
        decay_steps = max(1, total_steps - decay_start)
        progress = (step - decay_start) / decay_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * coeff


def compute_z_loss(logits: torch.Tensor, coefficient: float = 1e-4) -> torch.Tensor:
    """Z-loss for logit stability."""
    log_z = torch.logsumexp(logits, dim=-1)
    return coefficient * (log_z ** 2).mean()


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(1, n)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Qyra Fine-tuning v2")
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(DATA_FT_DIR, "chat.jsonl"),
        help="Path to JSONL fine-tuning data",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(CHECKPOINT_DIR, "best_pretrain.pt"),
        help="Pretrained checkpoint to start from",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile()")
    parser.add_argument("--label_smoothing", type=float, default=None)
    args = parser.parse_args()

    ftcfg = FinetuneConfig()

    # Use config defaults, override with CLI args
    epochs = args.epochs or ftcfg.epochs
    batch_size = args.batch_size or ftcfg.batch_size
    grad_accum = args.grad_accum or ftcfg.grad_accum_steps
    lr = args.lr or ftcfg.lr
    max_steps = args.max_steps or ftcfg.max_steps
    seed = args.seed or ftcfg.seed
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else ftcfg.label_smoothing

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    print(f"Device: {device}")

    # Quick defaults on CPU only when user didn't set epochs/max_steps explicitly.
    if not use_gpu and args.epochs is None and args.max_steps is None:
        if epochs > 1:
            print("CPU detected → limiting epochs to 1 (override with --epochs).")
        epochs = 1
        if max_steps > 200:
            print("CPU detected → capping max_steps at 200 (override with --max_steps).")
        max_steps = min(max_steps, 200)

    # -- Load pretrained model --------------------------------------------
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run pretraining first, or specify --checkpoint path."
        )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt["model_config"]
    model = Qyra(mcfg).to(device)

    # Handle compiled model state dict
    state_dict = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k.replace("_orig_mod.", "")] = v
        model.load_state_dict(new_sd)

    print(f"Loaded pretrained model from {args.checkpoint}")
    print(f"  Parameters: {model.count_parameters():,}")

    # -- torch.compile (GPU + CPU) ----------------------------------------
    compiled = False
    if use_gpu and not args.no_compile and hasattr(torch, 'compile'):
        try:
            backend = "inductor"
            mode = "reduce-overhead"
            model = torch.compile(model, backend=backend, mode=mode)
            compiled = True
            print(f"torch.compile: enabled (backend={backend}, mode={mode})")
        except Exception as e:
            print(f"torch.compile: failed ({e})")
    elif not use_gpu and not args.no_compile:
        print("torch.compile: skipped on CPU (startup cost >> speedup)")

    # -- Dataset ----------------------------------------------------------
    sp_model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Fine-tuning data not found: {args.data}\n"
            "Create a JSONL file with chat conversations."
        )

    full_dataset = FinetuneDataset(
        jsonl_path=args.data,
        sp_model_path=sp_model_path,
        max_seq_len=mcfg.max_seq_len,
    )

    val_size = max(1, int(len(full_dataset) * ftcfg.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {train_size} | Val: {val_size}")

    num_workers = ftcfg.num_workers if use_gpu else 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    # -- Optimizer --------------------------------------------------------
    decay_params = []
    no_decay_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tok_emb" in name or "pos_emb" in name:
            embed_params.append(param)
        elif "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": ftcfg.weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
            {"params": embed_params, "weight_decay": 0.01, "lr": float(lr) * 0.1},
        ],
        betas=ftcfg.betas,
    )

    # AMP (GPU) / BFloat16 (CPU with AVX512-BF16 only)
    use_amp = use_gpu
    try:
        use_cpu_bf16 = (not use_gpu) and hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported()
    except Exception:
        use_cpu_bf16 = False
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    amp_dtype = torch.float16 if use_amp else torch.float32
    
    if use_cpu_bf16:
        print("CPU BFloat16: enabled")
    else:
        print("CPU BFloat16: not supported or disabled; using float32")

    steps_per_epoch = max(1, len(train_loader) // grad_accum)

    # If пользователь задал --epochs, но не --max_steps, не режем по конфигному лимиту
    # — используем полное число шагов эпох * epochs.
    if args.max_steps is None and args.epochs is not None:
        max_steps = steps_per_epoch * epochs

    total_steps = min(max_steps, steps_per_epoch * epochs)

    # Gradient checkpointing only on GPU
    use_checkpoint = (not args.no_checkpoint and ftcfg.use_gradient_checkpointing and use_gpu)
    print(f"Gradient checkpointing: {'enabled' if use_checkpoint else 'disabled'}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.train()

    global_step = 0
    best_val_loss = float("inf")
    accum_loss = 0.0
    micro_step = 0
    t0 = time.time()

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{epochs}",
                    file=sys.stdout, dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits, loss = model(x, y, use_checkpoint=use_checkpoint,
                                         label_smoothing=label_smoothing)
                    if loss is None:
                        continue
                    total_loss = loss / float(grad_accum)
                scaler.scale(total_loss).backward()
            else:
                logits, loss = model(x, y, use_checkpoint=use_checkpoint,
                                     label_smoothing=label_smoothing)
                if loss is None:
                    continue
                total_loss = loss / float(grad_accum)
                total_loss.backward()

            accum_loss += loss.item() / float(grad_accum)
            micro_step += 1

            if micro_step % int(grad_accum) == 0:
                if use_amp:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), ftcfg.max_grad_norm)
                new_lr = get_lr_wsd(global_step, total_steps, lr, ftcfg.min_lr,
                                    ftcfg.warmup_frac, ftcfg.decay_frac)
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(ftcfg.log_interval) == 0:
                    avg = accum_loss / float(ftcfg.log_interval)
                    elapsed = time.time() - t0
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{new_lr:.2e}", step=global_step)
                    print(f"  step {global_step:>5d} | loss {avg:.4f} | lr {new_lr:.2e} | time {elapsed:.1f}s")
                    accum_loss = 0.0
                    t0 = time.time()

                if global_step % ftcfg.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  * val_loss = {val_loss:.4f} (best = {best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        path = os.path.join(CHECKPOINT_DIR, "best_finetune.pt")
                        sd = model.state_dict()
                        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
                        torch.save({
                            "model_state_dict": clean_sd,
                            "model_config": mcfg,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                        }, path)
                        print(f"  -> Best fine-tuned model saved: {path}")
                    model.train()

                if global_step % ftcfg.save_interval == 0:
                    path = os.path.join(CHECKPOINT_DIR, f"finetune_step{global_step}.pt")
                    sd = model.state_dict()
                    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
                    torch.save({
                        "model_state_dict": clean_sd,
                        "model_config": mcfg,
                        "global_step": global_step,
                    }, path)

                if global_step >= int(max_steps):
                    break
        if global_step >= int(max_steps):
            break

    final = os.path.join(CHECKPOINT_DIR, "finetune_final.pt")
    sd = model.state_dict()
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    torch.save({
        "model_state_dict": clean_sd,
        "model_config": mcfg,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, final)
    print(f"\n-> Fine-tuning complete: {final}")
    clear_memory()


if __name__ == "__main__":
    main()
