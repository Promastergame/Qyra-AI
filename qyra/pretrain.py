"""
pretrain.py v2 — Fast next-token pretraining on plain text.

Speed improvements:
  - torch.compile() for CPU/GPU kernel fusion
  - Automatic Mixed Precision (AMP) on GPU
  - Disabled gradient checkpointing on CPU (slower on CPU!)
  - Parallel data loading (num_workers=2)
  - Larger batch with less grad_accum -> fewer Python overhead steps
  - Reduced logging/eval frequency
  - Cosine schedule with warm restarts option

Quality improvements:
  - Label smoothing (0.1) prevents overconfidence
  - Larger effective context (512 tokens with 256 stride overlap)
  - Higher learning rate with shorter warmup for faster convergence
  - Z-loss only computed at log intervals (saves compute)
  - Better optimizer groups (separate LR for embeddings)

Usage:
    python pretrain.py [--epochs 5] [--batch_size 4] [--lr 6e-4] ...
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
    ModelConfig, PretrainConfig,
    DATA_RAW_DIR, TOKENIZER_PREFIX, CHECKPOINT_DIR,
)
from model import Qyra  # type: ignore
from dataset import PretrainDataset  # type: ignore


# -- Learning-rate schedule -- Warmup-Stable-Decay ------------------------

def get_lr_wsd(step: int, total_steps: int, peak_lr: float, min_lr: float,
               warmup_frac: float = 0.03, decay_frac: float = 0.30) -> float:
    """
    Warmup-Stable-Decay schedule.
    
    Phase 1 (0-3%):    linear warmup from 0 to peak_lr
    Phase 2 (3-70%):   stable at peak_lr
    Phase 3 (70-100%): cosine decay to min_lr
    """
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


# -- Z-loss for logit stability ------------------------------------------

def compute_z_loss(logits: torch.Tensor, coefficient: float = 1e-4) -> torch.Tensor:
    """
    Auxiliary loss to prevent logit drift.
    Penalizes large logit magnitudes.
    """
    log_z = torch.logsumexp(logits, dim=-1)
    return coefficient * (log_z ** 2).mean()


# -- Memory management ---------------------------------------------------

def setup_memory_efficient_cuda():
    """Configure CUDA for minimum memory usage."""
    if not torch.cuda.is_available():
        return
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.6"
    )
    torch.backends.cudnn.benchmark = True  # Enable for fixed input sizes


def clear_memory():
    """Free garbage collection and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# -- Validation -----------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    """Run model on validation set, return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(1, n)


# -- Main training loop ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qyra Pretraining v2")
    # Model
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile()")
    parser.add_argument("--label_smoothing", type=float, default=None)
    args = parser.parse_args()

    # -- Use config defaults, override with CLI args --------------------
    tcfg = PretrainConfig()
    mcfg_defaults = ModelConfig()

    # Model config
    d_model = args.d_model or mcfg_defaults.d_model
    n_heads = args.n_heads or mcfg_defaults.n_heads
    n_layers = args.n_layers or mcfg_defaults.n_layers
    d_ff = args.d_ff or mcfg_defaults.d_ff
    dropout = args.dropout if args.dropout is not None else mcfg_defaults.dropout
    vocab_size = args.vocab_size or mcfg_defaults.vocab_size
    max_seq_len = args.max_seq_len or tcfg.max_seq_len

    # Auto-derive n_kv_heads: use CLI value if given, else pick a valid divisor
    if args.n_kv_heads is not None:
        n_kv_heads = args.n_kv_heads
    else:
        # Try config default, fall back to a valid divisor of n_heads
        default_kv = mcfg_defaults.n_kv_heads
        if n_heads % default_kv == 0:
            n_kv_heads = default_kv
        else:
            # Find largest divisor of n_heads that's <= n_heads//2 (for GQA benefit)
            n_kv_heads = max(d for d in range(1, n_heads + 1) if n_heads % d == 0 and d <= n_heads // 2) if n_heads > 1 else 1
            print(f"  Auto-adjusted n_kv_heads to {n_kv_heads} (n_heads={n_heads})")

    # Training config
    batch_size = args.batch_size or tcfg.batch_size
    grad_accum = args.grad_accum or tcfg.grad_accum_steps
    lr = args.lr or tcfg.lr
    max_steps = args.max_steps or tcfg.max_steps
    stride = args.stride or tcfg.stride
    seed = args.seed or tcfg.seed
    data_dir = args.data_dir or DATA_RAW_DIR
    epochs = args.epochs or tcfg.epochs
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else tcfg.label_smoothing

    # -- Seed -------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -- Memory-efficient CUDA -------------------------------------------
    setup_memory_efficient_cuda()

    # -- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    print(f"Device: {device}")
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"  GPU: {gpu_name} ({vram_mb:.0f} MB)")

    # -- Tokenizer -------------------------------------------------------
    sp_model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(sp_model_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {sp_model_path}. "
            "Run `python train_tokenizer.py` first."
        )

    # -- Dataset ----------------------------------------------------------
    full_dataset = PretrainDataset(
        data_dir=data_dir,
        sp_model_path=sp_model_path,
        max_seq_len=max_seq_len,
        stride=stride,
    )

    val_size = max(1, int(len(full_dataset) * tcfg.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {train_size:,} samples | Val: {val_size:,} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        persistent_workers=(tcfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # val doesn't need parallel loading
        pin_memory=use_gpu,
    )

    # -- Model ------------------------------------------------------------
    mcfg = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )
    model = Qyra(mcfg).to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Model memory: ~{n_params * 4 / 1e6:.1f} MB (FP32)")

    # -- torch.compile for speed -----------------------------------------
    # NOTE: torch.compile on CPU/Windows requires MSVC (cl.exe).
    # Skip on CPU — the overhead of compilation isn't worth it there anyway.
    compiled = False
    if not args.no_compile and use_gpu and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compiled = True
            print("torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile: failed ({e}), continuing without")

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
            {"params": decay_params, "weight_decay": tcfg.weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
            {"params": embed_params, "weight_decay": 0.01, "lr": float(lr) * 0.3},
        ],
        betas=tcfg.betas,
        fused=False,  # fused only for CUDA
    )

    # -- AMP scaler for GPU -----------------------------------------------
    use_amp = use_gpu
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    amp_dtype = torch.float16 if use_amp else torch.float32

    # -- Resume from checkpoint -------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Handle compiled model state dict
        state_dict = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Strip _orig_mod. prefix from compiled model
            new_sd = {}
            for k, v in state_dict.items():
                new_sd[k.replace("_orig_mod.", "")] = v
            model.load_state_dict(new_sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # -- Training config info ---------------------------------------------
    # Guard against tiny datasets where len(loader) < grad_accum.
    # Without this, steps_per_epoch can be 0 → total_steps=0 → LR schedule divide-by-zero.
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = min(max_steps, steps_per_epoch * epochs)

    # Decide on gradient checkpointing
    # On CPU gradient checkpointing is SLOWER (recomputes forward pass)
    # Only enable on GPU where VRAM savings matter
    use_checkpoint = False
    if not args.no_checkpoint and tcfg.use_gradient_checkpointing and use_gpu:
        use_checkpoint = True

    print(f"Steps/epoch: {steps_per_epoch:,} | Total steps: {total_steps:,}")
    print(f"Effective batch size: {batch_size * grad_accum}")
    print(f"Gradient checkpointing: {'enabled' if use_checkpoint else 'disabled'}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"torch.compile: {'enabled' if compiled else 'disabled'}")
    if use_amp:
        print(f"Mixed precision: enabled (float16)")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.train()

    # -- Training loop ----------------------------------------------------
    accum_loss = 0.0
    accum_z_loss = 0.0
    micro_step = 0
    t0 = time.time()
    log_interval = tcfg.log_interval

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                    file=sys.stdout, dynamic_ncols=True)
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward + backward with optional AMP
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits, loss = model(x, y, use_checkpoint=use_checkpoint,
                                         label_smoothing=label_smoothing)
                    # Z-loss only at log steps to save compute
                    if tcfg.z_loss_coeff > 0 and (micro_step + 1) % int(grad_accum * log_interval) < grad_accum:
                        z_loss = compute_z_loss(logits, tcfg.z_loss_coeff)
                    else:
                        z_loss = torch.tensor(0.0, device=device)
                    total_loss = (loss + z_loss) / grad_accum

                scaler.scale(total_loss).backward()
            else:
                logits, loss = model(x, y, use_checkpoint=use_checkpoint,
                                     label_smoothing=label_smoothing)
                # Z-loss only at log steps to save compute
                if tcfg.z_loss_coeff > 0 and (micro_step + 1) % int(grad_accum * log_interval) < grad_accum:
                        z_loss = compute_z_loss(logits, tcfg.z_loss_coeff)
                else:
                    z_loss = torch.tensor(0.0, device=device)
                total_loss = (loss + z_loss) / grad_accum

                total_loss.backward()

            accum_loss += loss.item() / float(grad_accum)
            accum_z_loss += z_loss.item() / float(grad_accum)
            micro_step += 1

            # -- Optimizer step -------------------------------------------
            if micro_step % int(grad_accum) == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                    # Update LR
                    new_lr = get_lr_wsd(global_step, total_steps, lr, tcfg.min_lr,
                                       tcfg.warmup_frac, tcfg.decay_frac)
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                    new_lr = get_lr_wsd(global_step, total_steps, lr, tcfg.min_lr,
                                       tcfg.warmup_frac, tcfg.decay_frac)
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # -- Logging ----------------------------------------------
                if global_step % log_interval == 0:
                    elapsed = time.time() - t0
                    avg_loss = accum_loss / float(log_interval)
                    avg_z_loss = accum_z_loss / float(log_interval)
                    tokens_per_sec = (log_interval * grad_accum * batch_size * max_seq_len) / elapsed
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{new_lr:.2e}",
                        step=global_step,
                    )
                    print(
                        f"  step {global_step:>6d} | "
                        f"loss {avg_loss:.4f} (+z {avg_z_loss:.2e}) | "
                        f"lr {new_lr:.2e} | "
                        f"tok/s {tokens_per_sec:.0f} | "
                        f"time {elapsed:.1f}s"
                    )
                    accum_loss = 0.0
                    accum_z_loss = 0.0
                    t0 = time.time()

                # -- Validation -------------------------------------------
                if global_step % tcfg.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  * val_loss = {val_loss:.4f} (best = {best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(CHECKPOINT_DIR, "best_pretrain.pt")
                        # Get state dict (handle compiled model)
                        sd = model.state_dict()
                        clean_sd = {}
                        for k, v in sd.items():
                            clean_sd[k.replace("_orig_mod.", "")] = v
                        torch.save({
                            "model_state_dict": clean_sd,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "model_config": mcfg,
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_val_loss": best_val_loss,
                        }, save_path)
                        print(f"  -> New best model saved: {save_path}")

                    model.train()

                # -- Periodic checkpoint ----------------------------------
                if global_step % tcfg.save_interval == 0:
                    save_path = os.path.join(CHECKPOINT_DIR, f"pretrain_step{global_step}.pt")
                    sd = model.state_dict()
                    clean_sd = {}
                    for k, v in sd.items():
                        clean_sd[k.replace("_orig_mod.", "")] = v
                    torch.save({
                        "model_state_dict": clean_sd,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": mcfg,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                    }, save_path)
                    print(f"  -> Checkpoint saved: {save_path}")

                if global_step >= max_steps:
                    print(f"Reached max_steps ({max_steps}). Stopping.")
                    break

        if global_step >= max_steps:
            break

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} done | val_loss = {val_loss:.4f}")

    # -- Final save -------------------------------------------------------
    final_path = os.path.join(CHECKPOINT_DIR, "pretrain_final.pt")
    sd = model.state_dict()
    clean_sd = {}
    for k, v in sd.items():
        clean_sd[k.replace("_orig_mod.", "")] = v
    torch.save({
        "model_state_dict": clean_sd,
        "model_config": mcfg,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, final_path)
    print(f"\n-> Training complete. Final model: {final_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # Clear memory
    clear_memory()


if __name__ == "__main__":
    main()
