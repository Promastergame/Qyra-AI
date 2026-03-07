"""
config.py — CPU-first defaults tuned for старый Intel Core i5-3340.

Goals:
  - Keep parameter count low (~12–13M) so it fits in 8‑12 GB RAM and runs without AVX2.
  - Shorter context (256) to reduce memory bandwidth pressure.
  - Conservative batch sizes for single-socket desktop CPUs.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os

# -- Paths ----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_FT_DIR = os.path.join(PROJECT_ROOT, "data", "finetune")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "tokenizer")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "tok")

# -- Special tokens -------------------------------------------------------
SPECIAL_TOKENS: List[str] = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|tool_start|>",
    "<|tool_end|>",
    "<|tool_result|>",
]

# -- Tool Configuration ---------------------------------------------------
TOOL_CONFIG = {
    "max_tool_rounds": 5,
    "calculator_timeout": 2.0,
    "python_timeout": 10.0,
    "go_timeout": 30.0,
    "max_output_chars": 2000,
    "require_confirmation": True,
}


@dataclass
class ModelConfig:
    """Compact architecture ~12.5M parameters, safe for i5-3340 (no AVX2)."""
    vocab_size: int = 8000
    max_seq_len: int = 256          # shorter context keeps memory down
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2             # GQA: 3 query heads per KV head
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    tie_weights: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    use_qk_norm: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6


@dataclass
class PretrainConfig:
    """Training hyperparameters sized for a desktop CPU."""
    batch_size: int = 2              # keeps peak RAM under ~5 GB with 256 ctx
    grad_accum_steps: int = 4
    max_seq_len: int = 256
    stride: int = 128               # 50% overlap for coverage without big RAM hit
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.03
    decay_frac: float = 0.40
    max_steps: int = 5_000
    epochs: int = 5
    log_interval: int = 20
    eval_interval: int = 200
    save_interval: int = 1000
    val_split: float = 0.05
    seed: int = 42
    num_workers: int = 2
    use_gradient_checkpointing: bool = True
    z_loss_coeff: float = 1e-4
    label_smoothing: float = 0.1


@dataclass
class FinetuneConfig:
    """Fine-tuning setup that still fits on an i5 desktop."""
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_seq_len: int = 256
    lr: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.05
    decay_frac: float = 0.20
    max_steps: int = 1_000
    epochs: int = 3
    log_interval: int = 20
    eval_interval: int = 100
    save_interval: int = 500
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 2
    use_gradient_checkpointing: bool = False
    z_loss_coeff: float = 1e-4
    label_smoothing: float = 0.05
