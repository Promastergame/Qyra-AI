"""
config.py -- Central configuration for Qyra v2.

Architecture kept small for CPU training (~12.5M params).
Quality improvements: QK-norm, SDPA, label smoothing, better LR schedule.
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
    """CPU-friendly architecture -- ~12.5M parameters."""
    vocab_size: int = 8000
    max_seq_len: int = 256       # keep 256 for CPU speed
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2          # GQA: 3 query heads per KV head
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    tie_weights: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    use_qk_norm: bool = True     # NEW: QK-Norm for stability
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6


@dataclass
class PretrainConfig:
    """Training hyperparameters -- CPU-optimized for speed."""
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_seq_len: int = 256
    stride: int = 128            # 50% overlap for better coverage
    lr: float = 6e-4             # higher LR, faster convergence
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.03    # short warmup
    decay_frac: float = 0.30     # longer decay
    max_steps: int = 15_000
    epochs: int = 5
    log_interval: int = 25
    eval_interval: int = 250
    save_interval: int = 1000
    val_split: float = 0.05      # more training data
    seed: int = 42
    num_workers: int = 2          # parallel data loading
    use_gradient_checkpointing: bool = True
    z_loss_coeff: float = 1e-4
    label_smoothing: float = 0.1  # prevents overconfidence


@dataclass
class FinetuneConfig:
    """Training hyperparameters for fine-tuning."""
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
    max_steps: int = 5000
    epochs: int = 5
    log_interval: int = 20
    eval_interval: int = 200
    save_interval: int = 500
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 2
    use_gradient_checkpointing: bool = True
    z_loss_coeff: float = 1e-4
    label_smoothing: float = 0.05
