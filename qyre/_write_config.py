"""Helper script to write config.py with special tokens."""

CONTENT = r'''"""
config.py -- Central configuration for Qyra v2.

Improved:
  - Bigger model for better quality (d_model 512, 8 layers, 8 heads)
  - Longer context (max_seq_len 512)
  - Better training hyperparameters
  - Label smoothing support
  - CPU-optimized training settings
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
    """Larger CPU-optimized architecture (~30M parameters)."""
    vocab_size: int = 8000
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    n_layers: int = 8
    d_ff: int = 2048
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
    """Optimal training hyperparameters for CPU."""
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    stride: int = 512
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.03
    decay_frac: float = 0.30
    max_steps: int = 10_000
    epochs: int = 10
    log_interval: int = 25
    eval_interval: int = 250
    save_interval: int = 1000
    val_split: float = 0.05
    seed: int = 42
    num_workers: int = 4
    use_gradient_checkpointing: bool = True
    z_loss_coeff: float = 1e-4
    label_smoothing: float = 0.1


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration."""
    batch_size: int = 8
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    lr: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.05
    decay_frac: float = 0.20
    max_steps: int = 2000
    epochs: int = 10
    log_interval: int = 20
    eval_interval: int = 200
    save_interval: int = 500
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 4
    use_gradient_checkpointing: bool = True
    z_loss_coeff: float = 1e-4
    label_smoothing: float = 0.05
'''

if __name__ == "__main__":
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(CONTENT)
    print("Successfully updated config.py.")
