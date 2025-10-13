from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_fp16() -> bool:
    """Auto-detect if fp16 should be enabled based on device."""
    if torch.cuda.is_available():
        return True
    if torch.backends.mps.is_available():
        return False  # MPS doesn't support fp16 well
    return False  # CPU doesn't support fp16


@dataclass
class MLMConfig:
    """Configuration for MLM training."""

    corpus_path: Path
    output_dir: Path
    base_model: str = "jackaduma/SecBERT"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # MLM parameters
    mlm_probability: float = 0.15
    max_length: int = 512

    # Optimization
    gradient_accumulation_steps: int = 2
    fp16: bool = field(default_factory=_default_fp16)

    # Evaluation & checkpointing
    validation_split: float = 0.1
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2

    # Logging
    logging_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = "neuralshield"
    wandb_run_name: str | None = None
