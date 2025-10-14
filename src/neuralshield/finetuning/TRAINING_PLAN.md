# MLM Training Implementation Plan

**Date**: October 13, 2025  
**Task**: Domain adaptation of SecBERT on HTTP requests using Masked Language Modeling  
**GPU**: 40GB A100 (expected training time: 1-2 hours)

---

## Overview

**Goal**: Adapt SecBERT to HTTP domain by continuing pre-training on 47K valid HTTP requests

**Approach**:

- Unsupervised MLM training (no attack labels)
- Maintains anomaly detection paradigm
- Expected improvement: +4-5% recall on downstream task

---

## Decisions Made

### Architecture Choices

| Decision                    | Choice                      | Rationale                                                  |
| --------------------------- | --------------------------- | ---------------------------------------------------------- |
| **Framework**               | Hugging Face Trainer        | Battle-tested, handles distributed training, checkpointing |
| **Dataset Format**          | HF Dataset (in-memory)      | Built-in split, caching, efficient memory                  |
| **Masking Strategy**        | Standard 15%                | BERT default, proven, simple                               |
| **Train/Val Split**         | Random 90/10 (42.3K / 4.7K) | Simple, standard practice                                  |
| **Early Stopping**          | No                          | Want to see full learning curve (3 epochs)                 |
| **Baseline Comparison**     | Skip for now                | Focus on getting training working                          |
| **Catastrophic Forgetting** | Not monitoring initially    | 3 epochs unlikely to destroy knowledge                     |

### Training Configuration

**Default Hyperparameters** (will tune in next iteration):

```python
num_epochs = 3
batch_size = 8
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01
max_length = 512
mlm_probability = 0.15
gradient_accumulation_steps = 2  # Effective batch = 16
fp16 = True
```

**Why these defaults:**

- **3 epochs**: Standard for domain adaptation, avoids overfitting
- **lr 2e-5**: Lower than full pre-training (5e-5), prevents catastrophic forgetting
- **batch 8**: Conservative for A100, can increase if memory allows
- **gradient accum 2**: Effective batch of 16 for stability

---

## W&B Logging Strategy

### Project Configuration

```python
project = "neuralshield"
run_name = "secbert-http-mlm-{timestamp}"
tags = ["mlm", "domain-adaptation", "secbert", "http"]
```

### Metrics to Log

#### Training Metrics (Every 100 Steps)

- `train/loss` - MLM loss
- `train/learning_rate` - LR schedule
- `train/epoch` - Progress tracker
- `train/perplexity` - exp(loss), language quality metric
- `train/tokens_per_second` - Throughput
- `train/gpu_memory_allocated` - GPU usage

#### Evaluation Metrics (Every 500 Steps)

- `eval/loss` - Validation loss
- `eval/perplexity` - Validation quality
- `eval/runtime` - Inference speed
- `eval/samples_per_second` - Eval throughput

#### Per-Epoch Metrics

- `epoch/train_loss_avg` - Average training loss
- `epoch/eval_loss_avg` - Average eval loss
- `epoch/time_elapsed` - Training duration

#### Final Metrics (End of Training)

- `final/best_eval_loss` - Best checkpoint metric
- `final/best_perplexity` - Best language quality
- `final/total_training_time` - Full duration
- `final/total_steps` - Total optimization steps

### Artifacts to Log

- `config.json` - Model configuration
- `training_args.json` - Full training arguments
- `trainer_state.json` - Final trainer state
- `best_model/` - Best checkpoint by eval_loss

---

## Script Structure

### File: `train_mlm.py`

```
train_mlm.py
├── Imports (transformers, datasets, wandb, loguru)
├── MLMTrainingConfig (dataclass)
├── Helper Functions
│   ├── load_and_convert_corpus()
│   │   └── Load txt → split docs → HF Dataset → train_test_split
│   ├── tokenize_dataset()
│   │   └── Apply tokenization with truncation/padding
│   ├── compute_metrics()
│   │   └── Custom metrics (perplexity, accuracy)
│   ├── setup_training()
│   │   └── Model, tokenizer, data collator, trainer
│   └── log_final_metrics()
│       └── Aggregate and log to W&B
└── main()
    ├── 1. Parse CLI arguments
    ├── 2. Setup W&B
    ├── 3. Load and prepare corpus
    ├── 4. Load base model + tokenizer
    ├── 5. Setup trainer
    ├── 6. Train (W&B auto-logs)
    ├── 7. Evaluate final model
    ├── 8. Save best model
    └── 9. Log artifacts to W&B
```

---

## Implementation Components

### 1. Configuration Class

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MLMTrainingConfig:
    # Paths
    corpus_path: Path
    output_dir: Path
    base_model: str = "jackaduma/SecBERT"

    # Training
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # MLM
    mlm_probability: float = 0.15
    max_length: int = 512

    # Optimization
    gradient_accumulation_steps: int = 2
    fp16: bool = True

    # Evaluation
    validation_split: float = 0.1
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2

    # Logging
    logging_steps: int = 100
    wandb_project: str = "neuralshield"
    wandb_run_name: str = None
```

### 2. Dataset Preparation

**Input**: `http_corpus.txt` (37.48 MB, 47K docs)  
**Output**: `train_dataset` (42.3K), `val_dataset` (4.7K)

**Steps**:

1. Load text file
2. Split by `\n\n` (document separator)
3. Convert to HF Dataset: `Dataset.from_dict({"text": docs})`
4. Split: `dataset.train_test_split(test_size=0.1)`
5. Tokenize: `dataset.map(tokenize_fn, batched=True)`

### 3. Model & Tokenizer

**Base Model**: `jackaduma/SecBERT`

- Architecture: BERT-base (12 layers, 768 hidden, 12 heads)
- Vocab size: 52,000
- Max length: 512 tokens

**Load**:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT")
tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
```

### 4. Data Collator

**Purpose**: Handles dynamic padding and masking

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
```

**Masking Strategy**:

- 15% of tokens are selected for masking
- Of those 15%:
  - 80% replaced with `[MASK]`
  - 10% replaced with random token
  - 10% kept unchanged
- Special tokens never masked

### 5. Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    fp16=True,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=500,

    # Checkpointing
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # Logging
    logging_strategy="steps",
    logging_steps=100,
    report_to="wandb",

    # Performance
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)
```

### 6. Trainer Setup

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
```

### 7. Training Execution

```python
# Initialize W&B
import wandb
wandb.init(
    project="neuralshield",
    name=f"secbert-http-mlm-{timestamp}",
    tags=["mlm", "domain-adaptation", "secbert", "http"],
    config=config.__dict__
)

# Train
train_result = trainer.train()

# Evaluate
eval_result = trainer.evaluate()

# Save final model
trainer.save_model(output_dir / "final")
```

---

## Expected Training Metrics

### With 40GB A100

**Training Speed**:

- Tokens per second: ~10K-15K
- Samples per second: ~30-40
- Time per epoch: ~20-30 minutes
- **Total training time**: ~1-2 hours

**Memory Usage**:

- Model: ~3 GB
- Optimizer states: ~3 GB
- Activations (batch=8): ~2-4 GB
- **Total**: ~8-10 GB (plenty of headroom on A100)

**Disk Usage**:

- Checkpoint: ~3 GB each
- Keep 2 best: ~6 GB
- Final model: ~3 GB
- **Total**: ~9 GB

### Expected Loss Trajectory

**Baseline** (before training):

- Train perplexity: ~15-20 (cybersecurity corpus)
- Val perplexity: ~25-35 (HTTP domain mismatch)

**After Epoch 1**:

- Train perplexity: ~12-15
- Val perplexity: ~15-20 (significant drop)

**After Epoch 2**:

- Train perplexity: ~10-13
- Val perplexity: ~13-17

**After Epoch 3**:

- Train perplexity: ~9-12
- Val perplexity: ~12-16

**Target**: Val perplexity < 15 (good HTTP understanding)

---

## Output Structure

```
src/neuralshield/finetuning/models/secbert-http-adapted/
├── checkpoint-1000/
│   ├── config.json
│   ├── optimizer.pt
│   ├── pytorch_model.bin
│   ├── scheduler.pt
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-2000/
│   └── ... (same structure)
├── final/                      # Best model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── training_args.bin
├── trainer_state.json
└── README.md                   # Auto-generated by script
```

---

## CLI Interface

### Basic Usage

```bash
# Default training
uv run python -m neuralshield.finetuning.train_mlm

# Custom paths
uv run python -m neuralshield.finetuning.train_mlm \
  --corpus src/neuralshield/finetuning/data/http_corpus.txt \
  --output-dir src/neuralshield/finetuning/models/secbert-http-adapted
```

### Advanced Options

```bash
# Custom hyperparameters
uv run python -m neuralshield.finetuning.train_mlm \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 3e-5

# Different base model
uv run python -m neuralshield.finetuning.train_mlm \
  --base-model "bert-base-uncased"

# Disable W&B
uv run python -m neuralshield.finetuning.train_mlm \
  --no-wandb
```

---

## Post-Training Steps

After training completes:

1. **Verify model saved**

   ```bash
   ls -lh src/neuralshield/finetuning/models/secbert-http-adapted/final/
   ```

2. **Check W&B dashboard**

   - Training loss curve
   - Validation perplexity trend
   - GPU utilization

3. **Generate embeddings**

   ```bash
   uv run python -m neuralshield.encoding.dump_embeddings \
     src/neuralshield/data/CSIC/train.jsonl \
     src/neuralshield/finetuning/embeddings/train_adapted.npz \
     --encoder secbert \
     --model-path src/neuralshield/finetuning/models/secbert-http-adapted/final \
     --use-pipeline true
   ```

4. **Evaluate with Mahalanobis**

   ```bash
   uv run python experiments/06_mahalanobis_comparison/test_mahalanobis.py \
     src/neuralshield/finetuning/embeddings/train_adapted.npz \
     src/neuralshield/finetuning/embeddings/test_adapted.npz \
     src/neuralshield/finetuning/results \
     --max-fpr 0.05 --wandb
   ```

5. **Compare results**
   - Base SecBERT: 49.26% recall @ 5% FPR
   - HTTP-adapted SecBERT: Target 53-57% recall @ 5% FPR

---

## Success Criteria

### Training Success

- ✅ Training completes without errors
- ✅ Validation loss decreases consistently
- ✅ Final perplexity < 15 on HTTP corpus
- ✅ No NaN losses or gradient explosions
- ✅ Model checkpoints saved correctly

### Domain Adaptation Success

- ✅ Validation perplexity improves by >30% from baseline
- ✅ Training loss converges smoothly
- ✅ No signs of overfitting (train/val gap < 20%)
- ✅ GPU utilization > 80% (efficient training)

### Downstream Success (Next Phase)

- ✅ Recall improvement > 3% on anomaly detection
- ✅ FPR remains at 5%
- ✅ Embeddings cluster better (valid requests tighter)

---

## Troubleshooting

### Common Issues

**Out of Memory**:

- Reduce batch_size to 4
- Disable fp16 (unlikely on A100)
- Reduce max_length to 256

**Slow Training**:

- Increase batch_size to 16 (A100 can handle it)
- Check dataloader_num_workers
- Verify GPU utilization

**Poor Convergence**:

- Increase learning_rate to 3e-5
- Increase warmup_steps to 1000
- Train for more epochs

**NaN Loss**:

- Lower learning_rate to 1e-5
- Check for corrupted data
- Enable gradient clipping

---

## Next Iteration (Hyperparameter Tuning)

After baseline training, tune:

1. **Learning rate**: [1e-5, 2e-5, 3e-5, 5e-5]
2. **Batch size**: [8, 16, 32]
3. **Epochs**: [3, 5, 7]
4. **MLM probability**: [0.10, 0.15, 0.20]
5. **Warmup ratio**: [0.05, 0.1, 0.15]

Use W&B sweeps for efficient hyperparameter search.

---

## Timeline

| Phase                     | Duration  | Description                           |
| ------------------------- | --------- | ------------------------------------- |
| **Script implementation** | 45 min    | Write train_mlm.py                    |
| **Training**              | 1-2 hours | 3 epochs on A100                      |
| **Evaluation**            | 30 min    | Generate embeddings, test Mahalanobis |
| **Analysis**              | 30 min    | Compare results, document findings    |

**Total**: ~3-4 hours from start to finish

---

## Ready for Implementation

Plan approved with:

- ✅ Hugging Face Trainer
- ✅ HF Dataset (in-memory, random 90/10 split)
- ✅ Standard 15% masking
- ✅ Default hyperparameters (3 epochs, lr 2e-5)
- ✅ Detailed W&B logging
- ✅ No early stopping
- ✅ No baseline comparison (for now)
- ✅ 40GB A100 available

**Next step**: Implement `train_mlm.py`
