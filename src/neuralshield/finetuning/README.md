# Domain Adaptation: SecBERT HTTP MLM

Domain adaptation of SecBERT on HTTP requests using Masked Language Modeling.

## Quick Start

### 1. Prepare Training Data

```bash
uv run python -m neuralshield.finetuning.data.prepare_mlm_data \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/finetuning/data/http_corpus.txt
```

### 2. Train MLM

**Default training** (3 epochs, batch 8, lr 2e-5):

```bash
uv run python -m neuralshield.finetuning.train_mlm
```

**Custom configuration**:

```bash
uv run python -m neuralshield.finetuning.train_mlm \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --wandb-run-name "secbert-mlm-experiment-1"
```

**Without W&B logging**:

```bash
uv run python -m neuralshield.finetuning.train_mlm --no-wandb
```

## Module Structure

```
finetuning/
├── __init__.py
├── README.md                 # This file
├── TRAINING_PLAN.md          # Detailed implementation plan
├── train_mlm.py             # Main training script
├── data/
│   ├── README.md            # Data documentation
│   ├── prepare_mlm_data.py  # Data preparation script
│   └── http_corpus.txt      # Training corpus (47K docs)
└── models/                  # Trained models (created during training)
    └── secbert-http-adapted/
        ├── checkpoint-1000/
        ├── checkpoint-2000/
        └── final/           # Best model
```

## Training Configuration

| Parameter            | Default                       | Description               |
| -------------------- | ----------------------------- | ------------------------- |
| `--corpus`           | `data/http_corpus.txt`        | Training corpus path      |
| `--output-dir`       | `models/secbert-http-adapted` | Model output directory    |
| `--base-model`       | `jackaduma/SecBERT`           | Base model to adapt       |
| `--epochs`           | 3                             | Number of training epochs |
| `--batch-size`       | 8                             | Training batch size       |
| `--learning-rate`    | 2e-5                          | Learning rate             |
| `--validation-split` | 0.1                           | Validation set proportion |
| `--no-wandb`         | False                         | Disable W&B logging       |
| `--wandb-run-name`   | Auto-generated                | Custom W&B run name       |

## Expected Performance

**With 40GB A100**:

- Training time: ~1-2 hours (3 epochs)
- GPU memory: ~8-10 GB
- Disk space: ~9 GB (checkpoints + final model)

**Expected metrics**:

- Final validation loss: ~2.5-3.0
- Final perplexity: ~12-16
- Target: Perplexity < 15

## Post-Training Evaluation

### 1. Generate Embeddings

**Note**: Update SecBERT encoder to use adapted model first.

```bash
# Generate embeddings with adapted model
uv run python -m neuralshield.encoding.dump_embeddings \
  src/neuralshield/data/CSIC/train.jsonl \
  src/neuralshield/finetuning/embeddings/train_adapted.npz \
  --encoder secbert \
  --use-pipeline true
```

### 2. Train Mahalanobis Detector

```bash
uv run python src/scripts/train_mahalanobis_models.py \
  src/neuralshield/finetuning/embeddings/train_adapted.npz \
  src/neuralshield/finetuning/embeddings/test_adapted.npz \
  src/neuralshield/finetuning/models/mahalanobis_adapted.joblib
```

### 3. Evaluate Performance

```bash
uv run python experiments/06_mahalanobis_comparison/test_mahalanobis.py \
  src/neuralshield/finetuning/embeddings/train_adapted.npz \
  src/neuralshield/finetuning/embeddings/test_adapted.npz \
  src/neuralshield/finetuning/results \
  --max-fpr 0.05 --wandb
```

### 4. Compare Results

| Model                | Recall @ 5% FPR | Improvement |
| -------------------- | --------------- | ----------- |
| Base SecBERT         | 49.26%          | Baseline    |
| HTTP-adapted SecBERT | Target: 53-57%  | +4-8pp      |

## W&B Dashboard

Training metrics logged to W&B:

**Training**:

- `train/loss` - MLM loss
- `train/learning_rate` - LR schedule
- `train/perplexity` - Language quality

**Evaluation** (every 500 steps):

- `eval/loss` - Validation loss
- `eval/perplexity` - Validation quality

**Final**:

- `final/train_loss`
- `final/eval_loss`
- `final/perplexity`
- `final/total_steps`

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
uv run python -m neuralshield.finetuning.train_mlm --batch-size 4
```

### Slow Training

```bash
# Increase batch size (if GPU allows)
uv run python -m neuralshield.finetuning.train_mlm --batch-size 16
```

### Poor Convergence

```bash
# Increase learning rate or epochs
uv run python -m neuralshield.finetuning.train_mlm \
  --learning-rate 3e-5 --epochs 5
```

## Next Steps

After successful training:

1. ✅ Verify model saved in `models/secbert-http-adapted/final/`
2. ✅ Check W&B dashboard for training curves
3. ✅ Generate embeddings with adapted model
4. ✅ Evaluate on anomaly detection task
5. ✅ Compare with base SecBERT performance

## References

- [TRAINING_PLAN.md](TRAINING_PLAN.md) - Detailed implementation plan
- [data/README.md](data/README.md) - Training corpus documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Masked Language Modeling](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)
