# Running Experiment 02: Dense Embeddings Comparison

**Quick execution guide** for comparing TF-IDF vs dense semantic embeddings.

---

## Prerequisites

- ✓ Scripts created (`fastembed_embed.py`, `train_anomaly.py`, `test_anomaly.py`)
- ✓ Dependencies installed (fastembed, tqdm, matplotlib, seaborn)
- ✓ Directory structure ready

---

## Step 1: Generate Embeddings

### 1.1 Dense WITHOUT preprocessing

```bash
uv run python -m scripts.fastembed_embed \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/embeddings.npz \
  --model BAAI/bge-small-en-v1.5 \
  --batch-size 512 \
  --reader jsonl \
  --device cpu
```

### 1.2 Dense WITH preprocessing

```bash
uv run python -m scripts.fastembed_embed \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/embeddings.npz \
  --model BAAI/bge-small-en-v1.5 \
  --use-pipeline \
  --batch-size 512 \
  --reader jsonl \
  --device cpu
```

**Note**: Use `--device cuda` if you have a GPU available for faster processing

**Expected**: Each file ~20-30MB, (47000, 384) shape

---

## Step 2: Train Models (~1 minute)

### 2.1 Train WITHOUT preprocessing

```bash
uv run python -m scripts.train_anomaly \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model.joblib \
  --contamination 0.1 \
  --n-estimators 300 \
  --random-state 42
```

### 2.2 Train WITH preprocessing

```bash
uv run python -m scripts.train_anomaly \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/model.joblib \
  --contamination 0.1 \
  --n-estimators 300 \
  --random-state 42
```

**Watch for**: Training score statistics (mean, std dev)

---

## Step 3: Analyze Training Scores (~10 seconds)

```bash
uv run python experiments/02_dense_embeddings_comparison/analyze_scores.py \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model.joblib \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/model.joblib \
  --label "Dense (no pipeline)" \
  --label "Dense (with pipeline)" \
  --output experiments/02_dense_embeddings_comparison/training_scores.png
```

**Critical check**:

- If std dev < 0.001 → embeddings still too concentrated, won't work well
- If std dev > 0.01 → good discriminative potential ✓

---

## Step 4: Test Models (~5-10 minutes)

### 4.1 Test WITHOUT preprocessing

```bash
uv run python -m scripts.test_anomaly \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model.joblib \
  --encoder fastembed \
  --model-name BAAI/bge-small-en-v1.5 \
  --batch-size 512 \
  --device cpu \
  --wandb-project neuralshield \
  --wandb-run-name "exp02-dense-no-pipeline-cont0.1"
```

### 4.2 Test WITH preprocessing

```bash
uv run python -m scripts.test_anomaly \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/model.joblib \
  --encoder fastembed \
  --model-name BAAI/bge-small-en-v1.5 \
  --use-pipeline \
  --batch-size 512 \
  --device cpu \
  --wandb-project neuralshield \
  --wandb-run-name "exp02-dense-with-pipeline-cont0.1"
```

**Note**: W&B will log:

- Confusion matrix visualization
- Score distribution plots
- Sample true positives and false positives
- All metrics (precision, recall, F1, FPR, etc.)

---

## Step 5: Compare with TF-IDF Baseline

Results from **Experiment 01** (TF-IDF, contamination=0.1):

| Model                  | Precision | Recall | F1    | FPR   |
| ---------------------- | --------- | ------ | ----- | ----- |
| TF-IDF (no pipeline)   | 52.36%    | 0.85%  | 1.67% | 0.39% |
| TF-IDF (with pipeline) | 55.81%    | 0.96%  | 1.89% | 0.33% |

**What to look for in Dense results**:

- ✓ Recall > 10% (minimum viable success)
- ✓ Recall > 40% (strong success)
- ✓ F1-Score improvement
- ✓ Better score separation in W&B plots

---

## Optional: Test Other Contamination Values

If initial results are promising, test contamination=[0.15, 0.2, 0.25, 0.3]:

```bash
# Train with different contamination
uv run python -m scripts.train_anomaly \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model_contamination_0.15.joblib \
  --contamination 0.15 \
  --n-estimators 300 \
  --random-state 42

# Test
uv run python -m scripts.test_anomaly \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model_contamination_0.15.joblib \
  --encoder fastembed \
  --model-name BAAI/bge-small-en-v1.5 \
  --batch-size 512 \
  --device cpu \
  --wandb-project neuralshield \
  --wandb-run-name "exp02-dense-no-pipeline-cont0.15"
```

---

## Troubleshooting

**Q: Embedding generation too slow?**

- Reduce `--batch-size` to 256 or 128
- Use `--device cuda` if GPU available

**Q: Still getting low recall (<10%)?**

- Try larger model: `BAAI/bge-base-en-v1.5` (768 dimensions)
- Try higher contamination (0.3-0.5)
- Consider different anomaly detector (OCSVM, AutoEncoder)

**Q: High FPR (>20%)?**

- Lower contamination parameter
- Check if preprocessing helps

---

## Success Criteria Checklist

- [ ] Training scores have std dev > 0.01
- [ ] Recall > 10% (minimum)
- [ ] F1-Score > 15% (minimum)
- [ ] Score distributions in W&B show separation
- [ ] Preprocessing impact is clear (better or worse)

---

## Next Steps

After completing this experiment, document findings in `RESULTS.md` and update main `experiments/README.md`.
