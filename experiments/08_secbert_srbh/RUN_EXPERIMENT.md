# Running Experiment 08: SecBERT on SR_BH_2020

**Quick execution guide** for testing SecBERT embeddings on the large-scale SR_BH dataset.

---

## Prerequisites

- ✓ SR_BH dataset converted to JSONL
- ✓ Train/test split created (100K train / 807K test)
- ✓ SecBERT dependencies installed
- ✓ GPU recommended (optional but faster)

---

## Step 1: Generate Training Embeddings

### 1.1 WITH Preprocessing (Recommended)

```bash
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/train.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  --model jackaduma/SecBERT \
  --use-pipeline \
  --batch-size 32 \
  --device cuda
```

**Expected**:

- Time: ~5-10 min (GPU) / ~60-90 min (CPU)
- Output: `train_embeddings.npz` (~300MB, shape: 100000 × 768)

**Notes**:

- Change `--device cuda` to `--device cpu` if no GPU
- Reduce `--batch-size 16` if GPU OOM errors
- First run downloads SecBERT model (~400MB)

### 1.2 WITHOUT Preprocessing (Optional Comparison)

```bash
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/train.jsonl \
  experiments/08_secbert_srbh/without_preprocessing/train_embeddings.npz \
  --model jackaduma/SecBERT \
  --batch-size 32 \
  --device cuda
```

**Expected**: Same size/time as above

---

## Step 2: Train Anomaly Detector

### 2.1 WITH Preprocessing

```bash
uv run python -m scripts.train_anomaly \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib \
  --contamination 0.1 \
  --n-estimators 300 \
  --random-state 42
```

**Expected**:

- Time: ~2-5 min
- Output: `model.joblib` (~50-100MB)
- Watch for: Training score statistics

### 2.2 WITHOUT Preprocessing (Optional)

```bash
uv run python -m scripts.train_anomaly \
  experiments/08_secbert_srbh/without_preprocessing/train_embeddings.npz \
  experiments/08_secbert_srbh/without_preprocessing/model.joblib \
  --contamination 0.1 \
  --n-estimators 300 \
  --random-state 42
```

---

## Step 3: Generate Test Embeddings

### 3.1 WITH Preprocessing

```bash
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/test.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  --model jackaduma/SecBERT \
  --use-pipeline \
  --batch-size 32 \
  --device cuda
```

**Expected**:

- Time: ~40-80 min (GPU) / ~4-6 hours (CPU)
- Output: `test_embeddings.npz` (~2.4GB, shape: 807815 × 768)

**Notes**:

- This is the longest step (8x train size)
- Consider running overnight if using CPU
- Model already cached from Step 1

### 3.2 WITHOUT Preprocessing (Optional)

```bash
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/test.jsonl \
  experiments/08_secbert_srbh/without_preprocessing/test_embeddings.npz \
  --model jackaduma/SecBERT \
  --batch-size 32 \
  --device cuda
```

---

## Step 4: Evaluate Model

### 4.1 WITH Preprocessing

```bash
uv run python -m scripts.test_anomaly_precomputed \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib
```

**Expected output**:

```
Classification Report:
              precision    recall  f1-score   support

       valid       0.XX      0.XX      0.XX    425195
      attack       0.XX      0.XX      0.XX    382620

    accuracy                           0.XX    807815

Recall @ 5% FPR: XX.XX%
```

### 4.2 WITHOUT Preprocessing (Optional)

```bash
uv run python -m scripts.test_anomaly_precomputed \
  experiments/08_secbert_srbh/without_preprocessing/test_embeddings.npz \
  experiments/08_secbert_srbh/without_preprocessing/model.joblib
```

---

## Step 5: Compare Results

Create comparison summary:

```bash
# Save outputs
uv run python -m scripts.test_anomaly_precomputed \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib \
  > experiments/08_secbert_srbh/with_preprocessing/results.txt

uv run python -m scripts.test_anomaly_precomputed \
  experiments/08_secbert_srbh/without_preprocessing/test_embeddings.npz \
  experiments/08_secbert_srbh/without_preprocessing/model.joblib \
  > experiments/08_secbert_srbh/without_preprocessing/results.txt
```

Then create a markdown summary of key metrics.

---

## Quick Commands Summary

**Full pipeline WITH preprocessing** (recommended):

```bash
# 1. Train embeddings
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/train.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  --use-pipeline --batch-size 32 --device cuda

# 2. Train model
uv run python -m scripts.train_anomaly \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib

# 3. Test embeddings
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/test.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  --use-pipeline --batch-size 32 --device cuda

# 4. Evaluate
uv run python -m scripts.test_anomaly_precomputed \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  experiments/08_secbert_srbh/with_preprocessing/model.joblib
```

**Total time**:

- GPU: ~50-95 min
- CPU: ~5-7 hours

---

## Troubleshooting

### GPU Out of Memory

Reduce batch size:

```bash
--batch-size 16  # or even 8
```

### Process Killed / System Freeze

Your system may be out of RAM. Try:

1. Close other applications
2. Process test set in chunks (advanced - requires script modification)
3. Use a machine with more RAM

### Slow CPU Processing

Expected behavior. Options:

1. Use GPU
2. Run overnight
3. Test on smaller subset first (modify script)

### Import Errors

Ensure all dependencies installed:

```bash
uv pip list | grep -E "torch|transformers|scikit-learn"
```

---

## Expected Timeline

| Step                | GPU Time        | CPU Time       |
| ------------------- | --------------- | -------------- |
| 1. Train embeddings | 5-10 min        | 60-90 min      |
| 2. Train model      | 2-5 min         | 2-5 min        |
| 3. Test embeddings  | 40-80 min       | 4-6 hours      |
| 4. Evaluate         | 2-5 min         | 2-5 min        |
| **Total**           | **~50-100 min** | **~5-7 hours** |

---

## Next Steps

After completion:

1. Document results in experiment folder
2. Compare with CSIC baseline (Exp 03)
3. Analyze preprocessing impact
4. Consider testing other models (BGE-small, ByT5) on SR_BH
5. Update experiments/FINAL_RESULTS.md
