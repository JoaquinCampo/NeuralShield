# Experiment 02: Dense Embeddings vs TF-IDF Comparison

**Date**: September 30, 2025  
**Status**: 🔄 Planning  
**Author**: NeuralShield Team

---

## Objective

Determine if **dense semantic embeddings** (FastEmbed) improve anomaly detection performance compared to sparse TF-IDF embeddings, specifically for WAF/HTTP request classification.

## Hypothesis

Dense semantic embeddings will significantly improve recall (from <1% to >40%) because:

1. **Lower dimensionality** (384-1024 vs 5000 sparse dimensions)
2. **Semantic understanding** (similar attacks cluster together)
3. **Better variance** in training scores (not all concentrated at one value)

---

## Model Selection

### Criteria for Choosing FastEmbed Model

For **HTTP request anomaly detection**, we need:

1. **Good semantic understanding** of text patterns
2. **Efficient** (reasonable size and speed)
3. **English language** focus
4. **General-purpose** (not domain-specific like code or images)
5. **Proven performance** on text classification tasks

### Top Candidates

| Model                                  | Dim | Size (GB) | Tokens | Notes                                             |
| -------------------------------------- | --- | --------- | ------ | ------------------------------------------------- |
| **BAAI/bge-small-en-v1.5**             | 384 | 0.067     | 512    | ✨ **RECOMMENDED**: Fast, small, no prefix needed |
| BAAI/bge-base-en-v1.5                  | 768 | 0.21      | 512    | Good balance, slightly better quality             |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.09      | 256    | Popular baseline, older (2021)                    |
| snowflake/snowflake-arctic-embed-s     | 384 | 0.13      | 512    | 2024 model, good quality                          |
| nomic-ai/nomic-embed-text-v1.5-Q       | 768 | 0.13      | 8192   | Long context, quantized                           |

### Selected Model: `BAAI/bge-small-en-v1.5`

**Rationale**:

- ✅ **Smallest size** (67MB) → fastest to download and use
- ✅ **384 dimensions** → manageable for IsolationForest
- ✅ **No prefix required** → simpler to use
- ✅ **2023 model** → modern architecture
- ✅ **Optimized for retrieval** → good for finding similar patterns
- ✅ **512 token limit** → enough for HTTP requests

**Fallback**: If results are poor, try `BAAI/bge-base-en-v1.5` (768 dim, better quality)

---

## Experimental Design

### Controlled Variables

- Same dataset (CSIC train/test)
- Same anomaly detector (IsolationForest)
- Same hyperparameters (n_estimators=300)
- Same contamination values (0.1, 0.15, 0.2, 0.25, 0.3)

### Independent Variable

- **Embedding method**: TF-IDF (sparse) vs FastEmbed (dense)

### Dependent Variables

- Precision, Recall, F1-Score
- False Positive Rate (FPR)
- Specificity
- Training score distribution characteristics

---

## Methodology

### Phase 1: Embedding Generation

#### 1.1 Generate Dense Embeddings (WITHOUT preprocessing)

```bash
uv run python -m scripts.fastembed_embed \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/embeddings.npz \
  --model BAAI/bge-small-en-v1.5 \
  --batch-size 512
```

**Output**:

- Embeddings: (47000, 384) float32 array
- Metadata: model info, embedding dimension

#### 1.2 Generate Dense Embeddings (WITH preprocessing)

```bash
uv run python -m scripts.fastembed_embed \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/embeddings.npz \
  --model BAAI/bge-small-en-v1.5 \
  --use-pipeline \
  --batch-size 512
```

### Phase 2: Model Training

Train IsolationForest models with contamination=0.1 initially:

```bash
# WITHOUT preprocessing
uv run python -m scripts.train_anomaly \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model.joblib \
  --contamination 0.1 \
  --n-estimators 300

# WITH preprocessing
uv run python -m scripts.train_anomaly \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/embeddings.npz \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/model.joblib \
  --contamination 0.1 \
  --n-estimators 300
```

### Phase 3: Score Distribution Analysis

**Before testing**, analyze training score distributions:

```python
# Check if scores have better variance than TF-IDF
python experiments/02_dense_embeddings_comparison/analyze_scores.py
```

**Expected outcomes**:

- TF-IDF: mean=-0.292, std=0.0002 (extremely concentrated)
- Dense: mean=~0.15, std=~0.05 (better spread)

### Phase 4: Testing

Test on CSIC test set:

```bash
# WITHOUT preprocessing
uv run python -m scripts.test_anomaly \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/02_dense_embeddings_comparison/dense_without_preprocessing/model.joblib \
  --model-name BAAI/bge-small-en-v1.5

# WITH preprocessing
uv run python -m scripts.test_anomaly \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/02_dense_embeddings_comparison/dense_with_preprocessing/model.joblib \
  --model-name BAAI/bge-small-en-v1.5 \
  --use-pipeline
```

### Phase 5: Comparison

Compare 4 models:

1. TF-IDF without preprocessing (baseline from Experiment 01)
2. TF-IDF with preprocessing (baseline from Experiment 01)
3. Dense without preprocessing (new)
4. Dense with preprocessing (new)

---

## Success Criteria

### Minimum Viable Success

- ✅ Recall > 10% (at least 10x improvement over TF-IDF's <1%)
- ✅ F1-Score > 15%
- ✅ Training score std dev > 0.01 (better variance)

### Strong Success

- ✅ Recall > 40%
- ✅ F1-Score > 50%
- ✅ FPR < 10%
- ✅ Clear score separation between normal and attack samples

### Exceptional Success

- ✅ Recall > 70%
- ✅ F1-Score > 65%
- ✅ FPR < 5%
- ✅ Preprocessing shows clear additional benefit

---

## Implementation Steps

### Step 1: Create embedding script

- [ ] Create `scripts/fastembed_embed.py`
- [ ] Reuse existing `FastEmbedEncoder` from `neuralshield.encoding.models.fastembed`
- [ ] Add batch processing for 47,000 samples
- [ ] Save embeddings in .npz format

### Step 2: Create training script (generic)

- [ ] Create `scripts/train_anomaly.py`
- [ ] Load embeddings from .npz
- [ ] Train IsolationForest
- [ ] Save model with metadata

### Step 3: Create testing script (generic)

- [ ] Create `scripts/test_anomaly.py`
- [ ] Load model and generate embeddings on-the-fly
- [ ] Run predictions
- [ ] Generate metrics using evaluation module

### Step 4: Analysis script

- [ ] Create `analyze_scores.py` to compare score distributions
- [ ] Create visualization comparing TF-IDF vs Dense

### Step 5: Run experiment

- [ ] Generate embeddings (both with/without preprocessing)
- [ ] Train models
- [ ] Analyze score distributions
- [ ] Test models
- [ ] Compare results

### Step 6: Documentation

- [ ] Create `RESULTS.md` with findings
- [ ] Update main experiment README

---

## Expected Timeline

- **Day 1**: Create scripts and generate embeddings (~2 hours)
- **Day 2**: Train models and analyze distributions (~1 hour)
- **Day 3**: Test and analyze results (~2 hours)
- **Total**: ~5 hours of active work

---

## Risk Mitigation

### Potential Issues

1. **Dense embeddings still have low recall**

   - **Mitigation**: Try different contamination values (0.3, 0.4, 0.5)
   - **Mitigation**: Try larger model (bge-base-en-v1.5)

2. **Model too slow for 50k test samples**

   - **Mitigation**: Use quantized model (nomic-embed-text-v1.5-Q)
   - **Mitigation**: Batch processing optimization

3. **Memory issues with dense embeddings**
   - **Mitigation**: Process in smaller batches
   - **Mitigation**: Use float16 instead of float32

---

## Next Steps After This Experiment

**If dense embeddings work better**:

- Experiment 03: Optimal FastEmbed model comparison (small vs base vs large)
- Experiment 04: Preprocessing impact re-evaluation with dense embeddings

**If dense embeddings don't improve much**:

- Experiment 03: Try different anomaly detector (OCSVM, AutoEncoder)
- Experiment 04: Semi-supervised approach with small amount of attack data

---

## Notes

- Current implementation already has `FastEmbedEncoder` in `neuralshield.encoding.models.fastembed`
- Need to create command-line scripts for reproducibility
- Keep experiment fully isolated from Experiment 01 for clean comparison
