# Experiment 07: GPU-Accelerated OCSVM

Testing One-Class SVM with GPU acceleration using NVIDIA RAPIDS cuML.

## Objective

Test if OCSVM's non-linear RBF kernel can capture attack patterns better than:

- Mahalanobis distance (linear, 40% recall @ 5% FPR)
- IsolationForest (ensemble trees, 27% recall @ 5% FPR)

## Hypothesis

**OCSVM's RBF kernel may achieve better recall** by learning non-linear decision boundaries in embedding space.

## Baseline to Beat

BGE-small + Mahalanobis + Preprocessing:

- **Recall:** 40% @ 5% FPR
- **Time:** 5 seconds
- **Issue:** Attacks cluster with normal requests in score distribution

## Running on Colab A100

### 1. Setup Colab

1. Go to https://colab.research.google.com/
2. Runtime → Change runtime type → A100 GPU
3. Create new notebook

### 2. Install Dependencies

```python
!pip install cuml-cu11 loguru matplotlib seaborn scikit-learn wandb -q
```

### 3. Upload Files

Upload to Colab:

- `test_ocsvm_cuml.py` (this repo)
- `experiments/02_dense_embeddings_comparison/dense_with_preprocessing/embeddings.npz`
- `experiments/02_dense_embeddings_comparison/dense_with_preprocessing/test_embeddings.npz`

```python
from google.colab import files
uploaded = files.upload()
```

### 4. Login to Wandb (Optional)

```python
import wandb
wandb.login()
```

### 5. Run Test

```bash
!python test_ocsvm_cuml.py \
    embeddings.npz \
    test_embeddings.npz \
    results \
    --nu 0.05 \
    --gamma scale \
    --wandb \
    --wandb-run-name "ocsvm-bge-with-prep-gpu"
```

### 6. Download Results

```python
files.download('results/results.json')
files.download('results/score_distribution.png')
files.download('results/confusion_matrix.png')
```

## Expected Results

### Time

- **A100:** 5-10 minutes
- **CPU (local):** 30-60 minutes

### Performance Target

- **Recall:** >40% @ 5% FPR (beat Mahalanobis)
- **Why it might help:** Non-linear kernel can separate overlapping distributions

## Technical Details

### OCSVM Parameters

**Nu (ν):** Upper bound on fraction of outliers

- Default: 0.05 (5%)
- Controls training contamination tolerance

**Gamma (γ):** RBF kernel coefficient

- `scale`: 1 / (n_features × X.var()) = 1 / (384 × var)
- `auto`: 1 / n_features = 1 / 384
- Higher = more local, Lower = more global

**Kernel:** RBF (Radial Basis Function)

- K(x, y) = exp(-γ ||x - y||²)
- Non-linear decision boundary

### cuML vs sklearn

| Metric | cuML (GPU) | sklearn (CPU) |
| ------ | ---------- | ------------- |
| Time   | 5-10 min   | 30-60 min     |
| Memory | GPU VRAM   | System RAM    |
| API    | Compatible | Native        |

## Comparison to Other Detectors

| Detector        | Boundary   | Assumptions           | BGE Recall |
| --------------- | ---------- | --------------------- | ---------- |
| IsolationForest | Tree-based | Anomalies are rare    | 27%        |
| Mahalanobis     | Linear     | Gaussian distribution | 40%        |
| OCSVM           | Non-linear | Kernel separability   | TBD        |

## Why OCSVM Might Win

1. **Non-linear boundary:** Can wrap around normal cluster
2. **RBF kernel:** Captures local similarities
3. **Learns on support vectors:** Focuses on boundary samples

## Why OCSVM Might Lose

1. **Semantic overlap:** If attacks are semantically similar to normal, no boundary helps
2. **Overfitting risk:** nu=0.05 might be too strict
3. **Hyperparameter sensitivity:** Gamma choice matters

## Next Steps

### If OCSVM Wins (>40% recall)

- Test on ByT5 embeddings
- Hyperparameter search (nu, gamma grid)
- Production integration

### If OCSVM Loses (≤40% recall)

- Try ensemble: Mahalanobis + rule-based
- Investigate supervised classifier on embeddings
- Consider finetuning BGE for HTTP domain
