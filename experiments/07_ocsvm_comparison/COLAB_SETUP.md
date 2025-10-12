# Quick Colab Setup for OCSVM

Ready to test One-Class SVM with GPU acceleration!

## What You Need

### Files to Upload to Colab:

1. `test_ocsvm_cuml.py` (created)
2. `embeddings.npz` - from `experiments/02_dense_embeddings_comparison/dense_with_preprocessing/`
3. `test_embeddings.npz` - from `experiments/02_dense_embeddings_comparison/dense_with_preprocessing/`

### Expected Time:

- Upload: ~2 minutes
- Setup: ~1 minute
- Training: 5-10 minutes
- Total: ~15 minutes

## Step-by-Step

### 1. Open Colab

Go to: https://colab.research.google.com/

### 2. Change Runtime

- Runtime → Change runtime type → GPU (T4, V100, or A100)
- A100 is fastest but any GPU works

### 3. Install cuML

```python
!pip install cuml-cu11 loguru matplotlib seaborn scikit-learn wandb -q
```

### 4. Upload Files

```python
from google.colab import files

# Upload script
print("Upload test_ocsvm_cuml.py")
files.upload()

# Upload embeddings
print("Upload embeddings.npz (training)")
files.upload()

print("Upload test_embeddings.npz (test)")
files.upload()
```

### 5. Login to Wandb (Optional)

```python
import wandb
wandb.login()
```

### 6. Run

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

### 7. Download Results

```python
files.download('results/results.json')
files.download('results/score_distribution.png')
files.download('results/confusion_matrix.png')
```

## What We're Testing

**Current Leader:** BGE-small + Mahalanobis = 40% recall @ 5% FPR

**Question:** Can OCSVM's RBF kernel do better?

**Why it might:** Non-linear boundary could separate overlapping distributions

**Why it might not:** If attacks are semantically similar to normal, no boundary helps

## After Results

Save the outputs back to `experiments/07_ocsvm_comparison/with_preprocessing/`

We'll compare to Mahalanobis and decide next steps!
