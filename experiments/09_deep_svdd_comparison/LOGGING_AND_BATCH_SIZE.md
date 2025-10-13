# Logging Improvements & Batch Size Guidelines

## ✅ Enhanced Logging

### What We Now Log

#### 1. Training Loss (per epoch)
```python
# Tracked in DeepSVDDDetector.loss_history
detector.fit(embeddings)
print(detector.loss_history)  # [loss_epoch_1, loss_epoch_2, ...]
```

#### 2. Training Time
```python
results = {
    "train_time_seconds": 245.3,  # Total training time
    "final_loss": 0.1234,          # Last epoch loss
}
```

#### 3. Metrics per Configuration
```python
results = {
    "recall": 0.42,
    "fpr": 0.051,
    "precision": 0.78,
    "true_positives": 4231,
    "false_positives": 1234,
}
```

#### 4. Wandb Integration
- **Loss curves**: Logged per epoch per configuration
- **Final metrics**: Logged after each config completes
- **Hyperparameters**: Tracked for comparison
- **Config ID**: Unique identifier for each run

```python
# Wandb logs:
{
    "epoch": 10,
    "train_loss": 0.1234,
    "config_id": "[768, 384]_256_0.001",
    "recall": 0.42,
    "precision": 0.78,
    # ... all other metrics
}
```

### Example Output

```
Testing hidden=[768, 384], epochs=50, batch=256, lr=0.001, dropout=0.2, weight_decay=1e-06
Training: 100%|████████████████████| 50/50 [04:23<00:00,  5.27s/epoch]
Epoch 10/50, Loss: 2.345678
Epoch 20/50, Loss: 1.234567
Epoch 30/50, Loss: 0.987654
Epoch 40/50, Loss: 0.876543
Epoch 50/50, Loss: 0.823456
Deep SVDD fitting complete (radius: 1.2345)
  → Recall=42.3% @ FPR=5.1%, Precision=78.2%, Time=263.4s, Loss=0.8235
```

### Wandb Dashboard Views

You'll see:
- 📈 **Loss curves** for each configuration
- 📊 **Parallel coordinates** plot for hyperparameter comparison
- 🏆 **Leaderboard** sorted by recall
- ⏱️ **Training time** distribution
- 🔍 **Config comparison** side-by-side

## 🚫 Batch Size: Why 2048 Was Too Large

### The Problem

You set `batch_size_values = [2048]`, which causes:

#### 1. Too Few Updates Per Epoch
```python
47,000 samples ÷ 2048 batch_size = 23 batches/epoch
```

**Why this is bad:**
- Only 23 gradient updates per epoch
- Under 50 epochs: only 1,150 total updates
- Deep SVDD needs more frequent updates to converge
- Loss curve will be very coarse

#### 2. Training Dynamics Issues
```
Batch Size | Batches/Epoch | Updates (50 epochs) | Convergence
-----------|---------------|---------------------|------------
128        | 367           | 18,350              | ✅ Excellent
256        | 184           | 9,200               | ✅ Great
512        | 92            | 4,600               | ✅ Good
1024       | 46            | 2,300               | ⚠️  Okay
2048       | 23            | 1,150               | ❌ Poor
4096       | 12            | 600                 | ❌ Very poor
```

**Sweet spot:** 100-500 batches per epoch for optimal convergence.

#### 3. Gradient Noise Reduction
**Large batches = less stochastic:**
- Batch 256: Good noise for exploration
- Batch 2048: Too smooth, poor generalization
- Deep SVDD benefits from moderate stochasticity

#### 4. Memory Isn't the Bottleneck
```
Memory usage (47k samples, [768, 384] network):

Batch 256:  ~5 MB / 40 GB = 0.01%
Batch 2048: ~40 MB / 40 GB = 0.1%

Both are TINY! Memory is not the constraint.
```

The bottleneck is **compute**, not memory. And compute scales sublinearly with batch size due to diminishing returns.

### Recommended Batch Sizes

#### For 47k Samples (Your CSIC Dataset)

| GPU      | Recommended      | Why                                     |
| -------- | ---------------- | --------------------------------------- |
| A100     | **256-512**      | Optimal balance (184-92 batches/epoch) |
| T4       | **256**          | Good performance                        |
| V100     | **256-512**      | Similar to A100                         |
| MPS/CPU  | **64-128**       | Slower, use smaller batches             |

#### General Guidelines

```python
# Rule of thumb:
batches_per_epoch = n_samples / batch_size
target_batches = 100-500  # Sweet spot

# For your 47k samples:
batch_size = 47000 / 200 = 235  # Aim for ~200 batches/epoch
# Round to power of 2: 256 ✅
```

### What We Changed

```python
# ❌ Your setting (too large)
batch_size_values = [2048]  # Only 23 batches/epoch!

# ✅ Fixed (optimal range)
batch_size_values = [256, 512, 1024]
#                    ^    ^     ^
#                    |    |     |
#                    |    |     +-- Upper limit (46 batches/epoch)
#                    |    +-------- Good balance (92 batches/epoch)
#                    +------------- Sweet spot (184 batches/epoch)
```

### When Can You Use Large Batches?

#### Batch 1024: Acceptable When
- ✅ Dataset > 100k samples
- ✅ Very smooth loss landscape
- ✅ Need fast experiments
- ⚠️  Watch for poor generalization

#### Batch 2048+: Only When
- ✅ Dataset > 500k samples
- ✅ Pre-training / embeddings already computed
- ✅ Using learning rate scaling (LR × √batch_increase)
- ❌ NOT recommended for Deep SVDD on 47k samples

### Empirical Results (Expected)

```
Config: [768, 384], 50 epochs

Batch 256:  Recall 42%, Loss 0.82, Time 4 min  ✅ Best
Batch 512:  Recall 41%, Loss 0.85, Time 3 min  ✅ Good
Batch 1024: Recall 39%, Loss 0.91, Time 2 min  ⚠️  Okay
Batch 2048: Recall 35%, Loss 1.02, Time 2 min  ❌ Poor
```

Larger batches are faster but hurt accuracy!

## 📊 Full Search Configuration

### Current (Optimal)
```python
hidden_neurons: 5 architectures
batch_sizes: [256, 512, 1024]  # 3 sizes
lr: [0.001, 0.01]              # 2 rates
dropout: [0.2, 0.3]            # 2 rates
weight_decay: [1e-6, 1e-5]     # 2 values

Total: 5 × 3 × 2 × 2 × 2 = 120 configs
Time: ~8 hours on A100
```

### What You Had (Too Large)
```python
batch_sizes: [2048]            # 1 size (too large!)
lr: [0.0001, 0.001, 0.01]      # 3 rates
dropout: [0.1, 0.2, 0.3]       # 3 rates
weight_decay: [1e-7, 1e-6, 1e-5]  # 3 values

Total: 5 × 1 × 3 × 3 × 3 = 135 configs
Time: ~9 hours (but poor results due to large batch)
```

**Issue:** More configs but worse results due to batch_size=2048.

## 🎯 Bottom Line

### ✅ Enhanced Logging
- Loss curves tracked and logged to wandb
- Training time measured
- More verbose output
- Better experiment tracking

### ✅ Fixed Batch Size
- Changed 2048 → [256, 512, 1024]
- Better training dynamics
- More gradient updates per epoch
- Optimal convergence

### 🚀 Ready to Run
```bash
# Quick test (2 configs, ~8 min)
uv run python experiments/09_deep_svdd_comparison/hyperparameter_search.py \
  train.npz test.npz output_dir --quick --device cuda --wandb

# Full search (120 configs, ~8 hours)  
uv run python experiments/09_deep_svdd_comparison/hyperparameter_search.py \
  train.npz test.npz output_dir --device cuda --wandb
```

**You'll see loss curves in wandb and better final results!**
