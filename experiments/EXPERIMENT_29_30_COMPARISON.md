# Experiments 29 & 30: Comparison Guide

## Overview

Two complementary experiments for flag analysis and performance profiling:

- **Experiment 29**: Beautiful visualizations (images) for human analysis
- **Experiment 30**: Structured data (tables) for automated systems

## Quick Comparison

| Feature | Experiment 29 | Experiment 30 |
|---------|---------------|---------------|
| **Output Format** | Images + Metrics | Tables + Metrics |
| **Visualizations** | ✅ 6 custom images | ❌ None (WandB auto-generates from tables) |
| **WandB Tables** | ❌ None | ✅ 4 structured tables |
| **Time-Series Metrics** | ✅ Yes | ✅ Yes |
| **Summary Metrics** | ✅ Top 10 flags | ✅ Top 20 flags |
| **API Accessible** | ⚠️ Limited (images not queryable) | ✅ Full (all data queryable) |
| **Use Case** | Human analysis, presentations | Automated systems, data pipelines |
| **File Size** | Larger (images) | Smaller (tables only) |

## When to Use Which

### Use Experiment 29 When:
- ✅ You want beautiful visualizations for reports/presentations
- ✅ You're doing manual analysis
- ✅ You need publication-quality graphs
- ✅ You want to explore data visually

### Use Experiment 30 When:
- ✅ Your system reads WandB logs programmatically
- ✅ You need to query flag statistics via API
- ✅ You're building automated pipelines
- ✅ You want structured data for further processing
- ✅ You need to integrate with other tools

## Data Access Comparison

### Experiment 29 (Images)

**Accessible via API:**
- Time-series metrics (`performance/*`)
- Summary metrics (top 10 flags)
- JSON file

**Not Accessible via API:**
- Full flag statistics (only in images)
- Flag co-occurrence patterns (only in images)
- Flag count distributions (only in images)

### Experiment 30 (Structured)

**Accessible via API:**
- ✅ Time-series metrics (`performance/*`)
- ✅ Summary metrics (top 20 flags)
- ✅ **All flag statistics** (`flags/statistics_table`)
- ✅ **Flag co-occurrence** (`flags/cooccurrence_table`)
- ✅ **Flag distributions** (`flags/count_distribution_table`)
- ✅ **Batch metrics** (`performance/batch_metrics_table`)
- ✅ JSON file

## Example: Accessing Data

### Experiment 29 (Limited)

```python
import wandb

api = wandb.Api()
run = api.run("project/run-id")

# ✅ Can access
history = run.history()  # Time-series metrics
summary = run.summary    # Top 10 flags

# ❌ Cannot access programmatically
# - Full flag statistics (only in images)
# - Flag co-occurrence (only in images)
```

### Experiment 30 (Full Access)

```python
import wandb

api = wandb.Api()
run = api.run("project/run-id")

# ✅ Can access everything
history = run.history()  # Time-series metrics
summary = run.summary   # Top 20 flags

# ✅ Can access tables
flag_table = run.summary.get("flags/statistics_table")
cooccurrence_table = run.summary.get("flags/cooccurrence_table")
distribution_table = run.summary.get("flags/count_distribution_table")
batch_table = run.summary.get("performance/batch_metrics_table")

# ✅ Can query all flag statistics
for row in flag_table.data:
    flag = row[0]
    attack_rate = row[1]
    benign_rate = row[2]
    signal_strength = row[3]
    # Process programmatically...
```

## Recommendations

1. **For presentations/reports**: Use Experiment 29
2. **For automated systems**: Use Experiment 30
3. **For both**: Run both experiments (they're complementary)

## Running Both

```bash
# Run Experiment 29 (visualizations)
uv run python experiments/29_flag_performance_analysis/analyze_flags_performance.py \
  src/neuralshield/data/CSIC/test.jsonl \
  --wandb-run-name flag-analysis-visual

# Run Experiment 30 (structured)
uv run python experiments/30_flag_performance_structured/analyze_flags_performance.py \
  src/neuralshield/data/CSIC/test.jsonl \
  --wandb-run-name flag-analysis-structured
```

Both will log to the same WandB project, making it easy to compare and use both formats.

