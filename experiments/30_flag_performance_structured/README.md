# Experiment 30: Flag Analysis and Performance Profiling (Structured Logging)

**Purpose**: Comprehensive analysis of preprocessing flags and performance metrics with **structured WandB logging** optimized for programmatic access.

## Overview

This experiment processes HTTP requests in batches, extracts preprocessing flags, analyzes patterns, and logs **all data as structured WandB tables and metrics** (no images). This makes it perfect for systems that read WandB logs programmatically.

**Key Difference from Experiment 29**: This version logs data as WandB Tables instead of images, making it accessible via WandB API for automated systems.

## Features

### Time-Series Metrics (Logged per Batch)

- **Processing Time**: Average, P95, P99 per 1K requests
- **Throughput**: Requests per second
- **Breakdown**: Preprocessing times

### Flag Analysis (Logged as Tables)

- **Flag Statistics Table**: All flags with presence rates, signal strength, counts
- **Co-occurrence Table**: Flag pairs and their co-occurrence counts
- **Count Distribution Table**: Flag count distributions for attack vs benign

### Structured Data Output

All data is logged as:
- **WandB Tables**: Queryable structured data
- **WandB Metrics**: Time-series numeric values
- **WandB Summary**: Aggregated statistics
- **JSON File**: Complete results backup

## Usage

### Basic Usage

```bash
uv run python experiments/30_flag_performance_structured/analyze_flags_performance.py \
  src/neuralshield/data/CSIC/test.jsonl
```

### With Options

```bash
uv run python experiments/30_flag_performance_structured/analyze_flags_performance.py \
  src/neuralshield/data/CSIC/test.jsonl \
  --max-batches 10 \
  --wandb-project neuralshield-v2 \
  --wandb-run-name flag-analysis-structured-csic-test
```

### Parameters

- `dataset_path`: Path to JSONL dataset file (required)
- `--max-batches`: Maximum number of batches to process (default: all)
- `--output-dir`: Output directory for results (default: `experiments/30_flag_performance_structured/results`)
- `--wandb-project`: WandB project name (default: `neuralshield-v2`)
- `--wandb-run-name`: WandB run name (auto-generated if not provided)

## Output

### WandB Logs

**Time-Series Metrics** (logged per batch, queryable via API):
- `performance/batch_num`: Batch number
- `performance/avg_time_per_request_ms`: Average processing time per request
- `performance/batch_time_ms`: Total time for batch (1K requests)
- `performance/avg_time_per_1k_ms`: Average time per 1K requests
- `performance/avg_preprocessing_time_ms`: Average preprocessing time
- `performance/p95_preprocessing_time_ms`: P95 preprocessing time
- `performance/p99_preprocessing_time_ms`: P99 preprocessing time
- `performance/throughput_req_per_sec`: Throughput in requests/second

**WandB Tables** (structured data, queryable via API):
- `flags/statistics_table`: All flags with statistics
  - Columns: `flag`, `attack_presence_rate`, `benign_presence_rate`, `signal_strength`, `attack_count`, `benign_count`, `attack_per_request`, `benign_per_request`
- `flags/cooccurrence_table`: Flag co-occurrence patterns
  - Columns: `flag1`, `flag2`, `cooccurrence_count`
- `flags/count_distribution_table`: Flag count distributions
  - Columns: `label`, `flag_count`, `frequency`
- `performance/batch_metrics_table`: Batch-by-batch performance
  - Columns: `batch_num`, `batch_size`, `batch_time_ms`, `avg_time_per_request_ms`, `avg_preprocessing_time_ms`, `p95_preprocessing_time_ms`, `p99_preprocessing_time_ms`, `throughput_req_per_sec`

**Summary Metrics**:
- `performance/overall_avg_time_per_request_ms`: Overall average
- `performance/overall_throughput_req_per_sec`: Overall throughput
- `flags/total_unique_flags`: Number of unique flags found
- `flags/attack_requests`: Number of attack requests processed
- `flags/benign_requests`: Number of benign requests processed
- `flags/attack_avg_flags_per_request`: Average flags per attack request
- `flags/benign_avg_flags_per_request`: Average flags per benign request
- `flags/top_N_{flag}_signal_strength`: Top 20 flags by signal strength
- `flags/top_N_{flag}_attack_rate`: Top 20 flags attack rates
- `flags/top_N_{flag}_benign_rate`: Top 20 flags benign rates

### JSON Results

Results are saved to `results/results.json` with:
- Flag statistics (presence rates, signal strength)
- Summary statistics
- Performance metrics
- Batch-by-batch metrics

## Accessing Data via WandB API

### Python Example

```python
import wandb

api = wandb.Api()
run = api.run("your-project/your-run-id")

# Get flag statistics table
flag_table = run.summary.get("flags/statistics_table")
# Or access via run.files() or run.history()

# Get time-series metrics
history = run.history()
# Contains: performance/avg_time_per_request_ms, performance/throughput_req_per_sec, etc.

# Get summary statistics
summary = run.summary
print(f"Overall throughput: {summary['performance/overall_throughput_req_per_sec']}")
print(f"Total flags: {summary['flags/total_unique_flags']}")
```

### REST API Example

```bash
# Get run summary
curl https://api.wandb.ai/api/v1/runs/your-run-id

# Get run history (time-series)
curl https://api.wandb.ai/api/v1/runs/your-run-id/history
```

## Comparison with Experiment 29

| Feature | Experiment 29 | Experiment 30 |
|---------|---------------|---------------|
| **Visualizations** | ✅ Images (6 types) | ❌ None |
| **WandB Tables** | ❌ None | ✅ 4 tables |
| **Time-Series Metrics** | ✅ Yes | ✅ Yes |
| **Summary Metrics** | ✅ Top 10 flags | ✅ Top 20 flags |
| **API Accessible** | ⚠️ Limited | ✅ Full |
| **Use Case** | Human analysis | Automated systems |

## Architecture

```
experiments/30_flag_performance_structured/
├── __init__.py
├── analyze_flags_performance.py  # Main experiment script
├── flag_analysis.py              # Flag extraction and analysis
├── README.md                     # This file
└── results/                      # Output directory
    └── results.json              # JSON results
```

## Dependencies

- `wandb`: For experiment tracking and table logging
- `numpy`: For numerical operations
- `typer`: For CLI interface
- `loguru`: For logging

## Notes

- Processes requests in batches of 1,000 for efficient processing
- Flags are extracted from preprocessed request strings
- All data is logged as structured tables (no images)
- Time-series metrics are logged to WandB for automatic graphs
- All visualizations are automatically created by WandB from tables/metrics

