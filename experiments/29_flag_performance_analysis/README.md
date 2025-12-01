# Experiment 29: Flag Analysis and Performance Profiling

**Purpose**: Comprehensive analysis of preprocessing flags and performance metrics with beautiful time-series visualizations in WandB.

## Overview

This experiment processes HTTP requests in batches, extracts preprocessing flags, analyzes flag patterns, and logs time-series performance metrics to WandB. It provides actionable insights for:

- **Flag Analysis**: Which flags are most predictive of attacks vs benign traffic
- **Performance Profiling**: Processing time, throughput, and latency metrics over time
- **Decision Support**: Data-driven insights for model selection and threshold tuning

## Features

### Time-Series Metrics (Logged per Batch)

- **Processing Time**: Average, P95, P99 per 1K requests
- **Throughput**: Requests per second
- **Breakdown**: Preprocessing, encoding, detection times

### Flag Analysis

- **Flag Presence Rates**: Attack vs benign comparison
- **Signal Strength**: Ranking flags by discriminative power
- **Flag Count Distributions**: Violin plots showing flag density
- **Co-occurrence Patterns**: Which flags appear together

### Visualizations

1. **Flag Presence Comparison**: Side-by-side bar chart (attack vs benign)
2. **Signal Strength Ranking**: Ranked bar chart of discriminative flags
3. **Flag Count Distribution**: Violin plots comparing flag counts
4. **Co-occurrence Heatmap**: Matrix showing flag pairs
5. **Performance Time-Series**: Processing time and throughput over batches
6. **Performance Breakdown**: Pie chart of time allocation

## Usage

### Basic Usage

```bash
uv run python experiments/29_flag_performance_analysis/analyze_flags_performance.py \
  src/neuralshield/data/CSIC/test.jsonl
```

### With Options

```bash
uv run python experiments/29_flag_performance_analysis/analyze_flags_performance.py \
  src/neuralshield/data/CSIC/test.jsonl \
  --max-batches 10 \
  --wandb-project neuralshield-v2 \
  --wandb-run-name flag-analysis-csic-test
```

### Parameters

- `dataset_path`: Path to JSONL dataset file (required)
- `--max-batches`: Maximum number of batches to process (default: all)
- `--output-dir`: Output directory for results (default: `experiments/29_flag_performance_analysis/results`)
- `--wandb-project`: WandB project name (default: `neuralshield-v2`)
- `--wandb-run-name`: WandB run name (auto-generated if not provided)

## Output

### WandB Logs

**Time-Series Metrics** (logged per batch):
- `performance/batch_num`: Batch number
- `performance/avg_time_per_request_ms`: Average processing time per request
- `performance/batch_time_ms`: Total time for batch (1K requests)
- `performance/avg_time_per_1k_ms`: Average time per 1K requests
- `performance/avg_preprocessing_time_ms`: Average preprocessing time
- `performance/p95_preprocessing_time_ms`: P95 preprocessing time
- `performance/p99_preprocessing_time_ms`: P99 preprocessing time
- `performance/throughput_req_per_sec`: Throughput in requests/second

**Summary Metrics**:
- `performance/overall_avg_time_per_request_ms`: Overall average
- `performance/overall_throughput_req_per_sec`: Overall throughput
- `flags/total_unique_flags`: Number of unique flags found
- `flags/attack_requests`: Number of attack requests processed
- `flags/benign_requests`: Number of benign requests processed
- `flags/attack_avg_flags_per_request`: Average flags per attack request
- `flags/benign_avg_flags_per_request`: Average flags per benign request
- `flags/top_N_{flag}_signal_strength`: Top flags by signal strength

**Visualizations**:
- `visualizations/flag_presence_comparison`: Side-by-side comparison
- `visualizations/signal_strength_ranking`: Ranked flags
- `visualizations/flag_count_distribution`: Violin plots
- `visualizations/flag_cooccurrence`: Co-occurrence heatmap
- `visualizations/performance_time_series`: Time-series graphs
- `visualizations/performance_breakdown`: Pie chart

### JSON Results

Results are saved to `results/results.json` with:
- Flag statistics (presence rates, signal strength)
- Summary statistics
- Performance metrics
- Batch-by-batch metrics

## Example Insights

### Flag Analysis

- **QUOTE flag**: Appears in 45% of attacks vs 10% of benign → Strong signal
- **MULTIPLESLASH flag**: Appears in 62% of attacks vs 6% of benign → Very strong signal
- **HOME flag**: Appears in 5% of attacks vs 4% of benign → Weak signal

### Performance Metrics

- **Average processing time**: 1.2 ms per request
- **Throughput**: 800 req/sec
- **P95 latency**: 3.5 ms
- **P99 latency**: 8.2 ms

### Decision Support

- Which flags to prioritize in feature engineering
- Which flags cause false positives (high in benign)
- Flag combinations that indicate attacks
- Processing time impact of preprocessing
- Optimal batch sizes for production

## Architecture

```
experiments/29_flag_performance_analysis/
├── __init__.py
├── analyze_flags_performance.py  # Main experiment script
├── flag_analysis.py              # Flag extraction and analysis
├── visualizations.py             # Visualization functions
├── README.md                     # This file
└── results/                      # Output directory
    └── results.json              # JSON results
```

## Dependencies

- `wandb`: For experiment tracking and visualization
- `matplotlib`: For creating visualizations
- `seaborn`: For styling
- `numpy`: For numerical operations
- `typer`: For CLI interface
- `loguru`: For logging

## Notes

- Processes requests in batches of 1,000 for efficient processing
- Flags are extracted from preprocessed request strings
- Time-series metrics are logged to WandB for beautiful graphs
- All visualizations are automatically uploaded to WandB

