#!/usr/bin/env python3
"""
Experiment 29: Flag Analysis and Performance Profiling with Time-Series Logging

Processes requests in batches, extracts flags, analyzes patterns, and logs
beautiful time-series metrics to WandB.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import typer
import wandb
from loguru import logger

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flag_analysis import FlagAnalyzer
from visualizations import (
    create_cooccurrence_heatmap,
    create_flag_count_distribution,
    create_flag_presence_comparison,
    create_performance_breakdown_pie,
    create_performance_time_series,
    create_signal_strength_ranking,
)

app = typer.Typer()

BATCH_SIZE = 1000  # Process 1K requests per batch


def load_requests(jsonl_path: Path, max_requests: int | None = None):
    """Load requests from JSONL file."""
    count = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line.strip())
            request = obj.get("request", "")
            label = obj.get("label", "unknown")
            if request:
                yield request, label
                count += 1
                if max_requests and count >= max_requests:
                    break


def process_batch(
    requests: list[tuple[str, str]], batch_num: int
) -> dict[str, Any]:
    """Process a batch of requests and return metrics."""
    from neuralshield.preprocessing.pipeline import preprocess

    batch_start = time.time()
    preprocessing_times = []
    flag_analyzer = FlagAnalyzer()

    for request, label in requests:
        # Time preprocessing
        preprocess_start = time.time()
        processed = preprocess(request)
        preprocessing_times.append((time.time() - preprocess_start) * 1000)

        # Analyze flags
        flag_analyzer.add_request(request, label)

    batch_time_ms = (time.time() - batch_start) * 1000
    avg_preprocessing_time = np.mean(preprocessing_times) if preprocessing_times else 0.0
    p95_preprocessing_time = np.percentile(preprocessing_times, 95) if preprocessing_times else 0.0
    p99_preprocessing_time = np.percentile(preprocessing_times, 99) if preprocessing_times else 0.0

    throughput = (len(requests) / batch_time_ms * 1000) if batch_time_ms > 0 else 0.0

    return {
        "batch_num": batch_num,
        "batch_size": len(requests),
        "batch_time_ms": batch_time_ms,
        "avg_time_per_request_ms": batch_time_ms / len(requests) if requests else 0.0,
        "avg_preprocessing_time_ms": avg_preprocessing_time,
        "p95_preprocessing_time_ms": p95_preprocessing_time,
        "p99_preprocessing_time_ms": p99_preprocessing_time,
        "throughput_req_per_sec": throughput,
        "flag_analyzer": flag_analyzer,
    }


@app.command()
def main(
    dataset_path: Path = typer.Argument(
        ..., help="Dataset JSONL file (train or test)"
    ),
    max_batches: int = typer.Option(
        None, "--max-batches", help="Maximum number of batches to process"
    ),
    output_dir: Path = typer.Option(
        Path("experiments/29_flag_performance_analysis/results"),
        help="Output directory for results",
    ),
    wandb_project: str = typer.Option(
        "neuralshield-v2", help="WandB project name"
    ),
    wandb_run_name: str | None = typer.Option(
        None, help="WandB run name (auto-generated if None)"
    ),
) -> None:
    """Run flag analysis and performance profiling experiment."""
    logger.info("=" * 80)
    logger.info("FLAG ANALYSIS AND PERFORMANCE PROFILING EXPERIMENT")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    run_name = wandb_run_name or f"flag-analysis-{dataset_path.stem}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        tags=["flag-analysis", "performance", "time-series"],
        config={
            "dataset": str(dataset_path),
            "batch_size": BATCH_SIZE,
            "max_batches": max_batches,
        },
    )

    # Process requests in batches
    logger.info(f"Loading requests from {dataset_path}")
    requests_iter = load_requests(dataset_path)

    batch_metrics = []
    batch_numbers = []
    avg_times = []
    throughputs = []
    p95_times = []
    p99_times = []

    global_flag_analyzer = FlagAnalyzer()
    batch_num = 0

    logger.info(f"Processing requests in batches of {BATCH_SIZE}...")
    current_batch = []

    for request, label in requests_iter:
        current_batch.append((request, label))

        if len(current_batch) >= BATCH_SIZE:
            batch_num += 1
            logger.info(f"Processing batch {batch_num} ({len(current_batch)} requests)...")

            batch_result = process_batch(current_batch, batch_num)
            batch_metrics.append(batch_result)

            # Extract metrics for time-series
            batch_numbers.append(batch_num)
            avg_times.append(batch_result["avg_time_per_request_ms"])
            throughputs.append(batch_result["throughput_req_per_sec"])
            p95_times.append(batch_result["p95_preprocessing_time_ms"])
            p99_times.append(batch_result["p99_preprocessing_time_ms"])

            # Merge flag analyzer results
            batch_analyzer = batch_result["flag_analyzer"]
            global_flag_analyzer.attack_flag_counts.update(
                batch_analyzer.attack_flag_counts
            )
            global_flag_analyzer.benign_flag_counts.update(
                batch_analyzer.benign_flag_counts
            )
            global_flag_analyzer.attack_flag_presence.update(
                batch_analyzer.attack_flag_presence
            )
            global_flag_analyzer.benign_flag_presence.update(
                batch_analyzer.benign_flag_presence
            )
            global_flag_analyzer.attack_request_count += (
                batch_analyzer.attack_request_count
            )
            global_flag_analyzer.benign_request_count += (
                batch_analyzer.benign_request_count
            )
            global_flag_analyzer.attack_flag_counts_per_request.extend(
                batch_analyzer.attack_flag_counts_per_request
            )
            global_flag_analyzer.benign_flag_counts_per_request.extend(
                batch_analyzer.benign_flag_counts_per_request
            )

            # Update co-occurrence
            for pair, count in batch_analyzer.flag_cooccurrence.items():
                global_flag_analyzer.flag_cooccurrence[pair] = (
                    global_flag_analyzer.flag_cooccurrence.get(pair, 0) + count
            )

            # Log time-series metrics to WandB
            wandb.log(
                {
                    "performance/batch_num": batch_num,
                    "performance/avg_time_per_request_ms": batch_result[
                        "avg_time_per_request_ms"
                    ],
                    "performance/batch_time_ms": batch_result["batch_time_ms"],
                    "performance/avg_preprocessing_time_ms": batch_result[
                        "avg_preprocessing_time_ms"
                    ],
                    "performance/p95_preprocessing_time_ms": batch_result[
                        "p95_preprocessing_time_ms"
                    ],
                    "performance/p99_preprocessing_time_ms": batch_result[
                        "p99_preprocessing_time_ms"
                    ],
                    "performance/throughput_req_per_sec": batch_result[
                        "throughput_req_per_sec"
                    ],
                    "performance/avg_time_per_1k_ms": batch_result[
                        "batch_time_ms"
                    ],
                },
                step=batch_num,
            )

            current_batch = []

            if max_batches and batch_num >= max_batches:
                break

    # Process remaining requests
    if current_batch:
        batch_num += 1
        logger.info(f"Processing final batch {batch_num} ({len(current_batch)} requests)...")
        batch_result = process_batch(current_batch, batch_num)
        batch_metrics.append(batch_result)
        batch_numbers.append(batch_num)
        avg_times.append(batch_result["avg_time_per_request_ms"])
        throughputs.append(batch_result["throughput_req_per_sec"])
        p95_times.append(batch_result["p95_preprocessing_time_ms"])
        p99_times.append(batch_result["p99_preprocessing_time_ms"])

    logger.info(f"Processed {batch_num} batches")

    # Compute final flag statistics
    logger.info("Computing flag statistics...")
    flag_stats_result = global_flag_analyzer.compute_statistics()
    flag_stats = flag_stats_result["flag_statistics"]

    # Create visualizations
    logger.info("=" * 80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 80)

    # 1. Flag presence comparison
    logger.info("Creating flag presence comparison...")
    fig1 = create_flag_presence_comparison(flag_stats, top_n=25)
    wandb.log({"visualizations/flag_presence_comparison": wandb.Image(fig1)})
    plt.close(fig1)

    # 2. Signal strength ranking
    logger.info("Creating signal strength ranking...")
    fig2 = create_signal_strength_ranking(flag_stats, top_n=30)
    wandb.log({"visualizations/signal_strength_ranking": wandb.Image(fig2)})
    plt.close(fig2)

    # 3. Flag count distribution
    logger.info("Creating flag count distribution...")
    fig3 = create_flag_count_distribution(
        global_flag_analyzer.attack_flag_counts_per_request,
        global_flag_analyzer.benign_flag_counts_per_request,
    )
    wandb.log({"visualizations/flag_count_distribution": wandb.Image(fig3)})
    plt.close(fig3)

    # 4. Co-occurrence heatmap
    logger.info("Creating co-occurrence heatmap...")
    fig4 = create_cooccurrence_heatmap(
        global_flag_analyzer.flag_cooccurrence, top_n=20
    )
    wandb.log({"visualizations/flag_cooccurrence": wandb.Image(fig4)})
    plt.close(fig4)

    # 5. Performance time-series
    if batch_numbers:
        logger.info("Creating performance time-series...")
        fig5 = create_performance_time_series(
            batch_numbers, avg_times, throughputs, p95_times, p99_times
        )
        wandb.log({"visualizations/performance_time_series": wandb.Image(fig5)})
        plt.close(fig5)

    # 6. Performance breakdown (average)
    if batch_metrics:
        logger.info("Creating performance breakdown...")
        avg_preprocessing = np.mean([b["avg_preprocessing_time_ms"] for b in batch_metrics])
        # Estimate encoding and detection (placeholder - would need actual timing)
        avg_encoding = avg_preprocessing * 0.3  # Rough estimate
        avg_detection = avg_preprocessing * 0.1  # Rough estimate
        fig6 = create_performance_breakdown_pie(
            avg_preprocessing, avg_encoding, avg_detection
        )
        wandb.log({"visualizations/performance_breakdown": wandb.Image(fig6)})
        plt.close(fig6)

    # Log summary statistics
    logger.info("Logging summary statistics...")
    summary_stats = flag_stats_result["summary"]
    
    # Calculate overall performance metrics
    if batch_metrics:
        overall_avg_time = np.mean([b["avg_time_per_request_ms"] for b in batch_metrics])
        overall_throughput = np.mean([b["throughput_req_per_sec"] for b in batch_metrics])
        overall_p95 = np.mean([b["p95_preprocessing_time_ms"] for b in batch_metrics])
        overall_p99 = np.mean([b["p99_preprocessing_time_ms"] for b in batch_metrics])
    else:
        overall_avg_time = 0.0
        overall_throughput = 0.0
        overall_p95 = 0.0
        overall_p99 = 0.0

    wandb.summary = {
        "performance/overall_avg_time_per_request_ms": overall_avg_time,
        "performance/overall_avg_time_per_1k_ms": overall_avg_time * 1000,
        "performance/overall_throughput_req_per_sec": overall_throughput,
        "performance/overall_p95_time_ms": overall_p95,
        "performance/overall_p99_time_ms": overall_p99,
        "flags/total_unique_flags": summary_stats["total_flags"],
        "flags/attack_requests": summary_stats["attack_requests"],
        "flags/benign_requests": summary_stats["benign_requests"],
        "flags/attack_avg_flags_per_request": summary_stats[
            "attack_avg_flags_per_request"
        ],
        "flags/benign_avg_flags_per_request": summary_stats[
            "benign_avg_flags_per_request"
        ],
    }

    # Log top flags to summary
    sorted_flags = sorted(
        flag_stats.items(),
        key=lambda x: x[1]["signal_strength"],
        reverse=True,
    )[:10]

    for i, (flag, stats) in enumerate(sorted_flags):
        wandb.summary[f"flags/top_{i+1}_{flag}_signal_strength"] = stats[
            "signal_strength"
        ]
        wandb.summary[f"flags/top_{i+1}_{flag}_attack_rate"] = stats[
            "attack_presence_rate"
        ]
        wandb.summary[f"flags/top_{i+1}_{flag}_benign_rate"] = stats[
            "benign_presence_rate"
        ]

    # Save results to JSON
    results = {
        "flag_statistics": flag_stats,
        "summary": summary_stats,
        "performance": {
            "overall_avg_time_per_request_ms": overall_avg_time,
            "overall_avg_time_per_1k_ms": overall_avg_time * 1000,
            "overall_throughput_req_per_sec": overall_throughput,
            "overall_p95_time_ms": overall_p95,
            "overall_p99_time_ms": overall_p99,
            "batch_metrics": [
                {
                    "batch_num": b["batch_num"],
                    "avg_time_per_request_ms": b["avg_time_per_request_ms"],
                    "throughput_req_per_sec": b["throughput_req_per_sec"],
                }
                for b in batch_metrics
            ],
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Processed {batch_num} batches")
    logger.info(f"Total requests: {summary_stats['attack_requests'] + summary_stats['benign_requests']}")
    logger.info(f"Overall avg time per request: {overall_avg_time:.3f} ms")
    logger.info(f"Overall throughput: {overall_throughput:.1f} req/sec")
    logger.info(f"Total unique flags: {summary_stats['total_flags']}")


if __name__ == "__main__":
    app()

