#!/usr/bin/env python3
"""
Experiment 30: Flag Analysis and Performance Profiling with Structured WandB Logging

Processes requests in batches, extracts flags, analyzes patterns, and logs
all data as structured WandB tables and metrics (no images).

This version is optimized for systems that read WandB logs programmatically.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger

import wandb

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flag_analysis import FlagAnalyzer

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


def process_batch(requests: list[tuple[str, str]], batch_num: int) -> dict[str, Any]:
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
    avg_preprocessing_time = (
        np.mean(preprocessing_times) if preprocessing_times else 0.0
    )
    p95_preprocessing_time = (
        np.percentile(preprocessing_times, 95) if preprocessing_times else 0.0
    )
    p99_preprocessing_time = (
        np.percentile(preprocessing_times, 99) if preprocessing_times else 0.0
    )

    throughput = (len(requests) / batch_time_ms * 1000) if batch_time_ms > 0 else 0.0

    return {
        "batch_num": batch_num,
        "batch_size": len(requests),
        "batch_time_ms": batch_time_ms,
        "avg_time_per_request_ms": (batch_time_ms / len(requests) if requests else 0.0),
        "avg_preprocessing_time_ms": avg_preprocessing_time,
        "p95_preprocessing_time_ms": p95_preprocessing_time,
        "p99_preprocessing_time_ms": p99_preprocessing_time,
        "throughput_req_per_sec": throughput,
        "flag_analyzer": flag_analyzer,
    }


@app.command()
def main(
    dataset_path: Path = typer.Argument(..., help="Dataset JSONL file (train or test)"),
    max_batches: int = typer.Option(
        None, "--max-batches", help="Maximum number of batches to process"
    ),
    output_dir: Path = typer.Option(
        Path("experiments/30_flag_performance_structured/results"),
        help="Output directory for results",
    ),
    wandb_project: str = typer.Option("neuralshield-v2", help="WandB project name"),
    wandb_run_name: str | None = typer.Option(
        None, help="WandB run name (auto-generated if None)"
    ),
) -> None:
    """Run flag analysis and performance profiling experiment with structured logging."""
    logger.info("=" * 80)
    logger.info("FLAG ANALYSIS AND PERFORMANCE PROFILING (STRUCTURED LOGGING)")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    if wandb_run_name:
        run_name = wandb_run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"exp30-flag-analysis-structured-{dataset_path.stem}-{timestamp}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        tags=["flag-analysis", "performance", "structured", "tables", "exp30"],
        config={
            "dataset": str(dataset_path),
            "batch_size": BATCH_SIZE,
            "max_batches": max_batches,
            "logging_format": "structured_tables",
            "experiment": "30_flag_performance_structured",
        },
    )

    # Process requests in batches
    logger.info(f"Loading requests from {dataset_path}")
    requests_iter = load_requests(dataset_path)

    batch_metrics = []

    global_flag_analyzer = FlagAnalyzer()
    batch_num = 0

    logger.info(f"Processing requests in batches of {BATCH_SIZE}...")
    current_batch = []

    for request, label in requests_iter:
        current_batch.append((request, label))

        if len(current_batch) >= BATCH_SIZE:
            batch_num += 1
            logger.info(
                f"Processing batch {batch_num} ({len(current_batch)} requests)..."
            )

            batch_result = process_batch(current_batch, batch_num)
            batch_metrics.append(batch_result)

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

            # Log time-series metrics to WandB (structured data)
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
                    "performance/avg_time_per_1k_ms": batch_result["batch_time_ms"],
                },
                step=batch_num,
            )

            current_batch = []

            if max_batches and batch_num >= max_batches:
                break

    # Process remaining requests
    if current_batch:
        batch_num += 1
        logger.info(
            f"Processing final batch {batch_num} ({len(current_batch)} requests)..."
        )
        batch_result = process_batch(current_batch, batch_num)
        batch_metrics.append(batch_result)

    logger.info(f"Processed {batch_num} batches")

    # Compute final flag statistics
    logger.info("Computing flag statistics...")
    flag_stats_result = global_flag_analyzer.compute_statistics()
    flag_stats = flag_stats_result["flag_statistics"]

    # Compute new advanced analyses
    logger.info("Computing mutual information...")
    mi_scores = global_flag_analyzer.compute_mutual_information()

    logger.info("Computing correlation matrix...")
    flag_list, corr_matrix = global_flag_analyzer.compute_correlation_matrix()

    logger.info("Computing interaction effects...")
    interaction_effects = global_flag_analyzer.compute_interaction_effects()

    logger.info("Computing rarity statistics...")
    rarity_stats = global_flag_analyzer.compute_rarity_stats()

    logger.info("Computing family statistics...")
    family_stats = global_flag_analyzer.compute_family_stats()

    logger.info("Computing frequency distributions...")
    frequency_distributions = global_flag_analyzer.compute_frequency_distributions()

    logger.info("Computing sequence statistics...")
    sequence_stats = global_flag_analyzer.compute_sequence_stats(top_n=50)

    # Log flag statistics as WandB Table (structured data)
    logger.info("Logging flag statistics as WandB table...")
    sorted_flags = sorted(
        flag_stats.items(),
        key=lambda x: x[1]["signal_strength"],
        reverse=True,
    )

    # Create table data
    flag_table_rows = []
    for flag, stats in sorted_flags:
        flag_table_rows.append(
            [
                flag,
                float(stats["attack_presence_rate"]),
                float(stats["benign_presence_rate"]),
                float(stats["signal_strength"]),
                int(stats["attack_count"]),
                int(stats["benign_count"]),
                float(stats["attack_per_request"]),
                float(stats["benign_per_request"]),
            ]
        )

    flag_table = wandb.Table(
        columns=[
            "flag",
            "attack_presence_rate",
            "benign_presence_rate",
            "signal_strength",
            "attack_count",
            "benign_count",
            "attack_per_request",
            "benign_per_request",
        ],
        data=flag_table_rows,
    )
    wandb.log({"flags/statistics_table": flag_table})

    # Log co-occurrence as table
    logger.info("Logging flag co-occurrence as WandB table...")
    cooccurrence_rows = []
    for (flag1, flag2), count in sorted(
        global_flag_analyzer.flag_cooccurrence.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:500]:  # Top 500 pairs
        cooccurrence_rows.append([flag1, flag2, int(count)])

    cooccurrence_table = wandb.Table(
        columns=["flag1", "flag2", "cooccurrence_count"],
        data=cooccurrence_rows,
    )
    wandb.log({"flags/cooccurrence_table": cooccurrence_table})

    # Log flag count distributions as table
    logger.info("Logging flag count distributions as WandB table...")
    distribution_rows = []

    # Attack distribution
    attack_counts = global_flag_analyzer.attack_flag_counts_per_request
    if attack_counts:
        attack_unique, attack_freq = np.unique(attack_counts, return_counts=True)
        for count, freq in zip(attack_unique, attack_freq):
            distribution_rows.append(["attack", int(count), int(freq)])

    # Benign distribution
    benign_counts = global_flag_analyzer.benign_flag_counts_per_request
    if benign_counts:
        benign_unique, benign_freq = np.unique(benign_counts, return_counts=True)
        for count, freq in zip(benign_unique, benign_freq):
            distribution_rows.append(["benign", int(count), int(freq)])

    distribution_table = wandb.Table(
        columns=["label", "flag_count", "frequency"],
        data=distribution_rows,
    )
    wandb.log({"flags/count_distribution_table": distribution_table})

    # Log batch-by-batch performance as table
    logger.info("Logging batch performance as WandB table...")
    batch_table_rows = []
    for b in batch_metrics:
        batch_table_rows.append(
            [
                int(b["batch_num"]),
                int(b["batch_size"]),
                float(b["batch_time_ms"]),
                float(b["avg_time_per_request_ms"]),
                float(b["avg_preprocessing_time_ms"]),
                float(b["p95_preprocessing_time_ms"]),
                float(b["p99_preprocessing_time_ms"]),
                float(b["throughput_req_per_sec"]),
            ]
        )

    batch_table = wandb.Table(
        columns=[
            "batch_num",
            "batch_size",
            "batch_time_ms",
            "avg_time_per_request_ms",
            "avg_preprocessing_time_ms",
            "p95_preprocessing_time_ms",
            "p99_preprocessing_time_ms",
            "throughput_req_per_sec",
        ],
        data=batch_table_rows,
    )
    wandb.log({"performance/batch_metrics_table": batch_table})

    # Log mutual information as table
    if mi_scores:
        logger.info("Logging mutual information as WandB table...")
        mi_rows = []
        for flag, mi_score in sorted(
            mi_scores.items(), key=lambda x: x[1], reverse=True
        ):
            mi_rows.append([flag, float(mi_score)])

        mi_table = wandb.Table(
            columns=["flag", "mutual_information"],
            data=mi_rows,
        )
        wandb.log({"flags/mutual_information_table": mi_table})

    # Log correlation matrix as table
    if len(flag_list) > 0 and corr_matrix.size > 0:
        logger.info("Logging correlation matrix as WandB table...")
        corr_rows = []
        for i, flag1 in enumerate(flag_list):
            for j, flag2 in enumerate(flag_list):
                if i <= j:  # Upper triangle only (symmetric)
                    corr_rows.append([flag1, flag2, float(corr_matrix[i, j])])

        corr_table = wandb.Table(
            columns=["flag1", "flag2", "correlation"],
            data=corr_rows,
        )
        wandb.log({"flags/correlation_matrix_table": corr_table})

    # Log interaction effects as table
    if interaction_effects:
        logger.info("Logging interaction effects as WandB table...")
        interaction_rows = []
        for (flag1, flag2), stats in sorted(
            interaction_effects.items(),
            key=lambda x: x[1]["signal_strength"],
            reverse=True,
        )[:500]:  # Top 500 pairs
            interaction_rows.append(
                [
                    flag1,
                    flag2,
                    float(stats["attack_rate"]),
                    float(stats["benign_rate"]),
                    float(stats["signal_strength"]),
                    int(stats["attack_count"]),
                    int(stats["benign_count"]),
                    int(stats["total_count"]),
                ]
            )

        interaction_table = wandb.Table(
            columns=[
                "flag1",
                "flag2",
                "attack_rate",
                "benign_rate",
                "signal_strength",
                "attack_count",
                "benign_count",
                "total_count",
            ],
            data=interaction_rows,
        )
        wandb.log({"flags/interaction_effects_table": interaction_table})

    # Log rarity statistics as table
    if rarity_stats:
        logger.info("Logging rarity statistics as WandB table...")
        rarity_rows = []
        for flag, stats in sorted(
            rarity_stats.items(),
            key=lambda x: x[1]["rarity"],
            reverse=True,
        ):
            rarity_rows.append(
                [
                    flag,
                    float(stats["frequency"]),
                    float(stats["rarity"]),
                    int(stats["total_count"]),
                    int(stats["requests_with_flag"]),
                ]
            )

        rarity_table = wandb.Table(
            columns=[
                "flag",
                "frequency",
                "rarity",
                "total_count",
                "requests_with_flag",
            ],
            data=rarity_rows,
        )
        wandb.log({"flags/rarity_statistics_table": rarity_table})

    # Log family statistics as table
    if family_stats:
        logger.info("Logging family statistics as WandB table...")
        family_rows = []
        for family, stats in family_stats.items():
            family_rows.append(
                [
                    family,
                    int(stats["attack_count"]),
                    int(stats["benign_count"]),
                    float(stats["attack_presence_rate"]),
                    float(stats["benign_presence_rate"]),
                    float(stats["signal_strength"]),
                    int(stats["flags_in_family"]),
                ]
            )

        family_table = wandb.Table(
            columns=[
                "family",
                "attack_count",
                "benign_count",
                "attack_presence_rate",
                "benign_presence_rate",
                "signal_strength",
                "flags_in_family",
            ],
            data=family_rows,
        )
        wandb.log({"flags/family_statistics_table": family_table})

    # Log frequency distributions as table
    if frequency_distributions:
        logger.info("Logging frequency distributions as WandB table...")
        freq_dist_rows = []
        for flag, stats in sorted(
            frequency_distributions.items(),
            key=lambda x: x[1]["total_occurrences"],
            reverse=True,
        ):
            freq_dist_rows.append(
                [
                    flag,
                    float(stats["mean"]),
                    float(stats["median"]),
                    float(stats["p25"]),
                    float(stats["p75"]),
                    float(stats["p95"]),
                    float(stats["p99"]),
                    float(stats["std"]),
                    int(stats["min"]),
                    int(stats["max"]),
                    int(stats["total_occurrences"]),
                ]
            )

        freq_dist_table = wandb.Table(
            columns=[
                "flag",
                "mean",
                "median",
                "p25",
                "p75",
                "p95",
                "p99",
                "std",
                "min",
                "max",
                "total_occurrences",
            ],
            data=freq_dist_rows,
        )
        wandb.log({"flags/frequency_distributions_table": freq_dist_table})

    # Log sequence statistics as table
    if sequence_stats:
        logger.info("Logging sequence statistics as WandB table...")
        sequence_rows = []
        for sequence, count, attack_rate in sequence_stats:
            sequence_str = " → ".join(sequence)
            sequence_rows.append(
                [
                    sequence_str,
                    int(count),
                    float(attack_rate),
                    len(sequence),
                ]
            )

        sequence_table = wandb.Table(
            columns=[
                "sequence",
                "frequency",
                "attack_rate",
                "sequence_length",
            ],
            data=sequence_rows,
        )
        wandb.log({"flags/sequence_statistics_table": sequence_table})

    # Log summary statistics
    logger.info("Logging summary statistics...")
    summary_stats = flag_stats_result["summary"]

    # Calculate overall performance metrics
    if batch_metrics:
        overall_avg_time = np.mean(
            [b["avg_time_per_request_ms"] for b in batch_metrics]
        )
        overall_throughput = np.mean(
            [b["throughput_req_per_sec"] for b in batch_metrics]
        )
        overall_p95 = np.mean([b["p95_preprocessing_time_ms"] for b in batch_metrics])
        overall_p99 = np.mean([b["p99_preprocessing_time_ms"] for b in batch_metrics])
    else:
        overall_avg_time = 0.0
        overall_throughput = 0.0
        overall_p95 = 0.0
        overall_p99 = 0.0

    wandb.summary = {
        "performance/overall_avg_time_per_request_ms": float(overall_avg_time),
        "performance/overall_avg_time_per_1k_ms": float(overall_avg_time * 1000),
        "performance/overall_throughput_req_per_sec": float(overall_throughput),
        "performance/overall_p95_time_ms": float(overall_p95),
        "performance/overall_p99_time_ms": float(overall_p99),
        "flags/total_unique_flags": int(summary_stats["total_flags"]),
        "flags/attack_requests": int(summary_stats["attack_requests"]),
        "flags/benign_requests": int(summary_stats["benign_requests"]),
        "flags/attack_avg_flags_per_request": float(
            summary_stats["attack_avg_flags_per_request"]
        ),
        "flags/benign_avg_flags_per_request": float(
            summary_stats["benign_avg_flags_per_request"]
        ),
    }

    # Log top flags to summary (for quick access)
    sorted_flags_top = sorted(
        flag_stats.items(),
        key=lambda x: x[1]["signal_strength"],
        reverse=True,
    )[:20]  # Top 20 flags

    for i, (flag, stats) in enumerate(sorted_flags_top):
        wandb.summary[f"flags/top_{i + 1}_{flag}_signal_strength"] = float(
            stats["signal_strength"]
        )
        wandb.summary[f"flags/top_{i + 1}_{flag}_attack_rate"] = float(
            stats["attack_presence_rate"]
        )
        wandb.summary[f"flags/top_{i + 1}_{flag}_benign_rate"] = float(
            stats["benign_presence_rate"]
        )

    # Save results to JSON
    results = {
        "flag_statistics": flag_stats,
        "summary": summary_stats,
        "performance": {
            "overall_avg_time_per_request_ms": float(overall_avg_time),
            "overall_avg_time_per_1k_ms": float(overall_avg_time * 1000),
            "overall_throughput_req_per_sec": float(overall_throughput),
            "overall_p95_time_ms": float(overall_p95),
            "overall_p99_time_ms": float(overall_p99),
            "batch_metrics": [
                {
                    "batch_num": b["batch_num"],
                    "avg_time_per_request_ms": float(b["avg_time_per_request_ms"]),
                    "throughput_req_per_sec": float(b["throughput_req_per_sec"]),
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
    logger.info(
        f"Total requests: {summary_stats['attack_requests'] + summary_stats['benign_requests']}"
    )
    logger.info(f"Overall avg time per request: {overall_avg_time:.3f} ms")
    logger.info(f"Overall throughput: {overall_throughput:.1f} req/sec")
    logger.info(f"Total unique flags: {summary_stats['total_flags']}")
    logger.info("All data logged as structured WandB tables (no images)")


if __name__ == "__main__":
    app()
