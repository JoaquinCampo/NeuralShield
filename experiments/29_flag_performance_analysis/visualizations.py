"""Beautiful visualization functions for flag analysis and performance metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mutual_info_score

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# Color palette
COLOR_ATTACK = "#e74c3c"  # Red
COLOR_BENIGN = "#3498db"  # Blue
COLOR_SIGNAL = "#2ecc71"  # Green
COLOR_WARNING = "#f39c12"  # Orange


def create_flag_presence_comparison(
    flag_stats: dict[str, dict], top_n: int = 20
) -> plt.Figure:
    """Create side-by-side bar chart comparing flag presence in attacks vs benign."""
    # Sort by signal strength
    sorted_flags = sorted(
        flag_stats.items(),
        key=lambda x: x[1]["signal_strength"],
        reverse=True,
    )[:top_n]

    flags = [f[0] for f in sorted_flags]
    attack_rates = [f[1]["attack_presence_rate"] for f in sorted_flags]
    benign_rates = [f[1]["benign_presence_rate"] for f in sorted_flags]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(flags))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        attack_rates,
        width,
        label="Attack",
        color=COLOR_ATTACK,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        benign_rates,
        width,
        label="Benign",
        color=COLOR_BENIGN,
        alpha=0.8,
    )

    ax.set_xlabel("Flag", fontsize=12, fontweight="bold")
    ax.set_ylabel("Presence Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Flags: Attack vs Benign Presence Rates",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(flags, rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label if significant
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    return fig


def create_signal_strength_ranking(
    flag_stats: dict[str, dict], top_n: int = 25
) -> plt.Figure:
    """Create ranked bar chart of flag signal strength."""
    sorted_flags = sorted(
        flag_stats.items(),
        key=lambda x: x[1]["signal_strength"],
        reverse=True,
    )[:top_n]

    flags = [f[0] for f in sorted_flags]
    signal_strengths = [f[1]["signal_strength"] for f in sorted_flags]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [
        COLOR_SIGNAL if s > 0 else COLOR_WARNING for s in signal_strengths
    ]

    bars = ax.barh(flags, signal_strengths, color=colors, alpha=0.8)

    ax.set_xlabel("Signal Strength (Attack Rate - Benign Rate)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flag", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Flags by Signal Strength",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, strength) in enumerate(zip(bars, signal_strengths)):
        if abs(strength) > 0.01:
            ax.text(
                strength,
                i,
                f"{strength:.2%}",
                va="center",
                ha="left" if strength > 0 else "right",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


def create_flag_count_distribution(
    attack_counts: list[int], benign_counts: list[int]
) -> plt.Figure:
    """Create violin plot comparing flag count distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [benign_counts, attack_counts]
    positions = [1, 2]
    labels = [
        f"Benign\n(n={len(benign_counts):,})",
        f"Attack\n(n={len(attack_counts):,})",
    ]

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.6,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )

    # Color the violins
    for pc, color in zip(parts["bodies"], [COLOR_BENIGN, COLOR_ATTACK]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")
        pc.set_linewidth(1.5)

    # Style the means and medians
    parts["cmeans"].set_color("white")
    parts["cmeans"].set_linewidth(2)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Flags per Request", fontsize=12, fontweight="bold")
    ax.set_title(
        "Flag Count Distribution: Attack vs Benign",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def create_cooccurrence_heatmap(
    cooccurrence: dict[tuple[str, str], int], top_n: int = 15
) -> plt.Figure:
    """Create heatmap of flag co-occurrence patterns."""
    # Get top flags by frequency
    all_flags = set()
    for (f1, f2) in cooccurrence.keys():
        all_flags.add(f1)
        all_flags.add(f2)

    flag_freq = {}
    for flag in all_flags:
        freq = sum(
            count
            for (f1, f2), count in cooccurrence.items()
            if f1 == flag or f2 == flag
        )
        flag_freq[flag] = freq

    top_flags = sorted(flag_freq.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    top_flag_set = {f[0] for f in top_flags}

    # Build matrix
    matrix = np.zeros((len(top_flags), len(top_flags)))
    flag_list = [f[0] for f in top_flags]
    flag_to_idx = {flag: i for i, flag in enumerate(flag_list)}

    for (f1, f2), count in cooccurrence.items():
        if f1 in top_flag_set and f2 in top_flag_set:
            i = flag_to_idx[f1]
            j = flag_to_idx[f2]
            matrix[i][j] = count
            matrix[j][i] = count  # Symmetric

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(flag_list)))
    ax.set_yticks(np.arange(len(flag_list)))
    ax.set_xticklabels(flag_list, rotation=45, ha="right")
    ax.set_yticklabels(flag_list)

    # Add text annotations
    for i in range(len(flag_list)):
        for j in range(len(flag_list)):
            if matrix[i, j] > 0:
                text = ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="black" if matrix[i, j] < matrix.max() / 2 else "white",
                    fontsize=8,
                )

    ax.set_title(
        f"Flag Co-occurrence Heatmap (Top {top_n} Flags)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.colorbar(im, ax=ax, label="Co-occurrence Count")
    plt.tight_layout()
    return fig


def create_performance_time_series(
    batch_numbers: list[int],
    avg_times: list[float],
    throughputs: list[float],
    p95_times: list[float],
    p99_times: list[float],
) -> plt.Figure:
    """Create time-series plot of performance metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Processing time
    ax1.plot(
        batch_numbers,
        avg_times,
        label="Average",
        color=COLOR_BENIGN,
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        batch_numbers,
        p95_times,
        label="P95",
        color=COLOR_WARNING,
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    ax1.plot(
        batch_numbers,
        p99_times,
        label="P99",
        color=COLOR_ATTACK,
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    ax1.set_ylabel("Processing Time (ms)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Processing Time per 1K Requests (Time Series)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Throughput
    ax2.plot(
        batch_numbers,
        throughputs,
        label="Throughput",
        color=COLOR_SIGNAL,
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax2.set_xlabel("Batch Number (1K requests per batch)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Throughput (req/sec)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Throughput Over Time",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def create_performance_breakdown_pie(
    preprocessing_time: float,
    encoding_time: float,
    detection_time: float,
) -> plt.Figure:
    """Create pie chart of performance breakdown."""
    fig, ax = plt.subplots(figsize=(8, 8))

    sizes = [preprocessing_time, encoding_time, detection_time]
    labels = ["Preprocessing", "Encoding", "Detection"]
    colors = [COLOR_BENIGN, COLOR_SIGNAL, COLOR_ATTACK]
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.set_title(
        "Processing Time Breakdown",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    return fig

