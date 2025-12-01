"""Beautiful visualization functions for flag analysis and performance metrics."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
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


def create_mutual_information_ranking(
    mi_scores: dict[str, float], top_n: int = 30
) -> plt.Figure:
    """Create ranked bar chart of flag mutual information scores."""
    sorted_flags = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    flags = [f[0] for f in sorted_flags]
    mi_values = [f[1] for f in sorted_flags]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(flags, mi_values, color=COLOR_SIGNAL, alpha=0.8)

    ax.set_xlabel("Mutual Information", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flag", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Flags by Mutual Information",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, mi_val) in enumerate(zip(bars, mi_values)):
        ax.text(
            mi_val,
            i,
            f"{mi_val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def create_correlation_heatmap(
    flag_list: list[str], corr_matrix: np.ndarray, top_n: int = 20
) -> plt.Figure:
    """Create correlation heatmap for flags."""
    # Select top N flags by frequency or signal strength
    if len(flag_list) > top_n:
        # Use first top_n (already sorted)
        selected_flags = flag_list[:top_n]
        selected_indices = list(range(top_n))
        selected_matrix = corr_matrix[np.ix_(selected_indices, selected_indices)]
    else:
        selected_flags = flag_list
        selected_matrix = corr_matrix

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(selected_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(selected_flags)))
    ax.set_yticks(np.arange(len(selected_flags)))
    ax.set_xticklabels(selected_flags, rotation=45, ha="right")
    ax.set_yticklabels(selected_flags)

    # Add text annotations
    for i in range(len(selected_flags)):
        for j in range(len(selected_flags)):
            text = ax.text(
                j,
                i,
                f"{selected_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(selected_matrix[i, j]) > 0.5 else "black",
                fontsize=7,
            )

    ax.set_title(
        f"Flag Correlation Matrix (Top {len(selected_flags)} Flags)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.colorbar(im, ax=ax, label="Correlation Coefficient")
    plt.tight_layout()
    return fig


def create_interaction_effects_heatmap(
    interaction_effects: dict[tuple[str, str], dict[str, float]], top_n: int = 15
) -> plt.Figure:
    """Create heatmap showing interaction effects (attack rates) for flag pairs."""
    # Sort by signal strength
    sorted_pairs = sorted(
        interaction_effects.items(),
        key=lambda x: x[1]["signal_strength"],
        reverse=True,
    )[:top_n]

    # Get unique flags
    all_flags = set()
    for (f1, f2), _ in sorted_pairs:
        all_flags.add(f1)
        all_flags.add(f2)

    flag_list = sorted(all_flags)
    flag_to_idx = {flag: i for i, flag in enumerate(flag_list)}
    matrix = np.zeros((len(flag_list), len(flag_list)))

    for (f1, f2), stats in sorted_pairs:
        i = flag_to_idx[f1]
        j = flag_to_idx[f2]
        matrix[i][j] = stats["attack_rate"]
        matrix[j][i] = stats["attack_rate"]  # Symmetric

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

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
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.5 else "black",
                    fontsize=8,
                )

    ax.set_title(
        f"Flag Interaction Effects: Attack Rate for Pairs (Top {top_n})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.colorbar(im, ax=ax, label="Attack Rate")
    plt.tight_layout()
    return fig


def create_rarity_analysis(
    rarity_stats: dict[str, dict[str, float]],
    flag_stats: dict[str, dict[str, float]],
    top_n: int = 25,
) -> plt.Figure:
    """Create scatter plot: rarity vs signal strength."""
    # Combine rarity and signal strength
    combined = []
    for flag in rarity_stats.keys():
        if flag in flag_stats:
            combined.append(
                {
                    "flag": flag,
                    "rarity": rarity_stats[flag]["rarity"],
                    "signal_strength": flag_stats[flag]["signal_strength"],
                    "frequency": rarity_stats[flag]["frequency"],
                }
            )

    # Sort by signal strength
    combined.sort(key=lambda x: x["signal_strength"], reverse=True)
    combined = combined[:top_n]

    flags = [c["flag"] for c in combined]
    rarities = [c["rarity"] for c in combined]
    signal_strengths = [c["signal_strength"] for c in combined]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        rarities, signal_strengths, s=100, alpha=0.6, c=signal_strengths, cmap="RdYlGn"
    )

    # Add flag labels
    for i, flag in enumerate(flags):
        ax.annotate(
            flag,
            (rarities[i], signal_strengths[i]),
            fontsize=8,
            alpha=0.7,
            rotation=45,
        )

    ax.set_xlabel("Rarity (1 - Frequency)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Signal Strength (Attack Rate - Benign Rate)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Flag Rarity vs Signal Strength (Top {top_n})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Signal Strength")
    plt.tight_layout()
    return fig


def create_family_analysis(
    family_stats: dict[str, dict[str, Any]]
) -> plt.Figure:
    """Create grouped bar chart comparing flag families."""
    # Filter out families with no data (both attack and benign counts are 0)
    families_with_data = [
        f
        for f in family_stats.keys()
        if family_stats[f]["attack_count"] > 0 or family_stats[f]["benign_count"] > 0
    ]
    
    if not families_with_data:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No flag family data available", 
                ha="center", va="center", fontsize=14)
        ax.set_title("Flag Family Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
    
    families = families_with_data
    attack_rates = [family_stats[f]["attack_presence_rate"] for f in families]
    benign_rates = [family_stats[f]["benign_presence_rate"] for f in families]
    signal_strengths = [family_stats[f]["signal_strength"] for f in families]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Presence rates
    x = np.arange(len(families))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        attack_rates,
        width,
        label="Attack",
        color=COLOR_ATTACK,
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2,
        benign_rates,
        width,
        label="Benign",
        color=COLOR_BENIGN,
        alpha=0.8,
    )

    ax1.set_xlabel("Flag Family", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Presence Rate", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Flag Family Presence Rates",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=45, ha="right")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Signal strength
    colors = [COLOR_SIGNAL if s > 0 else COLOR_WARNING for s in signal_strengths]
    bars = ax2.bar(families, signal_strengths, color=colors, alpha=0.8)

    ax2.set_xlabel("Flag Family", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Signal Strength", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Flag Family Signal Strength",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xticklabels(families, rotation=45, ha="right")
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, strength in zip(bars, signal_strengths):
        if abs(strength) > 0.01:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                strength,
                f"{strength:.2%}",
                ha="center",
                va="bottom" if strength > 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    return fig


def create_frequency_distribution_boxplot(
    distributions: dict[str, dict[str, float | int]], top_n: int = 15
) -> plt.Figure:
    """Create box plot showing frequency distributions for top flags."""
    # Sort by total occurrences
    sorted_flags = sorted(
        distributions.items(),
        key=lambda x: x[1]["total_occurrences"],
        reverse=True,
    )[:top_n]

    flags = [f[0] for f in sorted_flags]
    means = [f[1]["mean"] for f in sorted_flags]
    p25s = [f[1]["p25"] for f in sorted_flags]
    p75s = [f[1]["p75"] for f in sorted_flags]
    medians = [f[1]["median"] for f in sorted_flags]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create box plot data
    positions = np.arange(len(flags))
    box_data = []
    for flag, stats in sorted_flags:
        # Create synthetic data from percentiles
        box_data.append([stats["p25"], stats["median"], stats["p75"]])

    bp = ax.boxplot(
        [[stats["p25"], stats["median"], stats["p75"]] for _, stats in sorted_flags],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )

    # Color boxes
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_BENIGN)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(flags, rotation=45, ha="right")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Flag Frequency Distributions (Top {top_n})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def create_sequence_analysis(
    sequence_stats: list[tuple[tuple[str, ...], int, float]], top_n: int = 15
) -> plt.Figure:
    """Create visualization of top flag sequences."""
    top_sequences = sequence_stats[:top_n]

    sequences_str = [
        " → ".join(seq[:5]) + ("..." if len(seq) > 5 else "")
        for seq, _, _ in top_sequences
    ]
    counts = [count for _, count, _ in top_sequences]
    attack_rates = [rate for _, _, rate in top_sequences]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Sequence frequency
    bars1 = ax1.barh(sequences_str, counts, color=COLOR_BENIGN, alpha=0.8)
    ax1.set_xlabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Flag Sequence", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Top {top_n} Flag Sequences by Frequency",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(True, alpha=0.3, axis="x")

    # Plot 2: Sequence attack rate
    colors = [COLOR_ATTACK if rate > 0.5 else COLOR_BENIGN for rate in attack_rates]
    bars2 = ax2.barh(sequences_str, attack_rates, color=colors, alpha=0.8)
    ax2.set_xlabel("Attack Rate", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Flag Sequence", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Top {top_n} Flag Sequences by Attack Rate",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlim(0, 1)
    ax2.axvline(x=0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, rate in zip(bars2, attack_rates):
        ax2.text(
            rate,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.2%}",
            va="center",
            ha="left" if rate > 0.5 else "right",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig

