#!/usr/bin/env python3
"""Analyze training score distributions to determine optimal contamination."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

# Paths
BASE_DIR = Path("experiments/preprocessing_impact")

print("=" * 70)
print("TRAINING SCORE DISTRIBUTION ANALYSIS")
print("=" * 70)
print()

# Load models
print("Loading models...")
without_prep_data = joblib.load(BASE_DIR / "without_preprocessing" / "model.joblib")
with_prep_data = joblib.load(BASE_DIR / "with_preprocessing" / "model.joblib")

without_prep_model = without_prep_data["model"]
with_prep_model = with_prep_data["model"]

# Load embeddings
print("Loading embeddings...")
without_prep_embeddings = np.load(
    BASE_DIR / "without_preprocessing" / "embeddings.npz"
)["embeddings"]
with_prep_embeddings = np.load(BASE_DIR / "with_preprocessing" / "embeddings.npz")[
    "embeddings"
]

print(
    f"WITHOUT preprocessing: {without_prep_embeddings.shape[0]} samples, {without_prep_embeddings.shape[1]} features"
)
print(
    f"WITH preprocessing:    {with_prep_embeddings.shape[0]} samples, {with_prep_embeddings.shape[1]} features"
)
print()

# Compute scores on training data
print("Computing anomaly scores on training data...")
without_prep_scores = without_prep_model.score_samples(without_prep_embeddings)
with_prep_scores = with_prep_model.score_samples(with_prep_embeddings)

print("Done!")
print()


# Analyze distributions
def analyze_distribution(scores, name):
    """Analyze and display statistics for a score distribution."""
    print(f"--- {name} ---")
    print(f"Min:        {np.min(scores):.6f}")
    print(f"Max:        {np.max(scores):.6f}")
    print(f"Mean:       {np.mean(scores):.6f}")
    print(f"Median:     {np.median(scores):.6f}")
    print(f"Std Dev:    {np.std(scores):.6f}")
    print()
    print("Percentiles:")
    percentiles = [1, 5, 10, 15, 20, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(scores, p)
        print(f"  {p:3d}th percentile: {value:.6f}")
    print()


print("=" * 70)
print("WITHOUT PREPROCESSING - Training Scores")
print("=" * 70)
analyze_distribution(without_prep_scores, "Raw Requests")

print("=" * 70)
print("WITH PREPROCESSING - Training Scores")
print("=" * 70)
analyze_distribution(with_prep_scores, "Preprocessed Requests")

# Create visualization
print("Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# WITHOUT preprocessing - histogram
axes[0, 0].hist(
    without_prep_scores, bins=100, alpha=0.7, color="blue", edgecolor="black"
)
axes[0, 0].axvline(
    without_prep_model.offset_,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Current offset (contam=0.1): {without_prep_model.offset_:.4f}",
)
axes[0, 0].set_title(
    "WITHOUT Preprocessing: Training Score Distribution", fontsize=12, fontweight="bold"
)
axes[0, 0].set_xlabel("Anomaly Score")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# WITHOUT preprocessing - percentile lines
axes[0, 1].hist(
    without_prep_scores, bins=100, alpha=0.7, color="blue", edgecolor="black"
)
for contam in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
    threshold = np.percentile(without_prep_scores, contam * 100)
    axes[0, 1].axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"contam={contam:.2f}: {threshold:.4f}",
    )
axes[0, 1].set_title(
    "WITHOUT Preprocessing: Contamination Options", fontsize=12, fontweight="bold"
)
axes[0, 1].set_xlabel("Anomaly Score")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# WITH preprocessing - histogram
axes[1, 0].hist(
    with_prep_scores, bins=100, alpha=0.7, color="purple", edgecolor="black"
)
axes[1, 0].axvline(
    with_prep_model.offset_,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Current offset (contam=0.1): {with_prep_model.offset_:.4f}",
)
axes[1, 0].set_title(
    "WITH Preprocessing: Training Score Distribution", fontsize=12, fontweight="bold"
)
axes[1, 0].set_xlabel("Anomaly Score")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# WITH preprocessing - percentile lines
axes[1, 1].hist(
    with_prep_scores, bins=100, alpha=0.7, color="purple", edgecolor="black"
)
for contam in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
    threshold = np.percentile(with_prep_scores, contam * 100)
    axes[1, 1].axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"contam={contam:.2f}: {threshold:.4f}",
    )
axes[1, 1].set_title(
    "WITH Preprocessing: Contamination Options", fontsize=12, fontweight="bold"
)
axes[1, 1].set_xlabel("Anomaly Score")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
output_path = BASE_DIR / "training_score_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to: {output_path}")
print()

# Recommendations
print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print()
print("The contamination parameter sets the offset at the Xth percentile of")
print("training scores. Lower scores are more anomalous.")
print()
print("Common contamination values:")
print("  - 0.01 (1%):  Very conservative, only flag extreme outliers")
print("  - 0.05 (5%):  Conservative, low false positive rate")
print("  - 0.10 (10%): Balanced approach (current setting)")
print("  - 0.15 (15%): More aggressive detection")
print("  - 0.20 (20%): High sensitivity, may have more false positives")
print()
print("Since we're training ONLY on normal traffic, the contamination parameter")
print("represents our assumption about how many 'outliers' exist in normal traffic")
print("due to natural variation.")
print()
print("NEXT STEP: Test both models with the SAME contamination value on the test")
print("set to fairly compare if preprocessing helps with attack detection.")
print("=" * 70)
