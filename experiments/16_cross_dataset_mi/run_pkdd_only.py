#!/usr/bin/env python3
"""Run Experiment 16 (cross-dataset MI) but only the SR-BH → PKDD direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from test_cross_dataset_mi import (
    load_pkdd_data,
    load_srbh_data,
    run_experiment,
)


def main() -> None:
    results_dir = Path("experiments/16_cross_dataset_mi/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SR-BH and PKDD datasets…")
    (
        srbh_all_normal,
        srbh_all_attacks,
        srbh_train_normal,
        srbh_test_requests,
        srbh_test_labels,
    ) = load_srbh_data()

    (
        pkdd_all_normal,
        pkdd_all_attacks,
        pkdd_train_normal,
        pkdd_test_requests,
        pkdd_test_labels,
    ) = load_pkdd_data()

    logger.info("Running SR-BH → PKDD cross-dataset MI experiment only…")
    run4_results = run_experiment(
        name="SR-BH → PKDD",
        mi_normal=srbh_all_normal,
        mi_attacks=srbh_all_attacks,
        train_normal=pkdd_train_normal,
        test_requests=pkdd_test_requests,
        test_labels=pkdd_test_labels,
        output_dir=results_dir / "run4_srbh_to_pkdd",
        strip_headers_for_mi=False,
    )

    best = max(run4_results, key=lambda r: r["recall"])
    logger.info(
        "Best run: k=%d recall=%.4f FPR=%.4f precision=%.4f",
        best["k"],
        best["recall"],
        best["fpr"],
        best["precision"],
    )


if __name__ == "__main__":
    main()
