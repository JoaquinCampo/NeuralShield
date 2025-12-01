from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from neuralshield.anomaly import MahalanobisDetector

app = typer.Typer(
    help="Evaluate Mahalanobis performance on original vs. compacted embeddings."
)


NORMAL_LABELS = {"valid", "normal"}
ATTACK_LABELS = {"attack", "anomaly"}


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return data["embeddings"].astype(np.float32, copy=False), data["labels"]


def evaluate_split(
    *,
    name: str,
    train_path: Path,
    test_path: Path,
) -> dict[str, float]:
    train_embeddings, train_labels = load_npz(train_path)
    test_embeddings, test_labels = load_npz(test_path)

    mask = np.isin(train_labels, list(NORMAL_LABELS))
    normal_embeddings = train_embeddings[mask]
    if normal_embeddings.size == 0:
        raise ValueError(f"No normal samples found in {train_path}.")

    detector = MahalanobisDetector()
    detector.fit(normal_embeddings)
    scores = detector.scores(test_embeddings)

    target = np.array(
        [1 if label in ATTACK_LABELS else 0 for label in test_labels],
        dtype=np.int32,
    )

    roc_auc = float(roc_auc_score(target, scores))
    average_precision = float(average_precision_score(target, scores))
    mean_attack = float(scores[target == 1].mean()) if np.any(target == 1) else 0.0
    mean_valid = float(scores[target == 0].mean()) if np.any(target == 0) else 0.0

    logger.info(
        "Evaluated {name} AUC={auc:.4f} AP={ap:.4f} attack={ma:.4f} valid={mv:.4f}",
        name=name,
        auc=roc_auc,
        ap=average_precision,
        ma=mean_attack,
        mv=mean_valid,
    )

    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "mean_attack_score": mean_attack,
        "mean_valid_score": mean_valid,
        "score_gap": mean_attack - mean_valid,
    }


@app.command()
def main(
    train_path: Path = typer.Option(
        Path("embeddings/SecBert/train_embeddings.npz"),
        help="Baseline training embeddings (.npz).",
    ),
    test_path: Path = typer.Option(
        Path("embeddings/SecBert/test_embeddings.npz"),
        help="Baseline test embeddings (.npz).",
    ),
    compact_train_path: Path | None = typer.Option(
        None,
        help="Optional compacted training embeddings (.npz).",
    ),
    compact_test_path: Path | None = typer.Option(
        None,
        help="Optional compacted test embeddings (.npz).",
    ),
    output: Path | None = typer.Option(
        None,
        help="Optional JSON file to persist results.",
    ),
) -> None:
    results: dict[str, dict[str, float]] = {}

    results["baseline"] = evaluate_split(
        name="baseline",
        train_path=train_path,
        test_path=test_path,
    )

    if compact_train_path is not None and compact_test_path is not None:
        results["compact"] = evaluate_split(
            name="compact",
            train_path=compact_train_path,
            test_path=compact_test_path,
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved results to {path}", path=str(output))

    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
