from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

app = typer.Typer(help="Entrena un baseline supervisado (logistic regression) sobre SRBH.")


def load_samples(
    path: Path,
    *,
    label_filter: str | None,
    limit: int,
    apply_preprocess: bool,
) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            label = obj["label"]
            if label_filter and label != label_filter:
                continue
            try:
                text = preprocess(obj["request"]) if apply_preprocess else obj["request"]
            except MalformedHttpRequestError:
                continue
            samples.append((text, 1 if label == "attack" else 0))
            if len(samples) >= limit:
                break
    return samples


@app.command()
def main(
    train_path: Path = typer.Option(
        Path("src/neuralshield/data/SR_BH_2020/train.jsonl"),
        help="Split normal (usado para muestrear válidos).",
    ),
    test_path: Path = typer.Option(
        Path("src/neuralshield/data/SR_BH_2020/test.jsonl"),
        help="Split con ataques (muestrear válidos y ataques).",
    ),
    valid_samples: int = typer.Option(40000, help="Cantidad de muestras válidas."),
    attack_samples: int = typer.Option(40000, help="Cantidad de muestras de ataque."),
    preprocess_data: bool = typer.Option(
        True,
        "--preprocess/--no-preprocess",
        help="Aplicar el pipeline HTTP antes de TF-IDF.",
    ),
    max_features: int = typer.Option(5000, help="Máximo de features TF-IDF."),
    ngram_min: int = typer.Option(1, help="n-gram mínimo."),
    ngram_max: int = typer.Option(3, help="n-gram máximo."),
    test_size: float = typer.Option(0.3, help="Proporción para el split de validación."),
    random_seed: int = typer.Option(42, help="Semilla de aleatoriedad."),
) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)

    logger.info("Recopilando válidos desde entrenamiento ({})", valid_samples)
    valid_from_train = load_samples(
        train_path,
        label_filter="valid",
        limit=valid_samples // 2,
        apply_preprocess=preprocess_data,
    )
    valid_from_test = load_samples(
        test_path,
        label_filter="valid",
        limit=valid_samples - len(valid_from_train),
        apply_preprocess=preprocess_data,
    )

    logger.info("Recopilando ataques ({})", attack_samples)
    attacks = load_samples(
        test_path,
        label_filter="attack",
        limit=attack_samples,
        apply_preprocess=preprocess_data,
    )

    dataset = valid_from_train + valid_from_test + attacks
    random.shuffle(dataset)
    texts = [text for text, _ in dataset]
    labels = [label for _, label in dataset]
    logger.info("Dataset supervisado listo: %d muestras (positivas=%d)", len(dataset), sum(labels))

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=5,
    )
    logger.info("Entrenando TF-IDF supervisado")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    logger.info("Entrenando LogisticRegression")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_val_vec)
    report = classification_report(
        y_val,
        y_pred,
        target_names=["valid", "attack"],
        digits=4,
    )
    logger.info("Resultados supervisados:\n{}", report)


if __name__ == "__main__":
    app()
