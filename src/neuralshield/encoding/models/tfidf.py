from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer

from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import register_encoder


class TFIDFEncoderConfig(BaseModel):
    """Hyperparameters for the TF-IDF text encoder."""

    max_features: int | None = 5000
    min_df: int | float = 1
    max_df: int | float = 1.0
    ngram_range: tuple[int, int] = (1, 1)
    lowercase: bool = True
    stop_words: str | None = None
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False
    binary: bool = False

    model_config = ConfigDict(validate_assignment=True)


@register_encoder("tfidf")
class TFIDFEncoder(RequestEncoder):
    """Vectorize requests using scikit-learn's TF-IDF implementation."""

    def __init__(
        self,
        *,
        model_name: str = "default",
        device: str = "cpu",
        settings: TFIDFEncoderConfig = TFIDFEncoderConfig(),
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self._settings = settings
        self._vectorizer: TfidfVectorizer
        self._state_path: Path | None = None
        self._fitted = False

        if model_name != "default":
            candidate = Path(model_name)
            self._state_path = candidate
            if candidate.exists():
                loaded = joblib.load(candidate)
                self._vectorizer = loaded
                self._fitted = True
                logger.info(
                    "Loaded TF-IDF vectorizer state from {path}",
                    path=str(candidate),
                )
            else:
                raise FileNotFoundError(
                    f"TF-IDF state file '{candidate}' not found; provide a valid path"
                )

        else:
            self._vectorizer = self._build_vectorizer()

        if device != "cpu":
            logger.debug(
                "TF-IDF encoder ignores device setting; defaulting to CPU",
            )

    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self._settings.max_features,
            min_df=self._settings.min_df,
            max_df=self._settings.max_df,
            ngram_range=self._settings.ngram_range,
            lowercase=self._settings.lowercase,
            stop_words=self._settings.stop_words,
            use_idf=self._settings.use_idf,
            smooth_idf=self._settings.smooth_idf,
            sublinear_tf=self._settings.sublinear_tf,
            binary=self._settings.binary,
            dtype=np.float32,
        )

    def fit_transform(self, corpus: Sequence[str]) -> np.ndarray:
        if not corpus:
            return np.empty((0, 0), dtype=np.float32)
        matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True
        return matrix.toarray().astype(np.float32, copy=False)

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        if not batch:
            feature_count = len(self._vectorizer.vocabulary_) if self._fitted else 0
            return np.empty((0, feature_count), dtype=np.float32)

        if not self._fitted:
            matrix = self._vectorizer.fit_transform(batch)
            self._fitted = True
        else:
            matrix = self._vectorizer.transform(batch)

        return matrix.toarray().astype(np.float32, copy=False)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self._state_path
        if target is None:
            raise ValueError(
                "No destination provided for TF-IDF vectorizer state; supply 'path'"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, target)
        self._state_path = target
        logger.info("Saved TF-IDF vectorizer state to {path}", path=str(target))
        return target

    def is_fitted(self) -> bool:
        return self._fitted
