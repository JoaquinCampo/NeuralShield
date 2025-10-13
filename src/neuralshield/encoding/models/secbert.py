from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import register_encoder


@register_encoder("secbert")
class SecBERTEncoder(RequestEncoder):
    """Wrap HuggingFace SecBERT model for request encoding.

    SecBERT is a BERT model pretrained on cybersecurity text from APTnotes,
    Stucco-Data, CASIE, and SemEval-2018 Task 8. It has a custom vocabulary
    optimized for security-related terms.

    Model: jackaduma/SecBERT
    Dimensions: 768
    Max tokens: 512
    """

    def __init__(self, *, model_name: str = "default", device: str = "cpu") -> None:
        super().__init__(model_name=model_name, device=device)

        # Use default SecBERT model
        model_id = "jackaduma/SecBERT" if model_name == "default" else model_name

        logger.info(f"Loading SecBERT tokenizer from {model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(f"Loading SecBERT model from {model_id}")
        self._model = AutoModel.from_pretrained(model_id)

        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            f"Initialized SecBERT encoder model={model_id} device={device} "
            f"embedding_dim=768"
        )

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        """Encode a batch of HTTP requests using SecBERT.

        Args:
            batch: Sequence of HTTP request strings

        Returns:
            NumPy array of shape (batch_size, 768) with embeddings
        """
        if not batch:
            return np.empty((0, 768), dtype=np.float32)

        logger.debug(
            f"Encoding batch size={len(batch)} with SecBERT model={self.model_name}"
        )

        encoded = self._tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move tensors to device
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)

            # Use [CLS] token embedding (first token)
            # Shape: (batch_size, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings = cls_embeddings.cpu().numpy().astype(np.float32)

        return embeddings

    def shutdown(self) -> None:
        """Clean up model resources."""
        logger.debug("Shutting down SecBERT encoder")
        # Move model to CPU and clear cache if using CUDA
        if self._device.type == "cuda":
            self._model.cpu()
            torch.cuda.empty_cache()
