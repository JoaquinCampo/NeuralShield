from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer, T5EncoderModel

from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import register_encoder


@register_encoder("byt5")
class ByT5Encoder(RequestEncoder):
    """Wrap ByT5 encoder for byte-level HTTP request encoding.

    ByT5 uses byte-level tokenization, making it ideal for HTTP anomaly detection:
    - Preserves single-character edits (%27 vs %28)
    - Captures unicode tricks, null bytes, CRLF injection
    - No preprocessing needed - raw strings work best

    Architecture:
    - Input: Raw HTTP request string (no preprocessing!)
    - ByT5 encoder: Byte-level tokens → hidden states
    - Pooling: Mean + Max concatenation
    - Output: L2-normalized fixed-size vector

    Default model: google/byt5-small
    Output dimensions: 2 × hidden_size (mean+max pooling)
    - byt5-small: 1472 × 2 = 2944 dims
    """

    def __init__(
        self,
        *,
        model_name: str = "default",
        device: str = "cpu",
        max_length: int = 1024,
    ) -> None:
        super().__init__(model_name=model_name, device=device)

        # Use default ByT5 model
        model_id = "google/byt5-small" if model_name == "default" else model_name

        logger.info(f"Loading ByT5 model from {model_id} on device={device}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = T5EncoderModel.from_pretrained(model_id)

        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        self._max_length = max_length

        # Determine output dimension (2x hidden size for mean+max pooling)
        self._hidden_size = self._model.config.d_model
        self._output_dim = self._hidden_size * 2

        logger.info(
            f"Initialized ByT5 encoder model={model_id} device={device} "
            f"hidden_size={self._hidden_size} output_dim={self._output_dim} "
            f"max_length={max_length}"
        )

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        """Encode a batch of HTTP requests using ByT5 + mean+max pooling.

        Args:
            batch: Sequence of HTTP request strings (raw, no preprocessing!)

        Returns:
            NumPy array of shape (batch_size, output_dim) with L2-normalized embeddings
        """
        if not batch:
            return np.empty((0, self._output_dim), dtype=np.float32)

        logger.debug(
            f"Encoding batch size={len(batch)} with ByT5 "
            f"model={self.model_name} output_dim={self._output_dim}"
        )

        # Tokenize (byte-level, preserves all characters)
        encoded = self._tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        # Move to device
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            # Get hidden states from encoder
            outputs = self._model(**encoded)
            hidden_states = (
                outputs.last_hidden_state
            )  # (batch_size, seq_len, hidden_size)

            # Mean pooling (global context)
            mean_pool = hidden_states.mean(dim=1)  # (batch_size, hidden_size)

            # Max pooling (captures rare spikes/anomalies)
            max_pool = hidden_states.max(dim=1)[0]  # (batch_size, hidden_size)

            # Concatenate mean and max
            combined = torch.cat(
                [mean_pool, max_pool], dim=1
            )  # (batch_size, 2*hidden_size)

            # L2 normalize (sensitive to subtle differences)
            normalized = F.normalize(combined, p=2, dim=1)

        # Convert to numpy
        embeddings = normalized.cpu().numpy().astype(np.float32)

        return embeddings

    def shutdown(self) -> None:
        """Clean up model resources."""
        logger.debug("Shutting down ByT5 encoder")
        if self._device.type == "cuda":
            self._model.cpu()
            torch.cuda.empty_cache()
