from __future__ import annotations

from pathlib import Path
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

    def __init__(
        self,
        *,
        model_name: str = "jackaduma/SecBERT",
        device: str = "cpu",
    ) -> None:
        super().__init__(model_name=model_name, device=device)

        # Use default SecBERT model
        model_id = "jackaduma/SecBERT"

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


@register_encoder("secbert-adapted")
class SecBERTAdaptedEncoder(RequestEncoder):
    """SecBERT model domain-adapted via MLM on HTTP requests.

    This encoder loads a SecBERT model that has been continued pre-training
    on valid HTTP traffic using Masked Language Modeling. The adapted model
    should better capture HTTP-specific syntax and patterns.

    Uses mean+max pooling over the last two hidden layers for optimal
    unsupervised anomaly detection:
    - Mean pooling captures the "style" of normal traffic
    - Max pooling catches rare/spiky tokens (weird encodings, dup headers)
    - Averaging layers -1 and -2 provides stability without labels

    Default path: src/neuralshield/finetuning/models/secbert-http-adapted/final
    Dimensions: 1536 (768 mean + 768 max)
    Max tokens: 512
    """

    def __init__(
        self,
        *,
        model_name: str = "secbert-adapted",
        device: str = "cpu",
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)

        # Default to the trained model path
        if model_path is None:
            model_path = Path(
                "src/neuralshield/finetuning/models/secbert-http-adapted/final"
            )
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise ValueError(f"Adapted model not found at {model_path}")

        logger.info(f"Loading adapted SecBERT tokenizer from {model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        logger.info(f"Loading adapted SecBERT model from {model_path}")
        self._model = AutoModel.from_pretrained(str(model_path))

        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

        logger.info(
            f"Initialized adapted SecBERT encoder model={model_path} device={device} "
            f"embedding_dim=768"
        )

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        """Encode a batch of HTTP requests using adapted SecBERT.

        Uses mean+max pooling over the last two hidden layers for better
        unsupervised anomaly detection. Mean captures the "style" of normal
        traffic, max catches rare/spiky tokens (weird encodings, dup headers).

        Args:
            batch: Sequence of HTTP request strings

        Returns:
            NumPy array of shape (batch_size, 1536) with embeddings
            (768 for mean + 768 for max)
        """
        if not batch:
            return np.empty((0, 1536), dtype=np.float32)

        logger.debug(
            f"Encoding batch size={len(batch)} with adapted SecBERT model={self.model_name}"
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
            outputs = self._model(**encoded, output_hidden_states=True)

            # Average last two hidden layers for stability
            # Shape: (batch_size, seq_len, 768)
            last_hidden = outputs.hidden_states[-1]
            second_last_hidden = outputs.hidden_states[-2]
            H = (last_hidden + second_last_hidden) / 2.0

            # Create mask: exclude special tokens ([CLS], [SEP], [PAD])
            attention_mask = encoded["attention_mask"]  # (batch_size, seq_len)
            input_ids = encoded["input_ids"]

            # Get special token IDs
            cls_token_id = self._tokenizer.cls_token_id
            sep_token_id = self._tokenizer.sep_token_id
            pad_token_id = self._tokenizer.pad_token_id

            # Create mask: 1 for content tokens, 0 for special/pad
            special_tokens_mask = (
                (input_ids == cls_token_id)
                | (input_ids == sep_token_id)
                | (input_ids == pad_token_id)
            )
            content_mask = attention_mask.bool() & ~special_tokens_mask
            content_mask = content_mask.unsqueeze(
                -1
            ).float()  # (batch_size, seq_len, 1)

            # Mean pooling: average over content tokens
            masked_H = H * content_mask
            sum_mask = content_mask.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
            mean_embedding = masked_H.sum(dim=1) / sum_mask  # (batch_size, 768)

            # Max pooling: max over content tokens (set padded to -inf)
            masked_H_max = H.masked_fill(content_mask == 0, float("-inf"))
            max_embedding = masked_H_max.max(dim=1)[0]  # (batch_size, 768)

            # Concatenate mean and max
            # Shape: (batch_size, 1536)
            combined_embedding = torch.cat([mean_embedding, max_embedding], dim=1)

        embeddings = combined_embedding.cpu().numpy().astype(np.float32)

        return embeddings

    def shutdown(self) -> None:
        """Clean up model resources."""
        logger.debug("Shutting down adapted SecBERT encoder")
        # Move model to CPU and clear cache if using CUDA
        if self._device.type == "cuda":
            self._model.cpu()
            torch.cuda.empty_cache()
