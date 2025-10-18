from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from loguru import logger

from neuralshield.encoding.models.factory import register_encoder
from neuralshield.encoding.models.secbert import SecBERTEncoder


@register_encoder("secbert-flag-weighted")
class SecBERTFlagWeightedEncoder(SecBERTEncoder):
    """SecBERT encoder that applies weighted pooling using Tier 1 flag tokens."""

    def __init__(
        self,
        *,
        model_name: str = "jackaduma/SecBERT",
        device: str = "cpu",
        token_weight_path: str | Path = Path(
            "src/neuralshield/encoding/data/secbert_flag_token_weights.json"
        ),
    ) -> None:
        super().__init__(model_name=model_name, device=device)

        weight_path = Path(token_weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Token weight file not found: {weight_path}")

        raw = json.loads(weight_path.read_text(encoding="utf-8"))
        self._token_weights: dict[str, float] = {
            token: float(weight) for token, weight in raw.items()
        }

        vocab_size = len(self._tokenizer)
        weight_tensor = torch.ones(vocab_size, dtype=torch.float32, device=self._device)

        unk_token_id = getattr(self._tokenizer, "unk_token_id", None)
        unk_token = getattr(self._tokenizer, "unk_token", None)
        missing_tokens: list[str] = []

        for token, weight in self._token_weights.items():
            token_id = self._tokenizer.convert_tokens_to_ids(token)
            if not isinstance(token_id, int):
                missing_tokens.append(token)
                continue
            if (
                unk_token_id is not None
                and token_id == unk_token_id
                and (unk_token is None or token != unk_token)
            ):
                missing_tokens.append(token)
                continue
            weight_tensor[token_id] = float(weight)

        mapped_count = int(weight_tensor.ne(1.0).sum().item())

        if missing_tokens:
            logger.warning(
                "Skipped {count} token weights not found in tokenizer vocabulary",
                count=len(missing_tokens),
            )

        self._token_weight_tensor = weight_tensor

        logger.info(
            "Loaded {count} token weights for flag-weighted pooling "
            "(mapped to {mapped} vocabulary entries)",
            count=len(self._token_weights),
            mapped=mapped_count,
        )

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        if not batch:
            return np.empty((0, 768), dtype=np.float32)

        logger.debug(
            "Encoding batch size={size} with SecBERT flag-weighted pooling",
            size=len(batch),
        )

        encoded = self._tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        attention_mask = encoded["attention_mask"].float()
        input_ids = encoded["input_ids"]

        token_weights = self._token_weight_tensor[input_ids].to(hidden_states.dtype)
        weights = token_weights * attention_mask

        weighted_hidden = hidden_states * weights.unsqueeze(-1)
        weight_sum = weights.sum(dim=1, keepdim=True)

        pooled = weighted_hidden.sum(dim=1) / weight_sum.clamp_min(1e-6)

        zero_mask = weight_sum.squeeze(1) <= 1e-6
        if zero_mask.any():
            pooled[zero_mask] = hidden_states[zero_mask, 0]

        return pooled.cpu().numpy().astype(np.float32)
