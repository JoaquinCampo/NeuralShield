from __future__ import annotations

from typing import Sequence

import numpy as np
from fastembed import LateInteractionTextEmbedding
from fastembed.postprocess import Muvera
from loguru import logger

from neuralshield.encoding.models.base import RequestEncoder
from neuralshield.encoding.models.factory import register_encoder


@register_encoder("colbert-muvera")
class ColBERTMuveraEncoder(RequestEncoder):
    """Wrap ColBERT with MUVERA postprocessing for request encoding.

    ColBERT produces multi-vector representations (one vector per token).
    MUVERA (Multi-Vector Representation Aggregation) converts these into
    a single fixed-size vector suitable for anomaly detection.

    This approach combines:
    - ColBERT's rich multi-vector token-level representations
    - MUVERA's learned compression to fixed-size vectors

    Default model: colbert-ir/colbertv2.0
    MUVERA output dimensions: r_reps * 2^k_sim * dim_proj
    Default: 20 * 2^5 * 16 = 10,240 dimensions
    """

    def __init__(
        self,
        *,
        model_name: str = "default",
        device: str = "cpu",
        k_sim: int = 5,  # MUVERA: number of clusters (2^k_sim)
        dim_proj: int = 16,  # MUVERA: projection dimension
        r_reps: int = 20,  # MUVERA: number of repetitions
    ) -> None:
        super().__init__(model_name=model_name, device=device)

        # Use default ColBERT model
        model_id = "colbert-ir/colbertv2.0" if model_name == "default" else model_name

        logger.info(f"Loading ColBERT model from {model_id} on device={device}")

        # Set up ONNX providers based on device
        # Check available providers first
        from onnxruntime import get_available_providers

        available = get_available_providers()

        if device == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CUDA acceleration")
        elif device == "mps" and "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            logger.info("Using CoreML acceleration")
        else:
            providers = ["CPUExecutionProvider"]
            if device != "cpu":
                logger.warning(
                    f"Requested device={device} but provider not available. "
                    f"Available: {available}. Falling back to CPU. "
                    f"Install onnxruntime-gpu for CUDA support."
                )

        self._embedder = LateInteractionTextEmbedding(
            model_name=model_id, providers=providers
        )

        logger.info(
            f"Initializing MUVERA postprocessor: k_sim={k_sim}, "
            f"dim_proj={dim_proj}, r_reps={r_reps}"
        )
        self._muvera = Muvera.from_multivector_model(
            model=self._embedder,
            k_sim=k_sim,
            dim_proj=dim_proj,
            r_reps=r_reps,
        )

        # Calculate MUVERA output dimension
        self._output_dim = r_reps * (2**k_sim) * dim_proj

        logger.info(
            f"Initialized ColBERT+MUVERA encoder model={model_id} device={device} "
            f"output_dim={self._output_dim}"
        )

    def encode(self, batch: Sequence[str]) -> np.ndarray:
        """Encode a batch of HTTP requests using ColBERT + MUVERA.

        Args:
            batch: Sequence of HTTP request strings

        Returns:
            NumPy array of shape (batch_size, output_dim) with embeddings
        """
        if not batch:
            return np.empty((0, self._output_dim), dtype=np.float32)

        logger.debug(
            f"Encoding batch size={len(batch)} with ColBERT+MUVERA "
            f"model={self.model_name} output_dim={self._output_dim}"
        )

        # Step 1: Get ColBERT multi-vector embeddings (one vector per token)
        multi_vecs = list(self._embedder.embed(batch))

        # Step 2: Apply MUVERA to convert multi-vectors to single vectors
        # Use process_document() for document-side encoding (asymmetric)
        embeddings_list = []
        for multi_vec in multi_vecs:
            fde = self._muvera.process_document(multi_vec)
            embeddings_list.append(fde)

        embeddings = np.vstack(embeddings_list).astype(np.float32)

        return embeddings

    def shutdown(self) -> None:
        """Clean up model resources."""
        logger.debug("Shutting down ColBERT+MUVERA encoder")
