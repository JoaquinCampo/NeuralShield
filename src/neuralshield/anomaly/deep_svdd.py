from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuralshield.anomaly.base import AnomalyDetector
from neuralshield.anomaly.factory import register_detector


class DeepSVDDNetwork(nn.Module):
    """Neural network for Deep SVDD."""

    def __init__(
        self,
        input_dim: int,
        hidden_neurons: list[int],
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_neurons:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


@register_detector("deep_svdd")
class DeepSVDDDetector(AnomalyDetector):
    """Deep Support Vector Data Description with GPU support.

    PyTorch-based implementation with full CUDA acceleration.
    Learns a hypersphere around normal data in learned representation space.

    Hyperparameters:
        hidden_neurons: Network architecture (e.g., [256, 128])
            Default: [128, 64]
        epochs: Training epochs
            Default: 100
        batch_size: Training batch size (use large batches on A100!)
            Default: 256 (optimal for A100, use 64 for T4/MPS)
        learning_rate: Adam optimizer learning rate
            Default: 0.001
        weight_decay: L2 regularization
            Default: 1e-6
        dropout_rate: Dropout for regularization
            Default: 0.2
        nu: Upper bound on fraction of outliers (for threshold)
            Default: 0.1
        device: Device ("cuda", "cpu", "mps", or None for auto)
            Default: None (auto-detect)
        verbose: Training progress verbosity
            Default: 1

    Training time (A100 GPU, batch_size=256):
        - 47k samples, 768 dims: ~2-3 seconds per epoch
        - Full training (100 epochs): ~4-6 minutes
    """

    def __init__(
        self,
        *,
        name: str = "default",
        hidden_neurons: list[int] | None = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-6,
        dropout_rate: float = 0.2,
        nu: float = 0.1,
        device: str | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(name=name)
        self.hidden_neurons = hidden_neurons or [128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.nu = nu
        self.verbose = verbose

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.network: DeepSVDDNetwork | None = None
        self.center: torch.Tensor | None = None
        self._threshold: float | None = None
        self._radius: float = 0.0
        self.loss_history: list[float] = []  # Track training loss

    def _init_center(self, data_loader: DataLoader) -> torch.Tensor:
        """Initialize hypersphere center as mean of network outputs using streaming computation."""
        assert self.network is not None, (
            "Network must be initialized before computing center"
        )

        self.network.eval()
        total = None
        n_samples = 0

        with torch.no_grad():
            for batch_x in data_loader:
                if isinstance(batch_x, list):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                outputs = self.network(batch_x)

                # Streaming mean computation
                if total is None:
                    total = outputs.sum(dim=0)
                else:
                    total += outputs.sum(dim=0)
                n_samples += len(outputs)

        assert total is not None, "No data processed"
        center = total / n_samples
        return center

    def fit(self, embeddings: NDArray[np.float32]) -> None:
        """Fit Deep SVDD on normal training embeddings."""
        n_samples, n_features = embeddings.shape

        logger.info(
            "Fitting Deep SVDD (PyTorch)",
            n_samples=n_samples,
            n_features=n_features,
            device=str(self.device),
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

        # Create network
        self.network = DeepSVDDNetwork(
            input_dim=n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Prepare data
        tensor_data = torch.FloatTensor(embeddings)
        dataset = TensorDataset(tensor_data)

        # Use multiple workers for data loading (except on MPS which has issues)
        num_workers = 0 if self.device.type == "mps" else 2
        pin_memory = self.device.type == "cuda"

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Initialize center
        self.center = self._init_center(data_loader)

        # Optimizer
        assert self.network is not None and self.center is not None
        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training loop
        self.network.train()
        for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch"):
            total_loss = 0.0
            n_batches = 0

            for batch_x in data_loader:
                if isinstance(batch_x, list):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)

                # Forward pass
                outputs = self.network(batch_x)

                # Deep SVDD loss: mean squared distance from center
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                loss = torch.mean(dist)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            self.loss_history.append(avg_loss)

            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        # Compute radius as (100 - nu) percentile of distances
        self.network.eval()
        distances = []
        with torch.no_grad():
            for batch_x in data_loader:
                if isinstance(batch_x, list):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                outputs = self.network(batch_x)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                distances.append(dist.cpu())

        distances_np = torch.cat(distances).numpy()
        self._radius = float(np.percentile(distances_np, 100 * (1 - self.nu)))

        self._fitted = True
        logger.info(f"Deep SVDD fitting complete (radius: {self._radius:.4f})")

    def scores(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores (distances from center)."""
        if not self.is_fitted or self.network is None or self.center is None:
            raise RuntimeError("Detector has not been fitted yet")

        self.network.eval()
        tensor_data = torch.FloatTensor(embeddings).to(self.device)

        scores_list = []
        with torch.no_grad():
            # Process in batches to avoid OOM
            for i in range(0, len(tensor_data), self.batch_size):
                batch = tensor_data[i : i + self.batch_size]
                outputs = self.network(batch)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                scores_list.append(dist.cpu())

        scores = torch.cat(scores_list).numpy()
        return scores.astype(np.float32)

    def set_threshold(
        self,
        normal_embeddings: NDArray[np.float32],
        max_fpr: float = 0.05,
    ) -> float:
        """Set threshold based on desired FPR."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before setting threshold")

        scores = self.scores(normal_embeddings)
        threshold = float(np.percentile(scores, (1 - max_fpr) * 100))
        self._threshold = threshold

        actual_fpr = np.mean(scores > threshold)
        logger.info(
            f"Threshold set to {threshold:.4f} "
            f"(target FPR={max_fpr:.1%}, actual={actual_fpr:.1%})"
        )

        return threshold

    def predict(
        self,
        embeddings: NDArray[np.float32],
        *,
        threshold: float | None = None,
    ) -> NDArray[np.bool_]:
        """Predict anomalies."""
        scores = self.scores(embeddings)

        limit = threshold if threshold is not None else self._threshold
        if limit is None:
            raise RuntimeError(
                "No threshold set. Call set_threshold() or provide threshold argument."
            )

        return (scores > limit).astype(bool)

    def save(self, path: str | Path) -> None:
        """Save trained model."""
        if not self.is_fitted or self.network is None or self.center is None:
            raise RuntimeError("Detector has not been fitted yet")

        payload = {
            "name": self.name,
            "hidden_neurons": self.hidden_neurons,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "nu": self.nu,
            "device": str(self.device),
            "verbose": self.verbose,
            "network_state": self.network.state_dict(),
            "center": self.center.cpu().numpy(),
            "threshold": self._threshold,
            "radius": self._radius,
            "loss_history": self.loss_history,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved Deep SVDD model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DeepSVDDDetector":
        """Load trained model."""
        payload = joblib.load(path)

        detector = cls(
            name=payload.get("name", "default"),
            hidden_neurons=payload.get("hidden_neurons", [128, 64]),
            epochs=int(payload.get("epochs", 100)),
            batch_size=int(payload.get("batch_size", 64)),
            learning_rate=float(payload.get("learning_rate", 0.001)),
            weight_decay=float(payload.get("weight_decay", 1e-6)),
            dropout_rate=float(payload.get("dropout_rate", 0.2)),
            nu=float(payload.get("nu", 0.1)),
            device=payload.get("device", "cpu"),
            verbose=int(payload.get("verbose", 1)),
        )

        # Reconstruct network
        center_np = payload["center"]
        input_dim = center_np.shape[0]

        # Infer input_dim from center or network state
        network_state = payload["network_state"]
        first_layer_key = [k for k in network_state.keys() if "0.weight" in k][0]
        input_dim = network_state[first_layer_key].shape[1]

        detector.network = DeepSVDDNetwork(
            input_dim=input_dim,
            hidden_neurons=detector.hidden_neurons,
            dropout_rate=detector.dropout_rate,
        ).to(detector.device)

        assert detector.network is not None
        detector.network.load_state_dict(network_state)
        detector.center = torch.FloatTensor(center_np).to(detector.device)
        detector._threshold = payload.get("threshold")
        detector._radius = float(payload.get("radius", 0.0))
        detector._fitted = True
        detector.loss_history = payload.get("loss_history", [])

        logger.info(f"Loaded Deep SVDD model from {path}")
        return detector
