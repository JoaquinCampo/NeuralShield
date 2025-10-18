# CSIC Flag-Weighted Embeddings

Generate CSIC train/test embeddings using the `secbert-flag-weighted` encoder so we can evaluate tier-1 flag pooling downstream.

## Colab Setup (A100 GPU)

1. Enable GPU: `Runtime → Change runtime type → GPU (A100)`.
2. Clone the repo and install dependencies:

```bash
!git clone https://github.com/<your-org>/neuralshield.git
%cd neuralshield
!pip install uv
!uv venv
!source .venv/bin/activate && uv sync
```

3. Dump embeddings with the flag-weighted encoder:

```bash
# Train split
!source .venv/bin/activate && \
  uv run python src/neuralshield/encoding/dump_embeddings.py \
  src/neuralshield/data/CSIC/train.jsonl \
  experiments/21_flag_correlation/csic_embeddings/train_flagweighted.npz \
  --encoder secbert-flag-weighted \
  --device cuda \
  --use-pipeline

# Test split
!source .venv/bin/activate && \
  uv run python src/neuralshield/encoding/dump_embeddings.py \
  src/neuralshield/data/CSIC/test.jsonl \
  experiments/21_flag_correlation/csic_embeddings/test_flagweighted.npz \
  --encoder secbert-flag-weighted \
  --device cuda \
  --use-pipeline
```

Artifacts will be saved under `experiments/21_flag_correlation/csic_embeddings/` and can be synced back from Colab afterward.
