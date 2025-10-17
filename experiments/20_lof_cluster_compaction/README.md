# Experiment 20 · LOF Cluster Compaction

Goal: tighten local neighbourhoods for normal HTTP requests so that density-based detectors like LOF get a clearer separation from malicious traffic.

Key idea:
- Start from the existing SecBERT embeddings stored in `embeddings/SecBert/train_embeddings.npz` and `test_embeddings.npz`.
- Operate only on the requests labelled as `valid`/`normal`.
- Learn a small projection head that pulls each embedding closer to its k-nearest neighbours (using a contrastive NT-Xent style loss) while keeping batches from collapsing. Only anchors whose baseline neighbourhood radius is above a configurable percentile are optimised so we avoid over-tightening already-compact samples.
- Export the projection so downstream anomaly detectors can re-encode requests and re-run LOF without touching the rest of the pipeline.

Workflow:
1. Train neighbourhood compaction with `uv run experiments/20_lof_cluster_compaction/train_cluster_compaction.py`.
   - Outputs a saved projector under `models/contrastive/lof_compaction_head.pt` (path configurable).
   - Logs neighbourhood radius metrics pre/post to validate that clusters compress rather than collapse. A variance-preserving penalty keeps the projected covariance from collapsing completely.
2. Apply the projector on train/test splits using `uv run experiments/20_lof_cluster_compaction/apply_projection.py`.
   - Produces `.npz` files mirroring the originals but with projected embeddings.
3. Re-run LOF (and optionally Mahalanobis) on the compacted embeddings with `uv run experiments/20_lof_cluster_compaction/evaluate_lof.py` to compare ROC/AUC and score distributions.

Success criteria:
- Mean intra-neighbour cosine distance on the validation split drops ≥10% without shrinking global variance to zero.
- LOF AUC improves over the baseline recorded in `embeddings/SecBert/results.json`.

Open questions / TODO:
- Try alternative sampling (session-aware pairs) if the simple k-NN approach blends unrelated traffic.
- Experiment with stronger regularisation (e.g. covariance penalty) if we observe collapse in practice.
