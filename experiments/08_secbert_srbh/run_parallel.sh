#!/bin/bash
# Run all 4 SecBERT embedding generation tasks in parallel
# Optimized for A100 GPU on Colab

echo "Starting parallel embedding generation..."
echo "Date: $(date)"

# 1. Train WITH preprocessing
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/train.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/train_embeddings.npz \
  --model jackaduma/SecBERT \
  --use-pipeline \
  --batch-size 128 \
  --device cuda \
  > experiments/08_secbert_srbh/train_with_prep.log 2>&1 &

PID1=$!
echo "Started: Train WITH preprocessing (PID: $PID1)"

# 2. Train WITHOUT preprocessing
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/train.jsonl \
  experiments/08_secbert_srbh/without_preprocessing/train_embeddings.npz \
  --model jackaduma/SecBERT \
  --batch-size 128 \
  --device cuda \
  > experiments/08_secbert_srbh/train_no_prep.log 2>&1 &

PID2=$!
echo "Started: Train WITHOUT preprocessing (PID: $PID2)"

# 3. Test WITH preprocessing
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/test.jsonl \
  experiments/08_secbert_srbh/with_preprocessing/test_embeddings.npz \
  --model jackaduma/SecBERT \
  --use-pipeline \
  --batch-size 128 \
  --device cuda \
  > experiments/08_secbert_srbh/test_with_prep.log 2>&1 &

PID3=$!
echo "Started: Test WITH preprocessing (PID: $PID3)"

# 4. Test WITHOUT preprocessing
uv run python -m scripts.secbert_embed \
  src/neuralshield/data/SR_BH_2020/test.jsonl \
  experiments/08_secbert_srbh/without_preprocessing/test_embeddings.npz \
  --model jackaduma/SecBERT \
  --batch-size 128 \
  --device cuda \
  > experiments/08_secbert_srbh/test_no_prep.log 2>&1 &

PID4=$!
echo "Started: Test WITHOUT preprocessing (PID: $PID4)"

echo ""
echo "All 4 processes started. PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Monitor progress with: tail -f experiments/08_secbert_srbh/*.log"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background jobs
wait $PID1 && echo "✓ Train WITH preprocessing completed"
wait $PID2 && echo "✓ Train WITHOUT preprocessing completed"
wait $PID3 && echo "✓ Test WITH preprocessing completed"
wait $PID4 && echo "✓ Test WITHOUT preprocessing completed"

echo ""
echo "All embedding generation complete!"
echo "Date: $(date)"
echo ""
echo "Generated files:"
ls -lh experiments/08_secbert_srbh/with_preprocessing/*.npz
ls -lh experiments/08_secbert_srbh/without_preprocessing/*.npz
