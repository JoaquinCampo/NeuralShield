#!/bin/bash
set -e

echo "================================================================"
echo "TESTING MULTIPLE CONTAMINATION VALUES"
echo "================================================================"
echo ""

CONTAMINATIONS=(0.15 0.2 0.25 0.3)

for CONTAM in "${CONTAMINATIONS[@]}"; do
    echo ""
    echo ">>> CONTAMINATION = $CONTAM <<<"
    echo ""
    
    # Train WITHOUT preprocessing
    echo "[1/4] Training WITHOUT preprocessing..."
    uv run python -m scripts.tfidf train \
        experiments/preprocessing_impact/without_preprocessing/embeddings.npz \
        experiments/preprocessing_impact/without_preprocessing/model_${CONTAM}.joblib \
        --valid-label valid \
        --detector isolation_forest \
        --contamination $CONTAM \
        --n-estimators 300 2>&1 | grep -q "Saved"
    
    # Train WITH preprocessing
    echo "[2/4] Training WITH preprocessing..."
    uv run python -m scripts.tfidf train \
        experiments/preprocessing_impact/with_preprocessing/embeddings.npz \
        experiments/preprocessing_impact/with_preprocessing/model_${CONTAM}.joblib \
        --valid-label valid \
        --detector isolation_forest \
        --contamination $CONTAM \
        --n-estimators 300 2>&1 | grep -q "Saved"
    
    # Test WITHOUT preprocessing
    echo "[3/4] Testing WITHOUT preprocessing..."
    uv run python -m scripts.tfidf test \
        src/neuralshield/data/CSIC/test.jsonl \
        experiments/preprocessing_impact/without_preprocessing/vectorizer.joblib \
        experiments/preprocessing_impact/without_preprocessing/model_${CONTAM}.joblib \
        --batch-size 512 2>&1 | tee experiments/preprocessing_impact/without_preprocessing/results_${CONTAM}.txt | grep -A 16 "Classification Metrics"
    
    # Test WITH preprocessing
    echo "[4/4] Testing WITH preprocessing..."
    uv run python -m scripts.tfidf test \
        src/neuralshield/data/CSIC/test.jsonl \
        experiments/preprocessing_impact/with_preprocessing/vectorizer.joblib \
        experiments/preprocessing_impact/with_preprocessing/model_${CONTAM}.joblib \
        --use-pipeline \
        --batch-size 512 2>&1 | tee experiments/preprocessing_impact/with_preprocessing/results_${CONTAM}.txt | grep -A 16 "Classification Metrics"
    
    echo "---"
done

echo ""
echo "================================================================"
echo "TESTING COMPLETE"
echo "================================================================"
