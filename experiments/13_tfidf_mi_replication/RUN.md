# Quick Execution Guide - Experiment 13

## Step 1: Run the main experiment

```bash
uv run python experiments/13_tfidf_mi_replication/test_tfidf_mi_ocsvm.py
```

**Expected output**:

```
[1/5] Loading data...
CSIC: 47000 train normal, 50065 test
SR_BH: 10000 attacks loaded

[2/5] Preprocessing (paper's 5 steps)...
Preprocessed 107,065 requests

[3/5] Computing MI on TF-IDF features...
TF-IDF shape: (57000, 5000)
Sparsity: 99.XX%
Computing mutual information...
Top 10 tokens by MI:
  1. 'union': 0.XXXXX
  2. 'select': 0.XXXXX
  3. '<script>': 0.XXXXX
  ...

[4/5] Testing different K values...
Testing k=50... Recall=0.XXXX, FPR=0.XXXX
Testing k=64... Recall=0.XXXX, FPR=0.XXXX
Testing k=100... Recall=0.XXXX, FPR=0.XXXX
Testing k=150... Recall=0.XXXX, FPR=0.XXXX
Testing k=200... Recall=0.XXXX, FPR=0.XXXX

SUMMARY
Best configuration: k=XXX
  Recall:    0.XXXX
  Precision: 0.XXXX
  FPR:       0.XXXX

COMPARISON TO SECBERT
SecBERT + Mahalanobis:  0.4926 recall @ 5.00% FPR
TF-IDF + MI (k=XXX):    0.XXXX recall @ X.XX% FPR
Difference:             +/-X.XXXX

❌ TF-IDF + MI is XX% WORSE than SecBERT
```

**Time**: ~5 minutes

---

## Step 2: Visualize results

```bash
uv run python experiments/13_tfidf_mi_replication/visualize_results.py
```

**Generates**:

- `results/plots/tfidf_mi_comparison.png` - 4-panel comparison
- `results/plots/mi_token_distribution.png` - Token analysis

---

## Step 3: Review selected tokens

```bash
# See top 100 MI-selected tokens
cat experiments/13_tfidf_mi_replication/results/selected_tokens_k100.txt
```

Look for:

- SQL injection keywords: `union`, `select`, `or`, `1=1`
- XSS patterns: `<script>`, `alert`, `onerror`
- Path traversal: `../`, `..%2f`
- Command injection: `;`, `|`, `&&`

---

## Decision Tree

**If recall ≥ 30%**:
→ **Promising!** Consider ensemble with SecBERT
→ Investigate why sparse+MI works well

**If recall = 10-30%**:
→ **Expected range** - validates MI helps sparse
→ But still worse than SecBERT (49%)
→ Document as comparison baseline

**If recall < 10%**:
→ **Paper didn't replicate**
→ Investigate differences (datasets, preprocessing)
→ Confirm sparse << dense

---

## Next Steps

**After results**:

1. Compare selected tokens to security expert knowledge
2. Analyze which attack types are detected
3. Document findings in RESULTS.md
4. Update main CONSOLIDATED_RESULTS.md
