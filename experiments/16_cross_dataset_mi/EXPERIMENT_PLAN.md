# Experiment 16: Cross-Dataset MI Feature Selection - Detailed Plan

**Goal**: Replicate the paper's semi-supervised anomaly detection framework using only SR-BH 2020 and CSIC 2010 datasets.

---

## Core Philosophy

The paper's key innovation is **transfer learning for attack detection**:

1. Use **generic labeled attacks** from any dataset to identify discriminative features
2. Train **only on normal traffic** from the target system (no target attacks needed)
3. Detect **novel attacks** on the target system

This is powerful because:

- No need for application-specific attack samples
- Works on zero-day attacks (unseen patterns)
- Minimal labeling effort (only normal traffic)

---

## Dataset Strategy

### Why These Two Datasets?

**CSIC 2010**:

- **Pros**: Clean labels, well-structured, synthetic attacks
- **Cons**: Old (2010), e-commerce specific, limited attack diversity
- **Use case**: Clear signal for feature selection

**SR-BH 2020**:

- **Pros**: Real-world, diverse, modern attacks, large scale
- **Cons**: Noisy, unlabeled normal traffic, honeypot artifacts
- **Use case**: Realistic target system

### The Two Runs

**Run 1: CSIC → SR-BH** (Primary validation)

- **Hypothesis**: Clean 2010 attack patterns can detect modern 2020 attacks
- **Challenge**: 10-year gap + synthetic → real-world transfer
- **Expected**: 70-85% TPR (harder transfer)

**Run 2: SR-BH → CSIC** (Sanity check)

- **Hypothesis**: Noisy real-world features work on clean synthetic data
- **Challenge**: Noisy feature selection
- **Expected**: 85-95% TPR (easier direction)

---

## Pipeline Details

### Step 1: Data Loading

**CSIC 2010** (from `src/neuralshield/data/CSIC/`):

- Train: ~14K requests (all normal)
- Test: ~21K requests (7K normal + 14K attacks)
- Total: ~35K requests
- Attack types: SQLi, XSS, path traversal, LDAP injection, etc.

**SR-BH 2020** (from `src/neuralshield/data/SR_BH_2020/`):

- Total: ~900K requests
- CAPEC categories: 13 types (SQLi, XSS, RCE, path traversal, etc.)
- Sampling: 15% from each CAPEC category for feature selection (following paper)
- Train/test split: 80/20 on normal traffic

**Sampling strategy** (for feature selection):

```python
# For CSIC → SR-BH (Run 1):
#   MI: All CSIC data (~35K)
#   Train: SR-BH normal only (~50K)
#   Test: SR-BH attacks (~50K)

# For SR-BH → CSIC (Run 2):
#   MI: SR-BH sample (15% per CAPEC ~20K attacks + 50K normal)
#   Train: CSIC train normal (~14K)
#   Test: CSIC test (~21K)
```

---

### Step 2: Preprocessing

**Paper's 5-step approach** (Section 3.2):

```python
def paper_preprocess(request: str) -> str:
    # 1. Header filtering (already done in datasets)

    # 2. URL decode
    decoded = unquote(request)

    # 3. UTF-8 decode (implicit in Python)

    # 4. URL decode again (double-encoding protection)
    decoded = unquote(decoded)

    # 5. Lowercase
    return decoded.lower()
```

**Why this is simpler than NeuralShield's 13 steps**:

- Paper focuses on **content-level** threats (SQLi, XSS)
- NeuralShield handles **protocol-level** threats too (HTTP smuggling, etc.)
- For feature selection, simpler is better (clearer signal)

**Critical tokenization rule**:

```python
CountVectorizer(
    token_pattern=r"\S+",  # Split ONLY on whitespace
    lowercase=False,       # Already lowercased
)
```

This preserves:

- `' OR '1'='1'` (SQL injection)
- `<script>alert(1)</script>` (XSS)
- `../../etc/passwd` (path traversal)
- `union select` (SQL keywords)

NLP tokenizers (spacy, nltk) would destroy these patterns.

---

### Step 3: Feature Selection (Algorithm 1 + 2)

**Algorithm 1: Dictionary Building**

```python
# Combine normal + attacks from source dataset
all_requests = normal_requests + attack_requests

# Build vocabulary with CountVectorizer
count_vec = CountVectorizer(
    max_features=5000,      # Paper's optimal
    token_pattern=r"\S+",
    lowercase=False,
)
count_vec.fit(all_requests)
dictionary = count_vec.get_feature_names_out()
```

**Algorithm 2: MI Computation**

```python
# TF-IDF with fixed vocabulary
tfidf_vec = TfidfVectorizer(
    vocabulary=dictionary,  # Use pre-built dictionary
    token_pattern=r"\S+",
    lowercase=False,
)
X = tfidf_vec.fit_transform(all_requests)

# Labels: 0 = normal, 1 = attack
labels = [0] * len(normal) + [1] * len(attacks)

# Compute MI for each token
mi_scores = mutual_info_classif(X, labels, random_state=42)

# Select top-K
top_k_indices = np.argsort(mi_scores)[-k:]
selected_tokens = [dictionary[i] for i in top_k_indices]
```

**Expected high-MI tokens**:

- SQL: `union`, `select`, `from`, `where`, `--`, `;--`
- XSS: `<script`, `onerror`, `alert(`, `javascript:`
- Path traversal: `../`, `..\\`, `/etc/passwd`
- Encoding tricks: `%00`, `0x`, `char(`
- Generic: `'`, `"`, `<`, `>`, `|`, `&`

---

### Step 4: Training (Algorithm 3)

**One-Class SVM parameters** (from paper's grid search):

- `nu = 0.05`: Expect 5% outliers in training data
- `gamma = 0.5`: RBF kernel width (tuned on validation set)
- `kernel = "rbf"`: Non-linear decision boundary

**Training procedure**:

```python
# Create BoW with selected tokens ONLY
bow_vec = CountVectorizer(
    vocabulary=selected_tokens,  # Top-K from MI
    token_pattern=r"\S+",
    lowercase=False,
)

# Vectorize normal traffic from target dataset
X_train = bow_vec.fit_transform(target_normal_requests)

# Train One-Class SVM
ocsvm = OneClassSVM(nu=0.05, gamma=0.5, kernel="rbf")
ocsvm.fit(X_train.toarray())
```

**Why toarray()?**

- OCSVM with RBF kernel requires dense matrices
- TF-IDF output is sparse
- Conversion is fast for K ≤ 200 features

---

### Step 5: Evaluation

**Metrics** (at ~5% FPR):

- **TPR (True Positive Rate)**: % of attacks detected
- **FPR (False Positive Rate)**: % of normal traffic flagged
- **Precision**: % of flagged requests that are attacks
- **F1-Score**: Harmonic mean of precision and recall

**Threshold calibration**:

```python
# Get scores on test set
test_scores = -ocsvm.decision_function(X_test)

# Predictions: -1 = attack, +1 = normal
predictions = ocsvm.predict(X_test)
is_attack = (predictions == -1)
```

**Confusion matrix**:

```
                 Predicted
                 Normal  Attack
Actual Normal      TN      FP
       Attack      FN      TP

TPR = TP / (TP + FN)  (want high)
FPR = FP / (FP + TN)  (want low, ~5%)
```

---

## Expected Outcomes

### Run 1: CSIC → SR-BH

**Best case** (k=100):

- TPR: 85%+ (matches paper's SR-BH result of 78.87%)
- FPR: 4-6%
- Interpretation: Clean synthetic attacks → modern real-world detection works

**Realistic case**:

- TPR: 70-80%
- FPR: 5-7%
- Interpretation: Good generalization, some 2020 attack patterns missed

**Worst case**:

- TPR: 50-60%
- FPR: 8-10%
- Interpretation: 10-year gap too large, or CSIC too specific

### Run 2: SR-BH → CSIC

**Expected** (k=100):

- TPR: 85-95%
- FPR: <3%
- Interpretation: Modern features easily detect old attacks

---

## Comparison to Paper's Results

**Paper's best results**:

| Dataset | TPR    | FPR   | K   |
| ------- | ------ | ----- | --- |
| Drupal  | 91.76% | 2.29% | 100 |
| SR-BH   | 78.87% | 5.18% | 100 |

**Our Run 1 target**:

- SR-BH: 75-85% TPR @ 5-6% FPR (comparable)

**Why we might do better**:

- We're testing the **full cross-dataset** scenario
- Paper's SR-BH may have been trained on SR-BH normal (same dataset)

**Why we might do worse**:

- No access to Drupal (their cleanest dataset)
- CSIC is older/simpler than their attack source

---

## Visualizations

### 1. MI Token Distribution

```python
plt.hist(mi_scores, bins=50)
plt.xlabel("MI Score")
plt.ylabel("Count")
plt.title("Distribution of MI Scores for 5000 TF-IDF Tokens")
```

**Expected shape**: Long tail (most tokens have low MI, few have very high MI)

### 2. Top-20 Tokens Bar Chart

```python
top_20_idx = np.argsort(mi_scores)[-20:]
plt.barh(range(20), mi_scores[top_20_idx])
plt.yticks(range(20), [tokens[i] for i in top_20_idx])
```

**Expected tokens**: SQL/XSS keywords, special chars

### 3. ROC Curves by K

```python
for k in [50, 100, 150, 200]:
    fpr, tpr, _ = roc_curve(y_true, scores[k])
    plt.plot(fpr, tpr, label=f"k={k}")
plt.xlabel("FPR")
plt.ylabel("TPR")
```

**Expected**: k=100 curve dominates (best AUC)

### 4. Token Overlap Heatmap

```python
# Compare selected tokens from Run 1 vs Run 2
csic_tokens = set(run1_tokens)
srbh_tokens = set(run2_tokens)
overlap = csic_tokens & srbh_tokens
```

**Expected**: 60-80% overlap (core attack patterns are universal)

---

## Success Criteria

### Critical (Must achieve):

1. ✅ Both runs complete without errors
2. ✅ TPR > baseline (random = 5%)
3. ✅ FPR ≈ 5% (threshold calibration works)

### Important (Should achieve):

1. ✅ Run 1 TPR > 60% (beats Experiment 13's 7.8%)
2. ✅ Run 2 TPR > 80% (easier direction)
3. ✅ k=100 outperforms k=50 and k=200

### Aspirational (Nice to have):

1. ✅ Run 1 TPR > 75% (matches paper)
2. ✅ Token overlap > 70% (universal patterns)
3. ✅ Clear narrative for thesis writeup

---

## Risk Mitigation

### Risk 1: Low TPR on Run 1 (<50%)

**Possible causes**:

- CSIC features too old/specific
- Preprocessing mismatch with paper
- Hyperparameter tuning needed

**Mitigation**:

- Compare selected tokens between runs (should overlap)
- Check if preprocessing preserves attack signatures
- Try paper's expert-selected 64 features as baseline

### Risk 2: High FPR (>10%)

**Possible causes**:

- Threshold calibration failed
- Normal traffic too noisy (SR-BH honeypot artifacts)
- Selected tokens too generic

**Mitigation**:

- Use cross-validation on training set to set threshold
- Filter SR-BH normal traffic (remove scanning noise)
- Increase K (more specific features)

### Risk 3: No improvement over Experiment 13

**Possible causes**:

- Implementation bug (not truly cross-dataset)
- Data loading error

**Mitigation**:

- Double-check: MI source ≠ test source
- Verify dataset sizes match expectations
- Add logging to track data flow

---

## Timeline & Milestones

**Phase 1: Implementation** (30 min)

- ✅ Create directory structure
- ✅ Copy preprocessing.py from Exp 13
- ✅ Implement data loaders with cross-dataset logic
- ✅ Implement Run 1 & Run 2

**Phase 2: Execution** (20 min)

- ✅ Run 1: CSIC → SR-BH
- ✅ Run 2: SR-BH → CSIC
- ✅ Save all results

**Phase 3: Analysis** (15 min)

- ✅ Generate visualizations
- ✅ Compare runs
- ✅ Analyze token overlap
- ✅ Write RESULTS.md

**Total**: ~65 minutes

---

## Deliverables

1. **Code**:

   - `test_cross_dataset_mi.py` (main experiment)
   - `preprocessing.py` (paper's 5 steps)
   - `visualize_results.py` (plots)

2. **Results**:

   - `RESULTS.md` (summary)
   - `metrics_comparison.json` (all K values)
   - `selected_tokens_k{50,100,150,200}.txt`

3. **Visualizations**:

   - MI distributions (both runs)
   - ROC curves (both runs)
   - Token overlap Venn diagram
   - TPR vs K curves

4. **Documentation**:
   - This plan (EXPERIMENT_PLAN.md)
   - README.md (high-level overview)
   - RESULTS.md (interpretation & conclusions)
