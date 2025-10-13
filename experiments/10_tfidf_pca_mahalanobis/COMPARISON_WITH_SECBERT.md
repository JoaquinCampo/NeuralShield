# TF-IDF+PCA+Mahalanobis vs SecBERT+Mahalanobis Comparison

## Performance Summary

### TF-IDF+PCA+Mahalanobis (with preprocessing)

- **Recall**: 27.08% (6,788 / 25,065 attacks detected)
- **Precision**: 84.45%
- **F1-Score**: 41.01%
- **FPR**: 5.00%
- **Total detections**: 8,038

### SecBERT+Mahalanobis (with preprocessing)

- **Recall**: 49.26% (12,347 / 25,065 attacks detected)
- **Precision**: 90.81%
- **F1-Score**: 63.87%
- **FPR**: 5.00%
- **Total detections**: 13,597

## Detection Overlap Analysis

### Agreement

- **Overall agreement**: 72.7% (models agree on 36,401 / 50,065 test samples)

### Attack Detection Breakdown

Total attacks: 25,065

- **Both models detect**: 3,376 attacks (13.5%)
- **SecBERT only**: 8,971 attacks (35.8%)
- **TF-IDF+PCA only**: 3,412 attacks (13.6%)
- **Neither detects**: 9,306 attacks (37.1%)

### Key Findings

1. **SecBERT is significantly stronger**:

   - 82% better recall (49.26% vs 27.08%)
   - SecBERT catches 8,971 unique attacks that TF-IDF+PCA misses
   - TF-IDF+PCA catches 3,412 unique attacks that SecBERT misses

2. **TF-IDF+PCA provides complementary coverage**:

   - 13.6% of attacks are only caught by TF-IDF+PCA
   - This suggests different attack patterns detected by each method

3. **False Positive Overlap**:

   - 612 false positives shared by both (49% of TF-IDF+PCA FPs)
   - 638 false positives unique to TF-IDF+PCA
   - 638 false positives unique to SecBERT

4. **False Negative Overlap**:
   - 9,306 attacks missed by both (37% of all attacks)
   - These represent hard-to-detect attacks for both methods

## Interpretation

### Why SecBERT Outperforms

1. **Dense semantic embeddings**: SecBERT captures contextual meaning
2. **No information loss**: Works on full 768-dimensional space
3. **Better feature representation**: Pre-trained on security-related text

### What TF-IDF+PCA Catches Uniquely

TF-IDF+PCA detects 3,412 attacks that SecBERT misses:

- Likely attacks with specific token patterns
- Cases where n-gram frequency is more diagnostic than semantics
- Possible URL-encoding or character-level anomalies preserved in TF-IDF

### PCA Limitation

- Reducing 5,000 TF-IDF dimensions to 300 loses critical sparse features
- Even with 91.94% variance explained, detection power drops severely
- Rare token patterns (often attack signatures) are discarded

## Conclusion

**SecBERT+Mahalanobis is the superior method**:

- 82% higher recall with similar precision
- Better overall F1-score (63.87% vs 41.01%)

**TF-IDF+PCA+Mahalanobis has limited value**:

- Catches only 13.6% of attacks uniquely
- PCA compression destroys critical detection features
- Not recommended as a standalone detector

**Potential ensemble benefit**:

- Combining both methods could catch additional 3,412 attacks (13.6% gain)
- Total coverage would increase from 49.26% to 62.86% recall
- Trade-off: slightly higher complexity
