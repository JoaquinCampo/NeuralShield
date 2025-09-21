# Security-First Preprocessing Principles

## Core Philosophy: Evidence Preservation Over Normalization

**Primary Goal**: Maintain complete evidence of how the original packet was constructed, as encoding/formatting choices often indicate malicious intent.

---

## Fundamental Principles

### 1. **Flag All Transformations**
- Every normalization, decoding, or structural change MUST be flagged
- The presence of encoded content is often more suspicious than the decoded content itself
- Example: `%20` in a URL path is more suspicious than a space character

### 2. **Preserve Suspicious Encodings**
- Keep percent-encoded suspicious characters: `%20`, `%00`, `%3C`, etc.
- Keep HTML entities that could indicate evasion: `&lt;`, `&#x2f;`, etc.
- Keep Unicode anomalies that indicate obfuscation attempts

### 3. **Minimal Necessary Decoding**
- Only decode when absolutely required for downstream processing
- Prefer flagging over silent normalization
- When decoding is necessary, always flag the transformation

### 4. **Attack Vector Awareness**
Common evasion techniques that must be preserved and flagged:
- **Percent encoding**: `%3Cscript%3E` instead of `<script>`
- **HTML entities**: `&#x3c;script&#x3e;` instead of `<script>`
- **Unicode normalization**: `％` (fullwidth) instead of `%`
- **Whitespace padding**: Multiple spaces/tabs to confuse parsers
- **Case variations**: `Content-type` vs `content-type`
- **Obs-fold**: Header continuations to bypass filters

---

## Security Rationale

### Why Preserve Encodings?

1. **Evasion Detection**: Attackers use encoding to bypass simple string matching
2. **Intent Analysis**: Choice to encode reveals deliberate obfuscation
3. **Pattern Recognition**: ML models can learn encoding-based attack patterns
4. **Forensic Value**: Original encoding preserved for incident analysis

### Example Attack Scenarios

**XSS Evasion**:
- Attacker sends: `/search?q=%3Cscript%3Ealert(1)%3C/script%3E`
- If decoded: `/search?q=<script>alert(1)</script>` (obvious attack)
- If preserved: `/search?q=%3Cscript%3Ealert(1)%3C/script%3E` + `PCTANGLE` flag
- **Benefit**: Model learns that percent-encoded angle brackets are suspicious

**SQL Injection Evasion**:
- Attacker sends: `/user?id=1&#x27;&#x20;OR&#x20;1=1--`
- If decoded: `/user?id=1' OR 1=1--` (obvious injection)
- If preserved: `/user?id=1&#x27;&#x20;OR&#x20;1=1--` + `HTMLENT` flag
- **Benefit**: Model learns that HTML-encoded SQL syntax is suspicious

---

## Implementation Guidelines

### For Each Step

1. **Detect First**: Identify anomalies before any transformation
2. **Flag Immediately**: Emit flags for detected anomalies
3. **Transform Minimally**: Only change what's absolutely necessary
4. **Preserve Evidence**: Keep original suspicious patterns when possible

### Flag Naming Convention

- **Encoding Flags**: `PCT*` for percent encoding, `HTMLENT` for entities
- **Structure Flags**: `MULTI*` for structural anomalies
- **Content Flags**: `CONTROL`, `FULLWIDTH`, etc. for character anomalies
- **Process Flags**: `*NORM`, `*MERGE` for transformations applied

---

## Step-Specific Applications

### Percent Decoding (Step 05)
- **Preserve**: `%20`, `%00`, `%3C`, `%3E`, control chars
- **Flag**: `PCTSPACE`, `PCTNULL`, `PCTANGLE`, `PCTCONTROL`
- **Decode**: Only safe alphanumeric encodings when necessary

### HTML Entity Decoding (Step 06)  
- **Preserve**: All entities that could indicate attacks
- **Flag**: `HTMLENT` for any entity presence
- **Decode**: Only when required for downstream parsing

### Unicode Normalization (Step 04)
- **Preserve**: Evidence of fullwidth/variant characters
- **Flag**: `FULLWIDTH` for any normalization changes
- **Normalize**: Apply NFKC but flag the transformation

### Header Processing (Steps 09-11)
- **Preserve**: Evidence of obs-fold, case variations, whitespace padding
- **Flag**: `OBSFOLD`, `HDRNORM`, `WSPAD` for transformations
- **Normalize**: Apply minimal changes for parsing consistency

---

## ML Model Benefits

1. **Feature Engineering**: Flags become boolean features indicating evasion attempts
2. **Pattern Learning**: Models learn that encoded content correlates with attacks
3. **Robust Detection**: Detection works even with novel encoding combinations
4. **Explainability**: Flags provide clear indicators of why a request was flagged

---

## Implementation Priority

**Security > Normalization > Performance**

When in doubt:
1. Preserve the original suspicious pattern
2. Flag the anomaly clearly
3. Apply minimal necessary transformation
4. Document the security rationale

This approach ensures that NeuralShield maintains maximum detection capability while providing clean, consistent input for machine learning models.
