### 14 Length Bucketing

**Step Contract:**
- Input lines considered: `[URL]`, `[HEADER]` lines, query metadata
- Transformations allowed: add length metrics with bucketing
- Flags emitted: `PLEN:{len}@{bucket}`, `PMAX:{len}@{bucket}`, `HCNT:{n}`, `HLEN:{len}@{bucket}`
- Dependencies: 08 (path), 10 (headers), 11 (whitespace collapse)
- Idempotence: multiple applications produce same result

---

### Scope

Calculate and report size metrics using bucketing for ML model stability:
- Path length and maximum segment length
- Header count and total header size
- Bucket values to prevent overfitting to exact sizes
- Provide consistent metrics across different request sizes

---

### Metrics Defined

**Path Metrics**:
- `PLEN`: Total length of normalized path
- `PMAX`: Length of longest path segment

**Header Metrics**:
- `HCNT`: Number of header lines after normalization
- `HLEN`: Total size of all headers in bytes

---

### Bucketing Strategy

**Default Buckets**:
- `0-15`, `16-31`, `32-63`, `64-127`, `128-255`, `256-511`, `512-1023`, `>1023`

**Bucket Assignment**:
- Inclusive lower and upper bounds except final bucket
- `>1023` for values exceeding largest defined bucket
- Deterministic assignment based on exact value

---

### Calculation Rules

1. **Path Length**: Count characters in normalized path (post step 08)
2. **Segment Length**: Split normalized path by `/`, measure longest segment
3. **Header Count**: Count `[HEADER]` lines after normalization and merging
4. **Header Size**: Sum `name: value` bytes (excluding CRLF line endings)
5. **Bucketing**: Apply bucket ranges to all length measurements

---

### Output Format

Metrics are emitted as informational lines (not flags):

```
[PLEN] {exact_length}@{bucket}
[PMAX] {exact_length}@{bucket}
[HCNT] {count}
[HLEN] {exact_length}@{bucket}
```

---

### Examples

Short path:
```
[URL] /api
```

Output:
```
[URL] /api
[PLEN] 4@0-15
[PMAX] 3@0-15
```

Longer path with segments:
```
[URL] /application/users/profile/settings
```

Output:
```
[URL] /application/users/profile/settings
[PLEN] 35@32-63
[PMAX] 11@0-15
```

Headers with various sizes:
```
[HEADER] host: example.com
[HEADER] user-agent: Mozilla/5.0 (long user agent string...)
[HEADER] accept: text/html,application/xml;q=0.9
```

Output:
```
[HEADER] host: example.com
[HEADER] user-agent: Mozilla/5.0 (long user agent string...)
[HEADER] accept: text/html,application/xml;q=0.9
[HCNT] 3
[HLEN] 156@128-255
```

Root path:
```
[URL] /
```

Output:
```
[URL] /
[PLEN] 1@0-15
[PMAX] 0@0-15
```

Large values:
```
[URL] /very/long/path/with/many/segments/that/exceeds/normal/lengths/and/continues/further
```

Output:
```
[URL] /very/long/path/with/many/segments/that/exceeds/normal/lengths/and/continues/further
[PLEN] 89@64-127
[PMAX] 9@0-15
```

---

### Implementation Details

1. **Post-Processing**: Calculate after all normalization steps complete
2. **Exact Values**: Store exact measurements for bucket assignment
3. **Deterministic**: Same input always produces same bucket assignment
4. **Efficient**: Single pass through normalized data
5. **Configurable**: Bucket ranges can be adjusted if needed

---

### Bucket Boundary Examples

| Length | Bucket |
|--------|--------|
| 0 | `0-15` |
| 15 | `0-15` |
| 16 | `16-31` |
| 63 | `32-63` |
| 64 | `64-127` |
| 255 | `128-255` |
| 256 | `256-511` |
| 1023 | `512-1023` |
| 1024 | `>1023` |
| 5000 | `>1023` |

---

### ML Model Benefits

- **Stability**: Reduces overfitting to exact lengths
- **Generalization**: Groups similar-sized requests
- **Feature Engineering**: Provides categorical size features
- **Anomaly Detection**: Large buckets indicate potential attacks

---

### Security Applications

- **DoS Detection**: Unusually large headers or paths
- **Buffer Overflow**: Extremely long components
- **Resource Consumption**: Requests with many headers
- **Baseline Establishment**: Normal size patterns for comparison

---

### Supersedes

- New step implementing `longitudes-bucketing.md` specification
- Extends `QLONG` logic from QueryProcessor to all components

---

### Good vs Bad Examples

Good (short path and few headers):
```
[URL] /a/b
[PLEN] 4@0-15
[PMAX] 1@0-15
[HEADER] host: example.com
[HCNT] 1
[HLEN] 21@16-31
```

Bad (extremely long components):
```
[URL] /very/very/very/long/path/that/exceeds/normal/usage/and/keeps/growing
[PLEN] 83@64-127
[PMAX] 9@0-15
[HEADER] user-agent: <huge string>
[HCNT] 8
[HLEN] 2048@>1023
```
