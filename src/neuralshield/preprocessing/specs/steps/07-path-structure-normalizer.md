### 08 Path Structure Normalizer

**Step Contract:**
- Input lines considered: `[URL]` lines
- Transformations allowed: normalize path structure without resolving traversal
- Flags emitted: `MULTIPLESLASH`, `DOTDOT`, `HOME`
- Dependencies: 06 (HTML entity decode once)
- Idempotence: multiple applications produce same result

---

### Scope

Normalize path structure while preserving security-relevant traversal indicators:
- Collapse multiple slashes (`//`) to single slash
- Remove current directory segments (`/.`)
- Preserve parent directory segments (`..`) without resolution
- Detect and flag structural anomalies

---

### Normalization Rules (flag all changes)

1. **Slash Collapsing**: Replace sequences of multiple `/` with single `/` → emit `MULTIPLESLASH`
2. **Current Directory Removal**: Remove `.` segments (e.g., `/./` becomes `/`) → emit `DOTCUR`
3. **Parent Directory Preservation**: Keep `..` segments for traversal detection → emit `DOTDOT`
4. **Root Canonicalization**: Empty or root-only paths become `/` → emit `HOME`
5. **Segment Processing**: Work on path segments without decoding `%2F`

---

### Flags Emitted

- `MULTIPLESLASH`: Multiple slash sequences (`//`) were collapsed
- `DOTCUR`: Current directory segments (`/.`) were removed
- `DOTDOT`: Parent directory segments (`..`) are present
- `HOME`: Normalized path is exactly `/`

---

### Examples

Multiple slashes:
```
[URL] /foo//bar///baz
```

Output:
```
[URL] /foo/bar/baz
MULTIPLESLASH
```

Current directory segments (flagged):
```
[URL] /path/./to/./file
```

Output:
```
[URL] /path/to/file
DOTCUR
```

Parent directory traversal:
```
[URL] /var/www/../../../etc/passwd
```

Output:
```
[URL] /var/www/../../../etc/passwd
DOTDOT
```

Root path:
```
[URL] /
```

Output:
```
[URL] /
HOME
```

Complex combination:
```
[URL] /app//data/.//config/../settings
```

Output:
```
[URL] /app/data/config/../settings
DOTDOT MULTIPLESLASH
```

Empty path normalization:
```
[URL] 
```

Output:
```
[URL] /
HOME
```

---

### Good vs Bad Examples

Good (already canonical):
```
[URL] /a/b/c
```

Bad (structure anomalies → flagged during normalization):
```
[URL] /a//b/./c/../d
[URL] /a/b/c/../d
DOTCUR DOTDOT MULTIPLESLASH
```

---

### Implementation Details

1. **Segmentation**: Split path by `/` without decoding percent-encoded slashes
2. **Empty Segment Detection**: Identify empty segments from multiple slashes
3. **Segment Classification**: Categorize as empty, current (`.`), parent (`..`), or regular
4. **Reconstruction**: Rebuild path with single slashes between valid segments
5. **Flag Generation**: Track anomalies during processing

---

### Edge Cases

- **Trailing Slashes**: Generally removed except for root path
- **Percent-Encoded Slashes**: `%2F` preserved as-is (not treated as delimiter)
- **Multiple Parent Dirs**: `../../../../` preserved entirely with single `DOTDOT` flag
- **Mixed Anomalies**: Multiple flags can be emitted for single path

---

### Security Considerations

- **No Resolution**: Never resolve `..` to prevent masking traversal attempts
- **Preservation**: Maintain all traversal indicators for downstream security analysis
- **Canonicalization**: Provide consistent format while preserving attack signatures

---

### Supersedes

- Original `PathStructureNormalizer` with same core functionality
