### 99 Flags Rollup Aggregator

**Step Contract:**
- Input lines considered: all lines with flags from previous steps
- Transformations allowed: collect and aggregate flags into single global line
- Flags emitted: `FLAGS:[...]` (global aggregation)
- Dependencies: all previous steps (final step in pipeline)
- Idempotence: multiple applications produce same result

---

### Scope

Aggregate all flags emitted by previous steps into a single global flags line:
- Collect flags from all individual flag lines
- Remove duplicate flags
- Sort flags alphabetically for consistency
- Emit single `FLAGS:[...]` line at end of output
- Remove individual flag lines to avoid duplication

---

### Aggregation Rules

1. **Flag Collection**: Scan all lines for flag patterns
2. **Deduplication**: Remove duplicate flags (same flag only appears once)
3. **Alphabetical Sorting**: Sort all flags consistently
4. **Line Removal**: Remove individual flag lines after collection
5. **Global Emission**: Add single `FLAGS:[...]` line at end

---

### Flag Line Detection

**Positive Patterns** (flag lines):
- Lines containing only uppercase flag names
- Lines with flag patterns like `QREPEAT:key`, `ARRAY:items[]`
- Lines matching regex: `^[A-Z_][A-Z0-9_]*(?::[A-Za-z0-9_<>.-]+)?(?:\s+[A-Z_][A-Z0-9_]*(?::[A-Za-z0-9_<>.-]+)?)*$`

**Negative Patterns** (not flag lines):
- Lines starting with `[` (structured content)
- Lines containing lowercase letters in non-parameter positions
- Metric lines like `[PLEN]`, `[HCNT]`

---

### Examples

Input with scattered flags:
```
[METHOD] GET
[URL] /path/../file
DOTDOT
[QPARAM] param QBARE
[QPARAM] encoded DOUBLEPCT
[QSEP] QRAWSEMI
[QMETA] count=3 QBARE DOUBLEPCT
[HEADER] host: example.com
SEMICOLON
```

Output:
```
[METHOD] GET
[URL] /path/../file
[QPARAM] param QBARE
[QPARAM] encoded DOUBLEPCT
[QSEP] QRAWSEMI
[QMETA] count=3 QBARE DOUBLEPCT
[HEADER] host: example.com
FLAGS:[DOTDOT DOUBLEPCT QBARE QRAWSEMI SEMICOLON]
```

Complex flags with parameters:
```
[URL] /path
MULTIPLESLASH
[QPARAM] items[] QARRAY:items[] QREPEAT:items[]
[QPARAM] items[] QARRAY:items[] QREPEAT:items[]
MIXEDSCRIPT
```

Output:
```
[URL] /path
[QPARAM] items[] QARRAY:items[] QREPEAT:items[]
[QPARAM] items[] QARRAY:items[] QREPEAT:items[]
FLAGS:[MIXEDSCRIPT MULTIPLESLASH QARRAY:items[] QREPEAT:items[]]
```

No flags case:
```
[METHOD] GET
[URL] /normal/path
[QPARAM] param
[HEADER] host: example.com
```

Output:
```
[METHOD] GET
[URL] /normal/path
[QPARAM] param
[HEADER] host: example.com
```

---

### Implementation Details

1. **Two-Pass Processing**: First pass collects flags, second pass emits clean output
2. **Pattern Matching**: Robust regex to identify flag vs content lines
3. **Set Operations**: Use set for automatic deduplication
4. **Stable Sorting**: Alphabetical sort for deterministic output
5. **Conditional Emission**: Only emit `FLAGS:` line if flags exist

---

### Flag Format

**Simple Flags**: `FULLWIDTH`, `CONTROL`, `HTMLENT`
**Parameterized Flags**: `QREPEAT:key`, `QARRAY:items[]`, `BADHDRNAME:x_custom`
**Sorting**: Alphabetical by flag name, parameters included in sort

---

### Integration Benefits

**For ML Models**:
- Single location for all anomaly indicators
- Consistent format across all requests
- Easy feature extraction from bracketed list

**For Security Analysis**:
- Centralized anomaly summary
- Quick identification of request characteristics
- Simplified alerting and filtering

**For Debugging**:
- Complete anomaly picture in one line
- Easy comparison between requests
- Reduced output verbosity

---

### Edge Cases

- **Empty Input**: No flags produces no `FLAGS:` line
- **Duplicate Flags**: Multiple instances of same flag deduplicated
- **Mixed Case**: All flags assumed uppercase (detection pattern enforces this)
- **Invalid Flags**: Malformed flag lines ignored (fail-safe)

---

### Alternative Formats

The rollup format can be configured:

**Bracketed List** (default):
```
FLAGS:[CONTROL FULLWIDTH HTMLENT]
```

**Space-Separated**:
```
FLAGS: CONTROL FULLWIDTH HTMLENT
```

**JSON Array**:
```
FLAGS:["CONTROL","FULLWIDTH","HTMLENT"]
```

---

### Supersedes

- New step providing global flag aggregation capability
- Complements individual step flag emission
