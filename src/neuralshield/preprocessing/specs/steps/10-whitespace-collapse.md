### 11 Whitespace Collapse

**Step Contract:**

- Input lines considered: `[HEADER]` lines (after normalization)
- Transformations allowed: collapse runs of tabs/spaces to single space
- Flags emitted: `WSPAD`
- Dependencies: 09-10 (header unfold and normalization)
- Idempotence: multiple applications produce same result

---

### Scope

Normalize whitespace within header values to eliminate formatting variations:

- Collapse sequences of spaces and tabs to single space
- Trim leading and trailing whitespace
- Preserve semantic spacing between tokens
- Flag when modifications are made

---

### Rules

1. **Evidence Preservation**: **ALWAYS flag whitespace anomalies** - indicates potential evasion
2. **Sequence Collapsing**: Replace `[\t ]+` with single space (0x20)
3. **Edge Trimming**: Remove leading and trailing whitespace
4. **Mandatory Flagging**: Emit `WSPAD` for ANY whitespace modification
5. **Token Preservation**: Maintain at least one space between distinct tokens
6. **Semantic Respect**: Don't alter whitespace that has meaning in specific headers
7. **Redaction Respect**: Skip processing for redacted values (`<SECRET:...>`)

---

### Whitespace Characters Targeted

- **Space** (0x20): Standard space character
- **Tab** (0x09): Horizontal tab character
- **Other**: Only these two characters are collapsed

---

### Flags Emitted

- `WSPAD`: Whitespace padding was detected and normalized

---

### Examples

Multiple spaces and tabs:

```
[HEADER] user-agent: Mozilla   5.0	(X11;  Linux)
```

Output:

```
[HEADER] user-agent: Mozilla 5.0 (X11; Linux) WSPAD
```

Leading/trailing whitespace:

```
[HEADER] x-custom:   value with spaces
```

Output:

```
[HEADER] x-custom: value with spaces WSPAD
```

No changes needed:

```
[HEADER] host: example.com
```

Output:

```
[HEADER] host: example.com
```

Complex whitespace patterns:

```
[HEADER] accept: text/html,		application/xml;  q=0.9,   image/webp
```

Output:

```
[HEADER] accept: text/html, application/xml; q=0.9, image/webp WSPAD
```

Redacted value (preserved):

```
[HEADER] authorization: <SECRET:bearer:64>
```

Output:

```
[HEADER] authorization: <SECRET:bearer:64>
```

---

### Implementation Details

1. **Pattern Matching**: Use regex `[\t ]+` to identify whitespace runs
2. **Boundary Detection**: Check for leading/trailing whitespace
3. **Change Tracking**: Flag only when actual modifications occur
4. **Preservation Logic**: Skip redacted and special values
5. **Idempotent Processing**: Second pass produces no changes
6. **Flag Format**: `WSPAD` emitted inline with affected headers (BERT-optimized for ML attention alignment)

---

### Interaction with Other Steps

- **After Obs-fold**: Operates on unfolded header values
- **After Normalization**: Works with normalized header names
- **Before Dangerous Chars**: Provides clean input for character analysis

---

### Edge Cases

- **Empty Values**: Headers with only whitespace become empty (flagged)
- **Single Spaces**: Preserved between meaningful tokens
- **Mixed Whitespace**: Both tabs and spaces in same run collapsed together
- **Nested Structures**: Whitespace within quoted strings or comments preserved

---

### Security Considerations

- **Evasion Prevention**: Eliminates whitespace-based filter bypass attempts
- **Normalization**: Provides consistent format for downstream analysis
- **Attack Surface**: Reduces variations that could confuse parsing

---

### Supersedes

- New step implementing `separadores-espacios.md` specification

---

### Good vs Bad Examples

Good (clean, minimal whitespace):

```
[HEADER] user-agent: Test/1.0 (X11; Linux)
```

Bad (runs of tabs/spaces → collapsed and flagged):

```
[HEADER] user-agent:  Test		1.0   (X11;  Linux) WSPAD
```

Good (redacted secrets preserved):

```
[HEADER] authorization: <SECRET:bearer:64>
```

Bad (leading/trailing whitespace → trimmed and flagged):

```
[HEADER] x-custom:   value with spaces    WSPAD
```
