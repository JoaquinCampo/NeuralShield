### 09 Header Unfold (Obs-fold)

**Step Contract:**
- Input lines considered: `[HEADER]` lines
- Transformations allowed: unfold obs-fold header continuations
- Flags emitted: `OBSFOLD`, `BADCRLF`, `BADHDRCONT`
- Dependencies: 01 (request structurer)
- Idempotence: multiple applications produce same result

---

### Scope

Detect and unfold obsolete header folding (obs-fold) to produce single-line headers:
- Unfold headers that continue on next line with SP/HTAB
- Normalize continuation separators to single space
- Detect and flag embedded CR/LF characters
- Handle malformed continuation lines

---

### Obs-fold Background

- **Historical**: HTTP/1.1 (RFC 7230) allowed header continuation with leading whitespace
- **Current Status**: Prohibited in RFC 9110 (HTTP semantics)
- **Security Risk**: Often used for evasion and header smuggling attacks
- **Detection**: Lines starting with SP (0x20) or HTAB (0x09)

---

### Rules

1. **Evidence Preservation**: **ALWAYS flag obs-fold detection** - this is a security indicator
2. **Continuation Detection**: Lines starting with SP/HTAB are continuations
3. **Unfolding Process**: Join continuation to previous header with single space
4. **Mandatory Flagging**: Emit `OBSFOLD` for ANY continuation detected
5. **Error Handling**: Flag orphaned continuations (no preceding header)
6. **Embedded CRLF**: Detect and flag CR/LF within header values
7. **Security Priority**: Obs-fold often indicates evasion attempts

---

### Flags Emitted

- `OBSFOLD`: Obs-fold continuation detected and unfolded
- `BADCRLF`: CR/LF characters embedded in header value
- `BADHDRCONT`: Continuation line without valid preceding header

---

### Examples

Basic obs-fold unfolding:
```
[HEADER] X-Custom: first line
[HEADER]  second part
[HEADER]	third part
```

Output:
```
[HEADER] X-Custom: first line second part third part
OBSFOLD
```

Orphaned continuation:
```
[HEADER]   orphaned continuation
[HEADER] Host: example.com
```

Output:
```
[HEADER] Host: example.com
BADHDRCONT
```

Embedded CRLF (malicious):
```
[HEADER] X-Evil: value1
value2
[HEADER] Host: example.com
```

Output:
```
[HEADER] X-Evil: value1 value2
[HEADER] Host: example.com
BADCRLF
```

Multiple continuations:
```
[HEADER] Accept: text/html,
[HEADER]  application/xml;q=0.9,
[HEADER]  image/webp,*/*;q=0.8
```

Output:
```
[HEADER] Accept: text/html, application/xml;q=0.9, image/webp,*/*;q=0.8
OBSFOLD
```

---

### Implementation Details

1. **Line-by-Line Processing**: Examine each header line for continuation markers
2. **State Tracking**: Maintain context of previous valid header for joining
3. **Whitespace Handling**: Trim leading whitespace from continuations
4. **Separator Insertion**: Add single space between original and continuation
5. **Error Detection**: Identify and flag various anomalies

---

### Edge Cases

- **Multiple Spaces/Tabs**: All leading whitespace collapsed to single space
- **Empty Continuations**: Lines with only whitespace still trigger `OBSFOLD`
- **Mixed Whitespace**: Both SP and HTAB treated as continuation markers
- **Long Headers**: No length limits imposed during unfolding

---

### Security Implications

- **Attack Vector**: Obs-fold commonly used in HTTP request smuggling
- **Evasion Technique**: Headers split across lines to bypass filters
- **Parsing Differences**: Different servers may handle obs-fold differently

---

### Supersedes

- New step implementing `lineas-plegadas-obs-fold.md` specification

---

### Good vs Bad Examples

Good (single-line headers):
```
[HEADER] host: example.com
[HEADER] user-agent: Test/1.0
```

Bad (obs-fold → unfolded and flagged):
```
[HEADER] x-custom: first
[HEADER]  second
[HEADER] 	third
OBSFOLD
```
