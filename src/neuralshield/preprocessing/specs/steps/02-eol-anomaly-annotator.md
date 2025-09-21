### 02 EOL Anomaly Annotator

**Step Contract:**
- Input lines considered: entire HTTP request string
- Transformations allowed: add EOL anomaly flags without modifying content
- Flags emitted: `EOL_BARELF`, `EOL_BARECR`, `EOLMIX`, `EOL_EOF_NOCRLF`
- Dependencies: 01 (request structurer)
- Idempotence: multiple applications produce same result

---

### Scope

Detect and flag End-Of-Line anomalies in HTTP requests without modifying content:
- Identify non-standard line endings (bare LF, bare CR, mixed)
- Flag files ending without proper line termination
- Add flag markers to help detect potential security issues

---

### EOL Token Types

- **CRLF**: Standard `\r\n` sequence (RFC compliant)
- **Bare LF**: `\n` not preceded by `\r` (Unix-style)
- **Bare CR**: `\r` not followed by `\n` (old Mac-style)
- **Mixed**: Combination of different EOL types in same request

---

### Flags Emitted

- `EOL_BARELF`: Line uses only bare LF endings
- `EOL_BARECR`: Line uses only bare CR endings  
- `EOLMIX`: Line mixes different EOL types
- `EOL_EOF_NOCRLF`: File ends without proper line ending

---

### Rules

1. **Content Preservation**: Never modify the actual content, only add flags
2. **Per-Line Analysis**: Analyze each line's EOL sequence independently
3. **Pattern Detection**: Identify consistent vs mixed EOL usage patterns
4. **Idempotent Flagging**: Don't duplicate flags on subsequent runs
5. **Position Tracking**: Maintain accurate position tracking through the text

---

### Examples

Input with mixed line endings:
```
GET /path HTTP/1.1\r\n
Host: example.com\n
User-Agent: test\r
```

Output:
```
GET /path HTTP/1.1\r\n
Host: example.com\n
EOL_BARELF
User-Agent: test\r
EOL_BARECR
```

Input ending without CRLF:
```
GET /path HTTP/1.1\r\n
Host: example.com
```

Output:
```
GET /path HTTP/1.1\r\n
Host: example.com
EOL_EOF_NOCRLF
```

---

### Implementation Notes

- Process character by character to detect EOL sequences
- Handle edge cases like consecutive empty lines
- Skip flagging lines that are already flag lines
- Maintain exact original content while adding annotations

---

### Supersedes

- `LineJumpCatcher` from original implementation

---

### Good vs Bad Examples

Good (consistent CRLF across lines):
```
GET /ok HTTP/1.1\r\n
Host: example.com\r\n
\r\n
```

Bad (mixed EOLs → flagged):
```
GET /ok HTTP/1.1\r\n
Host: example.com\n
EOL_BARELF
User-Agent: Test\r\n
EOL_BARECR
\n
EOL_EOF_NOCRLF
```
