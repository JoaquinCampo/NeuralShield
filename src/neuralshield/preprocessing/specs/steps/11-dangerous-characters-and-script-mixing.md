### 12 Dangerous Characters and Script Mixing

**Step Contract:**

- Input lines considered: `[URL]`, `[QUERY]`, `[HEADER]` lines
- Transformations allowed: none (detection only)
- Flags emitted: `ANGLE`, `QUOTE`, `SEMICOLON`, `PAREN`, `BRACE`, `PIPE`, `BACKSLASH`, `SPACE`, `NUL`, `QNUL`, `MIXEDSCRIPT`
- Dependencies: 06, 07, 10 (after decoding and normalization)
- Idempotence: multiple applications produce same result

---

### Scope

Detect dangerous characters and script mixing useful for attack detection:

- Identify characters commonly used in XSS, SQLi, RCE, and traversal attacks
- Detect homograph attacks through script mixing analysis
- Apply context-specific detection rules
- Flag both literal and percent-encoded forms

---

### Dangerous Character Categories

**Injection Characters**:

- `ANGLE`: `<`, `>` (HTML/XML tags)
- `QUOTE`: `'`, `"` (string escaping/injection)
- `SEMICOLON`: `;` (command separation/SQL injection)
- `PAREN`: `(`, `)` (function calls/SQL injection)
- `BRACE`: `{`, `}` (template injection/code blocks)

**Command Characters**:

- `PIPE`: `|` (command chaining/shell injection)
- `BACKSLASH`: `\` (path traversal/escaping)
- `SPACE`: ` ` (suspicious in URL paths)
- `NUL`: `\x00` (null byte injection/string termination)

**Special Flags**:

- `QNUL`: Null byte specifically in query values (emitted with `NUL`)
- `MIXEDSCRIPT`: Mixed alphabets (homograph attacks)

---

### Detection Patterns

Characters detected in both literal and percent-encoded forms:

| Character | Literal | Encoded     |
| --------- | ------- | ----------- | --- | ----- |
| `<` `>`   | `<` `>` | `%3C` `%3E` |
| `'` `"`   | `'` `"` | `%27` `%22` |
| `;`       | `;`     | `%3B`       |
| `(` `)`   | `(` `)` | `%28` `%29` |
| `{` `}`   | `{` `}` | `%7B` `%7D` |
| `         | `       | `           | `   | `%7C` |
| `\`       | `\`     | `%5C`       |
| ` `       | ` `     | `%20`       |
| `\x00`    | `\x00`  | `%00`       |

---

### Context-Specific Rules

**URL Context**:

- All dangerous characters flagged
- `SPACE` flag emitted (suspicious in paths)
- `SEMICOLON` flagged (unusual in paths)

**Query Context**:

- All dangerous characters flagged
- `QNUL` emitted along with `NUL` for null bytes
- `SEMICOLON` may be legitimate (query separator)

**Header Context**:

- Most dangerous characters flagged
- `SEMICOLON` NOT flagged (legitimate in cookies, Accept-Language)
- Script mixing checked in header values only

---

### Script Mixing Detection

**Target Scripts**:

- **Latin**: A-Z, a-z (0x0041-0x007A)
- **Cyrillic**: А-я (0x0400-0x04FF)
- **Greek**: Α-ω (0x0370-0x03FF)

**Detection Logic**:

1. Analyze alphabetic characters after percent-decoding
2. Identify script for each character using Unicode ranges
3. Flag if ≥2 different scripts found in same token
4. Ignore punctuation and Common/Inherited scripts

---

### Notes on `[QUERY]` values

- When the query parser emits `[QUERY] key=value` lines, detection should analyze the effective decoded token prior to any optional redaction. Preserve dangerous-character evidence via flags while respecting redaction policies.
- For `[QUERY]`, match detection on the effective value (decoded once by prior steps). If values are redacted (e.g., `<shape:len>` or `<SECRET:shape:len>`), do not reveal raw values; append flags only.

### Examples

URL examples

Input:

```
[URL] /path<script>alert(1)</script>
```

Output:

```
[URL] /path<script>alert(1)</script>
ANGLE PAREN
```

Percent-encoded and structural in URL:

```
[URL] /path%20with;param=1
```

Output:

```
[URL] /path%20with;param=1
SPACE SEMICOLON
```

Backslash in URL (traversal-style):

```
[URL] /..\\windows\\system32
```

Output:

```
[URL] /..\\windows\\system32
BACKSLASH
```

Script mixing in URL:

```
[URL] /раypal.com/login
```

Output:

```
[URL] /раypal.com/login
MIXEDSCRIPT
```

Query examples

Null byte in value:

```
[QUERY] file=../../../etc/passwd%00.jpg
```

Output:

```
[QUERY] file=../../../etc/passwd%00.jpg NUL QNUL
```

Angle brackets (encoded form preserved from prior steps):

```
[QUERY] q=%3Cscript%3Ealert(1)%3C/script%3E
```

Output:

```
[QUERY] q=%3Cscript%3Ealert(1)%3C/script%3E ANGLE PAREN
```

Angle brackets (literal form):

```
[QUERY] q=<script>alert(1)</script>
```

Output:

```
[QUERY] q=<script>alert(1)</script> ANGLE PAREN
```

Command-style payload with pipe and space:

```
[QUERY] cmd=cat /etc/passwd | grep root
```

Output:

```
[QUERY] cmd=cat /etc/passwd | grep root PIPE SPACE
```

Header examples

Quotes and parentheses:

```
[HEADER] etag: "abc-123"
[HEADER] user-agent: Test/1.0 (X11; Linux)
```

Output:

```
[HEADER] etag: "abc-123"
QUOTE
[HEADER] user-agent: Test/1.0 (X11; Linux)
PAREN
```

Legitimate semicolon (not flagged in headers):

```
[HEADER] cookie: session=abc123; theme=dark
```

Output:

```
[HEADER] cookie: session=abc123; theme=dark
```

Null byte in header value:

```
[HEADER] x-bin: test%00payload
```

Output:

```
[HEADER] x-bin: test%00payload
NUL
```

Mixed script in header value:

```
[HEADER] x-mixed: раypal.com
```

Output:

```
[HEADER] x-mixed: раypal.com
MIXEDSCRIPT
```

---

### Implementation Details

1. **Pattern Matching**: Regex patterns for each character type
2. **Context Awareness**: Different rules per component type
3. **Encoding Detection**: Handles both literal and percent-encoded forms
4. **Script Analysis**: Unicode category and range checking
5. **Flag Aggregation**: Collects all applicable flags per line

---

### Security Applications

**Attack Detection**:

- **XSS**: `ANGLE`, `QUOTE`, `PAREN`
- **SQLi**: `QUOTE`, `SEMICOLON`, `PAREN`
- **Command Injection**: `PIPE`, `SEMICOLON`, `BACKSLASH`
- **Path Traversal**: `BACKSLASH`, `NUL`
- **Homograph**: `MIXEDSCRIPT`

**Evasion Detection**:

- Percent-encoded attack payloads
- Unicode normalization attacks
- Mixed script domain spoofing

---

### Supersedes

- Original `DangerousCharacterDetector` with enhanced context awareness

---

### Good vs Bad Examples

Good (no dangerous characters, single script):

```
[URL] /files/readme.txt
[QUERY] q=search
[HEADER] accept: text/html
```

Bad (dangerous characters present → flagged):

```
[URL] /path<script>()
ANGLE PAREN
```

Bad (script mixing → flagged):

```
[URL] /раypal/login
MIXEDSCRIPT
```
