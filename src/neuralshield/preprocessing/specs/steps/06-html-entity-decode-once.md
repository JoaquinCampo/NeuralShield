### 06 HTML Entity Decode Once

**Step Contract:**
- Input lines considered: `[URL]`, `[QUERY]` lines
- Transformations allowed: decode HTML entities exactly once
- Flags emitted: `HTMLENT`
- Dependencies: 05 (percent decode once)
- Idempotence: multiple applications produce same result

---

### Scope

Decode HTML entities exactly once in URL and query components:
- Decode named entities like `&lt;`, `&gt;`, `&amp;`
- Decode numeric entities like `&#64;`, `&#x40;`
- Detect and flag when entities are found and decoded
- Respect delimiter preservation policy

---

### Rules

1. **Evidence Preservation**: **ALWAYS preserve original encoding evidence**
2. **Entity Detection**: Identify HTML entities using regex pattern
3. **Minimal Decoding**: Decode only when absolutely necessary for downstream processing
4. **Mandatory Flagging**: Flag ALL entity presence to maintain attack evidence
5. **Security Priority**: Preserving evasion evidence is more important than normalization

---

### Entity Types Supported

**Named Entities**:
- `&lt;` → `<`
- `&gt;` → `>`
- `&amp;` → `&`
- `&quot;` → `"`
- `&apos;` → `'`
- And all standard HTML5 named entities

**Numeric Entities**:
- `&#64;` → `@` (decimal)
- `&#x40;` → `@` (hexadecimal)
- `&#x2f;` → `/` (hexadecimal)

---

### Flags Emitted

- `HTMLENT`: HTML entities were found and decoded

---

### Examples

Named entities in URL (preserved and flagged):
```
[URL] /path&lt;script&gt;
```

Output:
```
[URL] /path&lt;script&gt;
HTMLENT
```

Numeric entities in query (preserved and flagged):
```
[QUERY] param=test&#x2f;path
```

Output:
```
[QUERY] param=test&#x2f;path
HTMLENT
```

No entities (no flag):
```
[URL] /normal/path
```

Output:
```
[URL] /normal/path
```

Mixed entities (preserved and flagged):
```
[QUERY] data=&lt;tag&gt;&#64;domain.com
```

Output:
```
[QUERY] data=&lt;tag&gt;&#64;domain.com
HTMLENT
```

---

### Implementation Notes

- Uses Python's `html.unescape()` for complete entity support
- Regex pattern `&(?:[a-zA-Z][a-zA-Z0-9]*|#(?:\d+|x[0-9a-fA-F]+));` detects entities
- Only processes `[URL]` and `[QUERY]` lines, passes others unchanged
- Maintains idempotency by checking for actual changes

---

### Supersedes

- `html_entity_decoder` from `FlagsNormalizer`

---

### Good vs Bad Examples

Good (no entities present):
```
[URL] /safe/path
```

Bad (entities present → flagged):
```
[QUERY] a=&lt;script&gt;
HTMLENT
```
