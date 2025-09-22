### 05 HTML Entity Detect

**Step Contract:**

- Input lines considered: `[URL]`, `[QUERY]` lines
- Transformations allowed: detect HTML entities (preservation-focused)
- Flags emitted: `HTMLENT`
- Dependencies: 04 (percent decode once)
- Idempotence: multiple applications produce same result

---

### Scope

Detect HTML entities in URL and query components:

- Identify named entities like `&lt;`, `&gt;`, `&amp;`
- Identify numeric entities like `&#64;`, `&#x40;`
- Preserve entities unchanged for evidence preservation
- Flag all detected entities with `HTMLENT`

---

### Rules

1. **Evidence Preservation**: **ALWAYS preserve original encoding evidence**
2. **Entity Detection**: Identify HTML entities using regex pattern
3. **Preservation Priority**: Never decode entities - preserve original encoding evidence
4. **Mandatory Flagging**: Flag ALL entity presence to maintain attack evidence
5. **Security Priority**: Preserving evasion evidence is more important than normalization

---

### Entity Types Supported

**Named Entities** (detected and preserved):

- `&lt;`, `&gt;`, `&amp;`, `&quot;`, `&apos;`
- And all standard HTML5 named entities

**Numeric Entities** (detected and preserved):

- `&#64;` (decimal), `&#x40;` (hexadecimal), `&#x2f;` (hexadecimal)
- All valid decimal and hexadecimal numeric entities

---

### Flags Emitted

- `HTMLENT`: HTML entities were detected and preserved

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

- Regex pattern `&(?:[a-zA-Z][a-zA-Z0-9]*|#(?:\d+|x[0-9a-fA-F]+));` detects entities
- Only processes `[URL]` and `[QUERY]` lines, passes others unchanged
- Maintains idempotency - multiple runs produce identical output
- Evidence preservation: entities remain unchanged in output

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
