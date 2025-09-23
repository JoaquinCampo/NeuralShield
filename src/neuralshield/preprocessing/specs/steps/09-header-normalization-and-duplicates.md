### 10 · Header Normalization & Duplicates (BERT-Optimized Emission)

**Step Contract:**

- Input lines considered: `[HEADER]` lines (after Step 09: obs-fold unfold)
- Transformations allowed: normalize names/values, validate format, handle duplicates/merges
- Flag emission: per-header inline via F: suffix; global/structural at end
- Dependencies: Step 09 (header unfold obs-fold)
- Idempotence: multiple applications produce the same output

---

### Scope & Intent

Normalize HTTP request headers and surface anomalies for downstream ML (BERT) using a single canonical string:

- Lowercase and canonicalize header names/values
- Validate names per RFC 9110 token rules
- Merge duplicates only where HTTP semantics allow
- Detect hop-by-hop headers in requests
- Emit headers in canonical order
- Attach header-specific flags inline (on the same line)
- Emit request-level aggregates/flags at the end

Why inline? For BERT, local flags next to the triggering header improve attention alignment while keeping a compact global summary.

---

### Output Grammar (Canonical)

**Header line (no flags):**

```
[HEADER] <lowercased-name>: <normalized-value>
```

**Header line (with flags):**

```
[HEADER] <lowercased-name>: <normalized-value> FLAG1,FLAG2
```

**End-of-request aggregates (counts/stats):**

```
[HAGG] h_count=<int> dup_names=<int> hopbyhop=<0|1> bad_names=<int> total_bytes=<int>
```

**End-of-request global flags (optional if none):**

```
[HGF] FLAG_A,FLAG_B
```

**Formatting rules:**

- One space before flags (i.e., value followed by single space then flags).
- No spaces inside the flag CSV lists.
- Stable, deterministic ordering everywhere.

---

### Normalization Rules

- **Name lowercasing**: all header names ➜ lowercase.
- **Whitespace trimming**: strip leading/trailing spaces in names/values.
- **Token validation (RFC 9110 token)**:
  - **Allowed**: `!#$%&'*+-.^_`|~` and alphanumerics.
  - **Forbidden**: controls, separators, spaces.
  - **Policy**: underscores may be flagged (BADHDRNAME) if present.
- **Canonical ordering**: sort headers by name except set-cookie which preserves arrival order.
- **Transformation evidence**: if normalization changed any header name case/spacing, add HDRNORM (global).
- **ASCII only** in emitted meta-syntax ([HEADER], [HAGG], [HGF]).
- **Determinism**: identical inputs after Step 09 produce identical outputs.

---

### Duplicate Handling

**Comma-mergeable** (join duplicates with `,`):

- `accept`, `accept-encoding`, `accept-language`
- `cache-control`, `pragma`, `link`
- `www-authenticate` (syntax-aware)

**Never merge**:

- `set-cookie` (always keep separate lines in arrival order)

**Other headers**:

- If duplicated and not in mergeable set → keep separate lines and flag DUPHDR inline on each affected line or on the first emitted instance (see policy below).

**Merge policy details**:

- **When merging**: Emit a single combined line with `DUPHDR,HDRMERGE`. Preserve comma/space canonicalization: `", "` between values.
- **When not merging duplicated names**: Emit multiple lines; attach `DUPHDR` to each or just first line (choose one policy and keep it consistent; we recommend first line only to reduce token bloat).

---

### Hop-by-Hop Headers (request context)

Flag presence in requests inline on the offending header: `HOPBYHOP`.

Common hop-by-hop names: `connection`, `te`, `upgrade`, `trailer`.

---

### Flags

**Per-header (inline)**

- `BADHDRNAME` — invalid token characters (or policy-violating underscore)
- `DUPHDR` — duplicate name encountered
- `HDRMERGE` — values were merged (comma-join) for this header
- `HOPBYHOP` — hop-by-hop header in request context

Do not repeat the header name inside the flag. The name is already on the line.

**Global (end of request)**

- `HDRNORM` — at least one header name was case/format-normalized

Additional structural flags are allowed (e.g., `HDRSET_ANOM`, size buckets).

---

### Emission Order

1. Collect and normalize headers (tracking per-header flags and global stats).
2. Emit set-cookie in arrival order; all other headers sorted by name.
3. Append flags with single space only if that line has any flags.
4. Emit `[HAGG]` … then `[HGF]` … (if any global flags).

**Flag ordering:**
Sort by severity, then alphabetically for determinism.

Recommended severity: `HOPBYHOP > BADHDRNAME > DUPHDR > HDRMERGE`.

---

### Examples

**Basic normalization:**

```
[HEADER] Host: Example.com
[HEADER] User-Agent: Mozilla/5.0
```

```
[HEADER] host: Example.com
[HEADER] user-agent: Mozilla/5.0
[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=40
```

---

**Duplicate mergeable headers:**

```
[HEADER] Accept: text/html
[HEADER] Accept: application/json
```

```
[HEADER] accept: text/html, application/json DUPHDR,HDRMERGE
[HAGG] h_count=1 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=52
[HGF] HDRNORM
```

---

**Invalid header name:**

```
[HEADER] X_Custom: value
```

```
[HEADER] x_custom: value BADHDRNAME
[HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=1 total_bytes=17
[HGF] HDRNORM
```

---

**Hop-by-hop in request:**

```
[HEADER] Connection: keep-alive
[HEADER] Host: example.com
```

```
[HEADER] connection: keep-alive HOPBYHOP
[HEADER] host: example.com
[HAGG] h_count=2 dup_names=0 hopbyhop=1 bad_names=0 total_bytes=39
[HGF] HDRNORM
```

---

**Set-cookie preservation:**

```
[HEADER] Set-Cookie: session=abc123
[HEADER] Set-Cookie: theme=dark
```

```
[HEADER] set-cookie: session=abc123
[HEADER] set-cookie: theme=dark
[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=44
[HGF] HDRNORM
```

---

**Non-mergeable duplicates:**

```
[HEADER] X-Trace: a
[HEADER] X-Trace: b
```

```
[HEADER] x-trace: a DUPHDR
[HEADER] x-trace: b
[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=26
[HGF] HDRNORM
```

---

### Implementation Notes

- **Two-pass**: collect → normalize/merge → emit.
- **Merging**: only for the allowlist; join with `", "`.
- **Ordering**: set-cookie in arrival order; others sorted by name.
- **Deterministic sorting**: names, flag lists, and merge order stabilized.

**Counters for [HAGG]:**

- `h_count`: emitted header lines count
- `dup_names`: # of distinct names seen with duplicates (post-normalization)
- `hopbyhop`: 1 if any hop-by-hop seen, else 0
- `bad_names`: # of distinct invalid names encountered
- `total_bytes`: length of concatenated emitted header values (optional; bucket later)

---

### Edge Cases

- **Empty values**: allowed (emit as empty after trimming).
- **Leading/trailing whitespace**: in values removed.
- **Mixed case names**: normalize to lowercase (HDRNORM global set if any changed).
- **Unknown headers**: allowed; if duplicated and not mergeable → DUPHDR policy applies.
- **Underscore policy**: if flagged, do not alter the underscore; just add BADHDRNAME.

---

### Security Considerations

- **Duplicates**: can induce parser ambiguity; surfaced via DUPHDR/HDRMERGE.
- **Invalid names**: often indicate injection attempts; surfaced via BADHDRNAME.
- **Hop-by-hop**: in requests may imply proxy manipulation; surfaced via HOPBYHOP.

---

### Backward Compatibility

Prior format ("flags only at end") can be supported via a legacy mode:

- Inline F: disabled; per-header flags accumulated and written under [HGF].
- Default mode for ML (this spec): inline per-header F: + global summary at end.

---

### Supersedes

- New step implementing `headers-nombres-orden-duplicados.md` specification
