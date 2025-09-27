### Comprehensive single HTTP packet example (ordered plan)

Purpose
- **Single packet** that triggers every spec under `src/neuralshield/preprocessing/specs/steps/`.
- **Order**: structural ML/canonicalization steps first, then security/attack-prevention.
- We will apply the steps incrementally and update the packet after each step.

---

## Ordered steps

Structural (ML/canonicalization first)
1. `steps/ml-rfc/url-absoluta-desde-relativa.md` — unify to absolute URL without changing semantics
2. `steps/ml-rfc/separadores-parseo-robusto.md` — robust parsing of parameters and separators
3. `steps/ml-rfc/separadores-espacios.md` — normalize spacing around separators (no decoding yet)
4. `steps/ml-rfc/normalizar-flags.md` — normalize/binarize bare flags consistently
5. `steps/embed-security/ordenar-claves-preservar-multiplicidad.md` — stable key ordering while preserving multiplicity
6. `steps/embed-security/shape-longitud-redaccion.md` — represent shapes/lengths, redact extremes (for ML features)
7. `steps/ml-rfc/longitudes-bucketing.md` — bucketize lengths into discrete ranges
8. `steps/ml-rfc/normalizar-paquete.md` — finalize canonical normalized package layout

Security (attack-prevention and signaling)
9. `steps/embed-security/percent-decode-una-vez.md` — percent-decode exactly once (no double-decoding)
10. `steps/embed-security/query-decodificar-una-vez.md` — decode query once with strict rules
11. `steps/embed-security/no-traducir-plus-espacio.md` — do not translate `+` as space in unsafe contexts
12. `steps/embed-security/encodings-y-decodificaciones-raras.md` — handle and flag rare/ambiguous encodings
13. `steps/embed-security/colapsar-slash-dot-no-resolver-dotdot.md` — collapse `//` and `/.` but do not resolve `..`
14. `steps/embed-security/lineas-plegadas-obs-fold.md` — normalize/remove obsolete header folding
15. `steps/embed-security/headers-nombres-orden-duplicados.md` — canonicalize header names/order; manage duplicates
16. `steps/embed-security/headers-valores-shape-aware.md` — shape-aware normalization of header values
17. `steps/embed-security/caracteres-peligrosos-script-mixing.md` — detect dangerous chars/script mixing (Unicode)
18. `steps/embed-security/flags-por-rarezas.md` — set rarity-derived security flags

Notes
- **Structural pass avoids decoding** content; it focuses on layout/canonical form for ML comparability.
- **Security pass** applies decoding, ambiguity resolution, detection, and flagging.

---

## Baseline raw packet (single example)

This single raw HTTP/1.1 request intentionally includes edge cases to trigger every step.

```http
GET /A/./B//C/%2e%2e/D%2F;param?q=foo&x=1&x=02; y ;a=1; b=2&bare&nul=%00&enc=a%2520b&plus=1+2&plus2=%2B&semi=a;b&mix=1%3B2&z=%25 HTTP/1.1
Host: EXAMPLE.com:80
User-Agent: TestClient/1.0
X-Obs: first line
\tcontinued part via obs-fold
x-dup-header:  one
X-Dup-Header:   two
X-Folded: alpha,
  beta,
   gamma
X-Mixed-Script: Latin a + Cyrillic а + Greek α
X-Encoded: %25%32%42
Cookie: a=1; b=2; c=%2B; d=1+2
X-Forwarded-For: 1.2.3.4
X-Forwarded-For: 5.6.7.8

```

Included triggers (at a glance)
- **Relative URL** to be made absolute via `Host` (1)
- **Mixed separators** `&` and `;`, with spaces around separators (2, 3)
- **Bare flag** `bare` and duplicate key `x` (4, 5)
- **Shapes/length extremes** and values for bucketing (6, 7)
- **Final canonical packaging** target (8)
- **Double-encoded** `a%2520b`, encoded percent `%25`, encoded plus `%2B` (9, 10, 12)
- **`+` present** in query and cookie for plus/space rule (11)
- **Path with `//`, `/.`, encoded slash, and `..`** (13)
- **Obsolete folding** in `X-Obs`/`X-Folded` and header duplicates/case (14, 15)
- **Header value shape quirks** and mixed casing/spacing (16)
- **Unicode script mixing** in `X-Mixed-Script` (17)
- **Rarity conditions** like null `\x00`, odd encodings (18)

---

## Stepwise record (to be filled as we confirm each step)

- After Step 1 (`url-absoluta-desde-relativa.md`):
  - Packet updated to absolute URL form; no decoding applied.
- After Step 2 (`separadores-parseo-robusto.md`):
  - Parameters parsed robustly; representation remains undecoded.
- After Step 3 (`separadores-espacios.md`):
  - Spacing around separators normalized.
- After Step 4 (`normalizar-flags.md`):
  - Bare flags normalized/binarized.
- After Step 5 (`ordenar-claves-preservar-multiplicidad.md`):
  - Stable ordering with multiplicity preserved.
- After Step 6 (`shape-longitud-redaccion.md`):
  - Shapes/lengths represented; extreme content redacted where needed.
- After Step 7 (`longitudes-bucketing.md`):
  - Length features bucketized.
- After Step 8 (`normalizar-paquete.md`):
  - Canonical package structure finalized.
- After Step 9–18 (security):
  - Decoding rules applied once; rare encodings handled; path collapse without `..` resolution; header folding removed and duplicates canonicalized; dangerous characters flagged; rarity flags set.

We will update this section incrementally after your confirmation at each step.



