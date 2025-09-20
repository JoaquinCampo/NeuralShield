### Caracteres peligrosos + Script mixing

Esta especificación define la detección y marcado de caracteres peligrosos (útiles para XSS/SQLi/RCE/traversal) y la detección de mezcla de alfabetos (script mixing) que habilitan ataques por homógrafos.

---

### Objetivos

- Señalar presencia de caracteres y patrones con alta correlación de riesgo.
- Dar al modelo señales explícitas sin eliminar información estructural.
- Mantener política consistente entre URL, query y headers.

---

### Conjunto de caracteres peligrosos (base)

- `ANGLE`: `<` o `>` (incluye formas percent-encoded `%3C`, `%3E`)
- `QUOTE`: `'` o `"` (incluye formas percent-encoded `%27`, `%22`)
- `SEMICOLON`: `;` (incluye forma percent-encoded `%3B`)
- `PAREN`: `(` o `)` (incluye formas percent-encoded `%28`, `%29`)
- `BRACE`: `{` o `}` (incluye formas percent-encoded `%7B`, `%7D`)
- `PIPE`: `|` (incluye forma percent-encoded `%7C`)
- `BACKSLASH`: `\` (incluye forma percent-encoded `%5C`)
- `SPACE`: espacio en URLs (incluye forma percent-encoded `%20`)
- `NUL`: `\x00` (incluye forma percent-encoded `%00`)
- `QNUL`: flag específica para NUL en valores de query (emitida junto con `NUL`)

Las flags se emiten por línea de contenido y aparecen inmediatamente después del contenido que las generó.

---

### Política por componente

- URL (`[URL]` lines)

  - Escanear el contenido de la URL post-normalización.
  - Emitir flags según caracteres detectados. `SPACE` en URL se considera sospechoso.
  - Aplicar detección de script mixing.

- Query (`[QUERY]` lines)

  - Escanear claves y valores por separado.
  - Para NUL (`\x00`) en valores, emitir `QNUL` además de `NUL`.
  - Aplicar detección de script mixing.

- Headers (`[HEADER]` lines)
  - Escanear valores normalizados salvo aquellos marcados como secretos redaccionados.
  - `SEMICOLON` no se considera peligroso en headers (legítimo en Accept-Language, Cookie, etc.).
  - Aplicar detección de script mixing para headers sensibles.

---

### Script mixing (MIXEDSCRIPT)

- Definición: presencia de caracteres pertenecientes a ≥2 escrituras entre Latín, Cirílico, Griego en un mismo token relevante.
- Detección:
  - Analizar caracteres alfabéticos en el contenido tras percent-decode.
  - Detectar scripts usando rangos Unicode: Latin (0x0041-0x007A), Cyrillic (0x0400-0x04FF), Greek (0x0370-0x03FF).
  - Si se detectan ≥2 scripts diferentes, emitir `MIXEDSCRIPT`.
- Notas:
  - Ignorar caracteres de puntuación, números y símbolos (clase `Common`/`Inherited`).
  - Emitir para ejemplos típicos de homógrafos: `раypal.com` (p cirílica + resto latín).
  - Se aplica a URLs, queries y headers.

---

### Interacciones

- Unicode (FULLWIDTH/CONTROL): estas flags se emiten por otros procesadores; aquí solo complementamos con `MIXEDSCRIPT`.
- Percent-decode: el análisis se hace tras decode automático; entidades HTML se tratan tras su decode.

---

### Ejemplos

URL con espacios:

```
[URL] /path%20with%20spaces/folder%20name/file.exe
SPACE
```

Query con NUL:

```
[QUERY] param=%00value
NUL QNUL
```

URL con script mixing:

```
[URL] /раypal.com/login
MIXEDSCRIPT
```

Query con script mixing:

```
[QUERY] user=аdmin
MIXEDSCRIPT
```

Header con script mixing:

```
[HEADER] Host: раypal.com
MIXEDSCRIPT
```

---

### Casos límite

- Valores redactados `<SECRET:...>`: no inspeccionar caracteres internos.
- Tokens con solo dígitos y puntuación: no emitir `MIXEDSCRIPT`.
- Contenido con longitud < 2 caracteres: no emitir `MIXEDSCRIPT`.
- Headers con semicolon: no emitir `SEMICOLON` (legítimo en headers).
- Espacios fuera de URLs: no emitir `SPACE` (solo en URLs).
- Coexistencia con otras flags: pueden aparecer múltiples flags por línea.

---

### Formato de salida

Las flags aparecen en líneas separadas inmediatamente después del contenido que las generó:

```
[URL] /content/with/flags
FLAG1 FLAG2 FLAG3
[QUERY] param=value
[QUERY] dangerous=<script>
ANGLE
```

---

### Requisitos de prueba

- Detección de cada flag individual (`ANGLE`, `QUOTE`, `SEMICOLON`, `PAREN`, `BRACE`, `PIPE`, `BACKSLASH`, `SPACE`, `NUL`).
- `QNUL` junto con `NUL` en valores de query.
- `MIXEDSCRIPT` en URLs, queries y headers con mezcla de scripts.
- Detección tanto de caracteres literales como percent-encoded.
- Respeto a la política específica por componente (URLs, queries, headers).
