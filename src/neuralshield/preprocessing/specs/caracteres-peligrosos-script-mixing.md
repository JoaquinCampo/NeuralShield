### Caracteres peligrosos + Script mixing

Esta especificación define la detección y marcado de caracteres peligrosos (útiles para XSS/SQLi/RCE/traversal) y la detección de mezcla de alfabetos (script mixing) que habilitan ataques por homógrafos.

---

### Objetivos

- Señalar presencia de caracteres y patrones con alta correlación de riesgo.
- Dar al modelo señales explícitas sin eliminar información estructural.
- Mantener política consistente entre path, query y headers.

---

### Conjunto de caracteres peligrosos (base)

- `ANGLE`: `<` o `>`
- `QUOTE`: `'` o `"`
- `SEMICOLON`: `;`
- `PAREN`: `(` o `)`
- `BRACE`: `{` o `}`
- `PIPE`: `|`
- `BACKSLASH`: `\\`
- `SPACE`: espacio dentro de segmentos sensibles (p. ej., en path tras decode)
- `NUL`: `\x00` (también puede aparecer como `%00` → ver `QNUL` en query)
- `PERCENT_IN_PATH`: `%` crudo en path tras decode (opcional; si se conserva `%`)

Estas flags se aplican por componente y se agregan al roll-up global `FLAGS:[...]`.

---

### Política por componente

- Path
	- Escanear el path canónico post-normalización (colapsado de `//`/`/.`, preservando `..`).
	- Emitir flags según caracteres detectados. `SPACE` en path se considera sospechoso.

- Query
	- Escanear claves y valores tras percent-decode por par.
	- Para NUL (`\x00`) en valores, emitir `QNUL` además de `NUL` global.

- Headers
	- Escanear valores normalizados (desplegados y con whitespace colapsado) salvo aquellos marcados como secretos redaccionados.

---

### Script mixing (MIXEDSCRIPT)

- Definición: presencia de caracteres pertenecientes a ≥2 escrituras entre Latín, Cirílico, Griego (extensible), en un mismo token relevante (host, segmento de path, clave/valor de query, header sensible como `host` o `referer`).
- Detección:
	- Calcular el conjunto de scripts dominantes por token (usando propiedades Unicode de cada codepoint).
	- Si |scripts| ≥ 2 y no es una combinación permitida (p. ej., Latin + Common), emitir `MIXEDSCRIPT`.
- Notas:
	- No emitir por combinación de Latin con signos de puntuación o dígitos (clase `Common`/`Inherited`).
	- Emitir para ejemplos típicos de homógrafos: `раypal.com` (p cirílica + resto latín).

---

### Interacciones

- Unicode (FULLWIDTH/CONTROL): estas flags se emiten en su spec; aquí solo complementamos con `MIXEDSCRIPT`.
- Percent-decode: el análisis se hace tras la pasada única correspondiente; entidades HTML se tratan tras su decode.

---

### Ejemplos

Path con `<script>`:
```
/a/<script>/b
```
Salida (fragmento):
```
P:/a/<script>/b
FLAGS:[ANGLE]
```

Query con NUL y comillas:
```
?name=O%27Brien%00
```
Salida (fragmento):
```
Q:1 KEYS:name
FLAGS:[QUOTE QNUL NUL]
```

Host con homógrafos:
```
U:http://раypal.com/
```
Salida (fragmento):
```
FLAGS:[MIXEDSCRIPT]
```

Backslash en path:
```
/a\b/c
```
Salida (fragmento):
```
P:/a\b/c
FLAGS:[BACKSLASH]
```


---

### Casos límite

- Valores redactados `<SECRET:...>`: no inspeccionar caracteres internos.
- Tokens con solo dígitos y puntuación: no `MIXEDSCRIPT`.
- Uso de `FULLWIDTH` junto a script mixing: pueden coexistir `FULLWIDTH` y `MIXEDSCRIPT`.

---

### Requisitos de prueba

- Detección de cada flag individual (`ANGLE`, `QUOTE`, etc.).
- `MIXEDSCRIPT` en dominios mixtos y en claves/valores de query.
- Coexistencia con `HTMLENT` y `DOUBLEPCT` (decode previo no rompe detección).
