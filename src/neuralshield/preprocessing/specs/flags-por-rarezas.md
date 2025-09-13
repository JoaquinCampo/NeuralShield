### Flags por rarezas - (HIGH)

Esta especificación centraliza las flags de rarezas/anomalías de alta prioridad, definiendo sus disparadores, alcance (path/query/headers) y reglas de emisión. Todas las flags listadas aquí deben agregarse al roll-up global `FLAGS:[...]` sin duplicados y ordenadas alfabéticamente.

---

### Principios

- No corregir silenciosamente: marcar y preservar la evidencia cuando sea posible.
- Idempotencia: un input que ya emitió una flag no debe generar nuevas flags al re-procesarse.
- Determinismo: mismas condiciones, mismas flags.

---

### Taxonomía principal

- Codificación / Decodificación
	- `DOUBLEPCT`: quedan `%hh` válidos tras una pasada de percent-decode.
	- `MULTIENC:<k>`: evidencia localizada de doble/triple encoding (por clave `<k>` o contexto). Opcional.
	- `HTMLENT`: se decodificaron entidades HTML en path o query.
	- `BADUTF8`: se detectaron secuencias inválidas al decodificar a UTF-8.
	- `PCTSLASH`: el path mantiene `%2F` tras la pasada única.
	- `PCTBACKSLASH`: el path mantiene `%5C` tras la pasada única.

- Estructura de Path
	- `MULTIPLESLASH`: se colapsaron `//`.
	- `DOTDOT`: presencia de `..` sin resolver.
	- `DOTDOTN:{n}`: opcional, cantidad de `..` consecutivos.
	- `HOME`: path canónico es `/`.

- Whitespace y líneas
	- `OBSFOLD`: se desplegaron líneas de continuación en headers.
	- `WSPAD`: se colapsaron tabs/espacios redundantes.
	- `BADCRLF`: CR/LF incrustados en valores de header.

- Unicode
	- `FULLWIDTH`: formas de ancho completo en campos estructurales.
	- `MIXEDSCRIPT`: mezcla de alfabetos relevantes (Latín, Cirílico, Griego) en un token.
	- `CONTROL`: caracteres de control presentes.
	- `IDNA`: host con etiquetas Unicode convertido a punycode.

- Query estructura/valores
	- `QREPEAT:<k>`: clave `<k>` aparece 2+ veces.
	- `QEMPTYVAL`: par con `=` y valor vacío.
	- `QBARE`: token sin `=`.
	- `QNUL`: valor contiene NUL tras decode.
	- `QNONASCII`: clave o valor con non-ASCII.
	- `QARRAY:<k>`: clave con sufijo `[]`.
	- `QSEMISEP`: `;` aceptado como separador por heurística.
	- `QRAWSEMI`: `;` presente pero no reconocido como separador.
	- `QLONG`: valor supera umbral configurado.

- Caracteres peligrosos
	- `ANGLE`, `QUOTE`, `SEMICOLON`, `PAREN`, `BRACE`, `PIPE`, `BACKSLASH`, `SPACE`, `NUL`.

- Headers
	- `DUPHDR:<name>`: header duplicado (lista separada por comas tras merge cuando aplica).
	- `BADHDRNAME:<name>`: nombre de header inválido.
	- `HOPBYHOP:<name>`: header hop-by-hop inesperado en request.
	- `UNKHDR:<name>`: header no en lista blanca (opcional).
	- `HOSTMISMATCH`: absolute-form difiere de `Host`.
	- `AUTHBEARER`/`AUTHBASIC`: credenciales en `authorization` (gestión en spec de headers).
	- `XFF`: `x-forwarded-for`/`forwarded` con IPs parseadas.
	- `COOKIE:{n}`: cantidad de cookies (resumen). Opcional.

---

### Reglas de emisión

- Unicidad: cada flag se emite como máximo una vez en `FLAGS:[...]` por request.
- Alcance local vs global: los módulos pueden incluir flags locales en su línea (`P:`, `Q:`, `H:`); el agregador siempre añade todas al roll-up `FLAGS:`.
- Orden: alfabético en el roll-up.

---

### Ejemplos de agregación

```
P:/a/b/../c%2Ejsp [DOTDOT] [DOUBLEPCT]
Q:3 KEYS:k,k,token [QREPEAT:k] [QSEMISEP]
H:x-test=valor1 valor2 [OBSFOLD]
FLAGS:[DOUBLEPCT DOTDOT OBSFOLD QREPEAT:k QSEMISEP]
```

---

### Casos límite

- Múltiples ocurrencias del mismo fenómeno → una única flag (p. ej., varias claves repetidas del mismo nombre → solo `QREPEAT:<k>` una vez).
- Flags con parámetros (`:<k>`, `:{n}`) deben mantener su parámetro canónico (case original de clave, o lowercased de forma determinista, según decisión global; recomendación: mantener tal cual aparece primero).

---

### Requisitos de prueba

- Dedupe correcto en roll-up.
- Orden alfabético estable.
- Convivencia de flags de categorías distintas.
