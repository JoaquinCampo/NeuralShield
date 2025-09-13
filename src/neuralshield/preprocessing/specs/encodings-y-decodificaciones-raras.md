### Encodings y decodificaciones “raras”

Esta especificación aborda casos anómalos de codificación y decodificación que suelen usarse para evasión de filtros: UTF-8 overlong/invalid, múltiples niveles de percent-encoding, combinaciones con entidades HTML, y otros.

---

### Objetivos

- Detectar y señalar codificaciones anómalas sin intentar “arreglarlas” completamente.
- Mantener la integridad del análisis canónico y la idempotencia.

---

### Casos cubiertos y flags

1. UTF-8 inválido / overlong → `BADUTF8`
	- Al decodificar bytes a Unicode, usar modo tolerante ("replace") para no perder estructura.
	- Si se detectan secuencias inválidas u overlong (ej., `C0 AF` para `/`), emitir `BADUTF8`.

2. Múltiple percent-encoding → `DOUBLEPCT` y/o `MULTIENC:<k>`
	- Tras una pasada única de percent-decode, si quedan `%hh` válidos, emitir `DOUBLEPCT`.
	- Si se identifica doble/triple encoding localizado (por clave o campo), opcionalmente `MULTIENC:<k>`.

3. HTML entities → `HTMLENT`
	- Si aparecen entidades HTML en path/query (p. ej., `&#x2f;`, `&lt;`), decodificar una sola vez y emitir `HTMLENT`.
	- Si tras decodificar entidades se producirían delimitadores estructurales (`/`, `&`, `=`), aplicar la misma política de preservación que en percent-decode (no romper estructura).

4. Mixtura de encodings
	- Combinar percent-encoding y entidades HTML en un mismo valor es típico de ofuscación. Si tras aplicar las pasadas únicas reglamentarias subsisten señales (residuos `%hh` válidos o entidades no decodificadas), se emiten las flags correspondientes (`DOUBLEPCT`, `HTMLENT`).

---

### Reglas normativas

1. Orden de decodificación
	- Primero percent-decode (pasada única), luego entidades HTML (pasada única) en path y query.
	- No reintentar decodificaciones adicionales si quedan residuos; señalar con flags.

2. Manejo de bytes → Unicode
	- Decodificar a Unicode con UTF-8 en modo tolerante ("replace"), sin descartar bytes.
	- Emitir `BADUTF8` si ocurre cualquier sustitución.

3. Preservación de delimitadores
	- Si una decodificación (percent o HTML entity) convertiría a delimitador estructural, aplicar política de preservación definida en sus respectivas specs (p. ej., mantener `%2F` en path y señalar `PCTSLASH`).

4. Idempotencia
	- Una segunda pasada no debe aplicar decodificaciones adicionales ni cambiar flags.

---

### Ejemplos

UTF-8 overlong:
```
GET /%C0%AFetc/passwd HTTP/1.1
```
Salida (fragmento):
```
P:/%C0%AFetc/passwd
FLAGS:[BADUTF8 DOUBLEPCT]
```

HTML entity en path:
```
/a&#x2f;b
```
Salida (fragmento):
```
P:/a/b
FLAGS:[HTMLENT]
```

Mixto percent + HTML entity con residuo:
```
?x=%2526y%3D1&#x26;z=2
```
Salida (fragmento):
```
Q:2 KEYS:x,z
FLAGS:[DOUBLEPCT HTMLENT]
```

---

### Casos límite

- Entidades HTML malformadas → ignorar y no emitir `HTMLENT` a menos que se produzca una sustitución válida.
- Percent-encoding inválido (`%2G`) → no decodificar, no contar para `DOUBLEPCT`.
- Secuencias mixtas complejas (percent → entidad → percent) → respetar una pasada única por tipo; no encadenar indefinidamente.

---

### Requisitos de prueba

- `BADUTF8` con overlong e inválidos.
- `HTMLENT` cuando haya sustituciones reales en path/query.
- `DOUBLEPCT` tras residuo `%hh` válido.
- Idempotencia de la secuencia de decodificación.
