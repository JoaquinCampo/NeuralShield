### Longitudes con bucketing

Esta especificación define cómo medir y reportar métricas de tamaño relevantes (path y headers) usando buckets para estabilizar la representación y resaltar outliers sin exponer números exactos.

---

### Objetivos

- Proveer señales de tamaño robustas para el modelo (`PLEN`, `PMAX`, `HLEN`, `HCNT`).
- Evitar sobreajuste a valores exactos mediante bucketing determinista.
- Mantener un cálculo consistente tras las normalizaciones estructurales.

---

### Métricas definidas

- `PLEN:{len}@{bucket}`: longitud total del path canónico.
- `PMAX:{len}@{bucket}`: longitud máxima de un segmento de path.
- `HCNT:{n}`: cantidad de headers tras normalización (una línea por header, excluyendo `set-cookie` merge si aplica).
- `HLEN:{len}@{bucket}`: tamaño total de headers en bytes (sumando `name: value\r\n` por línea emitida en salida canónica), o solo valores si se define así. Recomendación: contar `name: value` sin CRLF.

---

### Buckets

- Rangos por defecto: `0-15`, `16-31`, `32-63`, `64-127`, `128-255`, `256-511`, `512-1023`, `>1023`.
- Reglas:
	- Inclusivos por cada frontera inferior y superior excepto el último (`>1023`).
	- El bucket se representa literalmente (p. ej., `64-127`).

---

### Reglas normativas

1. Orden de cálculo
	- `PLEN` y `PMAX` se calculan después de aplicar:
		- Percent-decode por segmentos (según su spec) y preservación de delimitadores.
		- Colapsación de `//` y `/.` y preservación de `..` (esta spec no resuelve `..`).
	- `HCNT` y `HLEN` se calculan después del unfold (obs-fold), colapsación de whitespace y merge de duplicados definido en headers.

2. Determinismo
	- El conteo debe ser idéntico en segundas pasadas del preprocesador.

3. Emisión
	- `PLEN` y `PMAX` se emiten junto al `P:` del path.
	- `HCNT` y `HLEN` se emiten al final del bloque de headers.

---

### Ejemplos

Path corto:
```
P:/a/b PLEN:4@0-15 PMAX:1@0-15
```

Path más largo:
```
P:/alpha/beta/gamma PLEN:19@16-31 PMAX:5@0-15
```

Headers:
```
HCNT:12 HLEN:512@512-1023
```

---

### Casos límite

- Path vacío o `/` → `PLEN:1@0-15`, `PMAX:0@0-15`.
- Sin headers → `HCNT:0 HLEN:0@0-15`.
- Valores extremos (>64 KiB) → bucket superior (`>1023`) o ampliar buckets por configuración.

---

### Requisitos de prueba

- Cálculo correcto tras colapsar `//` y `/.` y con `..` presentes.
- Idempotencia después de segunda pasada.
- Buckets correctos en fronteras (15,16,31,32, ...).
