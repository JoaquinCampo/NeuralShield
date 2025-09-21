### Headers: nombres, orden y duplicados

Esta especificación define la normalización de nombres de headers, el orden de emisión, el tratamiento de duplicados, y las flags asociadas a anomalías.

---

### Objetivos

- Producir una representación canónica, determinista e idempotente del bloque de headers.
- Señalar duplicados, nombres inválidos y hop-by-hop inesperados.

---

### Normalización de nombres

- Lowercase de nombres de header.
- Trim de espacios alrededor del nombre y del valor.
- Validar nombre contra token de RFC 9110: caracteres permitidos `!#$%&'*+-.^_`|~` y alfanum.
- Si el nombre contiene caracteres prohibidos o underscores (según política), emitir `BADHDRNAME:<name>`.

---

### Orden y emisión

- Ordenar headers por nombre ascendente (binario) para emisión canónica.
- Excepción: `set-cookie` NO se mergea ni se reordena entre sí; se emiten en el orden de llegada.
- Formato de línea: `H:<name>=<value>`.

---

### Duplicados y merge

- Para headers que son listas separadas por comas (según RFC), mergear duplicados uniendo valores con `,` y emitir `DUPHDR:<name>`.
- No mergear `set-cookie`; cada instancia se mantiene en su propia línea.
- Para otros headers no listados como "comma-mergeable", política conservadora: mantener múltiples líneas y aún así emitir `DUPHDR:<name>` si hay más de una.

Headers típicamente mergeables: `accept`, `accept-encoding`, `accept-language`, `cache-control`, `pragma`, `link`, `www-authenticate` (respetando sintaxis), entre otros.

---

### Hop-by-hop en request

- Headers hop-by-hop: `connection`, `te`, `upgrade`, `trailer`.
- Si aparecen en requests donde no corresponden, emitir `HOPBYHOP:<name>`.

---

### Métricas

- `HCNT:{n}`: número de líneas `H:` emitidas (incluye múltiples `set-cookie`).
- `HLEN:{len}@{bucket}`: tamaño agregado de `name: value` (sin CRLF) en bytes.

---

### Interacciones

- `lineas-plegadas-obs-fold.md`: realizar unfold antes de esta fase.
- `separadores-espacios.md`: colapsar whitespace antes del merge.
- `headers-valores-shape-aware.md`: aplica normalizaciones específicas sobre valores después de esta base.

---

### Ejemplos

Duplicados mergeables:
```
Accept: text/html
Accept: */*
```
Salida:
```
H:accept=text/html, */*
FLAGS:[DUPHDR:accept]
```

`set-cookie` múltiple:
```
Set-Cookie: a=1; Path=/
Set-Cookie: b=2; Path=/
```
Salida:
```
H:set-cookie=a=1; Path=/
H:set-cookie=b=2; Path=/
```

Nombre inválido:
```
X_Custom: v
```
Salida (fragmento):
```
H:x_custom=v
FLAGS:[BADHDRNAME:x_custom]
```

Hop-by-hop en request:
```
Connection: keep-alive
```
Salida (fragmento):
```
H:connection=keep-alive
FLAGS:[HOPBYHOP:connection]
```

---

### Casos límite

- Valores con comas internas en headers no mergeables → no unir.
- `host` con discrepancia respecto a `U:` → `HOSTMISMATCH` (ver URL absoluta).
- Conteo de `HCNT` y `HLEN` tras todas las normalizaciones.

---

### Requisitos de prueba

- Merge correcto y emisión de `DUPHDR`.
- No mergear `set-cookie`.
- `BADHDRNAME` con caracteres prohibidos.
- `HOPBYHOP` en headers de request.
- Orden canónico estable por nombre.
