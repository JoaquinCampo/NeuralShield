### Colapsar `//` y `/.` pero NO resolver `..`

Esta especificación define cómo normalizar el path colapsando separadores redundantes sin alterar la semántica de navegación hacia arriba (`..`). También cubre flags de rareza asociadas a estructura de path.

---

### Objetivos

- Eliminar redundancias no semánticas del path (`//`, `/.`).
- Mantener la señal de potencial traversal (`..`) sin resolverla.
- Reportar métricas de longitud (PLEN, PMAX) y flags estructurales.

---

### Reglas normativas

1. Segmentación

   - Dividir el path en segmentos por `/` sin decodificar previamente `%2F` (ver spec de percent-decode).

2. Colapsación de redundancias

   - Reemplazar secuencias de múltiples `/` por un único `/`. Si ocurre al menos una vez → `MULTIPLESLASH`.
   - Eliminar segmentos `.` (punto actual) sin efectos colaterales → `/.` colapsa a `/` (forma canónica). Puede anotarse `DOTCUR` opcionalmente si se desea rastreo fino.

3. No resolución de `..`

   - No combinar ni eliminar segmentos `..`. Conservarlos tal cual, en su posición.
   - Emitir flag `DOTDOT` si aparece al menos un `..`.
   - Contabilizar runs de `..` consecutivos y, opcionalmente, emitir `DOTDOTN:{n}`.

4. Reconstrucción canónica

   - Unir los segmentos con un único `/` inicial si el path original comenzaba con `/`.
   - Si el path original estaba vacío, usar `/` como forma canónica.

5. Interacción con percent-decode

   - La colapsación estructural se realiza sin convertir `%2F`. Si se detectan `%2F` preservados tras el decode de segmentos, ya existe spec para `PCTSLASH`.

6. Flags adicionales

   - `HOME`: si el path canónico es exactamente `/`.
   - `PAREN`, `BRACE`, `BACKSLASH`, etc., se emiten desde la spec de caracteres peligrosos si aparecen en segmentos.

7. Métricas (opcional)
   - `PLEN:{len}@{bucket}`: longitud total del path canónico.
   - `PMAX:{len}@{bucket}`: longitud máxima de un segmento.
   - Buckets típicos: `0-15`, `16-31`, `32-63`, `64-127`, `128-255`, `>255`.

---

### Ejemplos

Entrada:

```
/foo//bar/.//baz
```

Salida (fragmento):

```
[URL] /foo/bar/baz
MULTIPLESLASH
```

Entrada con traversal:

```
/foo/../etc/passwd
```

Salida (fragmento):

```
[URL] /foo/../etc/passwd
DOTDOT
```

Entrada raíz:

```
/
```

Salida (fragmento):

```
[URL] /
HOME
```

Entrada con múltiples anomalías:

```
/foo//bar/../baz/.//qux
```

Salida (fragmento):

```
[URL] /foo/bar/../baz/qux
DOTDOT MULTIPLESLASH
```

---

### Casos límite

- Sufijos de `/` redundantes (`/a/b/`) → conservar o eliminar según política; recomendación: eliminar trailing vacío, salvo que el path sea `/`.
- Múltiples `..` seguidos (`../../../../`) → conservar y, opcionalmente, emitir `DOTDOTN:{n}`.
- Combinación de `%2F` con `/` reales: preservar `%2F` (ver percent-decode) y colapsar solo los `/` reales.

---

### Requisitos de prueba

- Colapsación de `//` y `/.`.
- Preservación de `..` sin resolución.
- Cálculo correcto de `PLEN` y `PMAX` con bucketing (si implementado).
- Idempotencia del proceso.
- Emisión inmediata de flags tras líneas donde se detecten anomalías.
