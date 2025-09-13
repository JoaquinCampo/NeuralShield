### Shape + longitud para valores (con redacción por sensibilidad) - (HIGH)

Esta especificación define cómo clasificar valores en "shapes" deterministas, redactar valores sensibles y reportar su longitud (y buckets) para maximizar señal al modelo protegiendo la privacidad.

---

### Objetivos

- Representar valores por su forma (shape) y tamaño, no por su literal.
- Redactar SIEMPRE valores sensibles.
- Mantener determinismo e idempotencia.

---

### Alcance

- Aplica a valores de query y a valores de headers cuando corresponda (p. ej., `authorization`, `cookie` resumido, etc.).
- No altera claves, salvo para clasificación auxiliar (no cambia case ni contenido de claves).

---

### Claves sensibles (redactar siempre)

Patrón (case-insensitive) para nombres de clave/encabezado considerados sensibles:

```
pass|pwd|token|auth|authorization|cookie|session|bearer|jwt|csrf|xsrf|apikey|api_key|access[_-]?token|id_token|refresh[_-]?token|sig|hmac|sso
```

- Si el nombre de la clave o del header coincide con el patrón anterior, el valor se redacta como `<SECRET:shape:len>`.
- Además, si el valor es un JWT válido o un patrón de credencial reconocido, se redacta por contenido aunque la clave no sea sensible (p. ej., `next=eyJ...` que es un JWT).

---

### Shapes (definiciones)

Orden de evaluación (precedencia): `jwt` → `uuid` → `ipv4/ipv6` → `b64url` → `b64` → `hex` → `email` → `uaxurl` → `num` → `lower/upper/alpha/alnum/lowernum/uppernum` → `mixed`.

- `jwt`: tres segmentos Base64URL separados por puntos, sin espacios: `^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$`
- `uuid`: formato 8-4-4-4-12 hex (case-insensitive): `^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`
- `ipv4`: usar validador numérico 0–255 por octeto; regex aproximada: `^([0-9]{1,3}\.){3}[0-9]{1,3}$` + chequeo de rangos
- `ipv6`: usar parser estándar (por ejemplo, `ipaddress.IPv6Address`)
- `b64`: Base64 con padding opcional correcto: `^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$`
- `b64url`: Base64URL sin `+`/`/`, padding opcional: `^[A-Za-z0-9_-]+={0,2}$`
- `hex`: `^[0-9a-fA-F]+$`
- `email`: heurística simple: `^[^@\s]+@[^@\s]+\.[^@\s]+$`
- `uaxurl`: parece URL (p. ej., `^https?://` o esquema válido `^[a-zA-Z][a-zA-Z0-9+.-]*:`)
- `num`: `^[0-9]+$`
- `lower`: `^[a-z]+$`
- `upper`: `^[A-Z]+$`
- `alpha`: `^[A-Za-z]+$`
- `alnum`: `^[A-Za-z0-9]+$`
- `lowernum`: `^[a-z0-9]+$`
- `uppernum`: `^[A-Z0-9]+$`
- `mixed`: todo lo demás (fallback determinista)

Extras opcionales:
- `rand{H}`: etiqueta de alta entropía con bits aproximados por carácter (Shannon). Se adjunta como sufijo si la entropía normalizada supera un umbral (p. ej., > 4.0 bits/char) y longitud >= 16.

---

### Redacción y longitud

- Formato general: `<shape:len>`.
- Si sensible (por clave o por contenido): `<SECRET:shape:len>`.
- Longitud `len` es la longitud en caracteres Unicode del valor tras percent-decode del query (y cualquier decode aplicable del flujo), ANTES de cualquier redacción.
- Opcionalmente adjuntar bucket: `<shape:len@bucket>`; recomendado mantener solo `len` y emitir bucket por separado si se requiere.

---

### Flags relacionadas

- `QLONG`: valor cuyo tamaño excede umbral (p. ej., > 1024). Se agrega al roll-up global.
- Para `jwt`, puede emitirse flag `JWT` opcional. En headers, `authorization` gestiona flags específicas (`AUTHBEARER`, etc.) en su propia spec.

---

### Representación sugerida

- En línea con la URL canónica:
	- `?login=<lower:6>&login=<lower:6>&modo=<lower:6>`
- Por clave (preservando orden de valores):
	- `QK:login=<lower:6>|<lower:6>`

---

### Ejemplos

- `pwd=visionario` → `pwd=<SECRET:alpha:10>`
- `token=eyJhbGciOi...` (JWT) → `token=<SECRET:jwt:836>`
- `id=12345` → `id=<num:5>`
- `hash=14d18cd98f...` → `hash=<hex:32>`
- `next=https://ex.com/a` → `next=<uaxurl:19>`
- `ip=192.168.0.1` → `ip=<ipv4:11>`

---

### Reglas normativas

1. Precedencia de shapes
	- Evaluar en el orden indicado para evitar clasificaciones ambiguas (p. ej., un JWT también es `b64url` pero debe clasificar como `jwt`).

2. Sensibilidad por contenido
	- Si el valor clasifica como `jwt` o evidencia credenciales (p. ej., encabezados `authorization`), tratar como secreto independientemente del nombre de clave.

3. Idempotencia
	- Si un valor ya está redactado (`<...>`), no volver a redaccionar ni transformar.

4. Longitud
	- Calcular `len` antes de redacción; mantener determinismo entre pasadas.

5. Entropía (opcional)
	- Si se activa, añadir sufijo `rand{H}` (p. ej., `<b64url:120 rand4.7>`). Debe ser determinista para el mismo input.

---

### Casos límite

- Valores vacíos → `<num:0>` no es apropiado; producir `<mixed:0>` (o mantener vacío si la representación así lo requiere) y considerar `QEMPTYVAL` (definido en query).
- Valores redactados previamente (`<...>`) → no modificar.
- `b64` con relleno extraño o inválido → no clasifica como `b64`; caerá a `mixed`.
- Emails con TLD no estándar → heurística simple; no forzar validación estricta.

---

### Requisitos de prueba

- Precedencia correcta (`jwt` sobre `b64url`).
- Redacción por clave sensible y por contenido (JWT en clave no sensible).
- Señalización de `QLONG` y entropía opcional.
- Idempotencia: redacción no se vuelve a aplicar.

---

### Interacciones

- Query parsing y percent-decode: esta spec opera después de decodificar tokens clave/valor.
- Headers valores shape-aware: se apoya en esta clasificación; flags adicionales (`AUTHBEARER`, `COOKIE`) se manejan en la spec de headers.
