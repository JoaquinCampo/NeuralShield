### Construir URL absoluta cuando es relativa

Esta especificación define cómo construir una URL absoluta canónica (`scheme://host[:port]/path?query`) cuando el request target viene en forma relativa (origin-form, p. ej., `GET /path HTTP/1.1`) y cómo tratar los casos de absolute-form y authority-form, así como discrepancias con el header `Host`.

---

### Objetivos

- Proveer una `U:` absoluta canónica para el modelo, evitando que aprenda a unir `Host` + path.
- Omitir puertos por defecto (80 en `http`, 443 en `https`), mantener puertos no estándar.
- Señalar inconsistencias entre absolute-form y `Host` con `HOSTMISMATCH`.

---

### Formas de target (RFC 9110)

- Origin-form: `/path?query` (típico en GET/POST…)
- Absolute-form: `http://host:port/path?query` (típico en proxies)
- Authority-form: `host:port` (típico en `CONNECT`)
- Asterisk-form: `*` (típico en `OPTIONS *`)

---

### Reglas normativas

1. Detección del esquema y host
	- Si el target es absolute-form, parsear `scheme`, `host`, `port`, `path`, `query` directamente del target.
	- Si es origin-form, extraer `host` (y opcionalmente `port`) del header `Host` (obligatorio en HTTP/1.1). El `scheme` puede inferirse del contexto (no disponible en crudo); por defecto use `http` si no hay metadatos externos. Se permite parametrizar el esquema por configuración.
	- Si `Host` no existe o es inválido, la `U:` no puede construirse completamente; emitir `BADHOST` y omitir `U:` o emitir `U://invalid/...` según política. Recomendación: no emitir `U:` y registrar flag.

2. Normalización de `host`
	- Lowercase.
	- IDNA: convertir a punycode (ASCII) para representación canónica; si el `Host` original contenía etiquetas Unicode, emitir `IDNA`.
	- IPv6 literales: deben ir entre corchetes `[::1]` en absolute-form.

3. Puertos
	- Omitir `:80` en `http` y `:443` en `https`.
	- Conservar cualquier otro puerto explícito.

4. Caminos y query
	- Mantener el path y query originales (antes de normalizaciones específicas). Otras specs se encargan de percent-decode, colapsado de `//` y `/.`, etc.
	- Si el origin-form carece de path (vacío), usar `/`.

5. Validaciones y flags
	- Si el target es absolute-form y existe `Host`, comparar `host[:port]` del target con el de `Host` tras normalizar; si difieren → `HOSTMISMATCH`.
	- Si `Host` contiene nombre inválido (caracteres prohibidos) → `BADHDRNAME:host` (ver spec de headers) y `BADHOST`.
	- Si `Host` tiene puerto no numérico o fuera de rango → `BADHOST`.

6. Emisión
	- Emitir una línea `U:scheme://host[:port]/path[?query]`.
	- Esta línea aparece inmediatamente después de `M:` en la salida canónica.

---

### Ejemplos

Origin-form con puerto por defecto:
```
GET /a/b.jsp HTTP/1.1
Host: ex.com:80
```
Salida:
```
U:http://ex.com/a/b.jsp
```

Origin-form con puerto no estándar:
```
GET /a/b.jsp HTTP/1.1
Host: ex.com:8080
```
Salida:
```
U:http://ex.com:8080/a/b.jsp
```

Absolute-form con `Host` coherente:
```
GET http://ex.com/a HTTP/1.1
Host: ex.com
```
Salida:
```
U:http://ex.com/a
```

Absolute-form con `Host` diferente:
```
GET http://ex.com/a HTTP/1.1
Host: other.com
```
Salida (fragmento):
```
U:http://ex.com/a
FLAGS:[HOSTMISMATCH]
```

Authority-form (`CONNECT`):
```
CONNECT db.example.com:5432 HTTP/1.1
```
Salida (fragmento):
```
M:CONNECT
# No se emite U:, o se emite forma canónica de autoridad según política
```

Asterisk-form:
```
OPTIONS * HTTP/1.1
Host: ex.com
```
Salida (fragmento):
```
U:http://ex.com/*
```

---

### Casos límite


- `Host` con espacios o tabs → colapsar/trim según spec de separadores y luego validar; si sigue inválido → `BADHOST`.
- `Host` con underscores o caracteres prohibidos → `BADHDRNAME:host` (ver headers) y `BADHOST`.
- IPv6 con puerto en `Host`: `Host: [2001:db8::1]:8080` → conservar en absolute-form.
- Origin-form sin path (vacío) → usar `/`.

---

### Requisitos de prueba

- Omitir puertos por defecto y conservar no estándar.
- `HOSTMISMATCH` en absolute-form vs `Host` distinto.
- `BADHOST` cuando el `Host` es inválido o ausente (en origin-form).
- IPv6 literales y punycode en dominios Unicode (emite `IDNA`).
