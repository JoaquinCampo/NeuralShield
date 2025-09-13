### Headers: valores “shape-aware”

Esta especificación define normalizaciones específicas por tipo de header para extraer forma y señales, sin exponer valores sensibles.

---

### Objetivos

- Reducir ruido y variabilidad de headers comunes.
- Exponer estructuras útiles (listas ordenadas, tokens clave/valor, shapes y longitudes) respetando privacidad.

---

### user-agent

- Tokenizar en `nombre/version` y plataformas conocidas; eliminar comentarios `()` y `[]`.
- Limitar a N tokens (configurable), recortar sufijos verbosos.
- Ejemplo:
```
H:user-agent=mozilla/5.0 konqueror/3.5 khtml/3.5.8 like:gecko
```

---

### accept*

- `accept`, `accept-charset`, `accept-language`, `accept-encoding` son listas separadas por comas/semicolons con parámetros `q`.

- accept
	- Normalizar a `type/subtype;qX`.
	- Ordenar por `type/subtype` y `q` descendente.
	- Ejemplo:
```
H:accept=text/html;q0.9 text/plain;q0.8 */*;q0.5
```

- accept-encoding
	- Conjunto de codificaciones (gzip, deflate, br, etc.), deduplicar, ordenar alfabéticamente.
	- Ejemplo:
```
H:accept-encoding=br deflate gzip
```

- accept-language
	- Normalizar a BCP 47 `ll-CC` (lowercase idioma, uppercase país) y ordenar por `q` descendente.
	- Ejemplo:
```
H:accept-language=es-AR;q1.0 en-US;q0.8 en;q0.5
```

---

### cookie

- No exponer valores. Extraer solo nombres y longitudes, en orden alfabético por nombre.
- Contar cookies → `COOKIE:{n}`.
- Ejemplo:
```
H:cookie=JSESSIONID<len:32> PREF<len:8> [COOKIE:2]
```

---

### authorization

- Redactar SIEMPRE:
	- `Bearer <token>` → `<SECRET:bearer:len>` y flag `AUTHBEARER`.
	- `Basic <base64>` → `<SECRET:basic:len>` y flag `AUTHBASIC`.
	- `Digest ...` → `<SECRET:digest:len>`.
- Ejemplo:
```
H:authorization=<SECRET:bearer:142> FLAGS:[AUTHBEARER]
```

---

### x-forwarded-for / forwarded

- Extraer lista de IPs, clasificar cada una (`ipv4`, `ipv6`, `private`).
- Emitir flag `XFF` si aparece alguno de estos headers.
- Ejemplo:
```
H:x-forwarded-for=ipv4,private,ipv4 FLAGS:[XFF]
```

---

### host

- Comparar con host de la `U:` absoluta construida.
- Si difieren, emitir `HOSTMISMATCH`.

---

### Reglas generales

- Aplicar `separadores-espacios.md` (colapsación) tras el unfold y antes de parseos específicos.
- Mantener idempotencia: volver a procesar no cambia el resultado.
- No romper sintaxis al reordenar: seguir RFC para cada header.

---

### Ejemplos combinados

```
H:user-agent=mozilla/5.0 like:gecko
H:accept=text/html;q0.9 */*;q0.5
H:accept-encoding=gzip br
H:accept-language=en-US;q1.0 es-AR;q0.8
H:cookie=JSESSIONID<len:32> [COOKIE:1]
H:authorization=<SECRET:bearer:128> FLAGS:[AUTHBEARER]
H:x-forwarded-for=ipv4,ipv4,private FLAGS:[XFF]
```
---

### Casos límite

- `user-agent` con cadenas extremadamente largas: truncar a N tokens.
- `accept` con comodines `*/*`: colocarlos al final por `q` y por especificidad.
- `cookie` con nombres repetidos: listar una vez con mayor longitud o todas (política consistente; recomendación: todas en orden alfabético).

---

### Requisitos de prueba

- Idempotencia de normalizadores por header.
- `AUTHBEARER`/`AUTHBASIC` en `authorization` y redacción correcta.
- `XFF` con clasificación de IPs.
- `HOSTMISMATCH` cuando `host` difiere de `U:`.
