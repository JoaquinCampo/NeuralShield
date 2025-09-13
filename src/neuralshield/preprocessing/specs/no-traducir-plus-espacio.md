### No traduzcas + a espacio (por defecto)

Esta especificación establece que el carácter `+` se trata como literal en componentes de URL (RFC 3986) y no se convierte a espacio durante el preprocesamiento del request target ni del query. Se documenta la excepción para bodies `application/x-www-form-urlencoded`.

---

### Objetivos

- Mantener semántica correcta de URLs: `+` es un carácter legal, no un separador de espacio.
- Evitar falsos positivos en tokens/hashes/base64url donde `+` puede ser significativo.

---

### Reglas normativas

1. URL y query del request target
	- No convertir `+` a espacio en el path ni en el query.
	- Percent-decode se aplica solo a secuencias `%hh`; `+` permanece literal.

2. Cuerpos `application/x-www-form-urlencoded` (futuro/si aplica)
	- En bodies con `Content-Type: application/x-www-form-urlencoded`, `+` representa un espacio y debe decodificarse como tal en el procesamiento del body, no del request target.
	- Emitir flag específica del origen (p. ej., `BODYFORM`) si se utiliza esta norma.

3. Idempotencia
	- Una segunda pasada no debe alterar `+` ya presente.

---

### Ejemplos

Query con `+` literal:
```
?token=abc+123
```
Salida (fragmento):
```
Q:1 KEYS:token
# token conserva '+'
```

Body form (no parte de esta pipeline de URL):
```
Content-Type: application/x-www-form-urlencoded

field=a+b
```
Procesamiento del body (no del query URL):
```
field = "a b"
```

---

### Casos límite

- Combinación con percent-decode: `a%2Bb` → decodifica a `a+b`, sigue sin traducirse a espacio.
- Valores redactados `<SECRET:...>` no se transforman.

---

### Requisitos de prueba

- `+` permanece literal en query y path.
- En bodies form-url-encoded, `+` → espacio.
- Idempotencia del tratamiento de `+`.
