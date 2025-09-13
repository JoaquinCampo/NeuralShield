### Extraer el query y decodificar % exactamente 1 vez

Esta especificación define el proceso de extracción del query del request target, su parseo robusto y la aplicación de percent-decode exactamente una vez a claves y valores, preservando multiplicidad y orden.

---

### Objetivos

- Parsear el query de forma determinista y tolerante.
- Aplicar percent-decode una sola vez por token (clave/valor) sin romper delimitadores.
- Exponer rarezas mediante flags `Q*` y `DOUBLEPCT` cuando corresponda.

---

### Reglas normativas

1. Extracción
	- Separar el query del path en el request target por el primer `?`. Si no existe, `query = ""`.

2. Separador principal
	- Por defecto usar `&`.
	- Heurística para `;`: si la cadena contiene `;` y el patrón dominante es `k=v(;k=v)+` con pocos `&`, usar separador mixto `;&` y emitir `QSEMISEP`. Si hay `;` pero no cumple patrón o se mezcla arbitrariamente, mantener `&` y emitir `QRAWSEMI`.

3. Tokenización
	- Dividir por el separador elegido en pares brutos `tok`.
	- Para cada `tok`:
		- Si contiene `=`: dividir en `key`, `value` por la primera aparición de `=`.
		- Si no contiene `=`: `key = tok`, `value = ""` y emitir `QBARE`.
	- Preservar el orden de llegada de pares.

4. Percent-decode por token
	- Aplicar percent-decode exactamente una vez a `key` y a `value` por separado, decodificando solo `%hh` válidos.
	- Si, tras la pasada, quedan `%hh` válidos en `key` o `value`, emitir `DOUBLEPCT`.
	- No decodificar `+` a espacio en URLs (literal en RFC 3986). Si en un futuro se trata un cuerpo `application/x-www-form-urlencoded`, esa regla se aplica al body.

5. Multiplicidad y resumen
	- Conservar múltiples ocurrencias de la misma clave en orden de llegada.
	- Emitir `QREPEAT:<k>` por cada clave repetida (una vez por clave que repite).
	- Emitir `QEMPTYVAL` cuando `value` sea cadena vacía por presencia explícita de `=`.
	- Emitir `Q: {count}` y `KEYS:{lista}` en orden de llegada.

6. Flags adicionales
	- `QNUL` si `value` contiene NUL (`\x00`) tras decode.
	- `QNONASCII` si clave o valor contienen non-ASCII.
	- `QARRAY:<k>` para claves con sufijo `[]`.
	- `QLONG` si algún `value` supera el umbral (p. ej., > 1024 bytes), parametrizable.

---

### Representación

Opciones de salida (compatibles):

- Compacta en línea con la URL canónica:
	- `U:... ?k=<shape>&k=<shape>&modo=<shape>`
- Por clave (preserva orden de valores):
	- `QK:login=<lower:6>|<lower:6>`
- Resumen
	- `Q:{n} KEYS:{k1,k2,...}`

La elección exacta de representación se define en la implementación, manteniendo idempotencia.

---

### Ejemplos

Separador `&` con clave repetida y vacío:
```
?login=alice&login=bob&empty=
```
Salida (fragmento):
```
Q:3 KEYS:login,login,empty
FLAGS:[QREPEAT:login QEMPTYVAL]
```

Separador `;` dominante:
```
?mode=1;user=alice;token=xyz
```
Salida (fragmento):
```
Q:3 KEYS:mode,user,token
FLAGS:[QSEMISEP]
```

Par sin `=` y con `%00`:
```
?justkey&name=%00
```
Salida (fragmento):
```
Q:2 KEYS:justkey,name
FLAGS:[QBARE QNUL]
```

Doble codificación en valor:
```
?next=%252Fadmin%253Fq%253D1
```
Salida (fragmento):
```
Q:1 KEYS:next
# next decodifica a %2Fadmin%3Fq%3D1 y quedan %hh válidos
FLAGS:[DOUBLEPCT]
```

---

### Casos límite

- Valores muy largos (1–8 KiB+): emiten `QLONG` configurable.
- Claves vacías `=v` → clave `""` válida; no es `QBARE`.
- `+` permanece literal; no convertir a espacio en query URL.

---

### Requisitos de prueba

- Heurística `;` vs `&`.
- Doble codificación detectada.
- `QREPEAT`, `QEMPTYVAL`, `QBARE`, `QNUL`, `QNONASCII`, `QARRAY`.
- Idempotencia del parseo y el decode por token.
