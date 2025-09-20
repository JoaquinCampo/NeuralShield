### Percent-decode exactamente 1 vez + flags

Esta especificación establece cómo y dónde aplicar percent-decoding (RFC 3986) exactamente una vez, cómo tratar delimitadores reservados, y qué flags emitir cuando persiste codificación o se detectan patrones anómalos.

---

### Objetivos

- Normalizar codificaciones percent a su forma textual una sola vez para una representación canónica.
- Preservar semántica: no convertir delimitadores que alteren la estructura del path o el parseo de query.
- Señalar evidencias de doble o múltiple codificación.

---

### Alcance

- Se aplica a:
	- Path (por segmentos) y query (claves y valores) del request target.
- No se aplica a:
	- Header values, salvo que contengan URLs y una spec dedicada lo defina explícitamente.

---

### Reglas normativas

1. Pasada única
	- Aplicar percent-decode exactamente una vez por componente (path, query). Esta pasada debe ser idempotente: una segunda ejecución no cambia el resultado.

2. Validación de secuencias
	- Solo decodificar secuencias válidas `%[0-9A-Fa-f]{2}`.
	- Dejar intactas secuencias inválidas (`%2G`, `%`, `%A`) y NO contarlas para flags de múltiple codificación.

3. Delimitadores reservados (preservación)
	- No decodificar a delimitadores estructurales en contextos donde cambien semántica:
		- En path: preservar `%2F` (`/`) y `%5C` (`\`) como literales codificados; NO convertirlos a `/` ni `\`. Emitir `PCTSLASH` y/o `PCTBACKSLASH` si aparecen tras la pasada.
		- En query: decodificar `%26` (`&`) y `%3D` (`=`) solo si ya se realizó el tokenizado del par clave/valor; si el decode antecede al parseo, preservar para no alterar el splitting. Recomendación: parsear tokens sobre la cadena sin decodificar delimitadores, luego decodificar en clave/valor.

4. Residuo tras la pasada
	- Si, tras la pasada, permanecen secuencias percent válidas (`%hh`), emitir `DOUBLEPCT` (señal de posible doble codificación u ofuscación). Esta flag es por componente (path/query) y se agrega al roll-up global.

5. Entidades HTML
	- La decodificación de entidades HTML (p. ej. `&#x2F;`) es un paso separado (ver spec correspondiente). Si se decodificaron entidades que resultan en delimitadores, aplicar la misma política de preservación de delimitadores.

6. Unicode y bytes
	- El decode percent produce bytes; decodificar a Unicode (UTF-8 tolerante) y seguir las reglas de normalización Unicode definidas en specs correspondientes.

---

### Path: procedimiento recomendado

- Dividir el path en segmentos por `/` (sin decodificar previamente `%2F`).
- En cada segmento, aplicar percent-decode una vez para secuencias válidas, excepto dejar intactas aquellas que correspondan a delimitadores preservados.
- Tras la pasada, si hay `%2F` o `%5C` aún presentes, emitir `PCTSLASH`/`PCTBACKSLASH`.
- Si quedan otras secuencias `%hh` válidas, emitir `DOUBLEPCT`.

---

### Query: procedimiento recomendado

- Tokenizar el query usando el separador decidido por heurística (`&` o `;&`), sin decodificar previamente `%26` o `%3D`.
- Para cada par:
	- Separar en `key[=value]` sin decodificar.
	- Aplicar percent-decode exactamente una vez a `key` y a `value` por separado.
	- Si, tras la pasada, aún hay `%hh` válidos, emitir `DOUBLEPCT` (a nivel query) y continuar.

---

### Flags

- `DOUBLEPCT`: quedan secuencias percent válidas tras una pasada de decode.
- `PCTSLASH`: el path contiene `%2F` (o variantes de case) tras la pasada.
- `PCTBACKSLASH`: el path contiene `%5C` tras la pasada.
- `MULTIENC:<k>`: opcional, cuando se observa evidencia de doble/triple codificación localizada (p. ej., en valor de la clave `k` del query). Puede complementarse con `DOUBLEPCT` global.

Todas ellas deben agregarse al roll-up `FLAGS:[...]`.

---

### Ejemplos

Path con doble codificación:
```
/foo%252Ejsp
```
Salida (fragmento):
```
P:/foo%2Ejsp
FLAGS:[DOUBLEPCT]
```

Path con `%2F` preservado:
```
/a%2Fb/c
```
Salida (fragmento):
```
P:/a%2Fb/c
FLAGS:[PCTSLASH]
```

Query con doble codificación:
```
?next=%252Fadmin%253Fq%253D1
```
Salida (fragmento):
```
Q:1 KEYS:next
U:...?next=%2Fadmin%3Fq%3D1
FLAGS:[DOUBLEPCT]
```

---

### Casos límite

- Secuencias `%` partidas o inválidas no se decodifican ni cuentan para `DOUBLEPCT`.
- Mezcla de mayúsculas/minúsculas en hex (`%2e`, `%2E`) se trata como equivalente.
- En query, `%26`/`%3D` se decodifican solo después del splitting; si se decodifican antes accidentalmente, el tokenizador podría alterar multiplicidad. Evitarlo.

---

### Requisitos de prueba

- Decode simple de `%2E` y residuo `%2E` desde `%252E` → `DOUBLEPCT`.
- `PCTSLASH`/`PCTBACKSLASH` cuando el path mantiene `%2F`/`%5C`.
- Secuencias inválidas (`%2G`) permanecen intactas; no generan `DOUBLEPCT`.
- Idempotencia de la pasada.
