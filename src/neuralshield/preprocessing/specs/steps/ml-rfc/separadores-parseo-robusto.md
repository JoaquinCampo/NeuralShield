### Separadores & parseo robusto (query)

Esta especificación formaliza el parseo robusto de separadores en el query (aclaración: la sección original estaba mal titulada como "(header)"). Se define cómo manejar `&` y `;` de forma tolerante, preservar multiplicidad y emitir flags de estructura.

---

### Objetivos

- Parsear queries que usan `&` y/o `;` como separadores.
- Mantener determinismo, idempotencia y preservación de orden.
- Señalar usos inusuales de `;`.

---

### Reglas normativas

1. Separador por defecto
	- Usar `&` como separador primario.

2. Heurística para `;`
	- Si `;` aparece y el patrón dominante es `k=v(;k=v)+` con poca o nula presencia de `&`, tratar `;` como separador adicional (conjunto separador `;&`) y emitir `QSEMISEP`.
	- En otro caso (presencia de `;` que no cumple patrón dominante), no considerarlo separador y emitir `QRAWSEMI`.

3. Tokenización determinista
	- Dividir la cadena por los caracteres del conjunto separador seleccionado.
	- Conservar tokens vacíos intermedios si aparecen; estos se descartan en fase de pares (no cuentan para `Q:`).

4. Construcción de pares
	- Para cada token no vacío:
		- Si contiene `=` → `key,value` por la primera `=`.
		- Si no contiene `=` → par `key` sin valor (`value = ""`) y `QBARE`.
	- Preservar orden de llegada de pares.

5. Flags de estructura
	- `QSEMISEP` si `;` se reconoce como separador.
	- `QRAWSEMI` si `;` aparece sin ser separador.
	- `QBARE` para pares sin `=`.
	- `QREPEAT:<k>` para claves repetidas (emitir una vez por clave repetida).

6. Interacción con percent-decode
	- No decodificar `%26` (`&`) ni `%3D` (`=`) antes de tokenizar.
	- Después del split, aplicar percent-decode una vez por token (ver spec correspondiente).

7. Resumen
	- `Q:{count}` (número de pares) y `KEYS:{lista}` con claves en orden.

---

### Ejemplos

Patrón dominante con `;`:
```
?x=1;y=2;z=3
```
Salida (fragmento):
```
Q:3 KEYS:x,y,z
FLAGS:[QSEMISEP]
```

`;&` mixto:
```
?x=1;y=2&z=3
```
Salida (fragmento):
```
Q:3 KEYS:x,y,z
FLAGS:[QSEMISEP]
```

`;` no separador:
```
?expr=a;b=c  # literal, no patrón dominante
```
Salida (fragmento):
```
Q:1 KEYS:expr
FLAGS:[QRAWSEMI]
```

---

### Casos límite

- Tokens vacíos consecutivos por `&&` o `;;` → no cuentan como pares.
- Claves vacías `=v` → par válido con clave `""` (no `QBARE`).
- Tokens con múltiples `=` → solo dividir por la primera.

---

### Requisitos de prueba

- Detección de patrón dominante con `;`.
- Mixto `;&`.
- `QRAWSEMI` cuando `;` no separa.
- Idempotencia del parseo.
