### Ordenar claves y preservar multiplicidad (LOW)

Esta especificación define cómo ofrecer una representación invariante del conjunto de claves del query, a la vez que se preserva la multiplicidad y el orden de llegada de los valores.

---

### Objetivos

- Invariancia al reordenamiento de pares en el query cuando sea conveniente para resumenes.
- Preservar la semántica de múltiple aparición de una misma clave y el orden de sus valores.

---

### Reglas normativas

1. Preservación de orden por defecto
	- La representación primaria (en línea o por pares) mantiene el orden de llegada (`arrival order`).
	- Para claves repetidas, los valores se listan en el orden recibido.

2. Resumen invariante de claves (opcional)
	- Generar un resumen `KEYS_SORTED:{k1,k2,...}` con claves ordenadas por colación binaria (ASCII) sin alterar el case original.
	- Este resumen es adicional a `KEYS:{lista}` que mantiene orden de llegada.

3. Representación por clave (compacta)
	- `QK:<key>=<shape>|<shape>|...` por cada clave, en el orden de `KEYS_SORTED` o en el orden de llegada de primeras apariciones (configurable, pero determinista).
	- No transformar el case de las claves; solo ordenar según colación binaria cuando corresponda.

4. Flags
	- `QREPEAT:<k>` cuando una clave aparece dos o más veces (emitir una vez por clave repetida).

---

### Ejemplos

Llegada: `login=aaa&modo=x&login=bbb`

Representación por pares (orden de llegada):
```
?login=<lower:3>&modo=<lower:1>&login=<lower:3>
Q:3 KEYS:login,modo,login
FLAGS:[QREPEAT:login]
```

Resumen ordenado:
```
KEYS_SORTED:login,modo
```

Compacta por clave:
```
QK:login=<lower:3>|<lower:3>
QK:modo=<lower:1>
```

---

### Casos límite

- Claves que difieren solo por case (`User` vs `user`): mantener como claves distintas; ordenar por colación binaria, sin case-folding.
- Claves vacías: permitidas; incluir en `KEYS`/`KEYS_SORTED`.

---

### Requisitos de prueba

- Detección de `QREPEAT` por clave.
- Invariancia del resumen ordenado ante permutaciones del query.
- Preservación del orden de llegada en la representación por pares.
