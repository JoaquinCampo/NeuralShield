### Líneas plegadas (obs-fold) → desplegar

Esta especificación define cómo detectar y desplegar (unfold) encabezados HTTP que usan la sintaxis obsoleta de continuación de línea (obs-fold). El objetivo es producir valores de headers en una única línea por nombre, preservando la semántica, y marcar la anomalía con `OBSFOLD`.

---

### Contexto

- En HTTP/1.1 antiguo (RFC 7230, ahora obsoleto) se permitía continuar un header en la línea siguiente si esta comenzaba con espacio (SP) o tab (HTAB). Esto es "obs-fold" y hoy está prohibido (RFC 9110).
- Algunos agentes lo siguen emitiendo o los atacantes lo usan para ofuscar payloads o realizar bypasses.

---

### Objetivos

- Unificar cualquier header representado en múltiples líneas en una única línea.
- Señalar el uso de obs-fold con `OBSFOLD` para que el modelo lo considere.
- Mantener la representación determinista e idempotente.
- Prevenir ambigüedades de CR/LF dentro de valores.

---

### Reglas normativas

1. Detección de obs-fold
	- Si una línea dentro de la sección de headers comienza con SP (0x20) o HTAB (0x09), se trata como continuación (obs-fold) del header inmediatamente anterior.
	- Si aparece una línea iniciada en SP/HTAB sin un header previo válido, emite `BADHDRCONT` y trata esa línea como ignorada para el unfolding (no se adjunta a nada).

2. Despliegue (unfold)
	- Para cada línea de continuación, recorta el prefijo de espacios/tabs iniciales y anéxala al valor previo separando con exactamente un espacio simple.
	- Si el valor previo ya terminaba en espacio(s), colapsa a un único espacio antes de anexar.
	- Marcar `OBSFOLD` si al menos se procesó una línea de continuación en toda la request.

3. Normalización mínima de espacios
	- El proceso de unfold NO realiza limpieza agresiva de espacios más allá de garantizar un único espacio como separador entre fragmentos.
	- La colapsación general de tabs/espacios a un espacio (flag `WSPAD`) pertenece a la spec de Separadores y puede aplicarse después.

4. Manejo de CR/LF embebidos
	- Cualquier CR (\r) o LF (\n) que aparezca dentro del valor de un header (fuera de los separadores de línea del protocolo) debe ser removido o escapado para la representación canónica y se emite `BADCRLF`.
	- Los caracteres de control adicionales (categoría Cc) activan la flag global `CONTROL` (ver spec núcleo) además de `BADCRLF` si corresponden a CR/LF.

5. Límites de robustez
	- Para evitar DoS por unfolding excesivo, se puede aplicar un límite de tamaño por header (p. ej., 16 KiB tras desplegar). Si se supera, emite `HLEN` acorde al bucket y continúa sin truncar en la representación (o truncar con política explícita si se define en otra spec).

6. Idempotencia
	- Tras desplegar, todas las líneas de headers quedan canónicas en una línea por nombre (antes de posibles fusiones por duplicados).
	- Una segunda pasada no debe encontrar nuevas líneas de continuación.

---

### Flags implicadas

- `OBSFOLD`: se emite si se ha detectado y procesado al menos una continuación de línea en headers.
- `BADCRLF`: se emite si se detectan CR/LF incrustados en el valor del header (no como separador de línea del protocolo).
- `BADHDRCONT`: línea de continuación (SP/HTAB) sin header previo válido.

Estas flags también deben aparecer en la línea global `FLAGS:[...]` de la salida.

---

### Interacciones con otras specs

- Separadores (tabs/espacios múltiples): la colapsación general a un espacio y `WSPAD` se aplican después del unfold.
- Headers: nombres, orden y duplicados: el proceso de merge de headers duplicados (`DUPHDR:<name>`) se realiza después de desplegar para operar sobre valores ya canónicos.
- Núcleo (normalizar y flags): `CONTROL` y el roll-up de `FLAGS:[...]` se aplican tras este paso.

---

### Ejemplos

Entrada:
```
X-Test: valor1\r\n
 val\tor2\r\n
```

Salida normalizada (fragmento):
```
H:x-test=valor1 valor2
FLAGS:[OBSFOLD]
```

Entrada con error de continuación:
```
\t  valor suelto\r\n
Host: ejemplo.com\r\n
```

Salida (fragmento):
```
H:host=ejemplo.com
FLAGS:[BADHDRCONT]
```

Entrada con CR/LF incrustados (maliciosa):
```
X-Evil: a\r\n\nInjected: b\r\n
```

Salida (fragmento):
```
H:x-evil=a Injected: b
FLAGS:[BADCRLF]
```

---

### Casos límite

- Múltiples niveles de continuación (varias líneas seguidas con SP/HTAB) → todas se despliegan en orden.
- Línea de continuación sin header previo → `BADHDRCONT` y la línea se ignora.
- Líneas con solo espacios/tabs tras el unfold → quedan como un único espacio separando tokens; limpieza más profunda es responsabilidad de `WSPAD`.
- Encabezados extremadamente largos tras el unfold → reportar en `HLEN` via buckets (ver spec de headers métricas).

---

### Requisitos de prueba

- Unfold simple con SP y con HTAB (ambos deben funcionar).
- Unfold con secuencia de 3+ líneas continuadas.
- `BADHDRCONT` cuando la primera línea de headers comienza con SP/HTAB.
- `BADCRLF` cuando CR/LF se encuentran dentro del valor.
- Idempotencia: segunda pasada no modifica el resultado.

---

### Relación con otras specs

- `separadores-espacios.md`: colapsación general de whitespace y `WSPAD`.
- `headers-nombres-orden-duplicados.md`: orden, merge y `DUPHDR`.
- `normalizar-flags.md`: política de flags globales.
