### Separadores (\t, espacios múltiples) → un espacio

Esta especificación define cómo normalizar el whitespace intra-línea en campos textuales (p. ej., valores de headers, partes del path representadas como texto, valores ya canónicos), colapsando secuencias de espacios/tabs a un único espacio, y emitiendo la flag `WSPAD` cuando ocurra.

---

### Objetivos

- Eliminar variabilidad irrelevante de formateo que no aporte semántica al modelo.
- Reducir superficie de ofuscación basada en padding irregular.
- Mantener idempotencia y no alterar delimitadores semánticos.

---

### Alcance

- Se aplica a:
	- Valores de headers ya desplegados (tras obs-fold) y normalizados de nombre/caso.
	- Fragmentos textuales donde los separadores no cambian estructura (descripciones, comentarios eliminados, etc.).
- No se aplica a:
	- Delimitadores sintácticos como `;`, `,`, `=` o `&` en query/headers.
	- Separadores de segmentos del path (`/`).
	- Contextos donde un TAB tiene semántica propia (no aplicable en HTTP estándar, pero mantener explícito).

---

### Reglas normativas

1. Colapsación
	- Reemplazar cualquier secuencia no vacía de `[\t ]+` por exactamente un espacio (`0x20`).
	- Conservar espacios únicos entre tokens al menos en uno.

2. Bordes y trimming
	- Recortar whitespace al inicio y fin del campo (left/right trim). Si se recortó, cuenta como colapsación y dispara `WSPAD`.

3. Idempotencia
	- Aplicar múltiples veces no debe producir cambios adicionales.

4. Emisión de flag
	- Si en un campo se realizó al menos una sustitución (colapsación o trimming), emitir `WSPAD`.
	- `WSPAD` puede emitirse a nivel global si ocurrió en cualquier campo relevante; además, algunas líneas pueden incluir anotación local si el formato lo admite.

5. Exclusiones explícitas
	- No colapsar dentro de valores marcados como secretos u opacos (`<SECRET:...>`). Estos ya están redaccionados y no deben cambiar.
	- No colapsar en claves/valores de query antes de su parseo; esta normalización puede aplicarse a la representación textual final, no al tokenizador.

---

### Interacciones con otras specs

- `lineas-plegadas-obs-fold.md`: ejecutar después del unfold. El unfold une con un espacio; esta spec puede colapsar runs mayores.
- `headers-nombres-orden-duplicados.md`: ejecutar antes de merges para que comparaciones y merges operen sobre valores canónicos.
- `normalizar-flags.md`: agrega `WSPAD` al roll-up global.

---

### Ejemplos

Entrada:
```
User-Agent:   Mozilla\t5.0   (X11;  Linux)
```

Salida (fragmento):
```
H:user-agent=Mozilla 5.0 (X11; Linux)
FLAGS:[WSPAD]
```

Entrada sin cambios necesarios:
```
X-Test: a b c
```

Salida (fragmento):
```
H:x-test=a b c
```
---

### Casos límite

- Secuencias mixtas de tabs y espacios → un único espacio.
- Campos vacíos o solo whitespace → cadena vacía (emite `WSPAD`).
- Espacios antes/después de delimitadores (`;`, `,`, `=`) dentro de un valor: sí se colapsan si están dentro del valor textual y no alteran el tokenizador del header específico.
- Campos que ya son `<SECRET:...>`: no tocar.

---

### Requisitos de prueba

- Colapsación básica con tabs y espacios múltiples.
- Trimming en ambos extremos.
- Idempotencia tras segunda pasada.
- No alterar delimitadores de estructura (p. ej., no eliminar `/` en path, ni `&`/`=` en query tokens antes del parseo).
