### Normalizar y agregar flags de rarezas (núcleo)

Esta especificación define el comportamiento base del preprocesador: producir una representación canónica e invariante de la request y, en paralelo, señalar cualquier rareza mediante FLAGS. Es el contrato transversal sobre el que se apoyan las demás especificaciones.

---

### Objetivos

- Proveer una forma estable (idempotente) de la request para el modelo.
- No ocultar señales de ataque/obfuscación: detectarlas y exponerlas como FLAGS.
- Proteger privacidad: nunca exponer secretos en claro; las demás specs detallan redacción por sensibilidad.

---


### Invariantes y principios

- Idempotencia: aplicar el preprocesador dos veces no cambia la salida.
- Determinismo: mismo input → misma salida (mismo orden y casing definidos).
- Minimalidad: normalizar lo necesario para invariancia; lo anómalo se marca, no se “arregla”.
- Componibilidad: esta spec define el marco; el detalle de path/query/headers se delega a sus specs específicas.

---

### Entrada y salida (resumen de formato)

- Entrada: request HTTP como bytes (método, target, versión, headers, y opcionalmente body no tratado aquí).
- Salida: líneas canónicas, por ejemplo:

```
M:GET
U:http://example.com/path.jsp
FLAGS:[FULLWIDTH DOUBLEPCT]
H:host=example.com
H:user-agent=mozilla/5.0
```

- `FLAGS:[...]` es la unión alfabética y sin duplicados de las flags detectadas en cualquier parte de la request. Las specs de cada componente (path, query, headers) pueden incluir flags en línea para contexto local; la línea `FLAGS:` es el roll-up global.

---

### Orden de operaciones (alto nivel)

1. Capturar método, target y headers sin interpretar (como bytes).
2. Decodificar a Unicode con UTF-8 tolerante ("replace") para análisis; si hubo secuencias inválidas → `BADUTF8`.
3. Normalización Unicode base (NFKC) en campos estructurales (método, nombres de headers, path y claves de query). Valores se tratan según su spec (ver redacción/shape).
4. Percent-decode exactamente una vez en path y query (según sus specs). Si tras eso persisten patrones `%[0-9A-Fa-f]{2}` válidos → `DOUBLEPCT`.
5. Decodificar entidades HTML una vez en path/query si aparecen → `HTMLENT`.
6. Detectar caracteres de control → `CONTROL`.
7. Detectar y registrar `FULLWIDTH` cuando se identifican formas de ancho completo o equivalentes normalizadas.
8. Generar salida canónica y la línea `FLAGS:[...]` (orden alfabético, sin repeticiones).

Notas:
- Decisiones detalladas (p. ej., no resolver `..`, colapsar `//`, manejo de `+`, cookies, etc.) están en las specs específicas y también pueden añadir flags. Esta spec solo define las flags nucleares de rareza y su política de emisión global.

---

### Taxonomía de flags nucleares en este ítem

- `FULLWIDTH`
  - Qué: presencia de caracteres de ancho completo (Fullwidth/Halfwidth Forms, variaciones de ancho mapeables por NFKC) en campos estructurales (p. ej., path y claves de query; nombres de header).
  - Cómo: tras NFKC, si la forma resultante difiere por mapeo de ancho, o se detecta en rangos U+FF00–U+FFEF → emitir.
  - Dónde: path, claves de query, nombres de headers.

- `DOUBLEPCT`
  - Qué: evidencia de doble codificación percent-encoding.
  - Cómo: aplicar percent-decode exactamente una vez; si quedan secuencias `%hh` válidas, emitir.
  - Dónde: path y query (por componente).

- `CONTROL`
  - Qué: aparición de caracteres de control Unicode (categoría Cc) o separadores de línea no estándar.
  - Cómo: escanear tras la decodificación a Unicode; excluir `TAB` y `SPACE` cuando sean separadores legales según cada spec; incluir `NUL` aunque aparezca via `%00` (otras specs añaden `QNUL` para query).
  - Dónde: global (si aparece en cualquier parte).

- `HTMLENT`
  - Qué: presencia de entidades HTML (p. ej., `&#x2f;`, `&lt;`).
  - Cómo: decodificar entidades una vez y, si hubo sustituciones, emitir.
  - Dónde: path y query.

Otras flags (p. ej., `MIXEDSCRIPT`, `DOTDOT`, `MULTIPLESLASH`, `Q*`, `DUPHDR:*`) se definen y emiten en sus respectivas specs; esta spec solo las agregará a la línea global `FLAGS:[...]` cuando estén presentes.

---

### Reglas normativas mínimas (para este ítem)

1. Percent-decode una vez por componente (path, query). No decodificar delimitadores reservados si cambia semántica (ver spec de path/query). Si tras la pasada persisten patrones `%hh` válidos → `DOUBLEPCT`.
2. Unicode (NFKC) aplicado a campos estructurales; si se detectan formas de ancho completo antes/después → `FULLWIDTH`.
3. Entidades HTML: una pasada de decodificación. Si se reemplazó al menos una entidad → `HTMLENT`.
4. Control chars: si `\p{Cc}` (o NUL de cualquier origen) aparece en cualquier segmento procesado → `CONTROL`.
5. La línea `FLAGS:[...]` debe contener todas las flags observadas, ordenadas alfabéticamente, sin duplicar, y emitirse siempre (vacía si no hubo flags no es necesario; si no hay flags, omitir la línea).

---

### Idempotencia

- Reaplicar el preprocesador no debe introducir ni eliminar flags (salvo la deduplicación natural). Las transformaciones son:
  - Una única decodificación percent.
  - Una única decodificación de entidades HTML.
  - Normalización Unicode determinista.
- La detección no depende del orden interno: solo del resultado normalizado tras las reglas establecidas.

---

### Ejemplos

1) FULLWIDTH + DOUBLEPCT

Entrada:
```
GET /％70ath%252Ejsp HTTP/1.1
Host: ex.com
```

Salida (fragmento):
```
M:GET
U:http://ex.com/path%2Ejsp
FLAGS:[DOUBLEPCT FULLWIDTH]
```

2) CONTROL + HTMLENT

Entrada:
```
GET /a&#x2f;b%00c HTTP/1.1
Host: ex.com
```

Salida (fragmento):
```
U:http://ex.com/a/b%00c
FLAGS:[CONTROL HTMLENT]
```

3) Sin rarezas nucleares

```
GET /a/b.jsp HTTP/1.1
Host: ex.com
```

```
M:GET
U:http://ex.com/a/b.jsp
```

---


### Casos límite y aclaraciones

- `DOUBLEPCT` no se emite por secuencias `%` inválidas (no hex); esas no cuentan.
- `FULLWIDTH` se evalúa sobre campos estructurales donde NFKC es aplicable; los valores opacos se tratan en la spec de shapes/redacción.
- `CONTROL` incluye NUL proveniente de `%00`; las specs de query/headers pueden añadir flags específicas adicionales (`QNUL`, etc.).
- Si no se detecta ninguna flag, no se emite la línea `FLAGS:`.

---

### Requisitos de pruebas (mínimos para este ítem)

- FULLWIDTH: ruta con caracteres U+FF01..U+FF60 produce `FULLWIDTH` y normaliza a ASCII cuando corresponda.
- DOUBLEPCT: "/a%252Ejsp" → tras una pasada queda `%2E` y se emite `DOUBLEPCT`.
- CONTROL: presencia de `\x00`, `\x01`… en target → `CONTROL`.
- HTMLENT: "/a&#x2f;b" → `HTMLENT` y barra decodificada según política del path.
- Idempotencia: aplicar dos veces no cambia salida ni FLAGS.

---

### Relación con otras specs

- Percent-decode (detalle y delimitadores): ver `percent-decode-una-vez.md`.
- Path (colapsar `//`, `/.`, no resolver `..`): ver `colapsar-slash-dot-no-resolver-dotdot.md`.
- Query (separadores, multiplicidad, shapes, sensibilidad): ver `query-decodificar-una-vez.md`, `separadores-parseo-robusto.md`, `shape-longitud-redaccion.md`.
- Headers (nombres, duplicados, normalización de valores): ver `headers-*`.
