### Caracteres peligrosos + Script mixing

Esta especificación define la detección y marcado de caracteres peligrosos (útiles para XSS/SQLi/RCE/traversal) y la detección de mezcla de alfabetos (script mixing) que habilitan ataques por homógrafos.

---

### Objetivos

- Señalar presencia de caracteres y patrones con alta correlación de riesgo.
- Dar al modelo señales explícitas sin eliminar información estructural.
- Mantener política consistente entre path, query y headers.

---

### Conjunto de caracteres peligrosos (base)

- `ANGLE`: `<` o `>`
- `QUOTE`: `'` o `"`
- `SEMICOLON`: `;`
- `PAREN`: `(` o `)`
- `BRACE`: `{` o `}`
- `PIPE`: `|`
- `BACKSLASH`: `\\`
- `SPACE`: espacio dentro de segmentos sensibles (p. ej., en path tras decode)
- `NUL`: `\x00` (también puede aparecer como `%00` → ver `QNUL` en query)
- `PERCENT_IN_PATH`: `%` crudo en path tras decode (opcional; si se conserva `%`)

Estas flags se aplican por componente y se agregan al roll-up global `FLAGS:[...]`.

---

### Política por componente

- Path
	- Escanear el path canónico post-normalización (colapsado de `//`/`/.`, preservando `..`).
	- Emitir flags según caracteres detectados. `SPACE` en path se considera sospechoso.

- Query
	- Escanear claves y valores tras percent-decode por par.
	- Para NUL (`\x00`) en valores, emitir `QNUL` además de `NUL` global.

- Headers
	- Escanear valores normalizados (desplegados y con whitespace colapsado) salvo aquellos marcados como secretos redaccionados.

---

### Script mixing (MIXEDSCRIPT)

- Definición: presencia de caracteres pertenecientes a ≥2 escrituras entre Latín, Cirílico, Griego (extensible), en un mismo token relevante (host, segmento de path, clave/valor de query, header sensible como `host` o `referer`).
- Detección:
	- Calcular el conjunto de scripts dominantes por token (usando propiedades Unicode de cada codepoint).
	- Si |scripts| ≥ 2 y no es una combinación permitida (p. ej., Latin + Common), emitir `MIXEDSCRIPT`.
- Notas:
	- No emitir por combinación de Latin con signos de puntuación o dígitos (clase `Common`/`Inherited`).
	- Emitir para ejemplos típicos de homógrafos: `раypal.com` (p cirílica + resto latín).

---

### Interacciones

- Unicode (FULLWIDTH/CONTROL): estas flags se emiten en su spec; aquí solo complementamos con `MIXEDSCRIPT`.
- Percent-decode: el análisis se hace tras la pasada única correspondiente; entidades HTML se tratan tras su decode.

---

### Ejemplos

Path con `<script>`:
```
/a/<script>/b
```
Salida (fragmento):
```
P:/a/<script>/b
FLAGS:[ANGLE]
```

Query con NUL y comillas:
```
?name=O%27Brien%00
```
Salida (fragmento):
```
Q:1 KEYS:name
FLAGS:[QUOTE QNUL NUL]
```

Host con homógrafos:
```
U:http://раypal.com/
```
Salida (fragmento):
```
FLAGS:[MIXEDSCRIPT]
```

Backslash en path:
```
/a\b/c
```
Salida (fragmento):
```
P:/a\b/c
FLAGS:[BACKSLASH]
```


---

### Casos límite

- Valores redactados `<SECRET:...>`: no inspeccionar caracteres internos.
- Tokens con solo dígitos y puntuación: no `MIXEDSCRIPT`.
- Uso de `FULLWIDTH` junto a script mixing: pueden coexistir `FULLWIDTH` y `MIXEDSCRIPT`.

---

### Requisitos de prueba

- Detección de cada flag individual (`ANGLE`, `QUOTE`, etc.).
- `MIXEDSCRIPT` en dominios mixtos y en claves/valores de query.
- Coexistencia con `HTMLENT` y `DOUBLEPCT` (decode previo no rompe detección).


### Colapsar `//` y `/.` pero NO resolver `..`

Esta especificación define cómo normalizar el path colapsando separadores redundantes sin alterar la semántica de navegación hacia arriba (`..`). También cubre flags de rareza asociadas a estructura de path.

---

### Objetivos

- Eliminar redundancias no semánticas del path (`//`, `/.`).
- Mantener la señal de potencial traversal (`..`) sin resolverla.
- Reportar métricas de longitud (PLEN, PMAX) y flags estructurales.

---

### Reglas normativas

1. Segmentación
	- Dividir el path en segmentos por `/` sin decodificar previamente `%2F` (ver spec de percent-decode).

2. Colapsación de redundancias
	- Reemplazar secuencias de múltiples `/` por un único `/`. Si ocurre al menos una vez → `MULTIPLESLASH`.
	- Eliminar segmentos `.` (punto actual) sin efectos colaterales → `/.` colapsa a `/` (forma canónica). Puede anotarse `DOTCUR` opcionalmente si se desea rastreo fino.

3. No resolución de `..`
	- No combinar ni eliminar segmentos `..`. Conservarlos tal cual, en su posición.
	- Emitir flag `DOTDOT` si aparece al menos un `..`.
	- Contabilizar runs de `..` consecutivos y, opcionalmente, emitir `DOTDOTN:{n}`.

4. Reconstrucción canónica
	- Unir los segmentos con un único `/` inicial si el path original comenzaba con `/`.
	- Si el path original estaba vacío, usar `/` como forma canónica.

5. Interacción con percent-decode
	- La colapsación estructural se realiza sin convertir `%2F`. Si se detectan `%2F` preservados tras el decode de segmentos, ya existe spec para `PCTSLASH`.

6. Flags adicionales
	- `HOME`: si el path canónico es exactamente `/`.
	- `PAREN`, `BRACE`, `BACKSLASH`, etc., se emiten desde la spec de caracteres peligrosos si aparecen en segmentos.

7. Métricas
	- `PLEN:{len}@{bucket}`: longitud total del path canónico.
	- `PMAX:{len}@{bucket}`: longitud máxima de un segmento.
	- Buckets típicos: `0-15`, `16-31`, `32-63`, `64-127`, `128-255`, `>255`.

---

### Ejemplos

Entrada:
```
/foo//bar/.//baz
```
Salida (fragmento):
```
P:/foo/bar/baz PLEN:11@0-15 PMAX:3@0-15
FLAGS:[MULTIPLESLASH]
```

Entrada con traversal:
```
/foo/../etc/passwd
```
Salida (fragmento):
```
P:/foo/../etc/passwd PLEN:20@16-31 PMAX:8@0-15
FLAGS:[DOTDOT]
```

Entrada raíz:
```
/
```
Salida (fragmento):
```
P:/ PLEN:1@0-15 PMAX:0@0-15
FLAGS:[HOME]
```

---

### Casos límite

- Sufijos de `/` redundantes (`/a/b/`) → conservar o eliminar según política; recomendación: eliminar trailing vacío, salvo que el path sea `/`.
- Múltiples `..` seguidos (`../../../../`) → conservar y, opcionalmente, emitir `DOTDOTN:{n}`.
- Combinación de `%2F` con `/` reales: preservar `%2F` (ver percent-decode) y colapsar solo los `/` reales.

---

### Requisitos de prueba

- Colapsación de `//` y `/.`.
- Preservación de `..` sin resolución.
- Cálculo correcto de `PLEN` y `PMAX` con bucketing.
- Idempotencia del proceso.


### Encodings y decodificaciones “raras”

Esta especificación aborda casos anómalos de codificación y decodificación que suelen usarse para evasión de filtros: UTF-8 overlong/invalid, múltiples niveles de percent-encoding, combinaciones con entidades HTML, y otros.

---

### Objetivos

- Detectar y señalar codificaciones anómalas sin intentar “arreglarlas” completamente.
- Mantener la integridad del análisis canónico y la idempotencia.

---

### Casos cubiertos y flags

1. UTF-8 inválido / overlong → `BADUTF8`
	- Al decodificar bytes a Unicode, usar modo tolerante ("replace") para no perder estructura.
	- Si se detectan secuencias inválidas u overlong (ej., `C0 AF` para `/`), emitir `BADUTF8`.

2. Múltiple percent-encoding → `DOUBLEPCT` y/o `MULTIENC:<k>`
	- Tras una pasada única de percent-decode, si quedan `%hh` válidos, emitir `DOUBLEPCT`.
	- Si se identifica doble/triple encoding localizado (por clave o campo), opcionalmente `MULTIENC:<k>`.

3. HTML entities → `HTMLENT`
	- Si aparecen entidades HTML en path/query (p. ej., `&#x2f;`, `&lt;`), decodificar una sola vez y emitir `HTMLENT`.
	- Si tras decodificar entidades se producirían delimitadores estructurales (`/`, `&`, `=`), aplicar la misma política de preservación que en percent-decode (no romper estructura).

4. Mixtura de encodings
	- Combinar percent-encoding y entidades HTML en un mismo valor es típico de ofuscación. Si tras aplicar las pasadas únicas reglamentarias subsisten señales (residuos `%hh` válidos o entidades no decodificadas), se emiten las flags correspondientes (`DOUBLEPCT`, `HTMLENT`).

---

### Reglas normativas

1. Orden de decodificación
	- Primero percent-decode (pasada única), luego entidades HTML (pasada única) en path y query.
	- No reintentar decodificaciones adicionales si quedan residuos; señalar con flags.

2. Manejo de bytes → Unicode
	- Decodificar a Unicode con UTF-8 en modo tolerante ("replace"), sin descartar bytes.
	- Emitir `BADUTF8` si ocurre cualquier sustitución.

3. Preservación de delimitadores
	- Si una decodificación (percent o HTML entity) convertiría a delimitador estructural, aplicar política de preservación definida en sus respectivas specs (p. ej., mantener `%2F` en path y señalar `PCTSLASH`).

4. Idempotencia
	- Una segunda pasada no debe aplicar decodificaciones adicionales ni cambiar flags.

---

### Ejemplos

UTF-8 overlong:
```
GET /%C0%AFetc/passwd HTTP/1.1
```
Salida (fragmento):
```
P:/%C0%AFetc/passwd
FLAGS:[BADUTF8 DOUBLEPCT]
```

HTML entity en path:
```
/a&#x2f;b
```
Salida (fragmento):
```
P:/a/b
FLAGS:[HTMLENT]
```

Mixto percent + HTML entity con residuo:
```
?x=%2526y%3D1&#x26;z=2
```
Salida (fragmento):
```
Q:2 KEYS:x,z
FLAGS:[DOUBLEPCT HTMLENT]
```

---

### Casos límite

- Entidades HTML malformadas → ignorar y no emitir `HTMLENT` a menos que se produzca una sustitución válida.
- Percent-encoding inválido (`%2G`) → no decodificar, no contar para `DOUBLEPCT`.
- Secuencias mixtas complejas (percent → entidad → percent) → respetar una pasada única por tipo; no encadenar indefinidamente.

---

### Requisitos de prueba

- `BADUTF8` con overlong e inválidos.
- `HTMLENT` cuando haya sustituciones reales en path/query.
- `DOUBLEPCT` tras residuo `%hh` válido.
- Idempotencia de la secuencia de decodificación.



### Flags por rarezas - (HIGH)

Esta especificación centraliza las flags de rarezas/anomalías de alta prioridad, definiendo sus disparadores, alcance (path/query/headers) y reglas de emisión. Todas las flags listadas aquí deben agregarse al roll-up global `FLAGS:[...]` sin duplicados y ordenadas alfabéticamente.

---

### Principios

- No corregir silenciosamente: marcar y preservar la evidencia cuando sea posible.
- Idempotencia: un input que ya emitió una flag no debe generar nuevas flags al re-procesarse.
- Determinismo: mismas condiciones, mismas flags.

---

### Taxonomía principal

- Codificación / Decodificación
	- `DOUBLEPCT`: quedan `%hh` válidos tras una pasada de percent-decode.
	- `MULTIENC:<k>`: evidencia localizada de doble/triple encoding (por clave `<k>` o contexto). Opcional.
	- `HTMLENT`: se decodificaron entidades HTML en path o query.
	- `BADUTF8`: se detectaron secuencias inválidas al decodificar a UTF-8.
	- `PCTSLASH`: el path mantiene `%2F` tras la pasada única.
	- `PCTBACKSLASH`: el path mantiene `%5C` tras la pasada única.

- Estructura de Path
	- `MULTIPLESLASH`: se colapsaron `//`.
	- `DOTDOT`: presencia de `..` sin resolver.
	- `DOTDOTN:{n}`: opcional, cantidad de `..` consecutivos.
	- `HOME`: path canónico es `/`.

- Whitespace y líneas
	- `OBSFOLD`: se desplegaron líneas de continuación en headers.
	- `WSPAD`: se colapsaron tabs/espacios redundantes.
	- `BADCRLF`: CR/LF incrustados en valores de header.

- Unicode
	- `FULLWIDTH`: formas de ancho completo en campos estructurales.
	- `MIXEDSCRIPT`: mezcla de alfabetos relevantes (Latín, Cirílico, Griego) en un token.
	- `CONTROL`: caracteres de control presentes.
	- `IDNA`: host con etiquetas Unicode convertido a punycode.

- Query estructura/valores
	- `QREPEAT:<k>`: clave `<k>` aparece 2+ veces.
	- `QEMPTYVAL`: par con `=` y valor vacío.
	- `QBARE`: token sin `=`.
	- `QNUL`: valor contiene NUL tras decode.
	- `QNONASCII`: clave o valor con non-ASCII.
	- `QARRAY:<k>`: clave con sufijo `[]`.
	- `QSEMISEP`: `;` aceptado como separador por heurística.
	- `QRAWSEMI`: `;` presente pero no reconocido como separador.
	- `QLONG`: valor supera umbral configurado.

- Caracteres peligrosos
	- `ANGLE`, `QUOTE`, `SEMICOLON`, `PAREN`, `BRACE`, `PIPE`, `BACKSLASH`, `SPACE`, `NUL`.

- Headers
	- `DUPHDR:<name>`: header duplicado (lista separada por comas tras merge cuando aplica).
	- `BADHDRNAME:<name>`: nombre de header inválido.
	- `HOPBYHOP:<name>`: header hop-by-hop inesperado en request.
	- `UNKHDR:<name>`: header no en lista blanca (opcional).
	- `HOSTMISMATCH`: absolute-form difiere de `Host`.
	- `AUTHBEARER`/`AUTHBASIC`: credenciales en `authorization` (gestión en spec de headers).
	- `XFF`: `x-forwarded-for`/`forwarded` con IPs parseadas.
	- `COOKIE:{n}`: cantidad de cookies (resumen). Opcional.

---

### Reglas de emisión

- Unicidad: cada flag se emite como máximo una vez en `FLAGS:[...]` por request.
- Alcance local vs global: los módulos pueden incluir flags locales en su línea (`P:`, `Q:`, `H:`); el agregador siempre añade todas al roll-up `FLAGS:`.
- Orden: alfabético en el roll-up.

---

### Ejemplos de agregación

```
P:/a/b/../c%2Ejsp [DOTDOT] [DOUBLEPCT]
Q:3 KEYS:k,k,token [QREPEAT:k] [QSEMISEP]
H:x-test=valor1 valor2 [OBSFOLD]
FLAGS:[DOUBLEPCT DOTDOT OBSFOLD QREPEAT:k QSEMISEP]
```

---

### Casos límite

- Múltiples ocurrencias del mismo fenómeno → una única flag (p. ej., varias claves repetidas del mismo nombre → solo `QREPEAT:<k>` una vez).
- Flags con parámetros (`:<k>`, `:{n}`) deben mantener su parámetro canónico (case original de clave, o lowercased de forma determinista, según decisión global; recomendación: mantener tal cual aparece primero).

---

### Requisitos de prueba

- Dedupe correcto en roll-up.
- Orden alfabético estable.
- Convivencia de flags de categorías distintas.



### Headers: nombres, orden y duplicados

Esta especificación define la normalización de nombres de headers, el orden de emisión, el tratamiento de duplicados, y las flags asociadas a anomalías.

---

### Objetivos

- Producir una representación canónica, determinista e idempotente del bloque de headers.
- Señalar duplicados, nombres inválidos y hop-by-hop inesperados.

---

### Normalización de nombres

- Lowercase de nombres de header.
- Trim de espacios alrededor del nombre y del valor.
- Validar nombre contra token de RFC 9110: caracteres permitidos `!#$%&'*+-.^_`|~` y alfanum.
- Si el nombre contiene caracteres prohibidos o underscores (según política), emitir `BADHDRNAME:<name>`.

---

### Orden y emisión

- Ordenar headers por nombre ascendente (binario) para emisión canónica.
- Excepción: `set-cookie` NO se mergea ni se reordena entre sí; se emiten en el orden de llegada.
- Formato de línea: `H:<name>=<value>`.

---

### Duplicados y merge

- Para headers que son listas separadas por comas (según RFC), mergear duplicados uniendo valores con `,` y emitir `DUPHDR:<name>`.
- No mergear `set-cookie`; cada instancia se mantiene en su propia línea.
- Para otros headers no listados como "comma-mergeable", política conservadora: mantener múltiples líneas y aún así emitir `DUPHDR:<name>` si hay más de una.

Headers típicamente mergeables: `accept`, `accept-encoding`, `accept-language`, `cache-control`, `pragma`, `link`, `www-authenticate` (respetando sintaxis), entre otros.

---

### Hop-by-hop en request

- Headers hop-by-hop: `connection`, `te`, `upgrade`, `trailer`.
- Si aparecen en requests donde no corresponden, emitir `HOPBYHOP:<name>`.

---

### Métricas

- `HCNT:{n}`: número de líneas `H:` emitidas (incluye múltiples `set-cookie`).
- `HLEN:{len}@{bucket}`: tamaño agregado de `name: value` (sin CRLF) en bytes.

---

### Interacciones

- `lineas-plegadas-obs-fold.md`: realizar unfold antes de esta fase.
- `separadores-espacios.md`: colapsar whitespace antes del merge.
- `headers-valores-shape-aware.md`: aplica normalizaciones específicas sobre valores después de esta base.

---

### Ejemplos

Duplicados mergeables:
```
Accept: text/html
Accept: */*
```
Salida:
```
H:accept=text/html, */*
FLAGS:[DUPHDR:accept]
```

`set-cookie` múltiple:
```
Set-Cookie: a=1; Path=/
Set-Cookie: b=2; Path=/
```
Salida:
```
H:set-cookie=a=1; Path=/
H:set-cookie=b=2; Path=/
```

Nombre inválido:
```
X_Custom: v
```
Salida (fragmento):
```
H:x_custom=v
FLAGS:[BADHDRNAME:x_custom]
```

Hop-by-hop en request:
```
Connection: keep-alive
```
Salida (fragmento):
```
H:connection=keep-alive
FLAGS:[HOPBYHOP:connection]
```

---

### Casos límite

- Valores con comas internas en headers no mergeables → no unir.
- `host` con discrepancia respecto a `U:` → `HOSTMISMATCH` (ver URL absoluta).
- Conteo de `HCNT` y `HLEN` tras todas las normalizaciones.

---

### Requisitos de prueba

- Merge correcto y emisión de `DUPHDR`.
- No mergear `set-cookie`.
- `BADHDRNAME` con caracteres prohibidos.
- `HOPBYHOP` en headers de request.
- Orden canónico estable por nombre.



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



### Shape + longitud para valores (con redacción por sensibilidad) - (HIGH)

Esta especificación define cómo clasificar valores en "shapes" deterministas, redactar valores sensibles y reportar su longitud (y buckets) para maximizar señal al modelo protegiendo la privacidad.

---

### Objetivos

- Representar valores por su forma (shape) y tamaño, no por su literal.
- Redactar SIEMPRE valores sensibles.
- Mantener determinismo e idempotencia.

---

### Alcance

- Aplica a valores de query y a valores de headers cuando corresponda (p. ej., `authorization`, `cookie` resumido, etc.).
- No altera claves, salvo para clasificación auxiliar (no cambia case ni contenido de claves).

---

### Claves sensibles (redactar siempre)

Patrón (case-insensitive) para nombres de clave/encabezado considerados sensibles:

```
pass|pwd|token|auth|authorization|cookie|session|bearer|jwt|csrf|xsrf|apikey|api_key|access[_-]?token|id_token|refresh[_-]?token|sig|hmac|sso
```

- Si el nombre de la clave o del header coincide con el patrón anterior, el valor se redacta como `<SECRET:shape:len>`.
- Además, si el valor es un JWT válido o un patrón de credencial reconocido, se redacta por contenido aunque la clave no sea sensible (p. ej., `next=eyJ...` que es un JWT).

---

### Shapes (definiciones)

Orden de evaluación (precedencia): `jwt` → `uuid` → `ipv4/ipv6` → `b64url` → `b64` → `hex` → `email` → `uaxurl` → `num` → `lower/upper/alpha/alnum/lowernum/uppernum` → `mixed`.

- `jwt`: tres segmentos Base64URL separados por puntos, sin espacios: `^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$`
- `uuid`: formato 8-4-4-4-12 hex (case-insensitive): `^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`
- `ipv4`: usar validador numérico 0–255 por octeto; regex aproximada: `^([0-9]{1,3}\.){3}[0-9]{1,3}$` + chequeo de rangos
- `ipv6`: usar parser estándar (por ejemplo, `ipaddress.IPv6Address`)
- `b64`: Base64 con padding opcional correcto: `^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$`
- `b64url`: Base64URL sin `+`/`/`, padding opcional: `^[A-Za-z0-9_-]+={0,2}$`
- `hex`: `^[0-9a-fA-F]+$`
- `email`: heurística simple: `^[^@\s]+@[^@\s]+\.[^@\s]+$`
- `uaxurl`: parece URL (p. ej., `^https?://` o esquema válido `^[a-zA-Z][a-zA-Z0-9+.-]*:`)
- `num`: `^[0-9]+$`
- `lower`: `^[a-z]+$`
- `upper`: `^[A-Z]+$`
- `alpha`: `^[A-Za-z]+$`
- `alnum`: `^[A-Za-z0-9]+$`
- `lowernum`: `^[a-z0-9]+$`
- `uppernum`: `^[A-Z0-9]+$`
- `mixed`: todo lo demás (fallback determinista)

Extras opcionales:
- `rand{H}`: etiqueta de alta entropía con bits aproximados por carácter (Shannon). Se adjunta como sufijo si la entropía normalizada supera un umbral (p. ej., > 4.0 bits/char) y longitud >= 16.

---

### Redacción y longitud

- Formato general: `<shape:len>`.
- Si sensible (por clave o por contenido): `<SECRET:shape:len>`.
- Longitud `len` es la longitud en caracteres Unicode del valor tras percent-decode del query (y cualquier decode aplicable del flujo), ANTES de cualquier redacción.
- Opcionalmente adjuntar bucket: `<shape:len@bucket>`; recomendado mantener solo `len` y emitir bucket por separado si se requiere.

---

### Flags relacionadas

- `QLONG`: valor cuyo tamaño excede umbral (p. ej., > 1024). Se agrega al roll-up global.
- Para `jwt`, puede emitirse flag `JWT` opcional. En headers, `authorization` gestiona flags específicas (`AUTHBEARER`, etc.) en su propia spec.

---

### Representación sugerida

- En línea con la URL canónica:
	- `?login=<lower:6>&login=<lower:6>&modo=<lower:6>`
- Por clave (preservando orden de valores):
	- `QK:login=<lower:6>|<lower:6>`

---

### Ejemplos

- `pwd=visionario` → `pwd=<SECRET:alpha:10>`
- `token=eyJhbGciOi...` (JWT) → `token=<SECRET:jwt:836>`
- `id=12345` → `id=<num:5>`
- `hash=14d18cd98f...` → `hash=<hex:32>`
- `next=https://ex.com/a` → `next=<uaxurl:19>`
- `ip=192.168.0.1` → `ip=<ipv4:11>`

---

### Reglas normativas

1. Precedencia de shapes
	- Evaluar en el orden indicado para evitar clasificaciones ambiguas (p. ej., un JWT también es `b64url` pero debe clasificar como `jwt`).

2. Sensibilidad por contenido
	- Si el valor clasifica como `jwt` o evidencia credenciales (p. ej., encabezados `authorization`), tratar como secreto independientemente del nombre de clave.

3. Idempotencia
	- Si un valor ya está redactado (`<...>`), no volver a redaccionar ni transformar.

4. Longitud
	- Calcular `len` antes de redacción; mantener determinismo entre pasadas.

5. Entropía (opcional)
	- Si se activa, añadir sufijo `rand{H}` (p. ej., `<b64url:120 rand4.7>`). Debe ser determinista para el mismo input.

---

### Casos límite

- Valores vacíos → `<num:0>` no es apropiado; producir `<mixed:0>` (o mantener vacío si la representación así lo requiere) y considerar `QEMPTYVAL` (definido en query).
- Valores redactados previamente (`<...>`) → no modificar.
- `b64` con relleno extraño o inválido → no clasifica como `b64`; caerá a `mixed`.
- Emails con TLD no estándar → heurística simple; no forzar validación estricta.

---

### Requisitos de prueba

- Precedencia correcta (`jwt` sobre `b64url`).
- Redacción por clave sensible y por contenido (JWT en clave no sensible).
- Señalización de `QLONG` y entropía opcional.
- Idempotencia: redacción no se vuelve a aplicar.

---

### Interacciones

- Query parsing y percent-decode: esta spec opera después de decodificar tokens clave/valor.
- Headers valores shape-aware: se apoya en esta clasificación; flags adicionales (`AUTHBEARER`, `COOKIE`) se manejan en la spec de headers.



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



### Longitudes con bucketing

Esta especificación define cómo medir y reportar métricas de tamaño relevantes (path y headers) usando buckets para estabilizar la representación y resaltar outliers sin exponer números exactos.

---

### Objetivos

- Proveer señales de tamaño robustas para el modelo (`PLEN`, `PMAX`, `HLEN`, `HCNT`).
- Evitar sobreajuste a valores exactos mediante bucketing determinista.
- Mantener un cálculo consistente tras las normalizaciones estructurales.

---

### Métricas definidas

- `PLEN:{len}@{bucket}`: longitud total del path canónico.
- `PMAX:{len}@{bucket}`: longitud máxima de un segmento de path.
- `HCNT:{n}`: cantidad de headers tras normalización (una línea por header, excluyendo `set-cookie` merge si aplica).
- `HLEN:{len}@{bucket}`: tamaño total de headers en bytes (sumando `name: value\r\n` por línea emitida en salida canónica), o solo valores si se define así. Recomendación: contar `name: value` sin CRLF.

---

### Buckets

- Rangos por defecto: `0-15`, `16-31`, `32-63`, `64-127`, `128-255`, `256-511`, `512-1023`, `>1023`.
- Reglas:
	- Inclusivos por cada frontera inferior y superior excepto el último (`>1023`).
	- El bucket se representa literalmente (p. ej., `64-127`).

---

### Reglas normativas

1. Orden de cálculo
	- `PLEN` y `PMAX` se calculan después de aplicar:
		- Percent-decode por segmentos (según su spec) y preservación de delimitadores.
		- Colapsación de `//` y `/.` y preservación de `..` (esta spec no resuelve `..`).
	- `HCNT` y `HLEN` se calculan después del unfold (obs-fold), colapsación de whitespace y merge de duplicados definido en headers.

2. Determinismo
	- El conteo debe ser idéntico en segundas pasadas del preprocesador.

3. Emisión
	- `PLEN` y `PMAX` se emiten junto al `P:` del path.
	- `HCNT` y `HLEN` se emiten al final del bloque de headers.

---

### Ejemplos

Path corto:
```
P:/a/b PLEN:4@0-15 PMAX:1@0-15
```

Path más largo:
```
P:/alpha/beta/gamma PLEN:19@16-31 PMAX:5@0-15
```

Headers:
```
HCNT:12 HLEN:512@512-1023
```

---

### Casos límite

- Path vacío o `/` → `PLEN:1@0-15`, `PMAX:0@0-15`.
- Sin headers → `HCNT:0 HLEN:0@0-15`.
- Valores extremos (>64 KiB) → bucket superior (`>1023`) o ampliar buckets por configuración.

---

### Requisitos de prueba

- Cálculo correcto tras colapsar `//` y `/.` y con `..` presentes.
- Idempotencia después de segunda pasada.
- Buckets correctos en fronteras (15,16,31,32, ...).
