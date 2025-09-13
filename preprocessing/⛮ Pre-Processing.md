- ### Normalizar y Agregar una flag de que es lo raro
	- 1. **Capturás la request** tal como vino.
	1. **Decodificás** percent-encoding una sola vez.
	2. **Normalizás Unicode** a ASCII donde sea posible.
	3. **Agregás flags** si había:
		- fullwidth chars
		- doble codificación
		- caracteres de control
		- entidades HTML
	4. **Salida**: versión estable + flags
	5. Ejemplo final:
		1. M:GET
		2. U:/path.jsp
		3. FLAGS: FULLWIDTH DOUBLEPCT
		4. H:host=example.com
		5. H:user-agent=mozilla/5.0

- ###  Líneas plegadas (obs-fold) → desplegar

		Qué es: Antes en HTTP/1.1 se permitía header folding = continuar una línea con un espacio o tab al inicio de la siguiente.
		Ejemplo:
		X-Test: valor1
			valor2
	- Significa `X-Test: valor1 valor2`.
	- **Problema**: Es raro hoy, a veces usado para _obfuscation_ o bypass de parsers.
	- **Solución**:
	    - Detectar líneas que empiezan con espacio/tab y unirlas a la anterior.
	    - Si ocurre, añadir `FLAGS:[OBSFOLD]` para que el modelo sepa que hubo plegado.

- ###  Separadores (\t, espacios múltiples) → un espacio
	Qué es: Simplificar cualquier espacio redundante (tabs, múltiples espacios) a un solo espacio.
	Por qué:
	Un modelo no necesita aprender diferencias irrelevantes como “\t” vs “ ”.
	Evita que un atacante esconda payloads con padding extraño.
	Cómo: Reemplazar cualquier secuencia \t + por " ".

	Ejemplo
	Entrada:
		User-Agent:   Mozilla\t5.0
    Salida:
		User-Agent: Mozilla 5.0
		WSPAD

  - Pasar metodo a mayusculas, si no es uno de los estandares poner OTHER

- ### **Construir URL absoluta cuando es relativa**
	- **Qué es:**  
     Si la request dice `GET /path/file.jsp HTTP/1.1`, ese `/path/file.jsp` es **relativo**. No incluye esquema (`http://` o `https://`) ni host.  
     Pero en los headers aparece `Host: mi-dominio.com`.
    
	- **Qué hacemos:**  
      Armamos una URL absoluta:
      http://mi-dominio.com/path/file.jsp
		- Si el puerto es `80` para `http` o `443` para `https`, lo omitimos:  
    `http://mi-dominio.com:80/...` → `http://mi-dominio.com/...`
	      - Si es otro puerto, lo mantenemos:  
	    `http://mi-dominio.com:8080/...`
		**Por qué ayuda:**
	- Evita que el modelo tenga que aprender a unir `Host` + path.
	- Siempre trabajamos con una forma estable: `scheme://host[:port]/path?query`.

- ### **Percent-decode exactamente 1 vez + flags**
	- **Qué es:**  
	    Las URLs pueden tener caracteres codificados: `%2E` = `.`  
    `%252E` = `%2E` codificado dos veces → se vuelve `.` sólo tras decodificar dos veces.
	- **Qué hacemos:**
    - Decodificamos **una sola vez** (`%2E` → `.`) para forma canónica.
    - Si tras eso sigue habiendo `%` → posible obfuscación, marcamos con `FLAGS:[DOUBLEPCT]`.
    - Ejemplo:
	    - /foo%2Ejsp        → /foo.jsp          (ok)
		 /foo%252Ejsp      → /foo%2Ejsp + [DOUBLEPCT]
	 **Por qué ayuda:**
		- El modelo ve siempre la forma más “limpia”.
		- La presencia de `DOUBLEPCT` alerta sobre intentos de bypass.

- ### **Colapsar `//` y `/.` pero NO resolver `..`**
	- **Qué es:**
    - `//` o `/.` no cambian la semántica → los colapsamos a `/`.
	- `..` sube un directorio → no lo resolvemos, sólo lo marcamos (`[DOTDOT]`), porque podría ser un ataque (path traversal).
	- Ejemplo:
		- /foo//bar/.//baz  → /foo/bar/baz + MULTIPLESLASH
		- /foo/../etc/passwd → /foo/../etc/passwd + DOTDOT

    FLAG:(HOME).

  - FLAG:(../*15)
      ../../../../../../../../../../../../../../  -> FLAG:(../*15)/etc/paswd
      * A expandir: 8.3

- ###  Longitudes con bucketing

  - Qué es: medir tamaños y ubicarlos en rangos (buckets) para que el modelo capte “lo raro” sin memorizar números exactos.
  - PLEN: longitud total del path (caracteres).
  - PMAX: longitud máxima de un segmento.
  - Buckets típicos: 0-15, 16-31, 32-63, 64-127, 128-255, >255 (ajusta a tu dataset).
  - Salida sugerida:
  - PLEN:{len}@{bucket} PMAX:{len}@{bucket}
  - Ejemplo

- ###  Caracteres peligrosos + Script mixing

  - Caracteres peligrosos: % (despues de decodificado en URL), <, >, ', ", ;, (, ), {, }, |, \, espacio, %00 (NUL).
  - Si aparecen en un segmento → marcamos con FLAGS:[ANGLE][QUOTE][SEMICOLON]...
  - Ejemplo: <script> en un query param → [ANGLE]
  - Script mixing: Mezcla de alfabetos (Latín + Cirílico + Griego) para homoglyph attacks:
  - раypal.com (con p cirílica) vs paypal.com
  - Si hay mezcla → FLAGS:[MIXEDSCRIPT]
  - Esto es clave para detectar XSS, SQLi, Unicode obfuscation, etc.

- ### Extraer el query y decodificar % exactamente 1 vez

  - Aplica percent-decode una sola vez a query (no al path aquí).
  - Si, tras esa decodificación, aún quedan % seguidos de hex válido (%2E, %3F, etc.), marca FLAGS:[DOUBLEPCT] (indicio de doble codificación/obfuscación).
  - No conviertas + → espacio por defecto (detalle más abajo).
  - Por qué: normalizas lo sano y exponés obfuscaciones sin “arreglarlas” del todo.

- ###  Separadores & parseo robusto (header)

  - Separa pares por &. Algunos servidores aceptan ; como separador; si ves ; repetidos con patrón k=v;k=v, puedes:
  - o bien tratarlos también como separadores y marcar FLAGS:[QSEMISEP],
  - o dejarlos literales y marcar FLAGS:[QRAWSEMI].
  - Cada par se divide en key[=value]. Si no hay = → FLAGS:[QBARE] (par “desnudo”).
  - Tip (Python): parse_qsl(query, keep_blank_values=True, strict_parsing=False, separator='&'). Si detectas ; dominante, usa separator=';&'.

- ###  No traduzcas + a espacio (por defecto)

  - En URLs (RFC 3986) el + es un carácter literal. - A CHEQUEAR
  - En cuerpos application/x-www-form-urlencoded (p. ej. POST) sí se usa + como espacio. Aquí no hay body, así que no lo traduzcas.
  - Si en el futuro procesás body con ese Content-Type, ahí sí traduce +→ y marca el origen.
  - Beneficio: evitas falsos positivos donde + tiene significado (tokens, hashes base64url, etc.).

- ###  Ordenar claves y preservar multiplicidad (LOW)

  - Ordena las claves alfabéticamente para invarianza de orden entre requests.
  - Preserva el orden de llegada de los valores por clave (muy importante: algunos frameworks aplican el primero, otros el último).
  - Representación recomendada (una de estas dos):
  - Compacta por clave (preserva orden de valores):
    - QK:login=<lower:6>|<lower:6>
    - QK:modo=<lower:6>
  - En línea única con la URL canónica (más corta):
    - ?login=<lower:6>&login=<lower:6>&modo=<lower:6>
  - Emite Q:{n} (total de pares) y KEYS:{lista} para resumen:
    - Q:5  KEYS:B1,login,modo,pwd,remember
  - Colación/case: en general no cambies el case de las claves (algunas apps distinguen User vs user). Ordená usando una colación binaria (ASCII) pero no toques las claves.

- ### Shape + longitud para valores (con redacción por sensibilidad) - (HIGH)

  - Clasifica cada valor en un shape y reemplázalo por <shape:len> (o <SECRET:shape:len> si sensible):
  - Claves sensibles (redactar siempre):
    - pass|pwd|token|auth|authorization|cookie|session|bearer|jwt|csrf|xsrf|apikey|api_key|access[_-]?token|id_token|refresh[_-]?token|sig|hmac|sso
  - Shapes útiles (elige subset razonable y determinista):
    - num (solo dígitos)
    - hex (0–9 a–f)
    - uuid (formato 8-4-4-4-12 válido)
    - ipv4, ipv6
    - b64 (Base64 con padding válido)
    - b64url (Base64URL sin +//, quizá sin =)
    - email (heurística simple)
    - uaxurl (parece URL)
    - lower, upper, alpha (solo letras), alnum
    - lowernum, uppernum, mixed (mezcla general)
  - Extras opcionales:
    - jwt (tres segmentos base64url: xxx.yyy.zzz)
    - Entropía alta → rand{H} (si calculas Shannon)
    - Longitud-bucket además del valor exacto (ej. len=37@32-63)
      - Ejemplos
      - pwd=visionario             → pwd=<SECRET:alpha:10>
      - token=eyJhbGciOi...        → token=<SECRET:jwt:836>
      - id=12345                   → id=<num:5>
      - hash=14d18cd98f...         → hash=<hex:32>
      - next=https://ex.com/a      → next=<uaxurl:19>
      - Por qué: el modelo ve forma y tamaño (muy informativo) sin ver el valor concreto (privacidad + generalización).

- ### Flags por rarezas - (HIGH)
  - Valores vacíos: k= → FLAGS:[QEMPTYVAL]
  - Clave repetida: k=a&k=b → FLAGS:[QREPEAT:k]
  - Par sin =: justkey → FLAGS:[QBARE]
  - Valores con NUL (%00) tras decode: FLAGS:[QNUL]
  - Non-ASCII en clave/valor: FLAGS:[QNONASCII]
  - Arrays/brackets: k[]=a&k[]=b → marca FLAGS:[QARRAY:k] (sin cambiar lógica de multiplicidad)
  - Pares con ; como separador: FLAGS:[QSEMISEP]
  - Valor muy largo (p. ej., >1 KB): FLAGS:[QLONG]

- ###  Encodings y decodificaciones “raras”

  - UTF-8 overlong o secuencias inválidas → FLAGS:[BADUTF8]
  - Doble/triple encoding (%252e, %255c) → FLAGS:[MULTIENC:k]
  - HTML entities si aparecen en path/query (&#x2f;) → decodificar 1 vez y marcar.
    → FLAGS:[HTMLENT]

- ###  Headers: nombres, orden y duplicados

  - Lowercase names, trim y colapsar espacios; ordenar por nombre.
  - Duplicados → unir con coma y marcar. → FLAGS:[DUPHDR:accept]
  - Nombres inválidos (caracteres prohibidos, subrayado en algunos entornos) → FLAGS:[BADHDRNAME:x_custom]
  - Hop-by-hop inesperados en request (connection, te, upgrade, trailer) → FLAGS:[HOPBYHOP:name] - (LOW)
  - Header count y tamaño total con bucketing (mitiga DoS / obfusc).
    → HCNT:12 HLEN:512@512-1023

- ### Headers: valores “shape-aware”

  - user-agent → tokenizar grosero (nombre/version + plataformas); cortar a N tokens, sin comentarios; shape del resto.
    - → H:user-agent=mozilla/5.0 konqueror/3.5 khtml/3.5.8 like:gecko
  - accept* → lista normalizada type/subtype;qX; ordenar por type y q descendente.
    - → H:accept=text/html;q0.9 text/plain;q0.8 */*;q0.5 …
  - accept-encoding → set de codificaciones.
    - → H:accept-encoding=gzip deflate x-gzip x-deflate
  - accept-language → estandarizar a ll-CC y ordenar por q.
  - cookie → solo nombres y longitudes (ordenados).
    - → H:cookie=JSESSIONID<len:32> [COOKIE:1]
  - authorization → siempre secreto y shape (Bearer, Basic, Digest).
    - → H:authorization=<SECRET:bearer:142> FLAGS:[AUTHBEARER]
  - x-forwarded-for / forwarded → extraer lista de IPs con shape.
    - → H:x-forwarded-for=ipv4,ipv4,private FLAGS:[XFF]
  - host → comparar con host del target absoluto; si difieren → FLAGS:[HOSTMISMATCH]
  - Headers inusuales (no en lista blanca) → marcar UNKHDR (opcional, o basada en vocabulario visto por app).
    - → FLAGS:[UNKHDR:x-foo]
