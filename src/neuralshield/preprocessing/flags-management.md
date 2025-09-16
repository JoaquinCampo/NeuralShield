### Gestión de flags en requests (Fase 0 y general)

Este documento define cómo detectar, nombrar e insertar flags de rarezas en requests HTTP sin romper la semántica del mensaje ni el resto del pipeline.

### Principios

- **Idempotencia**: cada ejecución produce el mismo resultado; no duplica flags.
- **No romper HTTP**: no tocar la request-line ni el body; sólo headers.
- **Separación de preocupaciones**: primero detectar/medir, luego normalizar, luego (opcional) insertar flags.
- **Compatibilidad**: mantener alias con nombres antiguos cuando aplique.

### Taxonomía y nombres

- Formato recomendado: `CATEGORIA:SUBTIPO[=valor]`, en mayúsculas.
- Ejemplos Fase 0 (terminadores de línea y líneas plegadas):
  - `EOL:MIX`
  - `EOL:BARELF`
  - `EOL:BARECR`
  - `EOL:EOF_NOCRLF`
  - `EOL:STAT:CRLF=<n>;LF=<n>;CR=<n>` (telemetría)
  - `OBS:FOLD`
  - `OBS:FOLD_N=<n>` (telemetría)
- Flags de calidad de headers (opcionales):
  - `HDR:MALFORMED=<n>`
  - `HDR:BADNAME=<n>`
  - `HDR:WSLINE=<n>`

Alias (compatibilidad):

- `EOLMIX → EOL:MIX`, `EOL_BARELF → EOL:BARELF`, `EOL_BARECR → EOL:BARECR`, `EOL_EOF_NOCRLF → EOL:EOF_NOCRLF`, `EOLSTAT → EOL:STAT`, `OBSFOLD → OBS:FOLD`, `OBSFOLD_N → OBS:FOLD_N`.

### Proceso recomendado (Fase 0)

1. `remove_framing_artifacts` (log-only): quita BOM y controles en bordes; registra telemetría con loguru.
2. Detección EOL en el raw: contar `CRLF`, `LF` sueltos y `CR` sueltos; emitir flags `EOL:*` y `EOL:STAT`.
3. Normalización EOL a CRLF (sólo en headers): elimina ambigüedades para el pipeline.
4. Detección y unfolding estricto de obs-fold (sólo en headers): unir continuaciones válidas con un espacio; emitir `OBS:*`.
5. (Opcional) Inserción in-band de flags en headers: ver siguiente sección.

Notas:

- El body no se toca en Fase 0 (puede ser binario o estar regido por `Content-Length`).
- La separación headers/body se detecta antes de normalizar (primer línea vacía).

### Inserción de flags dentro del request (segura)

- Se realiza sólo si hay flags presentes y si la configuración lo permite.
- Se agrega una cabecera sintética, justo antes de la línea vacía que separa headers del body:
  - `X-NS-Flags: EOL:MIX, EOL:BARELF, OBS:FOLD, OBS:FOLD_N=3`
- Reglas:
  - No duplicar: si ya existe `X-NS-Flags`, fusionar y desduplicar tokens.
  - Separador: coma `,` entre tokens; `=` para contadores.
  - Idempotente: reinserciones no alteran el resultado.
- Anotaciones locales (opcional):
  - `X-NS-Ann: hdr:3=OBS:FOLD`, `X-NS-Ann: hdr:7=HDR:BADNAME` para apuntar a líneas específicas.

Parámetros de configuración sugeridos:

- `flag_injection`: `off | header | annotated | both` (por defecto: `header`).
- `flag_injection_min_severity`: inyectar sólo si aparece alguna rareza de interés.
- `flag_injection_max_flags`: límite superior para la cantidad de tokens (p. ej., 16).

### Reglas de unfolding (strict)

- Sólo en headers. Nunca en request-line ni en body.
- Unir cuando la línea previa es `campo:"` válido y la siguiente inicia con SP o HTAB; sin línea en blanco entre ambas.
- Varias continuaciones seguidas son válidas; se unen con un único espacio.
- No unir y flaggear cuando:
  - Línea previa malformada (sin `:`) → `HDR:MALFORMED`.
  - Nombre de header inválido (no `tchar`) → `HDR:BADNAME`.
  - Línea de continuación es sólo whitespace → `HDR:WSLINE`.

### Logging (telemetría)

- Usar `loguru`.
- `debug`: resúmenes y estadísticas.
- `info`: anomalías detectadas (EOL mixto, unfolding efectuado, headers malformados).
- Los logs no forman parte del feature set; sólo auditoría.

### Ejemplo (antes → después)

Input (desordenado):

```
get /p%252Eath//../file.jsp?login=foo&login=bar HTTP/1.1
Host: Example.com
User-Agent: Mozilla\t5.0
Accept:
 text/html;q=0.9
 */*;q=0.5


```

Salida (Fase 0; EOL normalizado en headers, unfolding y flags in-band):

```
GET /p%252Eath//../file.jsp?login=foo&login=bar HTTP/1.1\r\n
Host: Example.com\r\n
User-Agent: Mozilla 5.0\r\n
Accept: text/html;q=0.9 */*;q=0.5\r\n
X-NS-Flags: EOL:MIX, OBS:FOLD, OBS:FOLD_N=1\r\n
\r\n
```

Notas del ejemplo:

- `get` → `GET` no es responsabilidad de Fase 0; se muestra sólo a modo ilustrativo de "canónico" posterior.
- `\t` en `User-Agent` se convierte en espacio como resultado del unfolding (continuación con HTAB) bajo reglas strict.
- `X-NS-Flags` refleja rarezas detectadas sin alterar semántica.

### Representación canónica y features

- La inserción de flags en headers permite que etapas posteriores emitan líneas canónicas (p. ej., `FU`, `H`, `Q`) con contexto estable.
- Si se requiere salida estilo "Canonical v2", los flags pueden mapearse a tokens del feature set (p. ej., `FU > OBS:FOLD`, `EOL > MIX`).

### Idempotencia y determinismo

- Reaplicar Fase 0 no cambia el texto ni duplica flags.
- La normalización a CRLF en headers garantiza un recorrido determinista para las etapas siguientes.
