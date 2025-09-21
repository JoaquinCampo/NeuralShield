### 01 Request Structurer

**Step Contract:**

- Input lines considered: cleaned raw HTTP request string
- Transformations allowed: parse into structured format with prefixes
- Flags emitted: syntax errors as exceptions (no flags)
- Dependencies: 00 (framing cleanup)
- Idempotence: multiple applications produce same result

---

### Scope

Transform raw HTTP request into structured canonical format:

- `[METHOD] <method>`
- `[URL] <path>`
- `[QUERY] <raw-param>` (repeated per query token)
- `[HEADER] <raw-header-line>` (repeated per header line)

Parse without interpreting values - just structure the request into components.

---

### Rules

1. **Request Line Parsing**:

   - Split into exactly 3 parts: method, URL, HTTP version
   - Validate method is in allowed list: GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD, TRACE, CONNECT
   - Validate HTTP version starts with "HTTP/"

2. **URL/Query Splitting**:

   - Split URL on first `?` into path and query string
   - If no `?`, query string is empty

3. **Query Tokenization**:

   - Split query string on `&` with HTML entity protection
   - Preserve HTML entities like `&#x3c;` during splitting
   - Each token becomes one `[QUERY]` line

4. **Header Processing**:
   - Split on literal `\n` until first empty line
   - Each header line becomes one `[HEADER]` line
   - Preserve original casing and spacing

---

### Examples

Input:

```
GET /path?param1=value1&param2=value2 HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0

```

Output:

```
[METHOD] GET
[URL] /path
[QUERY] param1=value1
[QUERY] param2=value2
[HEADER] Host: example.com
[HEADER] User-Agent: Mozilla/5.0
```

---

### HTML Entity Protection

When splitting query strings, protect HTML entities from being broken:

```
?param=a&#x26;b&other=c
```

Should split as: `["param=a&#x26;b", "other=c"]`, not break the entity.

---

### Error Handling

Raise `MalformedHttpRequestError` for:

- Wrong number of request line parts
- Invalid HTTP version format
- Invalid method
- Empty request

---

### Supersedes

- Original `RequestStructurer` functionality

---

### Good vs Bad Examples

Good (properly formed request):

```
GET /path?x=1&y=2 HTTP/1.1\n
Host: example.com\n
User-Agent: Test/1.0\n
\n
```

Output:

```
[METHOD] GET
[URL] /path
[QUERY] x=1
[QUERY] y=2
[HEADER] Host: example.com
[HEADER] User-Agent: Test/1.0
```

Bad (invalid request line - 2 parts only):

```
GET /missing-version\n
Host: example.com\n
\n
```

Result: raises `MalformedHttpRequestError`
