### 03 Bytes to Unicode Conversion

**Step Contract:**
- Input lines considered: raw request (if bytes) or structured request (if already string)
- Transformations allowed: convert bytes to Unicode with UTF-8 tolerant decoding
- Flags emitted: `BADUTF8`
- Dependencies: 01 (request structurer) or 00 (framing cleanup) for bytes input
- Idempotence: multiple applications produce same result

---

### Scope

Handle the conversion from bytes to Unicode strings with proper error detection:
- Apply UTF-8 decoding in tolerant mode ("replace")
- Detect and flag invalid or overlong UTF-8 sequences
- Ensure consistent Unicode representation for downstream processing

---

### Rules

1. **Input Detection**: Determine if input is bytes or already a string
2. **Tolerant Decoding**: Use UTF-8 with "replace" error handling to avoid crashes
3. **Invalid Sequence Detection**: Flag when replacement characters are inserted
4. **Overlong Sequence Detection**: Detect overlong encodings like `\xC0\xAF` for `/`
5. **No-op for Strings**: If input is already a string, pass through unchanged

---

### Flags Emitted

- `BADUTF8`: Invalid or overlong UTF-8 sequences detected during decoding

---

### Examples

Input with overlong UTF-8:
```
b'GET /%C0%AFetc/passwd HTTP/1.1\r\nHost: example.com\r\n\r\n'
```

Output:
```
GET /%C0%AFetc/passwd HTTP/1.1
Host: example.com

BADUTF8
```

Input with invalid UTF-8 bytes:
```
b'GET /path\xFF\xFE HTTP/1.1\r\nHost: example.com\r\n\r\n'
```

Output:
```
GET /path�� HTTP/1.1
Host: example.com

BADUTF8
```

Normal string input (no-op):
```
GET /path HTTP/1.1
Host: example.com

```

Output:
```
GET /path HTTP/1.1
Host: example.com

```

---

### Implementation Notes

- This step may be a no-op if the pipeline always receives string input
- Useful for scenarios where raw bytes are received from network sockets
- The "replace" strategy ensures processing can continue even with invalid input
- Flag emission helps identify potential encoding-based attacks

---

### Supersedes

- New step, no direct predecessor in original implementation
