# Preprocessing Snapshot Diff

Before: `docs/preprocessing_snapshots/snapshots/2026-03-08_after-wspad-order.jsonl`
After: `docs/preprocessing_snapshots/snapshots/2026-03-08_after-wspad-order-v2.jsonl`
Packets: 18

Changed: 1
Added: 0
Removed: 0

## Changes

### P13 - Duplicate mergeable headers (accept)

- before sha256: `c5879b6460ebdd0e3f73dd1687e6fcaef379d6e6ae4fed3efdaba48e31d3395f`
- after sha256: `edfd7a5afe21ea465f43caa6953f3462642f664c178922d6ba97142b5625ce40`

```diff
--- before
+++ after
@@ -1,9 +1,9 @@
 [METHOD] GET
 [URL] / HOME
 [URL_ABS] http://example.com/
-[HEADER] accept:text/html, application/xml DUPHDR HDRMERGE
+[HEADER] accept: text/html, application/xml DUPHDR HDRMERGE
 [HEADER] host: example.com
-[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=38
+[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=39
 [HGF] HDRNORM
 [QMETA] count=0
 [FLAGS] DUPHDR HDRMERGE HDRNORM HOME
```

