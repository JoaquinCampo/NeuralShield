# Preprocessing Snapshot Diff

Before: `docs/preprocessing_snapshots/snapshots/2026-03-07_after-plan.jsonl`
After: `docs/preprocessing_snapshots/snapshots/2026-03-07_after-plan-v2.jsonl`
Packets: 18

Changed: 1
Added: 0
Removed: 0

## Changes

### P06 - Absolute-form target + host mismatch

- before sha256: `8f8a92fb86f8402937c9ee9360248407343e369b7e37213ae29746cdf64f606d`
- after sha256: `9d82c4c214c61ed4d56114eb5846e2b96ecd30b1596fb1947717b85cee15001c`

```diff
--- before
+++ after
@@ -1,8 +1,8 @@
 [METHOD] GET
-[URL] http:/a.example/x MULTIPLESLASH
+[URL] http://a.example/x
 [URL_ABS] http://a.example/x
 [HEADER] host: b.example
 [HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=10
 [HGF] HDRNORM
-[FLAGS] HDRNORM HOSTMISMATCH MULTIPLESLASH
+[FLAGS] HDRNORM HOSTMISMATCH
 [QMETA] count=0
```

