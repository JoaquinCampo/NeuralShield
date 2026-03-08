# Preprocessing Snapshot Diff

Before: `docs/preprocessing_snapshots/snapshots/2026-03-07_after-plan-v2.jsonl`
After: `docs/preprocessing_snapshots/snapshots/2026-03-08_after-wspad-order.jsonl`
Packets: 18

Changed: 4
Added: 0
Removed: 0

## Changes

### P03 - Unusual method

- before sha256: `d7e883c4f172e2e75b5835f1a9399a93602b4a3d8762f258be2281e6d4753995`
- after sha256: `faaeceb7798cbf01d482a2ac6b2ff9eeaad2e2b0ae7aea83c15247c43eef79e8`

```diff
--- before
+++ after
@@ -1,8 +1,8 @@
 [METHOD] GOT
 [URL] / HOME
 [URL_ABS] http://example.com/
-[FLAGS] HDRNORM HOME UNUSUAL_METHOD
 [HEADER] host: example.com
 [HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=12
 [HGF] HDRNORM
 [QMETA] count=0
+[FLAGS] HDRNORM HOME UNUSUAL_METHOD
```

### P06 - Absolute-form target + host mismatch

- before sha256: `9d82c4c214c61ed4d56114eb5846e2b96ecd30b1596fb1947717b85cee15001c`
- after sha256: `bafe0a5514b8a4a79b6be118f8aa3e3fe9a58fa2023bccb9708187c403df3bd6`

```diff
--- before
+++ after
@@ -4,5 +4,5 @@
 [HEADER] host: b.example
 [HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=10
 [HGF] HDRNORM
+[QMETA] count=0
 [FLAGS] HDRNORM HOSTMISMATCH
-[QMETA] count=0
```

### P13 - Duplicate mergeable headers (accept)

- before sha256: `d62efe3e905c03421dd9273f0b9c1adeccc19f4d6726bcc08849694d3da6b94b`
- after sha256: `c5879b6460ebdd0e3f73dd1687e6fcaef379d6e6ae4fed3efdaba48e31d3395f`

```diff
--- before
+++ after
@@ -1,9 +1,9 @@
 [METHOD] GET
 [URL] / HOME
 [URL_ABS] http://example.com/
-[HEADER] accept: text/html, application/xml DUPHDR HDRMERGE
+[HEADER] accept:text/html, application/xml DUPHDR HDRMERGE
 [HEADER] host: example.com
-[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=40
+[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=38
 [HGF] HDRNORM
 [QMETA] count=0
 [FLAGS] DUPHDR HDRMERGE HDRNORM HOME
```

### P18 - Header whitespace anomalies (WSPAD)

- before sha256: `eb6444b53b054cf70918369c33b01c5e787d181f667fcec1f0ae2d27ecaa8cce`
- after sha256: `d2550eafd67f4ef94baaa65f0a0ef36ee0853c35d57664cb41fbe0077909ae8a`

```diff
--- before
+++ after
@@ -2,8 +2,8 @@
 [URL] / HOME
 [URL_ABS] http://example.com/
 [HEADER] host: example.com
-[HEADER] x-test: value with tabs
-[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=28
+[HEADER] x-test:   value\twith\t tabs WSPAD
+[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=33
 [HGF] HDRNORM
 [QMETA] count=0
-[FLAGS] HDRNORM HOME
+[FLAGS] HDRNORM HOME WSPAD
```

