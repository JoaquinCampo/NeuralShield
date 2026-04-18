# Preprocessing Snapshot Diff

Before: `docs/preprocessing_snapshots/snapshots/2026-03-07_baseline.jsonl`
After: `docs/preprocessing_snapshots/snapshots/2026-03-08_after-wspad-order-v2.jsonl`
Packets: 18

Changed: 12
Added: 0
Removed: 0

## Changes

### P02 - Normal GET (CRLF)

- before sha256: `9969e85a6c3bf800587dbaaad197ff87816f5b1922917b1d0d79557370887f2b`
- after sha256: `840355a9daf522569c6221389f4b34fe719692c6cb656a49e5252933e8bc2cbf`

```diff
--- before
+++ after
@@ -1,9 +1,9 @@
 [METHOD] GET
 [URL] / HOME
 [URL_ABS] http://example.com/
-[HEADER] host: example.com BADCRLF
-[HEADER] user-agent: curl/8.0 BADCRLF
-[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=39
+[HEADER] host: example.com
+[HEADER] user-agent: curl/8.0
+[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=21
 [HGF] HDRNORM
 [QMETA] count=0
-[FLAGS] BADCRLF HDRNORM HOME
+[FLAGS] HDRNORM HOME
```

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

### P04 - obs-fold continuation (X-Note + X-Admin)

- before sha256: `4198aaca4433eb08182cfd6794362bf9db28ba14d1a5f7337873910bd77fb450`
- after sha256: `9ae10bbbc2c3f7f6e23f6f936815b0c71e66cfc01ee1e177ff4e01a4a032619f`

```diff
--- before
+++ after
@@ -3,7 +3,7 @@
 [URL_ABS] http://example.com/
 [HEADER] host: example.com
 [HEADER] x-note: ok X-Admin: true OBSFOLD
-[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=37
+[HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=29
 [HGF] HDRNORM
 [QMETA] count=0
 [FLAGS] HDRNORM HOME OBSFOLD
```

### P05 - Orphan obs-fold continuation

- before sha256: `892df241ab82ffa9d17591af690b8ba57a3d95864cf2455327ebb046d5d8211c`
- after sha256: `9ed2a9846324e2399e0e3f894d9f9951b5159a5862c7add1190668e52b7aa09b`

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 [URL] / HOME
 [URL_ABS] http://example.com/
 [HEADER] host: example.com orphan-continuation OBSFOLD
-[HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=40
+[HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=32
 [HGF] HDRNORM
 [QMETA] count=0
 [FLAGS] HDRNORM HOME OBSFOLD
```

### P06 - Absolute-form target + host mismatch

- before sha256: `8f8a92fb86f8402937c9ee9360248407343e369b7e37213ae29746cdf64f606d`
- after sha256: `bafe0a5514b8a4a79b6be118f8aa3e3fe9a58fa2023bccb9708187c403df3bd6`

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
 [QMETA] count=0
+[FLAGS] HDRNORM HOSTMISMATCH
```

### P07 - Path normalization (dot + multiple slashes)

- before sha256: `7c1e91f6cee105c823055d6b783f69e28dc9bc5c87f04e7be99747f1b2dcb2e1`
- after sha256: `009490d4393f3ee2da51f70fc336283135ffe23abc4feca725a2815951ae5291`

```diff
--- before
+++ after
@@ -1,6 +1,6 @@
 [METHOD] GET
 [URL] /A/B/C/../D DOTCUR DOTDOT MULTIPLESLASH
-[URL_ABS] http://example.com/A/./B//C/../D
+[URL_ABS] http://example.com/A/B/C/../D
 [HEADER] host: example.com
 [HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=12
 [HGF] HDRNORM
```

### P09 - Query double-encoding and space evidence

- before sha256: `d0f02412f5a075ec470d7e37aae6447ff26234de87bc4304b33a7c77ef201ae7`
- after sha256: `007608f90cfdef09007711cf03d0c303d1d4f9e0c5b56f4ce8ddcad3d03d31f1`

```diff
--- before
+++ after
@@ -1,9 +1,9 @@
 [METHOD] GET
 [URL] / HOME
 [URL_ABS] http://example.com/
-[QUERY] enc=a b PCTSPACE
+[QUERY] enc=a%20b DOUBLEPCT PCTSPACE
 [HEADER] host: example.com
 [HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=12
 [HGF] HDRNORM
-[QMETA] count=1 PCTSPACE
-[FLAGS] HDRNORM HOME PCTSPACE
+[QMETA] count=1 DOUBLEPCT PCTSPACE
+[FLAGS] DOUBLEPCT HDRNORM HOME PCTSPACE
```

### P10 - Query NUL percent-encoding

- before sha256: `a348855edff35109f3ef6babf5b884cab0487a80371e56132f2719e7cb538f9a`
- after sha256: `e1fe9243f2578d9e9008e5bb45746e9cbaa58363836972e07cf5ef9eeccd0d2a`

```diff
--- before
+++ after
@@ -1,7 +1,7 @@
 [METHOD] GET
 [URL] / HOME
 [URL_ABS] http://example.com/
-[QUERY] nul=\x00 CONTROL NUL PCTNULL QNUL
+[QUERY] nul=%00 CONTROL NUL PCTNULL QNUL
 [HEADER] host: example.com
 [HAGG] h_count=1 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=12
 [HGF] HDRNORM
```

### P11 - Query with HTML entity ampersand

- before sha256: `f773c1fffe2622d0cddf5ef7fe81fd56836bcb877912845df89efd20f30c94ac`
- after sha256: `20d2be06a1446784ff8457f818c059757c4b95da6b3c50afbeb88113e4a0b369`

```diff
--- before
+++ after
@@ -8,4 +8,4 @@
 [HGF] HDRNORM
 [QSEP] MIXEDSEP QRAWSEMI
 [QMETA] count=2 HTMLENT SEMICOLON
-[FLAGS] HDRNORM HOME HTMLENT SEMICOLON
+[FLAGS] HDRNORM HOME HTMLENT MIXEDSEP QRAWSEMI SEMICOLON
```

### P12 - Mixed query separators (; and &)

- before sha256: `b7118bcc688f15feaaf8d6ba99a51640b9121ed3f4b0dbea6cd35f8930460782`
- after sha256: `f54dfadb883d31cb55ef52809ce58a190031092ba7400b833a1781852e08f003`

```diff
--- before
+++ after
@@ -9,4 +9,4 @@
 [HGF] HDRNORM
 [QSEP] QSEMISEP
 [QMETA] count=3 SEMICOLON
-[FLAGS] HDRNORM HOME SEMICOLON
+[FLAGS] HDRNORM HOME QSEMISEP SEMICOLON
```

### P13 - Duplicate mergeable headers (accept)

- before sha256: `fc12e7cfc605753d5bf3ff8811891557ec193c172fe81671dc61664121fc2267`
- after sha256: `edfd7a5afe21ea465f43caa6953f3462642f664c178922d6ba97142b5625ce40`

```diff
--- before
+++ after
@@ -1,9 +1,9 @@
 [METHOD] GET
 [URL] / HOME
 [URL_ABS] http://example.com/
-[HEADER] accept: text/html, application/xml DUPHDR,HDRMERGE WSPAD
+[HEADER] accept: text/html, application/xml DUPHDR HDRMERGE
 [HEADER] host: example.com
-[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=40
+[HAGG] h_count=2 dup_names=1 hopbyhop=0 bad_names=0 total_bytes=39
 [HGF] HDRNORM
 [QMETA] count=0
-[FLAGS] HDRNORM HOME WSPAD
+[FLAGS] DUPHDR HDRMERGE HDRNORM HOME
```

### P18 - Header whitespace anomalies (WSPAD)

- before sha256: `517fb7bc8e4804fd10f152f990e4699cc035dc0a393fa93fc3a013467a25b302`
- after sha256: `d2550eafd67f4ef94baaa65f0a0ef36ee0853c35d57664cb41fbe0077909ae8a`

```diff
--- before
+++ after
@@ -2,7 +2,7 @@
 [URL] / HOME
 [URL_ABS] http://example.com/
 [HEADER] host: example.com
-[HEADER] x-test: value with tabs WSPAD
+[HEADER] x-test:   value\twith\t tabs WSPAD
 [HAGG] h_count=2 dup_names=0 hopbyhop=0 bad_names=0 total_bytes=33
 [HGF] HDRNORM
 [QMETA] count=0
```

