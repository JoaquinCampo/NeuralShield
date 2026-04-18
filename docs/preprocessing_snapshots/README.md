# Preprocessing Snapshots

This folder stores reproducible "before/after" snapshots of the HTTP preprocessing pipeline.

Workflow:
1) Define packets in `repo/docs/preprocessing_snapshots/packets.jsonl`.
2) Run `repo/docs/preprocessing_snapshots/run_snapshot.py` to generate:
   - a machine-readable snapshot (`snapshots/*.jsonl`)
   - a human-readable report (`reports/*.md`)
3) After implementing pipeline changes, re-run the same packets and diff the outputs.

Notes:
- Packets are stored as `raw_escaped` strings and decoded using Python's `unicode_escape`.
- Run from `repo/` root to avoid current CWD-dependent config loading.
