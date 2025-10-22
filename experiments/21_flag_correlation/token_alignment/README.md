# Token Alignment Notes

Purpose: given the Tier 1 flag set identified in the parent experiment, capture which substrings of the original HTTP request triggered each flag so we can later map them onto encoder token positions.

Workflow:
1. Run each request through the standard preprocessing pipeline to learn which flags fire.
2. For requests containing Tier 1 flags, apply lightweight heuristics on the original text (regex/decoding) to locate the offending substrings (character offsets + snippets).
3. Persist the findings as JSONL for inspection and as input to any future token-to-flag weighting logic.

The heuristics intentionally stay simple. If a flag cannot be localised with these rules, the script records a `detector_not_implemented` note so we can decide whether deeper pipeline instrumentation is warranted.
