import argparse
import codecs
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    packet_id: str
    title: str
    raw_escaped: str
    processed: str


def decode_raw(raw_escaped: str) -> str:
    return codecs.decode(raw_escaped, "unicode_escape")


def visible_keep_newlines(s: str) -> str:
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if ch == "\n":
            out.append("\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code == 0:
            out.append("\\x00")
        elif code < 32 or code == 127:
            out.append(f"\\x{code:02x}")
        else:
            out.append(ch)
    return "".join(out)


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(
            Row(
                packet_id=obj["id"],
                title=obj.get("title", ""),
                raw_escaped=obj["raw_escaped"],
                processed=obj["processed"],
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a markdown report from an existing snapshot JSONL."
    )
    parser.add_argument("--snapshot", required=True, help="Snapshot JSONL path")
    parser.add_argument("--out", required=True, help="Output markdown path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    snap_path = (repo_root / args.snapshot).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(snap_path)

    lines: list[str] = []
    lines.append("# Preprocessing Snapshot")
    lines.append("")
    lines.append(f"Snapshot: `{snap_path.relative_to(repo_root)}`")
    lines.append(f"Packets: {len(rows)}")
    lines.append("")

    for r in rows:
        raw = decode_raw(r.raw_escaped)
        lines.append(f"## {r.packet_id} - {r.title}")
        lines.append("")
        lines.append("Raw packet (visible escapes):")
        lines.append("```http")
        lines.append(visible_keep_newlines(raw))
        lines.append("```")
        lines.append("")
        lines.append("Preprocessed artifact:")
        lines.append("```text")
        lines.append(visible_keep_newlines(r.processed))
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
