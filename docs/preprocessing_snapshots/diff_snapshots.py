import argparse
import json
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path


@dataclass(frozen=True)
class Row:
    packet_id: str
    title: str
    processed: str
    processed_sha256: str


def load_snapshot(path: Path) -> dict[str, Row]:
    rows: dict[str, Row] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        pid = obj["id"]
        rows[pid] = Row(
            packet_id=pid,
            title=obj.get("title", ""),
            processed=obj["processed"],
            processed_sha256=obj.get("processed_sha256", ""),
        )
    return rows


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


def diff_ready_lines(s: str) -> list[str]:
    """Return lines suitable for unified_diff (always newline-terminated)."""
    lines = s.splitlines()
    return [ln + "\n" for ln in lines]


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two preprocessing snapshots.")
    parser.add_argument("--before", required=True, help="Before snapshot JSONL")
    parser.add_argument("--after", required=True, help="After snapshot JSONL")
    parser.add_argument("--out", required=True, help="Output markdown report")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    before_path = (repo_root / args.before).resolve()
    after_path = (repo_root / args.after).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    before = load_snapshot(before_path)
    after = load_snapshot(after_path)

    all_ids = sorted(set(before) | set(after))
    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []

    for pid in all_ids:
        if pid not in before:
            added.append(pid)
        elif pid not in after:
            removed.append(pid)
        else:
            if before[pid].processed_sha256 != after[pid].processed_sha256:
                changed.append(pid)

    lines: list[str] = []
    lines.append("# Preprocessing Snapshot Diff")
    lines.append("")
    lines.append(f"Before: `{before_path.relative_to(repo_root)}`")
    lines.append(f"After: `{after_path.relative_to(repo_root)}`")
    lines.append(f"Packets: {len(all_ids)}")
    lines.append("")
    lines.append(f"Changed: {len(changed)}")
    lines.append(f"Added: {len(added)}")
    lines.append(f"Removed: {len(removed)}")
    lines.append("")

    if added:
        lines.append("## Added")
        lines.append("")
        for pid in added:
            lines.append(f"- {pid} - {after[pid].title}")
        lines.append("")

    if removed:
        lines.append("## Removed")
        lines.append("")
        for pid in removed:
            lines.append(f"- {pid} - {before[pid].title}")
        lines.append("")

    lines.append("## Changes")
    lines.append("")
    if not changed:
        lines.append("No differences.")
        lines.append("")
    else:
        for pid in changed:
            b = before[pid]
            a = after[pid]
            lines.append(f"### {pid} - {a.title or b.title}")
            lines.append("")
            lines.append(f"- before sha256: `{b.processed_sha256}`")
            lines.append(f"- after sha256: `{a.processed_sha256}`")
            lines.append("")

            diff = unified_diff(
                diff_ready_lines(visible_keep_newlines(b.processed)),
                diff_ready_lines(visible_keep_newlines(a.processed)),
                fromfile="before",
                tofile="after",
            )
            diff_text = "".join(diff).rstrip()
            lines.append("```diff")
            lines.append(diff_text)
            lines.append("```")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
