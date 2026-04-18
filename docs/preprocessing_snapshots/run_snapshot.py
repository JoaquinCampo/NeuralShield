import argparse
import codecs
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Packet:
    packet_id: str
    title: str
    raw_escaped: str


def _decode_raw(raw_escaped: str) -> str:
    # Interpret \n, \r, \t, \xHH, \uHHHH sequences.
    return codecs.decode(raw_escaped, "unicode_escape")


def _visible(s: str) -> str:
    # Make control characters visible for Markdown/code blocks.
    out = []
    for ch in s:
        code = ord(ch)
        if ch == "\n":
            out.append("\\n\n")
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


def _visible_keep_newlines(s: str) -> str:
    """Escape control characters but keep '\n' as line breaks."""
    out = []
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


def _sha256_text(s: str) -> str:
    # Use UTF-8 with surrogatepass to keep the transformation stable.
    b = s.encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(b).hexdigest()


def load_packets(path: Path) -> list[Packet]:
    packets: list[Packet] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        packets.append(
            Packet(
                packet_id=obj["id"],
                title=obj["title"],
                raw_escaped=obj["raw_escaped"],
            )
        )
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline on a packet set and save a snapshot."
    )
    parser.add_argument(
        "--packets",
        default="docs/preprocessing_snapshots/packets.jsonl",
        help="Path to packets.jsonl (relative to repo root)",
    )
    parser.add_argument(
        "--out",
        default=f"docs/preprocessing_snapshots/snapshots/{date.today().isoformat()}_baseline.jsonl",
        help="Snapshot JSONL output path (relative to repo root)",
    )
    parser.add_argument(
        "--report",
        default=f"docs/preprocessing_snapshots/reports/{date.today().isoformat()}_baseline.md",
        help="Markdown report path (relative to repo root)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    packets_path = (repo_root / args.packets).resolve()
    out_path = (repo_root / args.out).resolve()
    report_path = (repo_root / args.report).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    packets = load_packets(packets_path)
    if len(packets) < 15:
        raise SystemExit(f"Expected at least 15 packets, found {len(packets)}")

    # Import from repo root context (pipeline currently resolves config.toml relative to CWD).
    os.chdir(repo_root)
    from neuralshield.preprocessing.pipeline import preprocess

    snapshot_rows: list[dict] = []
    report_lines: list[str] = []
    report_lines.append("# Preprocessing Snapshot")
    report_lines.append("")
    report_lines.append(f"Date: {date.today().isoformat()}")
    report_lines.append(f"Packets: {len(packets)}")
    report_lines.append("")
    report_lines.append(f"Inputs: `{packets_path.relative_to(repo_root)}`")
    report_lines.append(f"Snapshot: `{out_path.relative_to(repo_root)}`")
    report_lines.append("")

    for pkt in packets:
        raw = _decode_raw(pkt.raw_escaped)
        processed = preprocess(raw)

        snapshot_rows.append(
            {
                "id": pkt.packet_id,
                "title": pkt.title,
                "raw_escaped": pkt.raw_escaped,
                "raw_sha256": _sha256_text(raw),
                "processed": processed,
                "processed_sha256": _sha256_text(processed),
            }
        )

        report_lines.append(f"## {pkt.packet_id} - {pkt.title}")
        report_lines.append("")
        report_lines.append("Raw packet (visible escapes):")
        report_lines.append("```http")
        report_lines.append(_visible(raw))
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("Preprocessed artifact:")
        report_lines.append("```text")
        report_lines.append(_visible_keep_newlines(processed))
        report_lines.append("```")
        report_lines.append("")

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in snapshot_rows) + "\n",
        encoding="utf-8",
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
