#!/usr/bin/env python3
"""
Convert SR_BH_2020 dataset from CSV to JSONL format.

Input: CSV with HTTP request/response fields and CAPEC multi-labels
Output: JSONL with {"request": "METHOD url HTTP/1.1\nHeaders...", "label": "valid"/"attack"}

Note: Request body is NOT included per project constraints.
"""

import csv
from pathlib import Path

from loguru import logger


def build_request_string(row: dict) -> str:
    """
    Build HTTP request string from CSV row fields.

    Format: METHOD url HTTP/version\nHeader: value\n...
    Body content is excluded but Content-Type/Content-Length headers included.
    """
    lines = []

    # Request line
    method = row.get("request_http_method", "GET").strip()
    url = row.get("request_http_request", "/").strip()
    protocol = row.get("request_http_protocol", "HTTP/1.1").strip()
    lines.append(f"{method} {url} {protocol}")

    # Headers - add in consistent order
    headers = [
        ("User-Agent", row.get("request_user_agent", "")),
        ("Pragma", ""),  # Not in SR_BH, but CSIC has it
        ("Cache-control", ""),  # Not in SR_BH
        ("Accept", row.get("request_accept", "")),
        ("Accept-Encoding", row.get("request_accept_encoding", "")),
        ("Accept-Charset", ""),  # Not in SR_BH
        ("Accept-Language", row.get("request_accept_language", "")),
        ("Host", row.get("request_host", "")),
        ("Origin", row.get("request_origin", "")),
        ("Referer", row.get("request_referer", "")),
        ("Cookie", row.get("request_cookie", "")),
        ("Do-Not-Track", row.get("request_do_not_track", "")),
        ("Content-Type", row.get("request_content_type", "")),
        ("Connection", row.get("request_connection", "")),
    ]

    # Add non-empty headers
    for header_name, header_value in headers:
        if header_value and header_value.strip():
            lines.append(f"{header_name}: {header_value.strip()}")

    # Add Content-Length if body exists (but don't include body content)
    body = row.get("request_body", "")
    if body and body.strip():
        body_length = len(body.encode("utf-8"))
        lines.append(f"Content-Length: {body_length}")

    return "\n".join(lines)


def convert_label(row: dict) -> str:
    """
    Convert CAPEC multi-labels to binary label.

    Column 25 "000 - Normal": 1 = valid, 0 = attack
    """
    normal_flag = row.get("000 - Normal", "0").strip()
    return "valid" if normal_flag == "1" else "attack"


def convert_csv_to_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> None:
    """
    Convert SR_BH CSV to JSONL format.

    Args:
        input_path: Input CSV file path
        output_path: Output JSONL file path
        limit: Optional limit on number of rows to process
    """
    logger.info("Converting {input} to {output}", input=input_path, output=output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    attack_count = 0
    total_count = 0

    with (
        open(input_path, "r", encoding="utf-8") as csv_file,
        open(output_path, "w", encoding="utf-8") as jsonl_file,
    ):
        reader = csv.DictReader(csv_file)

        for row in reader:
            if limit and total_count >= limit:
                break

            try:
                request_str = build_request_string(row)
                label = convert_label(row)

                if label == "valid":
                    valid_count += 1
                else:
                    attack_count += 1

                # Write JSONL (one JSON object per line)
                import json

                json_obj = {"request": request_str, "label": label}
                jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

                total_count += 1

                if total_count % 10000 == 0:
                    logger.info(
                        "Processed {count} samples ({valid} valid, {attack} attack)",
                        count=total_count,
                        valid=valid_count,
                        attack=attack_count,
                    )

            except Exception as e:
                logger.error(
                    "Error processing row {count}: {error}", count=total_count, error=e
                )
                continue

    logger.info("Conversion complete!")
    logger.info(
        "Total: {total} samples ({valid} valid, {attack} attack)",
        total=total_count,
        valid=valid_count,
        attack=attack_count,
    )
    logger.info("Output saved to {output}", output=output_path)


def main():
    """Main entry point."""
    input_file = Path(__file__).parent / "data_capec_multilabel.csv"
    output_file = Path(__file__).parent / "srbh_dataset.jsonl"

    convert_csv_to_jsonl(input_file, output_file)


if __name__ == "__main__":
    main()
