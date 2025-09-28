#!/usr/bin/env python3
"""
Convert CSIC dataset from TSV to JSONL format.

Input: TSV with columns [id, label, method, url, request_headers, custom_headers]
Output: JSONL with {"request": "METHOD url HTTP/1.1\nHeader1: value1\n...", "label": "valid"/"attack"}
"""

import csv
import json
import sys
from pathlib import Path


def parse_headers(headers_str):
    """Parse headers string into list of header lines."""
    if not headers_str or headers_str == "null":
        return []

    headers = []
    for line in headers_str.split("\\n"):
        line = line.strip()
        if line:
            headers.append(line)
    return headers


def create_request_string(method, url, headers_list):
    """Create the request string in the format: METHOD url HTTP/1.1\nHeader1\nHeader2\n..."""
    lines = [f"{method} {url} HTTP/1.1"]
    lines.extend(headers_list)
    return "\n".join(lines)


def convert_label(label):
    """Convert Valid/Invalid to valid/attack."""
    return "valid" if label == "Valid" else "attack"


def main():
    input_file = (
        Path(__file__).parent
        / "src"
        / "neuralshield"
        / "data"
        / "CSIC"
        / "csic_requests.tsv"
    )
    output_file = (
        Path(__file__).parent
        / "src"
        / "neuralshield"
        / "data"
        / "CSIC"
        / "csic_dataset.jsonl"
    )

    print(f"Converting {input_file} to {output_file}")

    with (
        open(input_file, "r", encoding="utf-8") as tsv_file,
        open(output_file, "w", encoding="utf-8") as jsonl_file,
    ):
        reader = csv.DictReader(tsv_file, delimiter="\t")

        for row in reader:
            headers_list = parse_headers(row["request_headers"])
            request_str = create_request_string(row["method"], row["url"], headers_list)
            label = convert_label(row["label"])

            json_obj = {"request": request_str, "label": label}

            jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"Conversion complete. Output saved to {output_file}")


if __name__ == "__main__":
    main()
