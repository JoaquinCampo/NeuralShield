from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer
from loguru import logger
from transformers import AutoTokenizer

from neuralshield.preprocessing.pipeline import preprocess
from neuralshield.preprocessing.steps.exceptions import MalformedHttpRequestError

TIER1_FLAGS = (
    "DOUBLEPCT",
    "PCTSUSPICIOUS",
    "PCTCONTROL",
    "QRAWSEMI",
    "QNUL",
    "QARRAY:",
    "QREPEAT:",
    "PCTSPACE",
    "PIPE",
    "QUOTE",
)


@dataclass
class FlagTokenSample:
    request_id: str
    label: str
    flag: str
    token_indices: list[int]
    tokens: list[str]
    processed_text: str


app = typer.Typer(
    help="Locate Tier 1 flag tokens in processed requests and map to token indices."
)


def iter_requests(jsonl_path: Path) -> Iterable[tuple[str, str, str]]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            request_id = str(obj.get("id") or obj.get("_id") or "")
            label = obj.get("label", "unknown")
            request = obj.get("request") or ""
            yield request_id, label, request


def find_flag_token_indices(
    processed: str, tokenizer: AutoTokenizer
) -> dict[str, list[int]]:
    tokens = processed.split()
    flag_positions: dict[str, list[int]] = {flag: [] for flag in TIER1_FLAGS}

    for idx, token in enumerate(tokens):
        for flag in TIER1_FLAGS:
            if flag.endswith(":"):
                if token.startswith(flag):
                    flag_positions[flag].append(idx)
            elif token == flag:
                flag_positions[flag].append(idx)

    if all(len(indices) == 0 for indices in flag_positions.values()):
        return {}

    encoding = tokenizer(
        processed,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_offsets = encoding["offset_mapping"]

    token_mappings = {flag: [] for flag in TIER1_FLAGS}
    for flag, positions in flag_positions.items():
        if not positions:
            continue
        for pos in positions:
            # Map whitespace-token index to nearest tokenizer output.
            approx_offset = sum(len(tokens[i]) + 1 for i in range(pos))
            best_idx = min(
                range(len(token_offsets)),
                key=lambda i: abs(token_offsets[i][0] - approx_offset),
            )
            token_mappings[flag].append(best_idx)

    return {flag: indices for flag, indices in token_mappings.items() if indices}


@app.command()
def main(
    dataset_root: Path = typer.Option(
        Path("src/neuralshield/data/SR_BH_2020"),
        help="Directory containing train.jsonl and test.jsonl",
    ),
    output: Path = typer.Option(
        Path("experiments/21_flag_correlation/token_alignment/samples.jsonl"),
        help="Destination JSONL file for flag-token samples.",
    ),
    model_name: str = typer.Option(
        "jackaduma/SecBERT",
        help="Tokenizer to use when mapping tokens.",
    ),
    max_samples: int = typer.Option(5000, help="Maximum samples to collect."),
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = 0

    with output.open("w", encoding="utf-8") as sink:
        for split in ("test", "train"):
            path = dataset_root / f"{split}.jsonl"
            if not path.exists():
                logger.warning("Skipping missing split {split}", split)
                continue

            for request_id, label, request in iter_requests(path):
                if collected >= max_samples:
                    break
                try:
                    processed = preprocess(request)
                except MalformedHttpRequestError:
                    logger.debug("Malformed request skipped", split=split)
                    continue

                mapping = find_flag_token_indices(processed, tokenizer)
                if not mapping:
                    continue

                encoding = tokenizer(processed, add_special_tokens=False)
                token_texts = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

                for flag, token_indices in mapping.items():
                    sample = {
                        "request_id": request_id,
                        "label": label,
                        "flag": flag,
                        "token_indices": token_indices,
                        "tokens": [token_texts[i] for i in token_indices],
                        "processed_text": processed,
                        "split": split,
                    }
                    sink.write(json.dumps(sample) + "\n")
                    collected += 1
                    if collected >= max_samples:
                        break

    logger.info("Collected {count} flag-token samples", count=collected)


if __name__ == "__main__":
    app()
