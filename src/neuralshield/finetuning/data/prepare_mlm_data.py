from pathlib import Path

import typer
from loguru import logger

from neuralshield.preprocessing.pipeline import preprocess

app = typer.Typer()


def load_valid_requests(jsonl_path: Path) -> list[str]:
    """Load valid HTTP requests from JSONL file."""
    import json

    valid_requests = []
    total_count = 0

    logger.info(f"Loading requests from {jsonl_path}")

    with open(jsonl_path) as f:
        for line in f:
            total_count += 1
            data = json.loads(line)

            # Only keep valid/normal requests (no attacks for domain adaptation)
            if data.get("label") in ("valid", "normal"):
                valid_requests.append(data["request"])

    logger.info(
        f"Loaded {len(valid_requests)} valid requests "
        f"(filtered from {total_count} total)"
    )
    return valid_requests


def apply_preprocessing(requests: list[str], use_pipeline: bool = True) -> list[str]:
    """Apply preprocessing to HTTP requests."""
    if not use_pipeline:
        logger.info("Skipping preprocessing (use raw requests)")
        return requests

    logger.info(f"Preprocessing {len(requests)} requests...")
    preprocessed = []

    for i, request in enumerate(requests):
        try:
            processed = preprocess(request)
            preprocessed.append(processed)

            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(requests)} requests")

        except Exception as e:
            logger.warning(f"Failed to preprocess request {i}: {e}")
            # Fallback: use original request
            preprocessed.append(request)

    logger.info(f"Preprocessing complete: {len(preprocessed)} requests ready")
    return preprocessed


def save_as_text_corpus(
    requests: list[str],
    output_path: Path,
    separator: str = "\n\n",
) -> None:
    """Save requests as text corpus for MLM training.

    Args:
        requests: List of HTTP request strings
        output_path: Where to save the corpus
        separator: String to separate documents (default: double newline)
    """
    logger.info(f"Saving corpus to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(separator.join(requests))

    # Statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Corpus saved: {file_size_mb:.2f} MB")
    logger.info(f"Total documents: {len(requests)}")


def analyze_corpus(corpus_path: Path) -> None:
    """Analyze the corpus for MLM training statistics."""
    from transformers import AutoTokenizer

    logger.info("Analyzing corpus with SecBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")

    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    documents = text.split("\n\n")
    logger.info(f"Loaded {len(documents)} documents from corpus")

    # Tokenize sample
    sample_size = min(1000, len(documents))
    sample_docs = documents[:sample_size]

    tokens = tokenizer(
        sample_docs,
        truncation=True,
        max_length=512,
        padding=False,
        return_length=True,
    )

    # Statistics
    lengths = tokens["length"]
    avg_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)

    truncated = sum(1 for l in lengths if l >= 512)
    truncated_pct = 100 * truncated / len(lengths)

    logger.info("Corpus Statistics:")
    logger.info(f"  Average sequence length: {avg_length:.1f} tokens")
    logger.info(f"  Min length: {min_length} tokens")
    logger.info(f"  Max length: {max_length} tokens")
    logger.info(f"  Truncated sequences (≥512): {truncated} ({truncated_pct:.1f}%)")

    # Vocabulary coverage
    all_tokens = set()
    for token_ids in tokens["input_ids"]:
        all_tokens.update(token_ids)

    vocab_size = tokenizer.vocab_size
    unique_tokens = len(all_tokens)
    coverage_pct = 100 * unique_tokens / vocab_size

    logger.info(
        f"  Unique tokens used: {unique_tokens}/{vocab_size} ({coverage_pct:.1f}%)"
    )


@app.command()
def main(
    input_jsonl: Path = typer.Argument(
        ...,
        help="Input JSONL file with HTTP requests (e.g., train.jsonl)",
    ),
    output_corpus: Path = typer.Argument(
        ...,
        help="Output text corpus file (e.g., http_corpus.txt)",
    ),
    use_preprocessing: bool = typer.Option(
        True,
        "--preprocess/--no-preprocess",
        help="Apply preprocessing pipeline",
    ),
    analyze: bool = typer.Option(
        True,
        "--analyze/--no-analyze",
        help="Analyze corpus statistics",
    ),
):
    """Prepare HTTP corpus for MLM domain adaptation.

    Example:
        uv run python -m neuralshield.finetuning.prepare_mlm_data \\
            src/neuralshield/data/CSIC/train.jsonl \\
            src/neuralshield/finetuning/data/http_corpus.txt
    """
    logger.info("=" * 80)
    logger.info("HTTP CORPUS PREPARATION FOR MLM TRAINING")
    logger.info("=" * 80)

    # Load valid requests
    requests = load_valid_requests(input_jsonl)

    # Apply preprocessing
    processed_requests = apply_preprocessing(requests, use_preprocessing)

    # Save corpus
    save_as_text_corpus(processed_requests, output_corpus)

    # Analyze
    if analyze:
        analyze_corpus(output_corpus)

    logger.info("=" * 80)
    logger.info("CORPUS PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Next step: Train MLM model using {output_corpus}")


if __name__ == "__main__":
    app()
