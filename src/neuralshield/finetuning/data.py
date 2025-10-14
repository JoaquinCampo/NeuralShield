from pathlib import Path

from datasets import Dataset
from loguru import logger


def load_corpus(corpus_path: Path) -> list[str]:
    """Load HTTP corpus from text file."""
    logger.info(f"Loading corpus from {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    documents = text.split("\n\n")
    logger.info(f"Loaded {len(documents)} documents")

    return documents


def prepare_datasets(
    documents: list[str], validation_split: float
) -> tuple[Dataset, Dataset]:
    """Convert documents to HuggingFace datasets with train/val split."""
    logger.info("Creating HuggingFace datasets...")

    dataset = Dataset.from_dict({"text": documents})
    logger.info(f"Created dataset with {len(dataset)} samples")

    split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")

    return train_dataset, val_dataset


def tokenize_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer,
    max_length: int,
) -> tuple[Dataset, Dataset]:
    """Tokenize datasets for MLM training."""
    logger.info("Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training set",
    )

    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation set",
    )

    logger.info("Tokenization complete")
    return train_tokenized, val_tokenized
