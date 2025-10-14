import math
from datetime import datetime

import torch
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import wandb
from neuralshield.encoding.observability import WandbSink
from neuralshield.finetuning.config import MLMConfig
from neuralshield.finetuning.data import (
    load_corpus,
    prepare_datasets,
    tokenize_datasets,
)


def setup_wandb(config: MLMConfig) -> None:
    """Initialize Weights & Biases logging."""
    if not config.use_wandb:
        logger.info("W&B logging disabled")
        return

    run_name = (
        config.wandb_run_name or f"secbert-http-mlm-{datetime.now():%Y%m%d-%H%M%S}"
    )

    wandb.init(
        project=config.wandb_project,
        name=run_name,
        tags=["mlm", "domain-adaptation", "secbert", "http"],
        config={
            "base_model": config.base_model,
            "task": "masked_language_modeling",
            "domain": "http_requests",
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "mlm_probability": config.mlm_probability,
            "max_length": config.max_length,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "fp16": config.fp16,
        },
    )

    logger.info(f"W&B initialized: {run_name}")


def create_trainer(
    config: MLMConfig,
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Trainer:
    """Create and configure HuggingFace Trainer."""
    logger.info("Setting up Trainer...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        report_to="wandb" if config.use_wandb else "none",
        dataloader_num_workers=4,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Trainer configured")
    return trainer


def save_training_summary(config: MLMConfig, train_result, eval_loss: float) -> None:
    """Save training summary to disk."""
    perplexity = math.exp(eval_loss)
    summary_path = config.output_dir / "training_summary.txt"

    with open(summary_path, "w") as f:
        f.write("MLM Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Base Model: {config.base_model}\n")
        f.write(f"Corpus: {config.corpus_path}\n")
        f.write(f"Epochs: {config.num_epochs}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write("\nResults:\n")
        f.write(f"Final Training Loss: {train_result.training_loss:.4f}\n")
        f.write(f"Final Validation Loss: {eval_loss:.4f}\n")
        f.write(f"Final Perplexity: {perplexity:.2f}\n")

    logger.info(f"Training summary saved to {summary_path}")


def train_model(config: MLMConfig) -> None:
    """Execute MLM training pipeline."""
    logger.info("=" * 80)
    logger.info("MLM TRAINING: SecBERT HTTP Domain Adaptation")
    logger.info("=" * 80)

    setup_wandb(config)

    documents = load_corpus(config.corpus_path)
    train_dataset, val_dataset = prepare_datasets(documents, config.validation_split)

    logger.info(f"Loading model: {config.base_model}")
    model = AutoModelForMaskedLM.from_pretrained(config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    logger.info("Model and tokenizer loaded")

    train_tokenized, val_tokenized = tokenize_datasets(
        train_dataset, val_dataset, tokenizer, config.max_length
    )

    trainer = create_trainer(config, model, tokenizer, train_tokenized, val_tokenized)

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    train_result = trainer.train()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")

    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    perplexity = math.exp(eval_loss)

    logger.info(f"Validation loss: {eval_loss:.4f}")
    logger.info(f"Validation perplexity: {perplexity:.2f}")

    if config.use_wandb:
        sink = WandbSink()
        sink.log(
            {
                "final/train_loss": train_result.training_loss,
                "final/eval_loss": eval_loss,
                "final/perplexity": perplexity,
                "final/total_steps": train_result.global_step,
            }
        )

    final_path = config.output_dir / "final"
    logger.info(f"Saving final model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    save_training_summary(config, train_result, eval_loss)

    if config.use_wandb:
        wandb.finish()

    logger.info("=" * 80)
    logger.info("MLM TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {final_path}")
