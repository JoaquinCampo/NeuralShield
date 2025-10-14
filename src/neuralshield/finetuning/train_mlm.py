from pathlib import Path

import typer

from neuralshield.finetuning.config import MLMConfig
from neuralshield.finetuning.training import train_model

app = typer.Typer()


@app.command()
def main(
    corpus: Path = typer.Option(
        "src/neuralshield/finetuning/data/http_corpus.txt",
        help="Path to HTTP corpus text file",
    ),
    output_dir: Path = typer.Option(
        "src/neuralshield/finetuning/models/secbert-http-adapted",
        help="Output directory for trained model",
    ),
    base_model: str = typer.Option(
        "jackaduma/SecBERT",
        help="Base model to adapt",
    ),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Training batch size"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
    validation_split: float = typer.Option(0.1, help="Validation set proportion (0-1)"),
    no_wandb: bool = typer.Option(False, help="Disable W&B logging"),
    wandb_run_name: str = typer.Option(None, help="Custom W&B run name"),
):
    """Train MLM for SecBERT HTTP domain adaptation.

    Example:
        uv run python -m neuralshield.finetuning.train_mlm

    Custom configuration:
        uv run python -m neuralshield.finetuning.train_mlm \\
            --epochs 5 --batch-size 16 --learning-rate 3e-5
    """
    config = MLMConfig(
        corpus_path=corpus,
        output_dir=output_dir,
        base_model=base_model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        use_wandb=not no_wandb,
        wandb_run_name=wandb_run_name,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_model(config)


if __name__ == "__main__":
    app()
