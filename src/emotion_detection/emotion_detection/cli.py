"""Command-line interface for the emotion detection package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer

from .config import EmotionSettings, load_settings
from .data import load_dataset, split_dataset
from .logging import configure_logging
from .models import get_model_cls, list_models
from .preprocess import clean_corpus
from .train import train as run_training


app = typer.Typer(add_completion=False, help="Emotion detection utilities")


def _prepare_settings(config: Optional[Path], overrides: Dict) -> EmotionSettings:
    settings = load_settings(config_path=config, overrides=overrides)
    configure_logging(settings.log_level)
    return settings


@app.command()
def train(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config file"),
    model_name: Optional[str] = typer.Option(None, help="Model to train"),
    data_dir: Optional[Path] = typer.Option(None, help="Dataset directory"),
    output_dir: Optional[Path] = typer.Option(None, help="Artifacts directory"),
):
    """Train a model and persist artefacts."""

    overrides = _build_overrides(
        model_name=model_name, data_dir=data_dir, output_dir=output_dir
    )
    settings = _prepare_settings(config, overrides)

    typer.echo(f"Training model: {settings.model_name}")
    metrics, artefact_dir = run_training(settings)
    typer.echo(json.dumps({"metrics": metrics, "artefact_dir": str(artefact_dir)}, indent=2))


@app.command()
def evaluate(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config file"),
    model_name: Optional[str] = typer.Option(None, help="Model name"),
    data_dir: Optional[Path] = typer.Option(None, help="Dataset directory"),
    artefact_dir: Optional[Path] = typer.Option(
        None, help="Directory containing a saved model"
    ),
):
    """Evaluate a trained model on the configured hold-out split."""

    overrides = _build_overrides(model_name=model_name, data_dir=data_dir)
    settings = _prepare_settings(config, overrides)

    model_path = artefact_dir or settings.output_dir / settings.model_name
    model_cls = get_model_cls(settings.model_name)
    model = model_cls.load(model_path)

    dataset = load_dataset(settings.data_dir)
    _, test_df = split_dataset(
        dataset,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    texts = _prepare_texts(settings.model_name, test_df["text"].tolist())

    from .evaluate import evaluate_model

    metrics = evaluate_model(
        model,
        texts,
        test_df["label"].tolist(),
        output_dir=model_path,
    )

    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def predict(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config file"),
    text: Optional[str] = typer.Option(None, help="Text to classify"),
    file: Optional[Path] = typer.Option(
        None, help="Optional CSV or JSONL file containing a 'text' column"
    ),
    artefact_dir: Optional[Path] = typer.Option(
        None, help="Directory containing a saved model"
    ),
    model_name: Optional[str] = typer.Option(None, help="Model to load"),
):
    """Predict emotion labels for a single text or a batch file."""

    if not text and not file:
        raise typer.BadParameter("Provide either --text or --file")

    overrides = _build_overrides(model_name=model_name)
    settings = _prepare_settings(config, overrides)
    model_path = artefact_dir or settings.output_dir / settings.model_name
    model_cls = get_model_cls(settings.model_name)
    model = model_cls.load(model_path)

    texts: List[str]
    if file:
        texts = _read_texts_from_file(file)
    else:
        texts = [text] if text else []

    texts = _prepare_texts(settings.model_name, texts)
    predictions = model.predict(texts)
    try:
        probabilities = model.predict_proba(texts)
    except AttributeError:
        probabilities = None

    response = {
        "predictions": predictions,
        "probabilities": probabilities,
    }
    typer.echo(json.dumps(response, indent=2))


@app.command()
def serve(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config file"),
    host: Optional[str] = typer.Option(None, help="API host"),
    port: Optional[int] = typer.Option(None, help="API port"),
    artefact_dir: Optional[Path] = typer.Option(
        None, help="Directory containing a saved model"
    ),
    model_name: Optional[str] = typer.Option(None, help="Model to serve"),
):
    """Launch the FastAPI prediction server."""

    overrides = _build_overrides(model_name=model_name)
    if host:
        overrides["api_host"] = host
    if port:
        overrides["api_port"] = port
    settings = _prepare_settings(config, overrides)

    from .api.server import create_app

    model_path = artefact_dir or settings.output_dir / settings.model_name
    app_instance = create_app(
        model_name=settings.model_name,
        artefact_dir=model_path,
        settings=settings,
    )

    import uvicorn

    uvicorn.run(
        app_instance,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


@app.command("models")
def list_available_models():
    """List registered models."""

    typer.echo("Registered models:")
    for name in list_models():
        typer.echo(f"- {name}")


def _build_overrides(
    *,
    model_name: Optional[str] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    overrides: Dict = {}
    if model_name:
        overrides["model_name"] = model_name
    if data_dir:
        overrides["data_dir"] = str(data_dir)
    if output_dir:
        overrides["output_dir"] = str(output_dir)
    return overrides


def _prepare_texts(model_name: str, texts: List[str]) -> List[str]:
    if model_name == "indobert":
        return texts
    return clean_corpus(texts)


def _read_texts_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if "text" not in frame.columns:
            raise typer.BadParameter("CSV file must contain a 'text' column")
        return frame["text"].astype(str).tolist()

    if path.suffix.lower() == ".jsonl":
        texts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if "text" not in payload:
                    raise typer.BadParameter("JSON lines must include a 'text' field")
                texts.append(str(payload["text"]))
        return texts

    raise typer.BadParameter("Unsupported file format. Use CSV or JSONL.")


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

