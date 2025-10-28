from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from toxicity_detection.data.loader import load_dataset
from toxicity_detection.evaluation.evaluator import ModelEvaluator
from toxicity_detection.models.bilstm_model import BiLSTMModel
from toxicity_detection.models.tfidf_model import TFIDFModel
from toxicity_detection.models.transformer_model import TransformerModel
from toxicity_detection.utils.logging import setup_logger
from toxicity_detection.utils.seed import set_seed

app = typer.Typer(
    name="toxicity-detection",
    help="Indonesian Toxicity Detection System CLI",
    add_completion=False,
)

console = Console()
logger = setup_logger(__name__)

@app.command()
def train(
    model_type: str = typer.Option(..., "--model", "-m", help="Model type (tfidf/bilstm/transformer)"),
    data_path: Path = typer.Option("dataset/indonesian_hate_speech.csv", "--data", "-d"),
    output_dir: Path = typer.Option("models", "--output", "-o"),
    seed: int = typer.Option(42, "--seed", "-s"),
) -> None:
    console.print(f"[bold blue]Training {model_type} model...[/bold blue]")
    set_seed(seed)
    console.print("Loading dataset...")
    train_ds, val_ds, test_ds = load_dataset(data_path)
    X_train = train_ds.get_texts()
    y_train = train_ds.data["labels"].values
    X_val = val_ds.get_texts()
    y_val = val_ds.data["labels"].values
    console.print(f"Initializing {model_type} model...")
    if model_type == "tfidf":
        model = TFIDFModel()
    elif model_type == "bilstm":
        model = BiLSTMModel(epochs=20)
    elif model_type == "transformer":
        model = TransformerModel(epochs=3)
    else:
        console.print(f"[bold red]Error: Unknown model type '{model_type}'[/bold red]")
        raise typer.Exit(1)

    console.print("Training...")
    metrics = model.train(X_train, y_train, X_val, y_val)

    model_output_path = output_dir / model_type
    model_output_path.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)

    console.print(f"[bold green]Training complete![/bold green]")
    console.print(f"Model saved to: {model_output_path}")
    console.print(f"Training time: {metrics.get('training_time', 0):.2f}s")


@app.command()
def evaluate(
    model_type: str = typer.Option(..., "--model", "-m"),
    model_path: Path = typer.Option(..., "--model-path", "-p"),
    data_path: Path = typer.Option("dataset/indonesian_hate_speech.csv", "--data", "-d"),
) -> None:
    console.print(f"[bold blue]Evaluating {model_type} model...[/bold blue]")
    console.print("Loading dataset...")
    _, _, test_ds = load_dataset(data_path)

    X_test = test_ds.get_texts()
    y_test = test_ds.data["labels"].values

    console.print("Loading model...")
    if model_type == "tfidf":
        model = TFIDFModel()
    elif model_type == "bilstm":
        model = BiLSTMModel()
    elif model_type == "transformer":
        model = TransformerModel()
    else:
        console.print(f"[bold red]Error: Unknown model type '{model_type}'[/bold red]")
        raise typer.Exit(1)

    model.load(model_path)

    console.print("Evaluating...")
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test)

    table = Table(title=f"{model_type.upper()} Model Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1 Score", f"{metrics.f1:.4f}")
    table.add_row("ROC-AUC", f"{metrics.roc_auc:.4f}")
    table.add_row("PR-AUC", f"{metrics.pr_auc:.4f}")
    table.add_row("ECE", f"{metrics.ece:.4f}")
    if metrics.avg_latency_ms:
        table.add_row("Avg Latency (ms)", f"{metrics.avg_latency_ms:.2f}")

    console.print(table)


@app.command()
def compare(
    data_path: Path = typer.Option("dataset/indonesian_hate_speech.csv", "--data", "-d"),
    models_dir: Path = typer.Option("models", "--models-dir"),
) -> None:
    console.print("[bold blue]Comparing all models...[/bold blue]")
    console.print("Loading dataset...")
    _, _, test_ds = load_dataset(data_path)

    X_test = test_ds.get_texts()
    y_test = test_ds.data["labels"].values

    models_config = [
        ("TF-IDF", "tfidf", TFIDFModel(), models_dir / "tfidf"),
        ("BiLSTM", "bilstm", BiLSTMModel(), models_dir / "bilstm"),
        ("IndoBERT", "transformer", TransformerModel(), models_dir / "transformer"),
    ]

    results = []

    for name, model_type, model, model_path in models_config:
        if not model_path.exists():
            console.print(f"[yellow]Skipping {name}: Model not found at {model_path}[/yellow]")
            continue

        console.print(f"\n[cyan]Evaluating {name}...[/cyan]")
        try:
            model.load(model_path)
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test, measure_latency=(name == "TF-IDF"))

            results.append({
                "name": name,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "pr_auc": metrics.pr_auc,
                "ece": metrics.ece,
                "latency": metrics.avg_latency_ms,
            })
        except Exception as e:
            console.print(f"[red]Error evaluating {name}: {e}[/red]")
            continue

    if not results:
        console.print("[bold red]No models were successfully evaluated![/bold red]")
        raise typer.Exit(1)

    table = Table(title="Model Comparison Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", width=12)
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="bold")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("PR-AUC", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Latency (ms)", justify="right")

    best_f1 = max(r["f1"] for r in results)

    for result in results:
        is_best = result["f1"] == best_f1
        style = "bold green" if is_best else ""

        name = f"🥇 {result['name']}" if is_best else result['name']
        latency_str = f"{result['latency']:.2f}" if result['latency'] else "N/A"

        table.add_row(
            name,
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['roc_auc']:.4f}",
            f"{result['pr_auc']:.4f}",
            f"{result['ece']:.4f}",
            latency_str,
            style=style,
        )

    console.print("\n")
    console.print(table)

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"Best model by F1 score: [bold green]{max(results, key=lambda x: x['f1'])['name']}[/bold green]")
    console.print(f"Best model by ROC-AUC: [bold green]{max(results, key=lambda x: x['roc_auc'])['name']}[/bold green]")


@app.command()
def predict(
    text: Optional[str] = typer.Argument(None, help="Text to predict (leave empty for interactive mode)"),
    model_type: str = typer.Option("transformer", "--model", "-m", help="Model to use (tfidf/bilstm/transformer)"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", "-p"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
) -> None:
    from rich.panel import Panel
    from rich.text import Text

    if model_path is None:
        models_dir = Path("models")
        if model_type == "tfidf":
            model_path = models_dir / "tfidf"
        elif model_type == "bilstm":
            model_path = models_dir / "bilstm"
        elif model_type == "transformer":
            model_path = models_dir / "transformer"

    console.print(f"[cyan]Loading {model_type} model...[/cyan]")
    if model_type == "tfidf":
        model = TFIDFModel()
    elif model_type == "bilstm":
        model = BiLSTMModel()
    elif model_type == "transformer":
        model = TransformerModel()
    else:
        console.print(f"[bold red]Error: Unknown model type '{model_type}'[/bold red]")
        raise typer.Exit(1)

    if not model_path.exists():
        console.print(f"[bold red]Error: Model not found at {model_path}[/bold red]")
        raise typer.Exit(1)

    model.load(model_path)
    console.print(f"[green]✓ Model loaded from {model_path}[/green]\n")

    def predict_text(input_text: str) -> None:
        if not input_text.strip():
            console.print("[yellow]Empty input, skipping...[/yellow]\n")
            return

        proba = model.predict_proba([input_text])[0]
        label = "TOXIC" if proba > 0.5 else "NON-TOXIC"
        confidence = proba if proba > 0.5 else (1 - proba)

        if label == "TOXIC":
            label_color = "bold red"
            emoji = "🚨"
        else:
            label_color = "bold green"
            emoji = "✅"

        result_text = Text()
        result_text.append(f"{emoji} ", style="bold")
        result_text.append(label, style=label_color)
        result_text.append(f" (confidence: {confidence:.2%})")

        panel = Panel(
            f"[bold]Input:[/bold] {input_text}\n\n{result_text}",
            title=f"Prediction Result ({model_type.upper()})",
            border_style="cyan",
        )
        console.print(panel)
        console.print()

    if interactive or text is None:
        console.print("[bold blue]Interactive Prediction Mode[/bold blue]")
        console.print("Type text to predict toxicity (or 'quit' to exit)\n")

        while True:
            try:
                user_input = console.input("[bold cyan]Enter text:[/bold cyan] ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Exiting...[/yellow]")
                    break

                predict_text(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
                break
            except EOFError:
                break

    else:
        predict_text(text)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
) -> None:
    import uvicorn
    from toxicity_detection.api.app import create_app
    
    console.print(f"[bold blue]Starting API server on {host}:{port}...[/bold blue]")
    app_instance = create_app(load_models=False)  
    uvicorn.run(app_instance, host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    app()
