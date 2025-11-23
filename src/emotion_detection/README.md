# Emotion Detection Module

A production-ready Indonesian emotion detection system supporting multiple model architectures (TF-IDF + classic ML, IndoBERT) with CLI, REST API, and Python interfaces.

## Features

- **Multiple Models**: TF-IDF + Logistic Regression/SVM/Naive Bayes, and IndoBERT fine-tuning
- **CLI Interface**: Train, evaluate, predict, and serve models via command-line
- **REST API**: FastAPI server with automatic OpenAPI documentation
- **Configuration Management**: YAML-based configs with environment variable overrides
- **Pretty Logging**: Rich-formatted logs with timing and progress tracking
- **Evaluation Metrics**: Accuracy, F1, precision/recall, confusion matrices, and classification reports
- **Notebook Support**: Jupyter notebook for interactive experimentation

## Installation

The module is part of the main project. Install dependencies:

```bash
pip install -e .
```

Required dependencies include:
- `scikit-learn` (for TF-IDF models)
- `transformers` + `torch` (for IndoBERT)
- `fastapi` + `uvicorn` (for API server)
- `typer` (for CLI)
- `rich` (for logging)
- `pydantic` + `pydantic-settings` (for configuration)

## Quick Start

### 1. Train a Model

```bash
# Train TF-IDF + Logistic Regression (default)
emotion-cli train --config configs/emotion.yaml

# Train a specific model
emotion-cli train --model-name svm --data-dir "dataset/Emotion Dataset from Indonesian Public Opinion"

# Train IndoBERT (requires GPU for reasonable speed)
emotion-cli train --model-name indobert
```

### 2. Evaluate a Trained Model

```bash
emotion-cli evaluate --model-name logreg --artefact-dir artifacts/emotion/logreg
```

### 3. Make Predictions

```bash
# Single text
emotion-cli predict --text "aku merasa sangat senang hari ini"

# Batch from CSV/JSONL
emotion-cli predict --file data/test.csv --model-name logreg
```

### 4. Serve API

```bash
emotion-cli serve --model-name logreg --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

## CLI Commands

### `train`

Train a model and save artifacts:

```bash
emotion-cli train [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --model-name TEXT      Model to train (logreg, svm, nb, indobert)
  --data-dir PATH        Dataset directory
  --output-dir PATH      Artifacts directory
```

### `evaluate`

Evaluate a trained model on the test set:

```bash
emotion-cli evaluate [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --model-name TEXT      Model name
  --data-dir PATH        Dataset directory
  --artefact-dir PATH    Directory containing saved model
```

### `predict`

Generate predictions for text(s):

```bash
emotion-cli predict [OPTIONS]

Options:
  --text TEXT            Single text to classify
  --file PATH            CSV or JSONL file with 'text' column
  --artefact-dir PATH    Directory containing saved model
  --model-name TEXT      Model to load
```

### `serve`

Launch the FastAPI prediction server:

```bash
emotion-cli serve [OPTIONS]

Options:
  --config PATH          Path to YAML config file
  --host TEXT            API host (default: 0.0.0.0)
  --port INTEGER         API port (default: 8000)
  --artefact-dir PATH    Directory containing saved model
  --model-name TEXT      Model to serve
```

### `models`

List all registered models:

```bash
emotion-cli models
```

## API Endpoints

### `GET /health`

Health check endpoint:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "logreg"
}
```

### `GET /models`

List available and loaded models:

```bash
curl http://localhost:8000/models
```

Response:
```json
{
  "loaded": "logreg",
  "available": ["logreg", "svm", "nb", "indobert"]
}
```

### `POST /predict`

Predict emotion for text(s):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "aku merasa sangat marah"}'
```

Batch prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["text1", "text2"]}'
```

Response:
```json
{
  "predictions": [
    {
      "text": "aku merasa sangat marah",
      "label": "anger",
      "scores": {
        "anger": 0.85,
        "fear": 0.10,
        "joy": 0.03,
        "sad": 0.02
      }
    }
  ]
}
```

## Python API

### Training

```python
from emotion_detection.config import load_settings
from emotion_detection.train import train

settings = load_settings(config_path="configs/emotion.yaml")
metrics, artefact_dir = train(settings)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### Evaluation

```python
from emotion_detection.models import get_model_cls
from emotion_detection.evaluate import evaluate_model
from emotion_detection.data import load_dataset, split_dataset

model_cls = get_model_cls("logreg")
model = model_cls.load(Path("artifacts/emotion/logreg"))

dataset = load_dataset("dataset/Emotion Dataset from Indonesian Public Opinion")
_, test_df = split_dataset(dataset, test_size=0.2, random_state=42)

metrics = evaluate_model(
    model,
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    output_dir=Path("artifacts/emotion/logreg")
)
```

### Prediction

```python
from emotion_detection.models import get_model_cls
from emotion_detection.preprocess import clean_corpus

model_cls = get_model_cls("logreg")
model = model_cls.load(Path("artifacts/emotion/logreg"))

texts = ["aku merasa senang", "saya sedih hari ini"]
processed = clean_corpus(texts)
predictions = model.predict(processed)
probabilities = model.predict_proba(processed)
```

## Configuration

Configuration is managed via YAML files (see `configs/emotion.yaml`) with environment variable overrides:

```yaml
data_dir: dataset/Emotion Dataset from Indonesian Public Opinion
output_dir: artifacts/emotion

model_name: logreg
random_state: 42
test_size: 0.2

max_features: 20000
ngram_min: 1
ngram_max: 2

pretrained_model_name: indobenchmark/indobert-base-p1
num_epochs: 3
learning_rate: 0.00002
batch_size: 16
warmup_ratio: 0.06
weight_decay: 0.01

api_host: 0.0.0.0
api_port: 8000

log_level: INFO
```

Environment variables use the `EMOTION_` prefix:

```bash
export EMOTION_MODEL_NAME=svm
export EMOTION_TEST_SIZE=0.15
export EMOTION_LOG_LEVEL=DEBUG
```

## Models

### TF-IDF + Logistic Regression (`logreg`)

Fast baseline model using TF-IDF vectorization and logistic regression.

```python
from emotion_detection.models import create_model

model = create_model("logreg", max_features=20000, ngram_range=(1, 2))
```

### TF-IDF + Linear SVM (`svm`)

SVM classifier with probability calibration for better probability estimates.

```python
model = create_model("svm", max_features=20000, ngram_range=(1, 2))
```

### TF-IDF + Multinomial Naive Bayes (`nb`)

Lightweight probabilistic classifier.

```python
model = create_model("nb", max_features=20000, ngram_range=(1, 2))
```

### IndoBERT (`indobert`)

Fine-tuned transformer model using `indobenchmark/indobert-base-p1`. Requires GPU for training.

```python
from emotion_detection.models.transformer_model import TransformerParams

params = TransformerParams(
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
)
model = create_model("indobert", params=params)
```

## Dataset Format

The module expects CSV files in a directory, one per emotion class:

```
dataset/
  Emotion Dataset from Indonesian Public Opinion/
    AngerData.csv
    FearData.csv
    JoyData.csv
    LoveData.csv
    NeutralData.csv
    SadData.csv
```

Each CSV should contain a text column (auto-detected from: `text`, `tweet`, `sentence`, `content`, `message`). Labels are inferred from filenames if not present (e.g., `AngerData.csv` → `anger`).

## Project Structure

```
src/emotion_detection/
├── __init__.py
├── cli.py                 # Typer CLI application
├── config.py              # Pydantic settings and config loading
├── evaluate.py            # Evaluation utilities
├── logging.py             # Rich logging setup
├── registry.py            # Model registry
├── train.py               # Training orchestration
├── api/
│   ├── __init__.py
│   └── server.py          # FastAPI application
├── data/
│   ├── __init__.py
│   └── dataset_loader.py   # CSV ingestion and splitting
├── models/
│   ├── __init__.py
│   ├── base.py            # BaseEmotionModel interface
│   ├── sklearn_models.py  # TF-IDF + classic ML models
│   └── transformer_model.py  # IndoBERT implementation
├── preprocess/
│   ├── __init__.py
│   └── text_cleaning.py    # Text normalization
└── utils/
    ├── __init__.py
    └── metrics.py          # Evaluation metrics and confusion matrices
```

## Artifacts

Trained models are saved under `artifacts/emotion/<model_name>/`:

- **Sklearn models**: `model.joblib`, `label_encoder.joblib`
- **Transformer models**: `pytorch_model.bin`, `config.json`, `tokenizer.json`, `metadata.json`
- **Evaluation**: `metrics.json`, `classification_report.txt`, `confusion_matrix.csv`, `confusion_matrix.png`
- **Metadata**: `split.json` (train/test sizes, label list)

## Notebook

See `notebooks/emotion_training.ipynb` for an interactive walkthrough covering:
- Dataset loading and exploration
- Baseline model training
- IndoBERT fine-tuning (optional)
- Prediction examples
- API integration

## License

Part of the Indonesian Hate Speech Detection project (MIT License).

