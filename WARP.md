# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Indonesian Toxicity Detection System using **BEAM (Boxology Extended Annotation Model)** architecture. A production-grade NLP system with 3-tier hierarchical detection:
- **Tier 1 (Basic)**: TF-IDF + Logistic Regression (~1-5ms latency)
- **Tier 2 (Contextual)**: BiLSTM (~10-50ms latency)  
- **Tier 3 (Sociolinguistic)**: IndoBERT (~50-200ms latency)

**Tech Stack**: Python 3.9+, PyTorch, Transformers, FastAPI, scikit-learn, Hydra

## Common Commands

### Environment Setup

```bash
# Preferred: Using uv (fast Python package manager)
uv venv
source .venv/bin/activate  # macOS/Linux
uv pip install -e ".[dev]"

# Alternative: Using pip
pip install -e ".[dev]"

# Verify installation
uv run python -c "import toxicity_detection; print('✅ Ready!')"
```

### Training Models

```bash
# Train individual models
uv run python -m toxicity_detection.cli train --model tfidf --data dataset/indonesian_hate_speech.csv
uv run python -m toxicity_detection.cli train --model bilstm --data dataset/indonesian_hate_speech.csv
uv run python -m toxicity_detection.cli train --model transformer --data dataset/indonesian_hate_speech.csv

# Without uv
python -m toxicity_detection.cli train --model tfidf --data dataset/indonesian_hate_speech.csv
```

### Evaluation

```bash
# Evaluate single model
python -m toxicity_detection.cli evaluate \
    --model tfidf \
    --model-path models/tfidf \
    --data dataset/indonesian_hate_speech.csv

# Compare all models
python -m toxicity_detection.cli compare \
    --data dataset/indonesian_hate_speech.csv \
    --models-dir models
```

### Running API Server

```bash
# Development server (with auto-reload)
uvicorn toxicity_detection.api.app:create_app --host 0.0.0.0 --port 8000 --reload --factory

# Or via CLI
uv run python -m toxicity_detection.cli serve --host 0.0.0.0 --port 8000

# API documentation: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### Testing

```bash
# Run all tests with coverage
pytest

# Detailed coverage report
pytest --cov=src/toxicity_detection --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v

# Run single test
pytest tests/unit/test_models.py::TestTFIDFModel::test_model_training -v
```

### Code Quality

```bash
# Format code (Black)
black src/ tests/ --line-length 100

# Lint (Ruff)
ruff check src/ tests/

# Type checking (MyPy)
mypy src/

# Run all quality checks
black src/ tests/ && ruff check src/ tests/ && mypy src/
```

## Architecture

### BEAM 3-Tier System

The system uses a hierarchical approach where each tier provides increasing sophistication:

1. **Input Layer**: Text preprocessing with Indonesian-specific normalization
2. **Detection Tiers**: Three parallel models with different complexity/accuracy tradeoffs
3. **Calibration Layer**: Isotonic calibration for reliable probability estimates (ECE monitoring)
4. **Fairness Layer**: Slice-based bias detection with demographic parity checks
5. **Output Layer**: Structured predictions with confidence levels and optional explanations

### Package Structure

```
src/toxicity_detection/
├── api/                    # FastAPI application
│   ├── app.py             # Endpoints, middleware, model registry
├── data/                   # Data loading and preprocessing
│   ├── loader.py          # ToxicityDataset, train/val/test splitting
│   ├── preprocessor.py    # Indonesian text cleaning, tokenization
├── models/                 # [TO BE IMPLEMENTED] Model implementations
│   ├── base.py            # BaseModel interface (referenced but missing)
│   ├── tfidf_model.py     # Tier 1: TF-IDF + LogReg
│   ├── bilstm_model.py    # Tier 2: BiLSTM
│   ├── transformer_model.py # Tier 3: IndoBERT
├── evaluation/            # Model evaluation
│   ├── evaluator.py       # ModelEvaluator with metrics computation
│   ├── metrics.py         # Custom metrics (ECE, fairness)
├── fairness/              # Bias auditing
│   ├── bias_audit.py      # Slice-based fairness analysis
├── explainability/        # Model interpretability
│   ├── error_analysis.py  # Misclassification analysis
├── utils/                 # Shared utilities
│   ├── logging.py         # Structured logging with trace IDs
│   ├── metrics.py         # Performance metrics
│   ├── seed.py            # Reproducibility utilities
├── cli.py                 # CLI commands (train/evaluate/serve/compare)
```

### Key Design Patterns

**Model Interface**: All models should implement `BaseModel` (currently referenced but not implemented):
- `train()`: Training with validation data
- `predict()`: Binary classification
- `predict_proba()`: Probability estimates (calibrated)
- `save()` / `load()`: Model persistence
- `is_trained`: Training state tracking

**API Design**: FastAPI with:
- **Model Registry**: Centralized model management with health checking
- **Trace IDs**: Request tracking via `generate_trace_id()` 
- **Tiered Endpoints**: `/api/v1/basic`, `/api/v1/contextual`, `/api/v1/sociolinguistic`
- **Structured Responses**: `ToxicityResponse` with confidence levels and optional explanations

**Configuration**: Hydra-based config management:
- `configs/config.yaml`: Main configuration
- `configs/training/default.yaml`: Training parameters

### Privacy & Ethics

**Built-in Privacy Controls**:
- `apply_pseudonymization()`: Automatic PII masking for emails, phones, user mentions
- All predictions include confidence scores for human-in-the-loop integration

**Fairness Monitoring**:
- Slice-based bias detection across demographic groups
- Calibration monitoring (ECE) to prevent overconfident predictions
- Fairness metrics: demographic parity, equalized odds

## Development Notes

### Missing Implementation

⚠️ **CRITICAL**: The `models/` package is referenced throughout but not yet implemented. When creating model files:
- Implement `BaseModel` abstract interface first
- All models must support calibration (use `sklearn.calibration.CalibratedClassifierCV`)
- Include proper error handling for untrained model access
- Support both file path and directory-based `save()`/`load()`

### Dataset Format

Expects CSV with columns:
- `text`: Indonesian text strings
- `labels`: Binary (0=non-toxic, 1=toxic)

Dataset source: [manueltonneau/indonesian-hate-speech-superset](https://huggingface.co/datasets/manueltonneau/indonesian-hate-speech-superset)

### Indonesian NLP Specifics

- **Preprocessing**: Uses Sastrawi stemmer (optional) for Indonesian text
- **Stopwords**: Common Indonesian words predefined in `IndonesianTextPreprocessor`
- **Transformer**: IndoBERT base model: `indobenchmark/indobert-base-p1`

### Testing Philosophy

- Unit tests focus on core functionality (models, preprocessing, evaluation)
- Tests use small sample datasets for speed
- Reproducibility is tested via fixed random seeds
- Model serialization is critical (train/save/load cycle)

### Code Style

- **Line length**: 100 characters (Black/Ruff configured)
- **Type hints**: Required for all functions (`mypy` enforces)
- **Logging**: Use structured logging via `loguru` (see `utils/logging.py`)
- **Imports**: Sorted via Ruff (I rule)

### Performance Considerations

- **Tier 1** should handle high throughput (lexical features only)
- **Tier 3** requires GPU for practical deployment
- API supports async endpoints for concurrent requests
- Latency is tracked per-request via `time.perf_counter()`

### Configuration System

Uses Hydra for hierarchical configuration:
- Override defaults: `python -m toxicity_detection.cli train model=bilstm data.test_size=0.3`
- Experiment outputs saved to `outputs/` (gitignored)
- All training runs should be reproducible via `seed: 42`

## Anti-Patterns to Avoid

- **Don't train without validation data**: The BEAM architecture requires calibration
- **Don't skip seed setting**: Reproducibility is critical for fair comparison
- **Don't ignore calibration**: Raw probabilities from classifiers are often poorly calibrated
- **Don't deploy without fairness audits**: Bias in toxicity detection has real social impact
- **Don't hardcode model paths**: Use CLI arguments or config files

## Quick Reference

**Model tiers map to API endpoints**:
- `tfidf` → `/api/v1/basic`
- `bilstm` → `/api/v1/contextual`  
- `transformer` → `/api/v1/sociolinguistic`

**Evaluation metrics priority**:
1. F1 Score (primary metric)
2. ROC-AUC (threshold-independent)
3. ECE (calibration quality)
4. Demographic parity (fairness)

**Common debugging**:
- Check model training state: `model.is_trained`
- Verify dataset: `dataset.get_statistics()`
- API health: `curl http://localhost:8000/health`
- Trace requests: Look for `trace_id` in logs
