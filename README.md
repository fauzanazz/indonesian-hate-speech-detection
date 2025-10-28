# Indonesian Toxicity Detection System 🇮🇩

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NLP system for detecting toxicity in Indonesian text using **BEAM (Boxology Extended Annotation Model)** architecture.

## 🎯 Features

### Multi-Tier Detection System
- **Tier 1 (Basic)**: TF-IDF + Logistic Regression - Fast lexical detection (~1-5ms)
- **Tier 2 (Contextual)**: BiLSTM - Context-aware detection (~10-50ms)
- **Tier 3 (Sociolinguistic)**: IndoBERT - Advanced stylistic detection (~50-200ms)

## 📊 BEAM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Indonesian Text                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   TIER 1     │  │   TIER 2     │  │   TIER 3     │
│   Basic      │  │  Contextual  │  │Sociolinguistic│
│              │  │              │  │              │
│ TF-IDF + LR  │  │   BiLSTM     │  │  IndoBERT    │
│ (~1-5ms)     │  │ (~10-50ms)   │  │ (~50-200ms)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────────────────────────────────────────┐
│           Calibration & Risk Control             │
│  • Isotonic calibration                          │
│  • ECE monitoring                                │
│  • Confidence thresholding                       │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│         Fairness & Ethics Layer                  │
│  • Slice-based bias detection                    │
│  • Demographic parity checks                     │
│  • Equalized odds evaluation                     │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            OUTPUT: Toxicity Score                │
│  • is_toxic: boolean                             │
│  • toxicity_score: [0, 1]                        │
│  • confidence: low/medium/high                   │
│  • explanation: feature importance/attention     │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

#### Option 1: Using uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/caernations/indonesian-hate-speech-detection.git
cd indonesian-hate-speech-detection

uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies 
uv pip install -e ".[dev]"

# Verify installation
uv run python -c "import toxicity_detection; print('✅ Ready!')"
```

#### Option 2: Using pip (Traditional)

```bash
git clone https://github.com/caernations/indonesian-hate-speech-detection.git
cd indonesian-hate-speech-detection
pip install -e .
pip install -e ".[dev]"
```

### Training Models

```bash
# With uv 
uv run python -m toxicity_detection.cli train --model tfidf --data dataset/indonesian_hate_speech.csv
uv run python -m toxicity_detection.cli train --model bilstm --data dataset/indonesian_hate_speech.csv
uv run python -m toxicity_detection.cli train --model transformer --data dataset/indonesian_hate_speech.csv

# manual
python -m toxicity_detection.cli train --model tfidf --data dataset/indonesian_hate_speech.csv
```

### Evaluation

```bash
# Evaluate model
python -m toxicity_detection.cli evaluate \
    --model tfidf \
    --model-path models/tfidf \
    --data dataset/indonesian_hate_speech.csv
```

### API Server

```bash
# With uv 
uv run python -m toxicity_detection.cli serve --host 0.0.0.0 --port 8000
make -f Makefile.uv run-api

# Or with uvicorn directly
uvicorn toxicity_detection.api.app:create_app --host 0.0.0.0 --port 8000 --reload --factory
```

Visit:
- 🌐 API Docs: http://localhost:8000/docs
- 📚 ReDoc: http://localhost:8000/redoc
- ❤️ Health: http://localhost:8000/health

### Example API Usage

```python
import requests

# Basic toxicity detection (Tier 1)
response = requests.post(
    "http://localhost:8000/api/v1/basic",
    json={"text": "Ini adalah contoh teks untuk dianalisis"}
)
print(response.json())

# Contextual detection (Tier 2)
response = requests.post(
    "http://localhost:8000/api/v1/contextual",
    json={
        "text": "Teks dengan konteks yang lebih kompleks",
        "return_explanation": True
    }
)
print(response.json())

# Advanced sociolinguistic detection (Tier 3)
response = requests.post(
    "http://localhost:8000/api/v1/sociolinguistic",
    json={"text": "Teks dengan nuansa linguistik yang halus"}
)
print(response.json())
```

## 📁 Project Structure

```
TODO: Tambahin project structure
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/toxicity_detection --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v
```

## 📊 Evaluation Metrics

### Performance Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class performance

### Calibration Metrics
- **ECE (Expected Calibration Error)**: Measures calibration quality
- **Reliability diagrams**: Visual calibration assessment

### Fairness Metrics
- **Demographic Parity**: P(Y_hat=1|A=a) across groups
- **Equalized Odds**: TPR and FPR across groups
- **Performance Disparity**: F1/accuracy gaps

### Operational Metrics
- **Latency**: Inference time (p50, p95, p99)
- **Throughput**: Requests per second
- **Error rates**: By error type (FP, FN)

## 🔬 Experiments

See `notebooks/01_experiment_and_evaluation.ipynb` for comprehensive experiments including:
- what
- what
- TODO: kerjain experiment notebook


## 🛠️ Development

### Configuration

Models and training can be configured via YAML files in `configs/`:

```yaml
# configs/models/tfidf.yaml
model:
  name: "tfidf_lr"
  max_features: 10000
  ngram_range: [1, 2]
  C: 1.0
  use_calibration: true
```

## 📈 Model Performance

TODO: tambahin model performance tgable

*Results on test set of BRPPPPP (x) samples*

## 🔒 Ethics & Safety

### Privacy Controls
- **Pseudonymization**: Automatic PII masking
- **Data minimization**: Only store necessary information
- **Audit trails**: Complete logging of predictions

### Fairness Considerations
- Regular bias audits across demographic slices
- Calibration monitoring to prevent discrimination
- Transparent reporting of model limitations

### Risk Controls
- **Confidence thresholding**: Flag low-confidence predictions
- **Human-in-the-loop**: Integration points for human review
- **Adversarial robustness**: Testing against edge cases


## 🙏 Acknowledgments

- Dataset: [manueltonneau/indonesian-hate-speech-superset](https://huggingface.co/datasets/manueltonneau/indonesian-hate-speech-superset)
- IndoBERT: [indobenchmark/indobert](https://huggingface.co/indobenchmark/indobert-base-p1)

## 📧 Authors

- TODO: Tambahin authors name nim github

---

**Built with ❤️ for safer Indonesian online spaces**
