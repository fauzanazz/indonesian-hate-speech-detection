# Architecture Documentation

## BEAM (Boxology Extended Annotation Model) Architecture

This document describes the architecture of the Indonesian Toxicity Detection System.

## System Overview

The system implements a **3-tier hierarchical detection architecture** with increasing complexity and accuracy:

```
┌─────────────────────────────────────────────────────────────────┐
│                        System Architecture                      │
└─────────────────────────────────────────────────────────────────┘

                          ┌──────────────┐
                          │   Input      │
                          │   Layer      │
                          └──────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │  Tier 1   │ │  Tier 2   │ │  Tier 3   │
            │  Basic    │ │Contextual │ │Socioling. │
            └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                  │             │             │
                  └─────────────┼─────────────┘
                                │
                          ┌─────┴──────┐
                          │Calibration │
                          │   Layer    │
                          └─────┬──────┘
                                │
                          ┌─────┴──────┐
                          │ Fairness   │
                          │   Layer    │
                          └─────┬──────┘
                                │
                          ┌─────┴──────┐
                          │   Output   │
                          │   Layer    │
                          └────────────┘
```

## Component Architecture

### 1. Data Pipeline

```
Data Sources → Loader → Preprocessor → Model
     │                                   │
     └─── Privacy Controls ──────────────┘
          (Pseudonymization)
```

**Components:**
- `ToxicityDataset`: Data container with validation
- `IndonesianTextPreprocessor`: Text cleaning and normalization
- `load_dataset()`: Train/val/test splitting with stratification

**Privacy Controls:**
- PII masking (emails, phones, user mentions)
- Pseudonymization hooks
- Data minimization

### 2. Model Tier Architecture

#### Tier 1: TF-IDF + Logistic Regression
```
Text → Preprocessing → TF-IDF → LogReg → Calibration → Prediction
                           │
                           └─→ Feature Importance
```

**Features:**
- Fast inference (~1-5ms)
- Interpretable (TF-IDF weights)
- Lexical features (n-grams)
- Isotonic calibration

#### Tier 2: BiLSTM
```
Text → Preprocessing → Tokenization → Embedding → BiLSTM → Dense → Prediction
                                                     │
                                                     └─→ Dropout + Regularization
```

**Features:**
- Context-aware (~10-50ms)
- Sequence modeling
- Bidirectional processing
- Early stopping

#### Tier 3: IndoBERT
```
Text → Tokenization → BERT Encoder → Classification Head → Prediction
                          │
                          └─→ Attention Weights (Explainability)
```

**Features:**
- Deep contextual understanding (~50-200ms)
- Pre-trained on Indonesian
- Transfer learning
- Attention-based explainability

### 3. Evaluation Architecture

```
Model → Evaluator → Metrics Calculator
   │                      │
   │                      ├─→ Performance (F1, ROC-AUC, PR-AUC)
   │                      ├─→ Calibration (ECE)
   │                      ├─→ Latency
   │                      └─→ Fairness (slice metrics)
   │
   └─→ Error Analyzer → Confusion Matrix
                      → Misclassification Analysis
                      → Error Patterns
```

### 4. API Architecture

```
Client Request
     │
     ▼
┌─────────────────┐
│  FastAPI        │
│  ┌───────────┐  │
│  │Middleware │  │  ← Logging, Tracing, CORS
│  └───────────┘  │
│  ┌───────────┐  │
│  │ Endpoints │  │
│  │ /basic    │  │  ← Tier 1 (TF-IDF)
│  │ /context  │  │  ← Tier 2 (BiLSTM)
│  │ /socio    │  │  ← Tier 3 (IndoBERT)
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Registry │  │  ← Model management
│  └───────────┘  │
└─────────────────┘
     │
     ▼
Response (JSON)
```

**Features:**
- Async endpoints
- Request tracing (trace_id)
- Latency monitoring
- Input validation
- Error handling

### 5. Fairness & Ethics Architecture

```
Model Predictions
     │
     ▼
┌───────────────────┐
│  Bias Auditor     │
│  ┌─────────────┐  │
│  │ Slice Maker │  │  ← Create demographic slices
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │   Metrics   │  │  ← Demographic parity, TPR, FPR
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ Gap Analysis│  │  ← Compute fairness gaps
│  └─────────────┘  │
└───────────────────┘
     │
     ▼
Fairness Report
```

## Design Principles

### 1. SOLID Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extendable without modifying existing code
- **Liskov Substitution**: All models implement `BaseModel`
- **Interface Segregation**: Clean interfaces for evaluation, error analysis
- **Dependency Inversion**: Depend on abstractions (`BaseModel`)

### 2. MLOps Practices

- **Reproducibility**: Fixed seeds, deterministic algorithms
- **Monitoring**: Structured logging with trace IDs
- **Versioning**: Model metadata and checkpoints
- **Testing**: Unit tests for all components
- **CI/CD**: Automated testing and deployment

### 3. Ethical AI

- **Transparency**: Explainability through feature importance and attention
- **Fairness**: Regular bias audits across slices
- **Privacy**: PII masking and pseudonymization
- **Safety**: Calibrated predictions with confidence scores

## Data Flow

### Training Flow
```
1. Dataset Loading
   └─→ CSV → DataFrame → ToxicityDataset

2. Preprocessing
   └─→ Clean → Normalize → Tokenize

3. Model Training
   └─→ Train → Validate → Early Stopping

4. Calibration
   └─→ Fit Calibrator → ECE Calculation

5. Evaluation
   └─→ Test Set → Metrics → Error Analysis → Fairness Audit

6. Model Saving
   └─→ Checkpoint → Metadata → Registry
```

### Inference Flow
```
1. API Request
   └─→ Validation → Trace ID Assignment

2. Preprocessing
   └─→ Clean → Normalize

3. Model Inference
   └─→ Feature Extraction → Prediction → Calibration

4. Post-processing
   └─→ Confidence Calculation → Explanation (optional)

5. Response
   └─→ JSON → Logging → Client
```

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Load balancing support
- Docker containerization
- Kubernetes-ready

### Performance Optimization
- Model caching
- Batch inference support
- GPU acceleration (Tier 3)
- Async processing

### Monitoring
- Request latency tracking
- Error rate monitoring
- Model performance drift detection
- Fairness metric tracking

## Security

### API Security
- Input validation
- Rate limiting hooks
- CORS configuration
- Authentication-ready

### Data Security
- PII masking
- Secure model storage
- Audit logging
- GDPR compliance hooks

## Future Enhancements

1. **Ensemble Methods**: Combine tiers for improved accuracy
2. **Active Learning**: Continuous model improvement
3. **Multi-language Support**: Extend to other Indonesian dialects
4. **Real-time Monitoring**: Prometheus + Grafana integration
5. **A/B Testing**: Framework for model experimentation
6. **Explainability**: LIME/SHAP integration for Tier 2/3

## References

- BEAM Architecture: Internal design document
- FastAPI: https://fastapi.tiangolo.com
- IndoBERT: https://huggingface.co/indobenchmark
- Fairness in ML: https://fairmlbook.org
