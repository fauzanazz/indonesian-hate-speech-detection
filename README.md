# Indonesian Hate Speech & Toxicity Detection

This project is a comprehensive solution for detecting, analyzing, and monitoring hate speech and toxic content in Indonesian text. It features a sophisticated multi-tiered machine learning backend served via a unified FastAPI server and a modern, user-friendly web interface built with Next.js.

![Web UI Screenshot](https://via.placeholder.com/1000x400.png?text=Project+UI+Screenshot)

## ✨ Core Features

This platform provides two primary AI-powered services:

1.  **Multi-Tiered Toxicity Detection**:
    -   Analyzes text to determine its toxicity level.
    -   Utilizes an ensemble of models, ranging from simple and fast (TF-IDF) to complex and context-aware (BiLSTM, IndoBERT), providing a nuanced and accurate prediction.
    -   The tiered approach allows for a trade-off between speed and accuracy depending on the use case.

2.  **Semantic Search for Toxic Content**:
    -   Finds semantically similar toxic comments or posts from a database.
    -   Powered by sentence-transformer models and the Qdrant vector database for efficient and scalable similarity search.

3.  **Interactive Web UI**:
    -   A clean and responsive interface to interact with the AI features.
    -   Test custom sentences, see model explanations, and search for content.
    -   Includes a dashboard for analytics (feature from `FeaturesSection.tsx`).

## 🛠️ Technology Stack

The project is a monorepo combining a Python backend with a TypeScript/Next.js frontend.

-   **Backend**:
    -   **Framework**: FastAPI
    -   **ML Models**: Scikit-learn, PyTorch, Transformers (Hugging Face)
    -   **Vector Database**: Qdrant
    -   **Logging**: Loguru

-   **Frontend**:
    -   **Framework**: Next.js (with App Router)
    -   **Language**: TypeScript
    -   **UI**: React, Tailwind CSS, shadcn/ui
    -   **Package Manager**: Bun

-   **DevOps / Tooling**:
    -   Python dependency management with `uv`.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.9+ and `uv` (`pip install uv`)
-   Node.js v18+ and Bun (`npm install -g bun`)
-   An instance of Qdrant running (e.g., via Docker).

### 1. Backend Setup

First, set up and run the FastAPI server.

```bash
# Navigate to the project root
cd indonesian-hate-speech-detection

# Create a virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# (Optional) You may need to download model artifacts.
# Please refer to the model storage solution for this project.

# Run the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API server should now be running at `http://localhost:8000`. You can access the documentation at `http://localhost:8000/docs`.

### 2. Frontend Setup

In a new terminal, set up and run the Next.js web application.

```bash
# Navigate to the webgui directory
cd src/webgui

# Install dependencies
bun install

# Run the development server
bun dev
```

The web interface should now be accessible at `http://localhost:3000`.

## 📂 Project Structure

The repository is organized as follows:

```
├── artifacts/          # Saved model artifacts (encoders, metrics)
├── configs/            # Configuration files for models and services
├── dataset/            # Raw and processed datasets
├── models/             # Trained model checkpoints
├── notebooks/          # Jupyter notebooks for experimentation
├── src/
│   ├── api/            # Unified FastAPI server source code
│   ├── toxic_search/   # Python module for semantic search logic
│   ├── toxicity_detection/ # Python module for toxicity classification logic
│   └── webgui/         # Next.js frontend application
├── tests/              # Unit and integration tests
└── pyproject.toml      # Python project configuration and dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
