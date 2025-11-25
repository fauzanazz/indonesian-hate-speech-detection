"""Ensure Qdrant is seeded with toxic-search vectors before the API starts."""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient

from toxic_search.config import get_config
from toxic_search.data.loader import load_dataset
from toxic_search.index.builder import build_index, index_documents
from toxic_search.models.encoder import load_encoder

MAX_RETRIES = int(os.getenv("QDRANT_BOOTSTRAP_RETRIES", "30"))
RETRY_DELAY = float(os.getenv("QDRANT_BOOTSTRAP_DELAY", "2.0"))
DATASET_PATH = Path(os.getenv("QDRANT_BOOTSTRAP_DATA", "dataset/indonesian_hate_speech.csv"))
TEXT_COLUMN = os.getenv("QDRANT_BOOTSTRAP_TEXT_COLUMN", "text")
LABEL_COLUMN = os.getenv("QDRANT_BOOTSTRAP_LABEL_COLUMN", "labels")
BATCH_SIZE = int(os.getenv("QDRANT_BOOTSTRAP_BATCH_SIZE", "256"))


def wait_for_qdrant(host: str, port: int) -> QdrantClient:
    """Wait until Qdrant is reachable, then return a connected client."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Connecting to Qdrant at {}:{} (attempt {}/{})",
                host,
                port,
                attempt,
                MAX_RETRIES,
            )
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            logger.info("Qdrant is reachable")
            return client
        except Exception as exc:  # pragma: no cover - defensive bootstrap
            last_error = exc
            logger.warning("Qdrant not ready: {}", exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Unable to connect to Qdrant after {MAX_RETRIES} attempts") from last_error


def ensure_collection() -> None:
    """Create the toxic-search collection if it does not exist."""
    config = get_config().qdrant
    host = os.getenv("QDRANT_HOST", config.host)
    port = int(os.getenv("QDRANT_PORT", config.port))

    client = wait_for_qdrant(host=host, port=port)
    build_index(client=client)
    index_dataset(client=client)
    logger.info("Qdrant collection ensured successfully")


def index_dataset(client: QdrantClient) -> None:
    """Populate Qdrant with toxic-search vectors if a dataset is available."""
    if not DATASET_PATH.exists():
        logger.warning("Bootstrap dataset not found at %s; skipping enrichment", DATASET_PATH)
        return

    logger.info("Loading bootstrap dataset from %s", DATASET_PATH)
    df = load_dataset(DATASET_PATH, text_column=TEXT_COLUMN, label_column=LABEL_COLUMN)

    texts = df[TEXT_COLUMN].astype(str).tolist()
    metadata = [{LABEL_COLUMN: label} for label in df[LABEL_COLUMN]]

    logger.info("Encoding %s rows and indexing into Qdrant", len(texts))
    encoder = load_encoder()
    index_documents(
        texts=texts,
        metadata=metadata,
        encoder=encoder,
        client=client,
        batch_size=BATCH_SIZE,
    )
    logger.info("Dataset indexed successfully")


if __name__ == "__main__":
    ensure_collection()

