"""Qdrant vector database indexing."""

from pathlib import Path

import pandas as pd
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from toxic_search.config import get_config
from toxic_search.models.encoder import ToxicEncoder


def build_index(
    client: QdrantClient | None = None,
    recreate: bool = False,
) -> QdrantClient:
    """Initialize Qdrant collection for toxic content vectors.
    
    Creates collection with appropriate distance metric and configuration.
    """
    config = get_config().qdrant
    
    client = client or QdrantClient(host=config.host, port=config.port)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == config.collection_name for c in collections)
    
    if collection_exists and recreate:
        logger.info(f"Deleting existing collection: {config.collection_name}")
        client.delete_collection(config.collection_name)
        collection_exists = False
    
    if not collection_exists:
        # Map config distance to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        
        logger.info(f"Creating collection: {config.collection_name}")
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.vector_size,
                distance=distance_map[config.distance],
            ),
        )
    
    logger.info(f"Collection ready: {config.collection_name}")
    return client


def index_documents(
    texts: list[str],
    metadata: list[dict] | None = None,
    encoder: ToxicEncoder | None = None,
    client: QdrantClient | None = None,
    batch_size: int = 100,
) -> None:
    """Index documents into Qdrant with their embeddings.
    
    Batches insertion for efficiency with large datasets.
    """
    from toxic_search.models.encoder import load_encoder
    
    config = get_config().qdrant
    
    encoder = encoder or load_encoder()
    client = client or build_index()
    metadata = metadata or [{} for _ in texts]
    
    logger.info(f"Indexing {len(texts)} documents...")
    
    # Encode all texts
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress=True)
    
    # Convert to list for Qdrant
    embeddings_list = embeddings.cpu().numpy().tolist()
    
    # Batch insertion
    points = []
    for idx, (text, embedding, meta) in enumerate(
        zip(texts, embeddings_list, metadata)
    ):
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={"text": text, **meta},
        )
        points.append(point)
        
        # Insert batch
        if len(points) >= batch_size:
            client.upsert(collection_name=config.collection_name, points=points)
            points = []
    
    # Insert remaining points
    if points:
        client.upsert(collection_name=config.collection_name, points=points)
    
    logger.info(f"Indexed {len(texts)} documents into {config.collection_name}")


def main() -> None:
    """CLI entry point for index-vectors command."""
    import argparse
    
    from toxic_search.data.loader import load_dataset
    from toxic_search.utils import setup_logging
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Index vectors into Qdrant")
    parser.add_argument("--data", type=Path, required=True, help="Data to index")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--model", type=Path, help="Fine-tuned model path (optional)")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    
    args = parser.parse_args()
    
    # Load data
    df = load_dataset(args.data, text_column=args.text_col, label_column=args.label_col)
    
    # Prepare metadata
    metadata = [{"label": label} for label in df[args.label_col]]
    
    # Load encoder
    from toxic_search.models.encoder import load_encoder
    encoder = load_encoder(model_name_or_path=args.model)
    
    # Build index
    client = build_index(recreate=args.recreate)
    
    # Index documents
    index_documents(
        texts=df[args.text_col].tolist(),
        metadata=metadata,
        encoder=encoder,
        client=client,
    )


if __name__ == "__main__":
    main()