"""Thin wrapper around sentence-transformers for toxic content encoding."""

from pathlib import Path

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from toxic_search.config import get_config


class ToxicEncoder:
    """Semantic encoder for toxic content using sentence-transformers."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode texts into embeddings.
        
        Returns normalized embeddings for cosine similarity by default.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=normalize,
        )
        return embeddings

    def save(self, path: str | Path) -> None:
        """Save fine-tuned model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Saved model to {path}")

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


def load_encoder(
    model_name_or_path: str | None = None,
    max_seq_length: int | None = None,
) -> ToxicEncoder:
    """Load encoder from pretrained model or checkpoint.
    
    Single source of truth for model loading across training/inference.
    """
    config = get_config().model
    
    model_name_or_path = model_name_or_path or config.base_model
    max_seq_length = max_seq_length or config.max_seq_length
    
    logger.info(f"Loading encoder: {model_name_or_path}")
    
    model = SentenceTransformer(model_name_or_path)
    model.max_seq_length = max_seq_length
    
    encoder = ToxicEncoder(model)
    logger.info(f"Encoder loaded: dim={encoder.embedding_dim}, device={encoder.device}")
    
    return encoder