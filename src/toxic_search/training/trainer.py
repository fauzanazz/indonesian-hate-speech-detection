"""Training loop for semantic encoder with triplet loss."""

from pathlib import Path

import torch
from loguru import logger
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from toxic_search.config import get_config
from toxic_search.data.triplets import Triplet
from toxic_search.models.encoder import ToxicEncoder


def train(
    encoder: ToxicEncoder,
    train_triplets: list[Triplet],
    val_triplets: list[Triplet] | None = None,
    output_path: str | Path = "models/fine_tuned",
) -> ToxicEncoder:
    """Train encoder with triplet loss.
    
    Uses sentence-transformers training utilities for efficiency.
    """
    config = get_config().training
    
    # Convert triplets to InputExample format
    train_examples = [
        InputExample(texts=[t.anchor, t.positive, t.negative])
        for t in train_triplets
    ]
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
    )
    
    # Define loss function
    train_loss = losses.TripletLoss(
        model=encoder.model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=config.margin,
    )
    
    # Training configuration
    warmup_steps = config.warmup_steps
    total_steps = len(train_dataloader) * config.num_epochs
    
    logger.info(f"Training configuration:")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  FP16: {config.fp16}")
    
    # Train model
    encoder.model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': config.learning_rate},
        output_path=str(output_path),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=config.fp16,
    )
    
    logger.info(f"Training complete. Model saved to {output_path}")
    
    return encoder


def main() -> None:
    """CLI entry point for train command."""
    import argparse
    
    from toxic_search.data.loader import load_dataset, split_dataset
    from toxic_search.data.triplets import TripletGenerator
    from toxic_search.models.encoder import load_encoder
    from toxic_search.utils import setup_logging
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Train toxic content encoder")
    parser.add_argument("--data", type=Path, required=True, help="Training data path")
    parser.add_argument("--output", type=Path, default="models/fine_tuned", help="Output path")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    
    args = parser.parse_args()
    
    # Load and split data
    logger.info("Loading dataset...")
    df = load_dataset(args.data, text_column=args.text_col, label_column=args.label_col)
    train_df, val_df, test_df = split_dataset(df)
    
    # Generate triplets
    logger.info("Generating triplets...")
    train_gen = TripletGenerator(train_df, args.text_col, args.label_col)
    train_triplets = []
    for batch in train_gen.generate(batch_size=1000):
        train_triplets.extend(batch)
    
    logger.info(f"Generated {len(train_triplets)} training triplets")
    
    # Load encoder
    encoder = load_encoder()
    
    # Train
    train(encoder, train_triplets, output_path=args.output)


if __name__ == "__main__":
    main()