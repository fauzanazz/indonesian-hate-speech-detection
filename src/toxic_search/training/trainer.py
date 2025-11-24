"""Training loop for semantic encoder with triplet loss."""

import json
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
        output_path=None,  # Don't save during training
        save_best_model=False,  # We'll save explicitly after
        show_progress_bar=True,
        use_amp=config.fp16,
    )
    
    # Explicitly save the model after training
    logger.info(f"Training complete. Saving model to {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save using the encoder's save method which wraps SentenceTransformer.save
    encoder.model.save(str(output_path))
    
    # Fix missing model_type in config.json
    config_path = output_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                model_config = json.load(f)
            
            if "model_type" not in model_config:
                model_type = None
                try:
                    transformer_module = encoder.model[0]
                    if hasattr(transformer_module, "auto_model"):
                        model_type = transformer_module.auto_model.config.model_type
                except Exception as e:
                    logger.warning(f"Could not detect model_type from encoder: {e}")

                if not model_type:
                    # Fallback for paraphrase-multilingual-mpnet-base-v2
                    logger.warning("Could not detect model_type, defaulting to 'xlm-roberta'")
                    model_type = "xlm-roberta"

                model_config["model_type"] = model_type
                logger.info(f"Added missing model_type='{model_type}' to config.json")
                
                with open(config_path, "w") as f:
                    json.dump(model_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not fix model_type in config.json: {e}")
    
    # Verify the save was successful by checking for required files
    required_files = ['config.json', 'modules.json']
    saved_files = list(output_path.iterdir())
    logger.info(f"Files saved: {[f.name for f in saved_files]}")
    
    missing_files = [f for f in required_files if not (output_path / f).exists()]
    if missing_files:
        logger.warning(f"Missing required files: {missing_files}")
    else:
        logger.info(f"Model successfully saved to {output_path}")
    
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