"""Training loop for IndoT5 counter speech generation."""

from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from counter_speech_generation.config import get_config
from counter_speech_generation.models.encoder import CounterSpeechGenerator, load_generator
from counter_speech_generation.training.experiment import ExperimentTracker, get_tracker
from counter_speech_generation.utils import clear_gpu_cache


def prepare_dataset(
    texts: list[str],
    counter_speeches: list[str],
    tokenizer,
    max_length: int = 512,
    max_target_length: int = 128,
    prefix: str = "Tuliskan counter speech untuk teks berikut: ",
) -> Dataset:
    """Prepare dataset for training.
    
    Args:
        texts: List of input texts
        counter_speeches: List of target counter speeches
        tokenizer: Tokenizer instance
        max_length: Maximum input length
        max_target_length: Maximum target length
        prefix: Prefix to add to inputs
    
    Returns:
        HuggingFace Dataset ready for training
    """
    # Add prefix to inputs
    inputs = [prefix + text for text in texts]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            counter_speeches,
            max_length=max_target_length,
            padding=True,
            truncation=True,
        )
    
    # Set labels
    model_inputs["labels"] = labels["input_ids"]
    
    # Convert to Dataset
    dataset = Dataset.from_dict(model_inputs)
    
    return dataset


def train(
    generator: CounterSpeechGenerator,
    train_texts: list[str],
    train_counters: list[str],
    val_texts: list[str] | None = None,
    val_counters: list[str] | None = None,
    output_path: str | Path = "models/counter_speech",
    track_experiment: bool = True,
    experiment_name: str | None = None,
) -> CounterSpeechGenerator:
    """Train IndoT5 model for counter speech generation.
    
    Args:
        generator: Initialized generator
        train_texts: Training input texts
        train_counters: Training target counter speeches
        val_texts: Validation input texts (optional)
        val_counters: Validation target counter speeches (optional)
        output_path: Path to save trained model
    
    Returns:
        Trained generator
    """
    config = get_config()
    train_config = config.training
    model_config = config.model
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize experiment tracking
    tracker = None
    if track_experiment:
        tracker = get_tracker()
        exp_id = tracker.start_experiment(experiment_name=experiment_name)
        # Save training config
        tracker.log_training_config({
            "output_path": str(output_path),
            "train_samples": len(train_texts),
            "val_samples": len(val_texts) if val_texts else 0,
        })
    
    logger.info("Preparing training dataset...")
    train_dataset = prepare_dataset(
        train_texts,
        train_counters,
        generator.tokenizer,
        max_length=model_config.max_length,
        max_target_length=model_config.max_target_length,
    )
    
    val_dataset = None
    if val_texts and val_counters:
        logger.info("Preparing validation dataset...")
        val_dataset = prepare_dataset(
            val_texts,
            val_counters,
            generator.tokenizer,
            max_length=model_config.max_length,
            max_target_length=model_config.max_target_length,
        )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=generator.tokenizer,
        model=generator.model,
        padding=True,
    )
    
    # Enable gradient checkpointing to save memory
    if train_config.gradient_checkpointing:
        generator.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled to save memory")
    
    # Clear GPU cache before training
    clear_gpu_cache()
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.batch_size,
        per_device_eval_batch_size=max(1, train_config.batch_size // 2),  # Smaller eval batch
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        max_grad_norm=train_config.max_grad_norm,
        logging_steps=train_config.logging_steps,
        eval_steps=train_config.evaluation_steps if val_dataset else None,
        save_steps=train_config.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        fp16=train_config.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=train_config.dataloader_num_workers,
        dataloader_pin_memory=train_config.dataloader_pin_memory,
        report_to="none",  # Can be changed to "tensorboard" or "wandb"
        remove_unused_columns=False,  # Keep all columns
        max_steps=-1,  # Use epochs instead
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=generator.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=generator.tokenizer,
    )
    
    logger.info("Starting training...")
    logger.info(f"  Batch size: {train_config.batch_size}")
    logger.info(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {train_config.num_epochs}")
    logger.info(f"  Learning rate: {train_config.learning_rate}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Gradient checkpointing: {train_config.gradient_checkpointing}")
    logger.info(f"  FP16: {train_config.fp16 and torch.cuda.is_available()}")
    if val_dataset:
        logger.info(f"  Validation samples: {len(val_dataset)}")
    
    # Train with error handling for OOM
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.error("CUDA out of memory error occurred!")
            logger.error("Try the following solutions:")
            logger.error("  1. Reduce batch_size to 1 in config")
            logger.error("  2. Reduce max_length in model config")
            logger.error("  3. Increase gradient_accumulation_steps")
            logger.error("  4. Set use_cpu=True to train on CPU (slower but works)")
            logger.error("  5. Use a smaller model or reduce sequence lengths")
            clear_gpu_cache()
            raise
        else:
            raise
    
    # Save final model
    trainer.save_model()
    generator.tokenizer.save_pretrained(str(output_path))
    
    # Save training config with model
    if tracker:
        import json
        config_summary = {
            "model": config.model.dict(),
            "training": config.training.dict(),
            "data": config.data.dict(),
        }
        with open(output_path / "training_config.json", "w") as f:
            json.dump(config_summary, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {output_path}")
    
    return generator


def main() -> None:
    """CLI entry point for train command."""
    import argparse
    
    from counter_speech_generation.data.loader import load_dataset, split_dataset
    from counter_speech_generation.models.encoder import load_generator
    from counter_speech_generation.utils import setup_logging
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Train counter speech generator")
    parser.add_argument("--data", type=Path, required=True, help="Training data path")
    parser.add_argument("--output", type=Path, default="models/counter_speech", help="Output path")
    parser.add_argument("--text-col", default=None, help="Text column name")
    parser.add_argument("--counter-col", default=None, help="Counter speech column name")
    
    args = parser.parse_args()
    
    # Load and split data
    logger.info("Loading dataset...")
    df = load_dataset(
        args.data,
        text_column=args.text_col,
        counter_column=args.counter_col,
    )
    train_df, val_df, test_df = split_dataset(df)
    
    # Load generator
    generator = load_generator()
    
    # Train
    train(
        generator,
        train_df["text"].tolist(),
        train_df["counter"].tolist(),
        val_df["text"].tolist(),
        val_df["counter"].tolist(),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

