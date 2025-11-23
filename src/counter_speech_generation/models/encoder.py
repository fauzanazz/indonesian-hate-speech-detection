"""IndoT5 model wrapper for counter speech generation."""

from pathlib import Path

import torch
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from counter_speech_generation.config import get_config


class CounterSpeechGenerator:
    """IndoT5-based generator for counter speech."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        text: str | list[str],
        max_length: int | None = None,
        num_beams: int | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        prefix: str = "Tuliskan counter speech untuk teks berikut: ",
    ) -> str | list[str]:
        """Generate counter speech for given toxic text.
        
        Args:
            text: Input toxic text(s)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
            repetition_penalty: Repetition penalty
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            prefix: Prefix to add to input text
        
        Returns:
            Generated counter speech(s)
        """
        config = get_config().model
        
        max_length = max_length or config.max_target_length
        num_beams = num_beams or config.num_beams
        length_penalty = length_penalty or config.length_penalty
        repetition_penalty = repetition_penalty or config.repetition_penalty
        temperature = temperature or config.temperature
        do_sample = do_sample if do_sample is not None else config.do_sample
        
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        # Add prefix
        inputs = [prefix + t for t in text]
        
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                early_stopping=True,
            )
        
        # Decode
        generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if is_single:
            return generated[0]
        return generated

    def save(self, path: str | Path) -> None:
        """Save model and tokenizer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"Saved model to {path}")

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)


def load_generator(
    model_name_or_path: str | None = None,
    device: str | None = None,
) -> CounterSpeechGenerator:
    """Load generator from pretrained model or checkpoint.
    
    Single source of truth for model loading across training/inference.
    """
    config = get_config()
    model_config = config.model
    train_config = config.training
    
    model_name_or_path = model_name_or_path or model_config.base_model
    
    # Determine device
    if device is None:
        if train_config.use_cpu:
            device = "cpu"
            logger.warning("Using CPU as specified in config (use_cpu=True)")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading generator: {model_name_or_path}")
    logger.info(f"Device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    generator = CounterSpeechGenerator(model, tokenizer, device)
    logger.info(f"Generator loaded: vocab_size={generator.vocab_size}, device={generator.device}")
    
    return generator

