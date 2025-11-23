"""Training orchestration for emotion detection models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .config import EmotionSettings
from .data import load_dataset, split_dataset
from .logging import configure_logging, get_logger, log_section
from .models import create_model
from .preprocess import clean_corpus


def train(settings: EmotionSettings) -> Tuple[Dict[str, float], Path]:
    """Train the selected model and persist artefacts.

    Returns a tuple of (metrics dict, artefact directory).
    """

    configure_logging(settings.log_level)
    logger = get_logger()

    with log_section("loading dataset", logger=logger):
        dataset = load_dataset(settings.data_dir)

    with log_section("splitting dataset", logger=logger):
        train_df, test_df = split_dataset(
            dataset,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )

    with log_section("preparing texts", logger=logger):
        if settings.model_name == "indobert":
            train_texts = train_df["text"].tolist()
            test_texts = test_df["text"].tolist()
        else:
            train_texts = clean_corpus(train_df["text"].tolist())
            test_texts = clean_corpus(test_df["text"].tolist())

    with log_section(f"initialising model '{settings.model_name}'", logger=logger):
        model = _create_model_from_settings(settings)

    with log_section("training", logger=logger):
        model.fit(train_texts, train_df["label"].tolist())

    with log_section("evaluating", logger=logger):
        from .evaluate import evaluate_model  # local import to avoid cycle

        metrics = evaluate_model(model, test_texts, test_df["label"].tolist())

    artefact_dir = settings.output_dir / settings.model_name
    artefact_dir.mkdir(parents=True, exist_ok=True)

    with log_section(f"saving artefacts to {artefact_dir}", logger=logger):
        model.save(artefact_dir)
        _write_json(artefact_dir / "metrics.json", metrics)

        split_info = {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "labels": sorted(dataset["label"].unique().tolist()),
        }
        _write_json(artefact_dir / "split.json", split_info)

    logger.info("Training complete: metrics=%s", metrics)

    return metrics, artefact_dir


def _create_model_from_settings(settings: EmotionSettings):
    if settings.model_name == "indobert":
        from .models.transformer_model import TransformerParams

        params = TransformerParams(
            pretrained_model_name=settings.pretrained_model_name,
            num_epochs=settings.num_epochs,
            learning_rate=settings.learning_rate,
            batch_size=settings.batch_size,
            warmup_ratio=settings.warmup_ratio,
            weight_decay=settings.weight_decay,
            random_state=settings.random_state,
        )
        return create_model("indobert", params=params)

    return create_model(
        settings.model_name,
        max_features=settings.max_features,
        ngram_range=settings.ngram_range,
        random_state=settings.random_state,
    )


def _write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


