"""Transformer-based emotion detection model using IndoBERT."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..registry import register_model
from .base import BaseEmotionModel


@dataclass
class TransformerParams:
    pretrained_model_name: str = "indobenchmark/indobert-base-p1"
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    max_length: int = 256
    random_state: int = 42


class EmotionDataset(Dataset):
    def __init__(self, encodings: dict, labels: Optional[List[int]] = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:  # type: ignore[override]
        return len(next(iter(self.encodings.values())))

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@register_model("indobert")
class IndoBERTEmotionModel(BaseEmotionModel):
    """Fine-tuned IndoBERT classifier."""

    def __init__(self, params: Optional[TransformerParams] = None) -> None:
        self.params = params or TransformerParams()
        self.model = None
        self.tokenizer = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- BaseEmotionModel API -------------------------------------------
    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> None:
        texts_list = list(texts)
        labels_list = list(labels)
        if not texts_list:
            raise ValueError("Training data is empty")

        unique_labels = sorted(set(labels_list))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.params.pretrained_model_name)
        encodings = self.tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            max_length=self.params.max_length,
        )
        label_ids = [self.label2id[label] for label in labels_list]
        dataset = EmotionDataset(encodings, label_ids)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.params.pretrained_model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

        training_args = TrainingArguments(
            output_dir=tempfile.mkdtemp(prefix="emotion_trainer_"),
            num_train_epochs=self.params.num_epochs,
            learning_rate=self.params.learning_rate,
            per_device_train_batch_size=self.params.batch_size,
            warmup_ratio=self.params.warmup_ratio,
            weight_decay=self.params.weight_decay,
            logging_strategy="epoch",
            save_strategy="no",
            seed=self.params.random_state,
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        self.model.eval()

    def predict(self, texts: Iterable[str]) -> List[str]:
        probs = self._predict_proba_tensor(list(texts))
        predicted = torch.argmax(probs, dim=-1).tolist()
        return [self.id2label[idx] for idx in predicted]

    def predict_proba(self, texts: Iterable[str]) -> List[dict[str, float]]:
        probs = self._predict_proba_tensor(list(texts))
        labels = [self.id2label[idx] for idx in range(probs.shape[-1])]
        return [dict(zip(labels, row.tolist())) for row in probs]

    def save(self, path: Path) -> None:
        model = self._ensure_model()
        tokenizer = self._ensure_tokenizer()
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        metadata = {
            "params": self.params.__dict__,
            "label2id": self.label2id,
        }
        with (path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle)

    @classmethod
    def load(cls, path: Path) -> "IndoBERTEmotionModel":
        with (path / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        params = TransformerParams(**metadata.get("params", {}))
        instance = cls(params=params)
        instance.label2id = metadata.get("label2id", {})
        instance.id2label = {int(v): k for k, v in instance.label2id.items()}
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.to(instance.device)
        instance.model.eval()
        return instance

    # --- Internal helpers ----------------------------------------------
    def _predict_proba_tensor(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, len(self.id2label)))

        model = self._ensure_model()
        tokenizer = self._ensure_tokenizer()
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.params.max_length,
        )
        dataset = EmotionDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=self.params.batch_size)

        collected: List[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                logits = model(**inputs).logits
                collected.append(F.softmax(logits, dim=-1).cpu())

        return torch.cat(collected, dim=0)

    def _ensure_model(self):
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.model

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialised.")
        return self.tokenizer

