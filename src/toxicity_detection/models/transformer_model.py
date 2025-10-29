import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from toxicity_detection.models.base import BaseModel
from toxicity_detection.utils.logging import setup_logger
from toxicity_detection.utils.seed import set_seed

logger = setup_logger(__name__)


class TransformerModel(BaseModel):
    def __init__(
        self,
        name: str = "indobert",
        model_name: str = "indobenchmark/indobert-base-p1",
        max_length: int = 64, 
        batch_size: int = 4, 
        epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,  
    ) -> None:
        super().__init__(name)

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer", model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.trainer: Optional[Trainer] = None

        logger.info(
            "Transformer model initialized",
            model=model_name,
            device=str(self.device),
            max_length=max_length,
        )

    def _prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[npt.NDArray[Any]] = None,
    ) -> HFDataset:
        data_dict = {"text": texts}
        if labels is not None:
            data_dict["label"] = labels.tolist()

        dataset = HFDataset.from_dict(data_dict)

        def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False, 
            )

        tokenized = dataset.map(tokenize_function, batched=True)
        return tokenized

    def _compute_metrics(self, eval_pred: tuple[npt.NDArray[Any], npt.NDArray[Any]]) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)

        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }

    def train(
        self,
        X_train: List[str],
        y_train: npt.NDArray[Any],
        X_val: Optional[List[str]] = None,
        y_val: Optional[npt.NDArray[Any]] = None,
        epochs: Optional[int] = None,
        output_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if epochs is None:
            epochs = self.epochs
        logger.info(
            "Training transformer model",
            n_samples=len(X_train),
            epochs=epochs,
            model=self.model_name,
        )
        start_time = time.time()

        set_seed(42)

        train_dataset = self._prepare_dataset(X_train, y_train)
        eval_dataset = None
        if X_val is not None and y_val is not None:
            eval_dataset = self._prepare_dataset(X_val, y_val)

        logger.info("Loading pre-trained model", model=self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification",
        )

        if output_dir is None:
            output_dir = Path("./outputs") / self.name

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            greater_is_better=True,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            save_total_limit=2,
            seed=42,
            fp16=torch.cuda.is_available(), 
            report_to="none",  
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics if eval_dataset else None,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        train_result = self.trainer.train()

        self.is_trained = True
        training_time = time.time() - start_time

        logger.info(
            "Training complete",
            training_time=f"{training_time:.2f}s",
            final_loss=f"{train_result.training_loss:.4f}",
        )

        metrics = {
            "training_time": training_time,
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics,
        }

        return metrics

    def predict(self, texts: List[str]) -> npt.NDArray[Any]:
        proba = self.predict_proba(texts)
        return (proba > 0.5).astype(int)

    def predict_proba(self, texts: List[str]) -> npt.NDArray[Any]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        dataset = self._prepare_dataset(texts)
        self.model.eval()

        if self.trainer is not None:
            predictions = self.trainer.predict(dataset)
            logits = predictions.predictions
        else:
            from torch.utils.data import DataLoader

            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["input_ids", "token_type_ids", "attention_mask"]])
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=data_collator,
            )

            all_logits = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    all_logits.append(outputs.logits.cpu())

            logits = torch.cat(all_logits, dim=0).numpy()

        proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
        return proba[:, 1] 

    def get_attention_weights(
        self,
        text: str,
        layer: int = -1,
    ) -> tuple[List[str], npt.NDArray[Any]]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting attention")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attention = outputs.attentions[layer][0]
        attention_avg = attention.mean(dim=0).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return tokens, attention_avg

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save")

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        import joblib

        metadata = {
            "name": self.name,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "is_trained": self.is_trained,
        }
        metadata_path = path / f"{self.name}_metadata.joblib"
        joblib.dump(metadata, metadata_path)

        logger.info("Model saved", path=str(path))

    def load(self, path: Path) -> None:
        import joblib
        path = Path(path)
        metadata_path = path / f"{self.name}_metadata.joblib"
        metadata = joblib.load(metadata_path)

        self.name = metadata["name"]
        self.model_name = metadata["model_name"]
        self.max_length = metadata["max_length"]
        self.batch_size = metadata.get("batch_size", 4)  
        self.is_trained = metadata["is_trained"]
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model.to(self.device)

        logger.info("Model loaded", path=str(path))

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        })
        return params
