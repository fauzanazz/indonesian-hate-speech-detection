import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from toxicity_detection.data.preprocessor import IndonesianTextPreprocessor
from toxicity_detection.models.base import BaseModel
from toxicity_detection.utils.logging import setup_logger
from toxicity_detection.utils.seed import set_seed

logger = setup_logger(__name__)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str = "lstm",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        # RNN layer
        rnn_class = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        if lengths is not None:
            packed = pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            rnn_out, hidden = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        else:
            rnn_out, hidden = self.rnn(embedded)

        if self.cell_type == "lstm":
            hidden = hidden[0] 

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        output = self.fc(hidden)

        return output


class BiLSTMModel(BaseModel):
    def __init__(
        self,
        name: str = "bilstm",
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = "lstm",
        bidirectional: bool = True,
        max_vocab_size: int = 20000,
        max_seq_length: int = 128,
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        **preprocessor_kwargs: Any,
    ) -> None:

        super().__init__(name)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.preprocessor = IndonesianTextPreprocessor(
            remove_punctuation=False,
            **preprocessor_kwargs,
        )

        self.tokenizer: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[BiLSTMClassifier] = None
        self.criterion = nn.BCEWithLogitsLoss()

        logger.info(
            "BiLSTM model initialized",
            cell_type=cell_type,
            hidden_dim=hidden_dim,
            device=str(self.device),
        )

    def _build_vocab(self, texts: List[str]) -> None:
        word_freq: Dict[str, int] = {}

        for text in texts:
            processed = self.preprocessor.preprocess(text)
            for word in processed.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[: self.max_vocab_size - 2]: 
            if word not in self.tokenizer:
                self.tokenizer[word] = len(self.tokenizer)

        logger.info("Vocabulary built", vocab_size=len(self.tokenizer))

    def _texts_to_sequences(self, texts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        lengths = []

        for text in texts:
            processed = self.preprocessor.preprocess(text)
            tokens = processed.split()[: self.max_seq_length]
            seq = [self.tokenizer.get(token, 1) for token in tokens]
            sequences.append(seq)
            lengths.append(len(seq))

        max_len = min(max(lengths) if lengths else 1, self.max_seq_length)
        padded = np.zeros((len(sequences), max_len), dtype=np.int64)

        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq

        return torch.LongTensor(padded), torch.LongTensor(lengths)

    def train(
        self,
        X_train: List[str],
        y_train: npt.NDArray[Any],
        X_val: Optional[List[str]] = None,
        y_val: Optional[npt.NDArray[Any]] = None,
        epochs: Optional[int] = None,
        patience: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if epochs is None:
            epochs = self.epochs
        logger.info("Training BiLSTM model", n_samples=len(X_train), epochs=epochs)
        start_time = time.time()

        set_seed(42)
        self._build_vocab(X_train)

        self.model = BiLSTMClassifier(
            vocab_size=len(self.tokenizer),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
            bidirectional=self.bidirectional,
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        X_train_seq, train_lengths = self._texts_to_sequences(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        train_dataset = TensorDataset(X_train_seq, train_lengths, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, val_lengths = self._texts_to_sequences(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_seq, val_lengths, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_lengths, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_lengths).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_lengths, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_x, batch_lengths).squeeze()
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info("Early stopping", epoch=epoch + 1)
                    break

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}" if val_loss else "N/A",
                )

        self.is_trained = True
        training_time = time.time() - start_time

        logger.info("Training complete", training_time=f"{training_time:.2f}s")

        return {
            "training_time": training_time,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "history": history,
        }

    def predict(self, texts: List[str]) -> npt.NDArray[Any]:
        proba = self.predict_proba(texts)
        return (proba > 0.5).astype(int)

    def predict_proba(self, texts: List[str]) -> npt.NDArray[Any]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        X_seq, lengths = self._texts_to_sequences(texts)
        dataset = TensorDataset(X_seq, lengths)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []

        with torch.no_grad():
            for batch_x, batch_lengths in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x, batch_lengths).squeeze()
                proba = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(proba if proba.ndim > 0 else [proba.item()])

        return np.array(predictions)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"{self.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        import joblib

        metadata = {
            "name": self.name,
            "tokenizer": self.tokenizer,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "cell_type": self.cell_type,
            "bidirectional": self.bidirectional,
            "max_seq_length": self.max_seq_length,
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
        self.tokenizer = metadata["tokenizer"]
        self.is_trained = metadata["is_trained"]

        self.model = BiLSTMClassifier(
            vocab_size=len(self.tokenizer),
            embedding_dim=metadata["embedding_dim"],
            hidden_dim=metadata["hidden_dim"],
            num_layers=metadata["num_layers"],
            dropout=metadata["dropout"],
            cell_type=metadata["cell_type"],
            bidirectional=metadata["bidirectional"],
        ).to(self.device)

        model_path = path / f"{self.name}_model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        logger.info("Model loaded", path=str(path))

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "cell_type": self.cell_type,
            "vocab_size": len(self.tokenizer),
        })
        return params
