from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from toxicity_detection.utils.logging import setup_logger
from toxicity_detection.utils.seed import set_seed

logger = setup_logger(__name__)


class ToxicityDataset:
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "labels",
    ) -> None:
        self.data = data
        self.text_column = text_column
        self.label_column = label_column

        self._validate()

        logger.info(
            "Dataset initialized",
            n_samples=len(self.data),
            n_toxic=int(self.data[label_column].sum()),
            n_non_toxic=int((self.data[label_column] == 0).sum()),
        )

    def _validate(self) -> None:
        if self.text_column not in self.data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataset")
        if self.label_column not in self.data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataset")

        if self.data[self.text_column].isna().any():
            n_missing = self.data[self.text_column].isna().sum()
            logger.warning(f"Found {n_missing} missing text values, removing them")
            self.data = self.data.dropna(subset=[self.text_column])

        unique_labels = self.data[self.label_column].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")

        empty_texts = self.data[self.text_column].str.strip().str.len() == 0
        if empty_texts.any():
            n_empty = empty_texts.sum()
            logger.warning(f"Found {n_empty} empty text values, removing them")
            self.data = self.data[~empty_texts]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        row = self.data.iloc[idx]
        return str(row[self.text_column]), int(row[self.label_column])

    def get_texts(self) -> list[str]:
        return self.data[self.text_column].tolist()

    def get_labels(self) -> list[int]:
        return self.data[self.label_column].tolist()

    def get_statistics(self) -> dict[str, int | float]:
        labels = self.data[self.label_column]
        return {
            "n_samples": len(self.data),
            "n_toxic": int(labels.sum()),
            "n_non_toxic": int((labels == 0).sum()),
            "toxicity_rate": float(labels.mean()),
            "avg_text_length": float(self.data[self.text_column].str.len().mean()),
            "median_text_length": float(self.data[self.text_column].str.len().median()),
        }


def load_dataset(
    data_path: Path | str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[ToxicityDataset, ToxicityDataset, ToxicityDataset]:
    set_seed(random_state)

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info("Loading dataset", path=str(data_path))
    df = pd.read_csv(data_path)

    df = df.dropna(subset=["text", "labels"])
    df = df[df["text"].str.strip().str.len() > 0]

    logger.info(
        "Dataset loaded",
        n_samples=len(df),
        n_toxic=int(df["labels"].sum()),
        toxicity_rate=f"{df['labels'].mean():.2%}",
    )

    # Split into train and test
    stratify_col = df["labels"] if stratify else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Split train into train and validation
    stratify_col = train_val_df["labels"] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    train_dataset = ToxicityDataset(train_df.reset_index(drop=True))
    val_dataset = ToxicityDataset(val_df.reset_index(drop=True))
    test_dataset = ToxicityDataset(test_df.reset_index(drop=True))

    logger.info(
        "Dataset split complete",
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )

    return train_dataset, val_dataset, test_dataset


def apply_pseudonymization(text: str) -> str:
    import re

    # Replace user mentions
    text = re.sub(r"@\w+(\.\w+)*", "[USER]", text)

    # Replace email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)

    # Replace phone numbers
    text = re.sub(r"\b(\+62|0)\d{9,12}\b", "[PHONE]", text)

    # Replace URLs
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "[URL]", text)

    return text
