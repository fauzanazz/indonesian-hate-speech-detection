"""Dataset ingestion helpers for the emotion detection project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TEXT_CANDIDATES: Tuple[str, ...] = (
    "text",
    "tweet",
    "sentence",
    "content",
    "message",
)

LABEL_CANDIDATES: Tuple[str, ...] = ("label", "emotion", "target")


def load_dataset(
    data_dir: Path | str,
    *,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Load and concatenate per-emotion CSV files into a clean DataFrame."""

    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    csv_files = sorted(base_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {base_path}")

    frames = [
        _load_single_csv(path, text_column=text_column, label_column=label_column)
        for path in csv_files
    ]

    dataset = pd.concat(frames, ignore_index=True)
    dataset.dropna(subset=["text", "label"], inplace=True)
    dataset.drop_duplicates(subset=["text", "label"], inplace=True)
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset["label"] = dataset["label"].astype(str).str.strip().str.lower()
    dataset = dataset[dataset["text"].str.len() > 0]

    return dataset.reset_index(drop=True)


def split_dataset(
    dataset: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return stratified train/test splits."""

    train_df, test_df = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["label"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _load_single_csv(
    path: Path,
    *,
    text_column: Optional[str],
    label_column: Optional[str],
) -> pd.DataFrame:
    # Try to detect delimiter (tab or comma)
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    delimiter = '\t' if '\t' in first_line else ','
    frame = pd.read_csv(path, delimiter=delimiter)

    # Identify text column
    text_col = _select_column(frame, text_column, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError(f"Could not infer text column for file: {path}")

    # Identify label column
    label_col = _select_column(frame, label_column, LABEL_CANDIDATES)
    if label_col is None:
        inferred_label = _infer_label_from_filename(path)
        frame["label"] = inferred_label
    else:
        frame["label"] = frame[label_col]

    frame["text"] = frame[text_col]

    return frame[["text", "label"]]


def _select_column(
    frame: pd.DataFrame,
    preferred: Optional[str],
    candidates: Sequence[str],
) -> Optional[str]:
    if preferred and preferred in frame.columns:
        return preferred

    normalized = {col.lower(): col for col in frame.columns}

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for column, dtype in frame.dtypes.items():
        if dtype == "object":
            return column

    return None


def _infer_label_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.lower().endswith("data"):
        stem = stem[:-4]
    return stem.strip().lower()

