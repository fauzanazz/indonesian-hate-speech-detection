"""Dataset loading and splitting utilities."""

from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


def load_dataset(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """Load dataset from CSV or JSON file.
    
    Single source of truth for data loading across the project.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    required_cols = {text_column, label_column}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(df)} samples from {path}")
    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratify_column: str | None = "label",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test sets.
    
    Ensures reproducible splits with optional stratification.
    """
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("Split sizes must sum to 1.0")
    
    stratify = df[stratify_column] if stratify_column else None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify,
        random_state=random_state,
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    temp_stratify = temp_df[stratify_column] if stratify_column else None
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_stratify,
        random_state=random_state,
    )
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df