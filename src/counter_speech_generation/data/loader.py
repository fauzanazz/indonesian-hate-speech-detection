"""Dataset loading and splitting utilities for counter speech generation."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from counter_speech_generation.config import get_config


def load_dataset(
    path: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    counter_column: str | None = None,
) -> pd.DataFrame:
    """Load dataset from CSV or JSON file.
    
    Single source of truth for data loading across the project.
    """
    config = get_config().data
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Use config defaults if not provided
    text_column = text_column or config.text_column
    label_column = label_column or config.label_column
    counter_column = counter_column or config.counter_column
    
    required_cols = {text_column, counter_column}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out rows with missing counter speech
    initial_len = len(df)
    df = df.dropna(subset=[text_column, counter_column])
    df = df[df[counter_column].str.strip().astype(bool)]
    
    if len(df) < initial_len:
        logger.warning(f"Removed {initial_len - len(df)} rows with missing counter speech")
    
    logger.info(f"Loaded {len(df)} samples from {path}")
    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float | None = None,
    val_size: float | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test sets.
    
    Ensures reproducible splits.
    """
    config = get_config().data
    
    train_size = train_size or config.train_split
    val_size = val_size or config.val_split
    test_size = test_size or config.test_split
    random_state = random_state or config.random_state
    
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("Split sizes must sum to 1.0")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
    )
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df

