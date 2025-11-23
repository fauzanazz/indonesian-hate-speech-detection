"""Data mining utilities for counter speech generation.

Optional utilities for data augmentation, filtering, or preprocessing.
"""

from typing import Any

import pandas as pd
from loguru import logger


def filter_by_length(
    df: pd.DataFrame,
    text_column: str,
    counter_column: str,
    min_length: int = 10,
    max_length: int = 512,
) -> pd.DataFrame:
    """Filter dataset by text and counter speech length.
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        counter_column: Name of counter speech column
        min_length: Minimum character length
        max_length: Maximum character length
    
    Returns:
        Filtered dataframe
    """
    initial_len = len(df)
    
    # Filter by text length
    df = df[
        (df[text_column].str.len() >= min_length) &
        (df[text_column].str.len() <= max_length)
    ]
    
    # Filter by counter speech length
    df = df[
        (df[counter_column].str.len() >= min_length) &
        (df[counter_column].str.len() <= max_length)
    ]
    
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Filtered out {removed} samples by length")
    
    return df


def add_prefix(
    df: pd.DataFrame,
    text_column: str,
    prefix: str = "Tuliskan counter speech untuk teks berikut: ",
) -> pd.DataFrame:
    """Add prefix to text column for better model conditioning.
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        prefix: Prefix to add
    
    Returns:
        Dataframe with prefixed text
    """
    df = df.copy()
    df[text_column] = prefix + df[text_column]
    return df

