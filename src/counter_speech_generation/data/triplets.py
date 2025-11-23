"""Optional triplet construction for contrastive learning.

This module is kept for consistency with toxic_search structure,
but may not be used for text generation tasks.
"""

from dataclasses import dataclass
from typing import Iterator

import pandas as pd
from loguru import logger


@dataclass
class Triplet:
    """Container for (anchor, positive, negative) triplet."""

    anchor: str
    positive: str
    negative: str


class TripletGenerator:
    """Generate training triplets from labeled data.
    
    Note: This may not be directly applicable to counter speech generation,
    but is included for structural consistency.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        counter_column: str = "counter",
    ):
        self.df = df
        self.text_column = text_column
        self.counter_column = counter_column

    def generate(self, batch_size: int = 32) -> Iterator[list[Triplet]]:
        """Generate batches of triplets.
        
        For counter speech: anchor=toxic text, positive=counter speech, negative=random counter
        """
        triplets = []

        for _, row in self.df.iterrows():
            anchor_text = row[self.text_column]
            positive_text = row[self.counter_column]

            # Sample negative (different counter speech)
            negative_row = self.df[self.df[self.counter_column] != positive_text].sample(1)
            if negative_row.empty:
                continue

            negative_text = negative_row[self.counter_column].iloc[0]

            triplets.append(
                Triplet(
                    anchor=anchor_text,
                    positive=positive_text,
                    negative=negative_text,
                )
            )

            # Yield batch when ready
            if len(triplets) >= batch_size:
                yield triplets
                triplets = []

        # Yield remaining triplets
        if triplets:
            yield triplets

    def __len__(self) -> int:
        """Estimate number of possible triplets."""
        return len(self.df)

