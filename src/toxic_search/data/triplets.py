"""Triplet construction for contrastive learning."""

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
    """Generate training triplets from labeled data."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
    ):
        self.df = df
        self.text_column = text_column
        self.label_column = label_column
        
        # Group by label for efficient sampling
        self.label_groups = df.groupby(label_column)[text_column].apply(list).to_dict()
        self.labels = list(self.label_groups.keys())

    def generate(self, batch_size: int = 32) -> Iterator[list[Triplet]]:
        """Generate batches of triplets.
        
        For each anchor:
        - Positive: same label, different text
        - Negative: different label (from hard negatives if available)
        """
        triplets = []
        
        for _, row in self.df.iterrows():
            anchor_text = row[self.text_column]
            anchor_label = row[self.label_column]
            
            # Sample positive (same label, different text)
            positives = [
                text for text in self.label_groups[anchor_label] 
                if text != anchor_text
            ]
            
            if not positives:
                continue
            
            positive_text = positives[0] if len(positives) == 1 else \
                pd.Series(positives).sample(1).iloc[0]
            
            # Sample negative (different label)
            negative_labels = [label for label in self.labels if label != anchor_label]
            
            if not negative_labels:
                continue
            
            negative_label = pd.Series(negative_labels).sample(1).iloc[0]
            negative_text = pd.Series(self.label_groups[negative_label]).sample(1).iloc[0]
            
            triplets.append(Triplet(
                anchor=anchor_text,
                positive=positive_text,
                negative=negative_text,
            ))
            
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


def generate_triplets(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 32,
) -> Iterator[list[Triplet]]:
    """Generate triplets from a DataFrame.
    
    Convenience function wrapping TripletGenerator.
    """
    generator = TripletGenerator(df, text_column, label_column)
    logger.info(f"Generating triplets from {len(df)} samples")
    
    yield from generator.generate(batch_size)