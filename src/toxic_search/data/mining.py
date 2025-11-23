"""Hard negative mining using BM25 and semantic similarity."""

import numpy as np
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi

from toxic_search.config import get_config


def mine_hard_negatives(
    anchor_texts: list[str],
    candidate_texts: list[str],
    anchor_labels: list[int],
    candidate_labels: list[int],
    k: int | None = None,
) -> list[list[str]]:
    """Mine hard negatives using BM25 lexical similarity.
    
    Hard negatives are texts that are lexically similar but semantically different
    (i.e., different labels). This improves contrastive learning.
    """
    config = get_config().mining
    k = k or config.negatives_per_anchor
    
    # Tokenize for BM25
    tokenized_candidates = [text.lower().split() for text in candidate_texts]
    bm25 = BM25Okapi(tokenized_candidates)
    
    hard_negatives = []
    
    for anchor_text, anchor_label in zip(anchor_texts, anchor_labels):
        tokenized_query = anchor_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get candidates with different labels (true negatives)
        negative_mask = np.array(candidate_labels) != anchor_label
        negative_indices = np.where(negative_mask)[0]
        
        if len(negative_indices) == 0:
            logger.warning(f"No negatives found for anchor: {anchor_text[:50]}...")
            hard_negatives.append([])
            continue
        
        # Score only negative candidates
        negative_scores = scores[negative_indices]
        
        # Apply temperature sampling for diversity
        if config.sampling_temperature > 0:
            probs = np.exp(negative_scores / config.sampling_temperature)
            probs /= probs.sum()
            sampled_idx = np.random.choice(
                len(negative_indices),
                size=min(k, len(negative_indices)),
                replace=False,
                p=probs,
            )
        else:
            # Greedy: take top-k by score
            sampled_idx = np.argsort(negative_scores)[-k:]
        
        selected_negatives = [candidate_texts[negative_indices[i]] for i in sampled_idx]
        hard_negatives.append(selected_negatives)
    
    logger.info(f"Mined {sum(len(negs) for negs in hard_negatives)} hard negatives")
    return hard_negatives


def main() -> None:
    """CLI entry point for mine-negatives command."""
    import argparse
    from pathlib import Path
    
    from toxic_search.data.loader import load_dataset
    from toxic_search.utils import setup_logging
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Mine hard negatives using BM25")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Output path for mined negatives")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    
    args = parser.parse_args()
    
    # Load data
    df = load_dataset(args.input, text_column=args.text_col, label_column=args.label_col)
    
    # Mine negatives
    negatives = mine_hard_negatives(
        anchor_texts=df[args.text_col].tolist(),
        candidate_texts=df[args.text_col].tolist(),
        anchor_labels=df[args.label_col].tolist(),
        candidate_labels=df[args.label_col].tolist(),
    )
    
    # Save results
    result_df = pd.DataFrame({
        "anchor": df[args.text_col].tolist(),
        "hard_negatives": negatives,
    })
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_json(args.output, orient="records", indent=2)
    
    logger.info(f"Saved hard negatives to {args.output}")


if __name__ == "__main__":
    main()