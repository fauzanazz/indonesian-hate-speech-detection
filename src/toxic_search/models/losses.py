"""Custom loss functions for metric learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss with configurable margin.
    
    Encourages anchor to be closer to positive than negative by margin.
    L = max(0, d(a,p) - d(a,n) + margin)
    """

    def __init__(self, margin: float = 0.5, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
        """
        if self.distance == "cosine":
            # Cosine distance: 1 - cosine_similarity
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        elif self.distance == "euclidean":
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

        # Triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class OnlineTripletLoss(nn.Module):
    """Online triplet mining with batch-level hard negative selection.
    
    Mines hardest negatives within each batch for efficiency.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute online triplet loss with hard negative mining.
        
        Args:
            embeddings: All embeddings in batch (batch_size, embedding_dim)
            labels: Corresponding labels (batch_size,)
        """
        # Compute pairwise cosine distances
        similarity = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        )
        distance = 1 - similarity

        # Create masks for positives and negatives
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal

        # For each anchor, find hardest positive and negative
        pos_distances = distance.clone()
        pos_distances[~labels_equal] = -float('inf')
        hardest_positive = pos_distances.max(dim=1)[0]

        neg_distances = distance.clone()
        neg_distances[~labels_not_equal] = float('inf')
        hardest_negative = neg_distances.min(dim=1)[0]

        # Compute triplet loss
        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        
        # Only consider valid triplets (where positive and negative exist)
        valid_triplets = (hardest_positive > -float('inf')) & (hardest_negative < float('inf'))
        
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return losses[valid_triplets].mean()