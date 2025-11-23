"""Custom loss functions for counter speech generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss for text generation.
    
    Helps prevent overconfidence and improves generalization.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label smoothing cross-entropy loss.
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            targets: Target token ids (batch_size, seq_len)
        
        Returns:
            Scalar loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)
        batch_size, seq_len, vocab_size = logits.shape
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            true_dist.scatter_(
                2,
                targets.unsqueeze(-1),
                (1.0 - self.smoothing),
            )
            true_dist.masked_fill_(
                (targets.unsqueeze(-1) == self.ignore_index),
                0.0,
            )
        
        # Compute KL divergence
        kl_div = F.kl_div(
            log_probs.view(-1, vocab_size),
            true_dist.view(-1, vocab_size),
            reduction="none",
        )
        
        # Mask out ignored indices
        mask = (targets.view(-1) != self.ignore_index).float()
        loss = (kl_div.sum(dim=-1) * mask).sum() / mask.sum()
        
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in generation.
    
    Focuses learning on hard examples.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            targets: Target token ids (batch_size, seq_len)
        
        Returns:
            Scalar loss value
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Get probabilities of target tokens
        target_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        
        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute cross-entropy
        ce_loss = F.nll_loss(
            log_probs.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
            reduction="none",
        )
        
        # Apply focal weight
        focal_loss = (focal_weight.view(-1) * ce_loss).sum() / (
            (targets.view(-1) != self.ignore_index).sum() + 1e-8
        )
        
        return self.alpha * focal_loss

