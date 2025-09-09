"""
Loss functions for AlphaScrabble training.

Implements the combined loss function for policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AlphaScrabbleLoss(nn.Module):
    """Combined loss function for AlphaScrabble training."""
    
    def __init__(self, value_loss_weight: float = 1.0, policy_loss_weight: float = 1.0, 
                 l2_weight: float = 1e-4):
        """Initialize loss function.
        
        Args:
            value_loss_weight: Weight for value loss
            policy_loss_weight: Weight for policy loss  
            l2_weight: Weight for L2 regularization
        """
        super().__init__()
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.l2_weight = l2_weight
        
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, policy_logits: torch.Tensor, value_pred: torch.Tensor,
                policy_targets: torch.Tensor, value_targets: torch.Tensor,
                model: nn.Module) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Args:
            policy_logits: Predicted policy logits (batch_size, num_moves)
            value_pred: Predicted values (batch_size, 1)
            policy_targets: Target policy probabilities (batch_size, num_moves)
            value_targets: Target values (batch_size, 1)
            model: Model for L2 regularization
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Value loss (MSE)
        value_loss = self.mse_loss(value_pred, value_targets)
        
        # Policy loss (cross-entropy with soft targets)
        policy_loss = self._policy_loss(policy_logits, policy_targets)
        
        # L2 regularization
        l2_loss = self._l2_loss(model)
        
        # Combined loss
        total_loss = (self.value_loss_weight * value_loss + 
                     self.policy_loss_weight * policy_loss + 
                     self.l2_weight * l2_loss)
        
        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'l2_loss': l2_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _policy_loss(self, policy_logits: torch.Tensor, 
                    policy_targets: torch.Tensor) -> torch.Tensor:
        """Compute policy loss using KL divergence."""
        # Convert logits to probabilities
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Compute KL divergence: KL(target || predicted)
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
        kl_div = torch.sum(policy_targets * torch.log(policy_targets + 1e-8), dim=-1) - \
                 torch.sum(policy_targets * F.log_softmax(policy_logits, dim=-1), dim=-1)
        
        return torch.mean(kl_div)
    
    def _l2_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute L2 regularization loss."""
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
        return l2_loss


class PolicyLoss(nn.Module):
    """Policy loss only."""
    
    def __init__(self):
        """Initialize policy loss."""
        super().__init__()
    
    def forward(self, policy_logits: torch.Tensor, 
                policy_targets: torch.Tensor) -> torch.Tensor:
        """Compute policy loss."""
        # Use KL divergence
        policy_probs = F.softmax(policy_logits, dim=-1)
        kl_div = torch.sum(policy_targets * torch.log(policy_targets + 1e-8), dim=-1) - \
                 torch.sum(policy_targets * F.log_softmax(policy_logits, dim=-1), dim=-1)
        return torch.mean(kl_div)


class ValueLoss(nn.Module):
    """Value loss only."""
    
    def __init__(self):
        """Initialize value loss."""
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, value_pred: torch.Tensor, 
                value_targets: torch.Tensor) -> torch.Tensor:
        """Compute value loss."""
        return self.mse_loss(value_pred, value_targets)
