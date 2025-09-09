"""
Neural network architecture for AlphaScrabble.

Implements a CNN-based architecture with policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm."""
    
    def __init__(self, channels: int, dropout: float = 0.1):
        """Initialize residual block."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.LayerNorm([channels, 15, 15])
        self.norm2 = nn.LayerNorm([channels, 15, 15])
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = x + residual
        x = F.relu(x)
        return x


class BoardEncoder(nn.Module):
    """Encodes board state into features."""
    
    def __init__(self, input_channels: int = 32, hidden_channels: int = 64, 
                 num_blocks: int = 8, dropout: float = 0.1):
        """Initialize board encoder."""
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, 3, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, dropout) for _ in range(num_blocks)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(hidden_channels, 256)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Board features of shape (batch_size, input_channels, 15, 15)
            
        Returns:
            Tuple of (board_features, global_embedding)
        """
        # Initial convolution
        x = F.relu(self.conv_in(x))
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global embedding
        global_emb = self.global_pool(x).squeeze(-1).squeeze(-1)  # (batch_size, hidden_channels)
        global_emb = F.relu(self.global_fc(global_emb))  # (batch_size, 256)
        
        return x, global_emb


class RackEncoder(nn.Module):
    """Encodes rack state into features."""
    
    def __init__(self, input_dim: int = 27, hidden_dim: int = 128, output_dim: int = 128):
        """Initialize rack encoder."""
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Rack features of shape (batch_size, input_dim)
            
        Returns:
            Rack embedding of shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PolicyHead(nn.Module):
    """Policy head for move selection."""
    
    def __init__(self, board_channels: int = 64, rack_dim: int = 128, 
                 move_feature_dim: int = 64, hidden_dim: int = 256):
        """Initialize policy head."""
        super().__init__()
        
        # Board feature processing
        self.board_conv = nn.Conv2d(board_channels, 32, 1)
        self.board_fc = nn.Linear(32 * 15 * 15, 128)
        
        # Combined features
        combined_dim = 128 + rack_dim + move_feature_dim
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, board_features: torch.Tensor, rack_emb: torch.Tensor, 
                move_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            board_features: Board features (batch_size, board_channels, 15, 15)
            rack_emb: Rack embedding (batch_size, rack_dim)
            move_features: Move features (batch_size, num_moves, move_feature_dim)
            
        Returns:
            Policy logits (batch_size, num_moves)
        """
        batch_size, num_moves, _ = move_features.shape
        
        # Process board features
        board_conv = F.relu(self.board_conv(board_features))  # (batch_size, 32, 15, 15)
        board_flat = board_conv.view(batch_size, -1)  # (batch_size, 32*15*15)
        board_emb = F.relu(self.board_fc(board_flat))  # (batch_size, 128)
        
        # Expand board and rack embeddings for each move
        board_expanded = board_emb.unsqueeze(1).expand(-1, num_moves, -1)  # (batch_size, num_moves, 128)
        rack_expanded = rack_emb.unsqueeze(1).expand(-1, num_moves, -1)  # (batch_size, num_moves, rack_dim)
        
        # Combine features
        combined = torch.cat([board_expanded, rack_expanded, move_features], dim=-1)
        
        # Process through MLP
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # (batch_size, num_moves, 1)
        
        return x.squeeze(-1)  # (batch_size, num_moves)


class ValueHead(nn.Module):
    """Value head for position evaluation."""
    
    def __init__(self, board_dim: int = 256, rack_dim: int = 128, hidden_dim: int = 256):
        """Initialize value head."""
        super().__init__()
        
        combined_dim = board_dim + rack_dim
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, board_emb: torch.Tensor, rack_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            board_emb: Board embedding (batch_size, board_dim)
            rack_emb: Rack embedding (batch_size, rack_dim)
            
        Returns:
            Value estimate (batch_size, 1)
        """
        # Combine embeddings
        combined = torch.cat([board_emb, rack_emb], dim=-1)
        
        # Process through MLP
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))  # Output in [-1, 1]
        
        return x


class AlphaScrabbleNet(nn.Module):
    """Main neural network for AlphaScrabble."""
    
    def __init__(self, board_channels: int = 32, rack_dim: int = 27, 
                 move_feature_dim: int = 64, hidden_channels: int = 64,
                 num_blocks: int = 8, dropout: float = 0.1):
        """Initialize AlphaScrabble network."""
        super().__init__()
        
        # Encoders
        self.board_encoder = BoardEncoder(board_channels, hidden_channels, num_blocks, dropout)
        self.rack_encoder = RackEncoder(rack_dim, 128, 128)
        
        # Heads
        self.policy_head = PolicyHead(hidden_channels, 128, move_feature_dim)
        self.value_head = ValueHead(256, 128)
        
    def forward(self, board_features: torch.Tensor, rack_features: torch.Tensor, 
                move_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            board_features: Board features (batch_size, board_channels, 15, 15)
            rack_features: Rack features (batch_size, rack_dim)
            move_features: Move features (batch_size, num_moves, move_feature_dim)
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Encode board and rack
        board_conv, board_emb = self.board_encoder(board_features)
        rack_emb = self.rack_encoder(rack_features)
        
        # Get policy and value
        policy_logits = self.policy_head(board_conv, rack_emb, move_features)
        value = self.value_head(board_emb, rack_emb)
        
        return policy_logits, value
    
    def predict(self, board_features: np.ndarray, rack_features: np.ndarray, 
                move_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for single state.
        
        Args:
            board_features: Board features (board_channels, 15, 15)
            rack_features: Rack features (rack_dim,)
            move_features: Move features (num_moves, move_feature_dim)
            
        Returns:
            Tuple of (policy_logits, value)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            board_tensor = torch.FloatTensor(board_features).unsqueeze(0)
            rack_tensor = torch.FloatTensor(rack_features).unsqueeze(0)
            move_tensor = torch.FloatTensor(move_features).unsqueeze(0)
            
            # Move to device if available
            if next(self.parameters()).is_cuda:
                board_tensor = board_tensor.cuda()
                rack_tensor = rack_tensor.cuda()
                move_tensor = move_tensor.cuda()
            
            # Forward pass
            policy_logits, value = self.forward(board_tensor, rack_tensor, move_tensor)
            
            # Convert back to numpy
            policy_logits = policy_logits.cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
            
            return policy_logits, value
    
    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'board_channels': 32,
                'rack_dim': 27,
                'move_feature_dim': 64,
                'hidden_channels': 64,
                'num_blocks': 8,
                'dropout': 0.1
            }
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'AlphaScrabbleNet':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['model_config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
