"""
Training module for AlphaScrabble neural network.

Implements the training loop with self-play data.
"""

import os
import time
from typing import List, Dict, Optional, Tuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .model import AlphaScrabbleNet
from .loss import AlphaScrabbleLoss
from .dataset import GameDataset, TrainingDataManager
from ..engine.features import FeatureExtractor
from ..utils.logging import get_logger


class Trainer:
    """Trainer for AlphaScrabble neural network."""
    
    def __init__(self, model: AlphaScrabbleNet, config: dict, 
                 data_manager: TrainingDataManager, device: str = 'auto'):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            config: Training configuration
            data_manager: Data manager for loading training data
            device: Device to use for training ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.config = config
        self.data_manager = data_manager
        self.feature_extractor = FeatureExtractor()
        self.logger = get_logger(__name__)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Initialize loss function
        self.criterion = AlphaScrabbleLoss(
            value_loss_weight=config.get('value_loss_weight', 1.0),
            policy_loss_weight=config.get('policy_loss_weight', 1.0),
            l2_weight=config.get('l2_weight', 1e-4)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            policy_logits, value_pred = self.model(
                batch['board_features'],
                batch['rack_features'], 
                batch['move_features']
            )
            
            # Compute loss
            loss, loss_dict = self.criterion(
                policy_logits, value_pred,
                batch['policy_targets'], batch['value_targets'],
                self.model
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                policy_logits, value_pred = self.model(
                    batch['board_features'],
                    batch['rack_features'],
                    batch['move_features']
                )
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    policy_logits, value_pred,
                    batch['policy_targets'], batch['value_targets'],
                    self.model
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'val_loss': avg_loss}
    
    def train(self, train_games: List, val_games: List, 
              epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_games: Training game records
            val_games: Validation game records
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Training games: {len(train_games)}, Validation games: {len(val_games)}")
        
        # Create datasets
        train_dataset = self.data_manager.create_training_dataset(train_games, self.feature_extractor)
        val_dataset = self.data_manager.create_training_dataset(val_games, self.feature_extractor)
        
        # Create data loaders
        train_loader = self.data_manager.get_data_loader(train_dataset, batch_size, shuffle=True)
        val_loader = self.data_manager.get_data_loader(val_dataset, batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                self.save_checkpoint(epoch + 1)
        
        self.logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, epoch: int, filename: Optional[str] = None) -> None:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.data_manager.data_dir, "checkpoints", filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def get_model(self) -> AlphaScrabbleNet:
        """Get the trained model."""
        return self.model
