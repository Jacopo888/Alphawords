"""
Dataset and replay buffer for AlphaScrabble training.

Handles storage and loading of self-play data.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from ..rules.board import Board, Move, GameState
from ..rules.tiles_en import Tile


class GameRecord:
    """Record of a single game."""
    
    def __init__(self, game_id: str, moves: List[Move], scores: List[int], 
                 winner: int, game_length: int):
        """Initialize game record."""
        self.game_id = game_id
        self.moves = moves
        self.scores = scores
        self.winner = winner
        self.game_length = game_length
        self.timestamp = pd.Timestamp.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'moves': self.moves,
            'scores': self.scores,
            'winner': self.winner,
            'game_length': self.game_length,
            'timestamp': self.timestamp
        }


class ReplayBuffer:
    """Replay buffer for storing self-play data."""
    
    def __init__(self, max_size: int = 10000, data_dir: str = "data"):
        """Initialize replay buffer."""
        self.max_size = max_size
        self.data_dir = data_dir
        self.buffer: List[GameRecord] = []
        self.current_size = 0
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    def add_game(self, game_record: GameRecord) -> None:
        """Add a game to the buffer."""
        if self.current_size >= self.max_size:
            # Remove oldest game
            self.buffer.pop(0)
            self.current_size -= 1
        
        self.buffer.append(game_record)
        self.current_size += 1
    
    def add_games(self, game_records: List[GameRecord]) -> None:
        """Add multiple games to the buffer."""
        for game_record in game_records:
            self.add_game(game_record)
    
    def get_games(self, num_games: Optional[int] = None) -> List[GameRecord]:
        """Get games from buffer."""
        if num_games is None:
            return self.buffer.copy()
        else:
            return self.buffer[-num_games:].copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.current_size = 0
    
    def save(self, filename: str) -> None:
        """Save buffer to file."""
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert to serializable format
        data = {
            'games': [game.to_dict() for game in self.buffer],
            'max_size': self.max_size,
            'current_size': self.current_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filename: str) -> None:
        """Load buffer from file."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Buffer file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.max_size = data['max_size']
        self.current_size = data['current_size']
        
        # Reconstruct games
        self.buffer = []
        for game_data in data['games']:
            game = GameRecord(
                game_id=game_data['game_id'],
                moves=game_data['moves'],
                scores=game_data['scores'],
                winner=game_data['winner'],
                game_length=game_data['game_length']
            )
            game.timestamp = game_data['timestamp']
            self.buffer.append(game)


class GameDataset(Dataset):
    """Dataset for training neural network."""
    
    def __init__(self, game_records: List[GameRecord], feature_extractor, 
                 max_moves_per_game: int = 50):
        """Initialize dataset."""
        self.game_records = game_records
        self.feature_extractor = feature_extractor
        self.max_moves_per_game = max_moves_per_game
        
        # Flatten all training examples
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[dict]:
        """Create training examples from game records."""
        examples = []
        
        for game in self.game_records:
            # Create examples for each move in the game
            for i, move in enumerate(game.moves):
                if i >= self.max_moves_per_game:
                    break
                
                # Create example
                example = {
                    'game_id': game.game_id,
                    'move_index': i,
                    'move': move,
                    'final_score': game.scores[0] - game.scores[1],  # Score difference
                    'winner': game.winner
                }
                examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get training example."""
        example = self.examples[idx]
        
        # This is a simplified version - in practice you'd need to reconstruct
        # the board state and rack for each move
        return {
            'game_id': example['game_id'],
            'move_index': example['move_index'],
            'move': example['move'],
            'final_score': example['final_score'],
            'winner': example['winner']
        }


class TrainingDataManager:
    """Manages training data storage and loading."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data manager."""
        self.data_dir = data_dir
        self.replay_buffer = ReplayBuffer(data_dir=data_dir)
        
        # Create subdirectories
        os.makedirs(os.path.join(data_dir, "selfplay"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "checkpoints"), exist_ok=True)
    
    def save_selfplay_data(self, games: List[GameRecord], 
                          filename: str = "selfplay_games.pkl") -> None:
        """Save self-play data."""
        filepath = os.path.join(self.data_dir, "selfplay", filename)
        self.replay_buffer.buffer = games
        self.replay_buffer.save(filename)
    
    def load_selfplay_data(self, filename: str = "selfplay_games.pkl") -> List[GameRecord]:
        """Load self-play data."""
        filepath = os.path.join(self.data_dir, "selfplay", filename)
        self.replay_buffer.load(filename)
        return self.replay_buffer.get_games()
    
    def create_training_dataset(self, games: List[GameRecord], 
                               feature_extractor) -> GameDataset:
        """Create training dataset from games."""
        return GameDataset(games, feature_extractor)
    
    def get_data_loader(self, dataset: GameDataset, batch_size: int = 32, 
                       shuffle: bool = True) -> DataLoader:
        """Create data loader."""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
