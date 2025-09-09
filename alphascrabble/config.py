"""
Configuration settings for AlphaScrabble.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    """Main configuration class for AlphaScrabble."""
    
    # Board settings
    BOARD_SIZE: int = 15
    RACK_SIZE: int = 7
    BINGO_BONUS: int = 50
    
    # MCTS settings
    MCTS_SIMULATIONS: int = 160
    CPUCT: float = 1.5
    TEMPERATURE: float = 1.0
    TEMPERATURE_THRESHOLD: int = 10  # First N moves use high temperature
    
    # Neural network settings
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.1
    
    # Training settings
    SELFPLAY_GAMES: int = 1000
    TRAINING_EPOCHS: int = 10
    EVAL_GAMES: int = 50
    SAVE_FREQUENCY: int = 100
    
    # Paths
    LEXICA_CACHE_DIR: str = "lexica_cache"
    DATA_DIR: str = "data"
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    
    # Random seed
    SEED: Optional[int] = None
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.LEXICA_CACHE_DIR, self.DATA_DIR, 
                        self.CHECKPOINT_DIR, self.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @property
    def dawg_path(self) -> str:
        """Path to DAWG lexicon file."""
        return os.path.join(self.LEXICA_CACHE_DIR, "english_enable1.dawg")
    
    @property
    def gaddag_path(self) -> str:
        """Path to GADDAG lexicon file."""
        return os.path.join(self.LEXICA_CACHE_DIR, "english_enable1.gaddag")
    
    @property
    def enable1_path(self) -> str:
        """Path to ENABLE1 word list."""
        return os.path.join(self.LEXICA_CACHE_DIR, "enable1.txt")


# Default configuration instance
DEFAULT_CONFIG = Config()
