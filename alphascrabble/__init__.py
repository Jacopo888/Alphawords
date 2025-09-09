"""
AlphaScrabble: AlphaZero-style Scrabble engine with MCTS and neural networks.

A complete implementation of a Scrabble engine using Monte Carlo Tree Search
with neural network guidance, similar to AlphaZero but adapted for Scrabble.
"""

__version__ = "0.1.0"
__author__ = "AlphaScrabble Team"

from .config import Config
from .rules.board import Board, Move, GameState
from .rules.tiles_en import TileBag, TILE_SCORES, TILE_COUNTS
from .engine.movegen import MoveGenerator
from .nn.model import AlphaScrabbleNet

__all__ = [
    "Config",
    "Board", 
    "Move",
    "GameState",
    "TileBag",
    "TILE_SCORES",
    "TILE_COUNTS", 
    "MoveGenerator",
    "AlphaScrabbleNet",
]
