"""
Lexicon module for AlphaScrabble.

Provides access to Quackle's DAWG and GADDAG lexicons for move generation.
"""

from .gaddag import GADDAG, GADDAGNode, WordListLoader
from .move_generator import MoveGenerator, Move, MoveTile, Position, Direction
from .gaddag_loader import GaddagLoader

__all__ = [
    'GADDAG', 'GADDAGNode', 'WordListLoader',
    'MoveGenerator', 'Move', 'MoveTile', 'Position', 'Direction',
    'GaddagLoader'
]
