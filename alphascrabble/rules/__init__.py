"""
Scrabble rules implementation.

Contains all the core game rules, board representation, and scoring logic.
"""

from .board import Board, Move, GameState, Position, PlacedTile, PremiumType
from .tiles_en import Tile, TileBag, TILE_SCORES, TILE_COUNTS, get_tile_score, is_valid_letter
from .bag import ScrabbleBag

__all__ = [
    "Board",
    "Move", 
    "GameState",
    "Position",
    "PlacedTile",
    "PremiumType",
    "Tile",
    "TileBag",
    "TILE_SCORES",
    "TILE_COUNTS", 
    "get_tile_score",
    "is_valid_letter",
    "ScrabbleBag",
]
