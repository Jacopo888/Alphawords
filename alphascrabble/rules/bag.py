"""
Tile bag management for Scrabble.

Handles drawing tiles, returning tiles, and tracking remaining tiles.
"""

from typing import List, Optional
import random
from .tiles_en import Tile, TileBag


class ScrabbleBag:
    """Enhanced tile bag with additional functionality."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the tile bag."""
        self.tile_bag = TileBag(seed)
        self.drawn_tiles: List[Tile] = []  # Track all drawn tiles
    
    def draw_tiles(self, count: int) -> List[Tile]:
        """Draw tiles from the bag."""
        drawn = self.tile_bag.draw_tiles(count)
        self.drawn_tiles.extend(drawn)
        return drawn
    
    def return_tiles(self, tiles: List[Tile]) -> None:
        """Return tiles to the bag."""
        self.tile_bag.return_tiles(tiles)
        # Remove from drawn tiles tracking
        for tile in tiles:
            if tile in self.drawn_tiles:
                self.drawn_tiles.remove(tile)
    
    def tiles_remaining(self) -> int:
        """Get number of tiles remaining in bag."""
        return self.tile_bag.tiles_remaining()
    
    def is_empty(self) -> bool:
        """Check if bag is empty."""
        return self.tile_bag.is_empty()
    
    def get_remaining_tiles(self) -> List[Tile]:
        """Get list of all remaining tiles in bag."""
        return self.tile_bag.tiles.copy()
    
    def get_drawn_tiles(self) -> List[Tile]:
        """Get list of all tiles drawn so far."""
        return self.drawn_tiles.copy()
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the bag to initial state."""
        self.tile_bag = TileBag(seed)
        self.drawn_tiles = []
    
    def get_tile_frequency(self) -> dict:
        """Get frequency of each tile type remaining."""
        frequency = {}
        for tile in self.tile_bag.tiles:
            letter = tile.letter
            frequency[letter] = frequency.get(letter, 0) + 1
        return frequency
    
    def get_drawn_frequency(self) -> dict:
        """Get frequency of each tile type drawn."""
        frequency = {}
        for tile in self.drawn_tiles:
            letter = tile.letter
            frequency[letter] = frequency.get(letter, 0) + 1
        return frequency
