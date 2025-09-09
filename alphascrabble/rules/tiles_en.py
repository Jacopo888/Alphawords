"""
English Scrabble tile distribution and scoring rules.

Based on official English Scrabble rules with:
- 100 tiles total (98 letters + 2 blanks)
- Official point values and distribution
"""

from typing import Dict, List, Tuple
import random
from dataclasses import dataclass


# Official English Scrabble tile distribution and scores
TILE_SCORES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10, '?': 0  # Blank tile
}

TILE_COUNTS = {
    'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
    'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
    'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
    'Y': 2, 'Z': 1, '?': 2  # Blank tiles
}

# Verify total count is 100
assert sum(TILE_COUNTS.values()) == 100, f"Total tiles should be 100, got {sum(TILE_COUNTS.values())}"


@dataclass
class Tile:
    """Represents a Scrabble tile."""
    letter: str
    is_blank: bool = False
    blank_letter: str = ''  # What the blank represents when played
    
    def __post_init__(self):
        if self.letter == '?':
            self.is_blank = True
    
    @property
    def score(self) -> int:
        """Get the score value of this tile."""
        return TILE_SCORES[self.letter]
    
    @property
    def display_letter(self) -> str:
        """Get the letter to display (blank shows as lowercase)."""
        if self.is_blank and self.blank_letter:
            return self.blank_letter.lower()
        return self.letter
    
    def set_blank_letter(self, letter: str) -> None:
        """Set what letter this blank represents."""
        if self.is_blank:
            self.blank_letter = letter.upper()
        else:
            raise ValueError("Can only set blank letter for blank tiles")


class TileBag:
    """Manages the bag of tiles for drawing and returning."""
    
    def __init__(self, seed: int = None):
        """Initialize the tile bag with all tiles."""
        self.tiles: List[Tile] = []
        self._initialize_tiles()
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.tiles)
    
    def _initialize_tiles(self) -> None:
        """Create all tiles according to official distribution."""
        for letter, count in TILE_COUNTS.items():
            for _ in range(count):
                self.tiles.append(Tile(letter))
    
    def draw_tiles(self, count: int) -> List[Tile]:
        """Draw up to count tiles from the bag."""
        drawn = []
        for _ in range(min(count, len(self.tiles))):
            drawn.append(self.tiles.pop())
        return drawn
    
    def return_tiles(self, tiles: List[Tile]) -> None:
        """Return tiles to the bag."""
        for tile in tiles:
            if tile.is_blank:
                tile.blank_letter = ''  # Reset blank
        self.tiles.extend(tiles)
        random.shuffle(self.tiles)
    
    def tiles_remaining(self) -> int:
        """Get number of tiles remaining in bag."""
        return len(self.tiles)
    
    def is_empty(self) -> bool:
        """Check if bag is empty."""
        return len(self.tiles) == 0


def get_tile_score(letter: str) -> int:
    """Get the score for a single letter."""
    return TILE_SCORES.get(letter.upper(), 0)


def is_valid_letter(letter: str) -> bool:
    """Check if a letter is valid for English Scrabble."""
    return letter.upper() in TILE_SCORES


def get_letter_frequency() -> Dict[str, int]:
    """Get the frequency of each letter in the tile bag."""
    return TILE_COUNTS.copy()
