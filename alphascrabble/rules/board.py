"""
Scrabble board implementation with official rules.

Includes:
- 15x15 board with premium squares
- Move representation and validation
- Scoring with word multipliers and bonuses
- Game state management
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import copy
from .tiles_en import Tile, TILE_SCORES


class PremiumType(Enum):
    """Types of premium squares."""
    NONE = "none"
    DLS = "dls"  # Double Letter Score
    TLS = "tls"  # Triple Letter Score  
    DWS = "dws"  # Double Word Score
    TWS = "tws"  # Triple Word Score
    CENTER = "center"  # Center square (first move)


@dataclass
class Position:
    """Represents a position on the board."""
    row: int
    col: int
    
    def __post_init__(self):
        if not (0 <= self.row < 15 and 0 <= self.col < 15):
            raise ValueError(f"Position ({self.row}, {self.col}) is out of bounds")
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        return isinstance(other, Position) and self.row == other.row and self.col == other.col


@dataclass
class PlacedTile:
    """A tile placed on the board."""
    tile: Tile
    position: Position
    is_new: bool = True  # Whether this tile was placed this turn


@dataclass
class Move:
    """Represents a Scrabble move."""
    tiles: List[PlacedTile]  # Tiles being placed
    direction: str  # 'across' or 'down'
    main_word: str  # The main word formed
    cross_words: List[str] = field(default_factory=list)  # Words formed by crossing
    score: int = 0
    bingo_bonus: int = 0
    total_score: int = 0
    
    def __post_init__(self):
        """Calculate total score after initialization."""
        self.total_score = self.score + self.bingo_bonus
    
    @property
    def is_bingo(self) -> bool:
        """Check if this is a bingo (using all 7 tiles)."""
        return len(self.tiles) == 7
    
    @property
    def notation(self) -> str:
        """Get move notation (e.g., 'HELLO A8')."""
        if not self.tiles:
            return "PASS"
        
        start_pos = self.tiles[0].position
        word = self.main_word
        pos_str = f"{chr(65 + start_pos.row)}{start_pos.col + 1}"
        return f"{word} {pos_str}{self.direction[0].upper()}"


class Board:
    """15x15 Scrabble board with premium squares."""
    
    # Premium square layout (row, col, type)
    PREMIUM_SQUARES = [
        # Triple Word Score
        (0, 0), (0, 7), (0, 14),
        (7, 0), (7, 14),
        (14, 0), (14, 7), (14, 14),
        
        # Double Word Score  
        (1, 1), (1, 13), (2, 2), (2, 12), (3, 3), (3, 11),
        (4, 4), (4, 10), (10, 4), (10, 10), (11, 3), (11, 11),
        (12, 2), (12, 12), (13, 1), (13, 13),
        
        # Triple Letter Score
        (1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
        (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9),
        
        # Double Letter Score
        (0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
        (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
        (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
        (12, 6), (12, 8), (14, 3), (14, 11),
    ]
    
    def __init__(self):
        """Initialize empty board."""
        self.grid: List[List[Optional[PlacedTile]]] = [
            [None for _ in range(15)] for _ in range(15)
        ]
        self.premium_map: Dict[Position, PremiumType] = self._create_premium_map()
        self.move_count = 0
    
    def _create_premium_map(self) -> Dict[Position, PremiumType]:
        """Create mapping of positions to premium types."""
        premium_map = {}
        
        # Center square
        premium_map[Position(7, 7)] = PremiumType.CENTER
        
        # Triple Word Score
        tws_positions = [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)]
        for row, col in tws_positions:
            premium_map[Position(row, col)] = PremiumType.TWS
        
        # Double Word Score
        dws_positions = [(1, 1), (1, 13), (2, 2), (2, 12), (3, 3), (3, 11), (4, 4), (4, 10),
                        (10, 4), (10, 10), (11, 3), (11, 11), (12, 2), (12, 12), (13, 1), (13, 13)]
        for row, col in dws_positions:
            premium_map[Position(row, col)] = PremiumType.DWS
        
        # Triple Letter Score
        tls_positions = [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
                        (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)]
        for row, col in tls_positions:
            premium_map[Position(row, col)] = PremiumType.TLS
        
        # Double Letter Score
        dls_positions = [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                        (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
                        (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
                        (12, 6), (12, 8), (14, 3), (14, 11)]
        for row, col in dls_positions:
            premium_map[Position(row, col)] = PremiumType.DLS
        
        return premium_map
    
    def get_tile(self, row: int, col: int) -> Optional[PlacedTile]:
        """Get tile at position, or None if empty."""
        if 0 <= row < 15 and 0 <= col < 15:
            return self.grid[row][col]
        return None
    
    def place_tile(self, tile: Tile, row: int, col: int, is_new: bool = True) -> None:
        """Place a tile on the board."""
        if not (0 <= row < 15 and 0 <= col < 15):
            raise ValueError(f"Position ({row}, {col}) is out of bounds")
        
        if self.grid[row][col] is not None:
            raise ValueError(f"Position ({row}, {col}) is already occupied")
        
        self.grid[row][col] = PlacedTile(tile, Position(row, col), is_new)
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if position is empty."""
        return self.get_tile(row, col) is None
    
    def get_premium_type(self, row: int, col: int) -> PremiumType:
        """Get premium type for position."""
        pos = Position(row, col)
        return self.premium_map.get(pos, PremiumType.NONE)
    
    def is_center_occupied(self) -> bool:
        """Check if center square is occupied."""
        return not self.is_empty(7, 7)
    
    def get_word_at_position(self, row: int, col: int, direction: str) -> Optional[str]:
        """Get the word formed at a position in given direction."""
        if direction == 'across':
            return self._get_word_across(row, col)
        elif direction == 'down':
            return self._get_word_down(row, col)
        return None
    
    def _get_word_across(self, row: int, col: int) -> Optional[str]:
        """Get word going across from position."""
        if self.is_empty(row, col):
            return None
        
        # Find start of word
        start_col = col
        while start_col > 0 and not self.is_empty(row, start_col - 1):
            start_col -= 1
        
        # Build word
        word = ""
        for c in range(start_col, 15):
            tile = self.get_tile(row, c)
            if tile is None:
                break
            word += tile.display_letter
        
        return word if len(word) > 1 else None
    
    def _get_word_down(self, row: int, col: int) -> Optional[str]:
        """Get word going down from position."""
        if self.is_empty(row, col):
            return None
        
        # Find start of word
        start_row = row
        while start_row > 0 and not self.is_empty(start_row - 1, col):
            start_row -= 1
        
        # Build word
        word = ""
        for r in range(start_row, 15):
            tile = self.get_tile(r, col)
            if tile is None:
                break
            word += tile.display_letter
        
        return word if len(word) > 1 else None
    
    def calculate_move_score(self, move: Move) -> int:
        """Calculate score for a move."""
        if not move.tiles:
            return 0
        
        total_score = 0
        word_multiplier = 1
        
        # Calculate main word score
        main_word_score = 0
        for placed_tile in move.tiles:
            tile_score = placed_tile.tile.score
            premium = self.get_premium_type(placed_tile.position.row, placed_tile.position.col)
            
            if premium == PremiumType.DLS:
                tile_score *= 2
            elif premium == PremiumType.TLS:
                tile_score *= 3
            
            if premium == PremiumType.DWS:
                word_multiplier *= 2
            elif premium == PremiumType.TWS:
                word_multiplier *= 3
            
            main_word_score += tile_score
        
        total_score += main_word_score * word_multiplier
        
        # Add bingo bonus if applicable
        if move.is_bingo:
            total_score += 50
        
        return total_score
    
    def apply_move(self, move: Move) -> None:
        """Apply a move to the board."""
        for placed_tile in move.tiles:
            self.place_tile(placed_tile.tile, placed_tile.position.row, 
                          placed_tile.position.col, is_new=True)
        self.move_count += 1
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.grid = copy.deepcopy(self.grid)
        new_board.move_count = self.move_count
        return new_board
    
    def display(self) -> str:
        """Get string representation of board."""
        lines = []
        lines.append("   " + "".join(f"{i:2d}" for i in range(15)))
        
        for row in range(15):
            line = f"{chr(65 + row):2s} "
            for col in range(15):
                tile = self.get_tile(row, col)
                if tile:
                    line += f" {tile.display_letter}"
                else:
                    premium = self.get_premium_type(row, col)
                    if premium == PremiumType.CENTER:
                        line += " *"
                    elif premium == PremiumType.TWS:
                        line += "3W"
                    elif premium == PremiumType.DWS:
                        line += "2W"
                    elif premium == PremiumType.TLS:
                        line += "3L"
                    elif premium == PremiumType.DLS:
                        line += "2L"
                    else:
                        line += " ."
            lines.append(line)
        
        return "\n".join(lines)


@dataclass
class GameState:
    """Complete game state including board, players, and scores."""
    board: Board
    players: List[str]  # Player names
    scores: List[int]  # Current scores
    racks: List[List[Tile]]  # Player racks
    current_player: int  # Index of current player
    tile_bag: 'TileBag'
    game_over: bool = False
    winner: Optional[int] = None
    last_move: Optional[Move] = None
    
    def __post_init__(self):
        """Initialize game state."""
        if len(self.players) != len(self.scores) or len(self.players) != len(self.racks):
            raise ValueError("Players, scores, and racks must have same length")
    
    def get_current_rack(self) -> List[Tile]:
        """Get current player's rack."""
        return self.racks[self.current_player]
    
    def get_current_score(self) -> int:
        """Get current player's score."""
        return self.scores[self.current_player]
    
    def next_player(self) -> None:
        """Move to next player."""
        self.current_player = (self.current_player + 1) % len(self.players)
    
    def add_score(self, player: int, points: int) -> None:
        """Add points to player's score."""
        self.scores[player] += points
    
    def draw_tiles(self, count: int) -> List[Tile]:
        """Draw tiles for current player."""
        drawn = self.tile_bag.draw_tiles(count)
        self.racks[self.current_player].extend(drawn)
        return drawn
    
    def remove_tiles_from_rack(self, tiles: List[Tile]) -> None:
        """Remove tiles from current player's rack."""
        for tile in tiles:
            if tile in self.racks[self.current_player]:
                self.racks[self.current_player].remove(tile)
    
    def check_game_over(self) -> bool:
        """Check if game is over."""
        # Game ends when a player uses all tiles or all players pass consecutively
        for i, rack in enumerate(self.racks):
            if not rack:  # Player has no tiles
                self.game_over = True
                self.winner = i
                return True
        
        # Check if bag is empty and no valid moves possible
        if self.tile_bag.is_empty():
            # This would require checking for valid moves, simplified for now
            pass
        
        return False
    
    def calculate_final_scores(self) -> None:
        """Calculate final scores including penalties for remaining tiles."""
        if not self.game_over:
            return
        
        for i, rack in enumerate(self.racks):
            if rack:  # Player has remaining tiles
                penalty = sum(tile.score for tile in rack)
                self.scores[i] -= penalty
                
                # Add penalty to other players
                for j in range(len(self.players)):
                    if i != j:
                        self.scores[j] += penalty
