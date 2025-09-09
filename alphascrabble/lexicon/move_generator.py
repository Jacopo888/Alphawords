"""
Move generation for Scrabble using GADDAG lexicon.

This module provides efficient move generation by combining the GADDAG data structure
with Scrabble board analysis to find all valid moves.
"""

from typing import List, Dict, Set, Tuple, Optional, Iterator
from dataclasses import dataclass
from enum import Enum
import itertools

from .gaddag import GADDAG
from ..rules.board import Board, Tile


class Direction(Enum):
    """Direction for word placement."""
    ACROSS = "across"
    DOWN = "down"


@dataclass
class Position:
    """Position on the board."""
    row: int
    col: int
    
    def __add__(self, other):
        if isinstance(other, tuple):
            return Position(self.row + other[0], self.col + other[1])
        return Position(self.row + other.row, self.col + other.col)
    
    def __sub__(self, other):
        if isinstance(other, tuple):
            return Position(self.row - other[0], self.col - other[1])
        return Position(self.row - other.row, self.col - other.col)
    
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __hash__(self):
        return hash((self.row, self.col))


@dataclass
class MoveTile:
    """A tile placement in a move."""
    tile: Tile
    position: Position
    is_new: bool = True  # Whether this tile is newly placed


@dataclass
class Move:
    """A complete move in Scrabble."""
    tiles: List[MoveTile]
    direction: Direction
    main_word: str
    all_words: List[str]
    total_score: int
    is_bingo: bool = False
    
    def __post_init__(self):
        """Calculate bingo bonus."""
        if len(self.tiles) == 7:  # Used all 7 tiles
            self.is_bingo = True
            self.total_score += 50


class MoveGenerator:
    """Generates valid Scrabble moves using GADDAG."""
    
    def __init__(self, gaddag: GADDAG):
        self.gaddag = gaddag
        self.board_size = 15
    
    def generate_moves(self, board: Board, rack: List[Tile]) -> List[Move]:
        """Generate all valid moves for the given board and rack."""
        moves = []
        
        # If board is empty, place first word
        if self._is_empty_board(board):
            moves.extend(self._generate_first_moves(rack))
        else:
            # Generate moves that connect to existing tiles
            moves.extend(self._generate_connected_moves(board, rack))
        
        # Sort moves by score (highest first)
        moves.sort(key=lambda m: m.total_score, reverse=True)
        
        return moves
    
    def _is_empty_board(self, board: Board) -> bool:
        """Check if the board is empty."""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if not board.is_empty(i, j):
                    return False
        return True
    
    def _generate_first_moves(self, rack: List[Tile]) -> List[Move]:
        """Generate moves for the first turn (must go through center)."""
        moves = []
        center = Position(7, 7)
        
        # Get all possible words from rack
        letters = ''.join(tile.letter for tile in rack)
        words = self.gaddag.get_words_with_letters(letters, min_length=2)
        
        for word in words:
            # Try placing word across (horizontal)
            if len(word) <= 7:  # Can't exceed rack size
                move = self._create_first_move(word, center, Direction.ACROSS, rack)
                if move:
                    moves.append(move)
            
            # Try placing word down (vertical)
            if len(word) <= 7:
                move = self._create_first_move(word, center, Direction.DOWN, rack)
                if move:
                    moves.append(move)
        
        return moves
    
    def _create_first_move(self, word: str, start_pos: Position, 
                          direction: Direction, rack: List[Tile]) -> Optional[Move]:
        """Create a move for the first turn."""
        # Check if we can form this word with our rack
        if not self._can_form_word(word, rack):
            return None
        
        # Create move tiles
        move_tiles = []
        current_pos = start_pos
        
        for i, letter in enumerate(word):
            # Find tile in rack
            tile = self._find_tile_in_rack(letter, rack, move_tiles)
            if not tile:
                return None
            
            move_tiles.append(MoveTile(tile, current_pos, True))
            
            # Move to next position
            if direction == Direction.ACROSS:
                current_pos = Position(current_pos.row, current_pos.col + 1)
            else:
                current_pos = Position(current_pos.row + 1, current_pos.col)
        
        # Calculate score
        score = self._calculate_move_score(move_tiles, word, direction)
        
        return Move(
            tiles=move_tiles,
            direction=direction,
            main_word=word,
            all_words=[word],
            total_score=score
        )
    
    def _generate_connected_moves(self, board: Board, rack: List[Tile]) -> List[Move]:
        """Generate moves that connect to existing tiles."""
        moves = []
        
        # Find all positions adjacent to existing tiles
        adjacent_positions = self._find_adjacent_positions(board)
        
        for pos in adjacent_positions:
            # Try placing words in both directions
            moves.extend(self._generate_moves_at_position(board, rack, pos, Direction.ACROSS))
            moves.extend(self._generate_moves_at_position(board, rack, pos, Direction.DOWN))
        
        return moves
    
    def _find_adjacent_positions(self, board: Board) -> Set[Position]:
        """Find all positions adjacent to existing tiles."""
        adjacent = set()
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if not board.is_empty(i, j):
                    # Check all 4 directions
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.board_size and 
                            0 <= nj < self.board_size and 
                            board.is_empty(ni, nj)):
                            adjacent.add(Position(ni, nj))
        
        return adjacent
    
    def _generate_moves_at_position(self, board: Board, rack: List[Tile], 
                                   start_pos: Position, direction: Direction) -> List[Move]:
        """Generate moves starting at a specific position."""
        moves = []
        
        # Get all possible words from rack
        letters = ''.join(tile.letter for tile in rack)
        words = self.gaddag.get_words_with_letters(letters, min_length=2)
        
        for word in words:
            # Try placing word at this position
            move = self._create_connected_move(board, word, start_pos, direction, rack)
            if move:
                moves.append(move)
        
        return moves
    
    def _create_connected_move(self, board: Board, word: str, start_pos: Position,
                              direction: Direction, rack: List[Tile]) -> Optional[Move]:
        """Create a move that connects to existing tiles."""
        # Check if word fits on board
        if not self._word_fits_on_board(word, start_pos, direction):
            return None
        
        # Check if word connects to existing tiles
        if not self._word_connects_to_board(board, word, start_pos, direction):
            return None
        
        # Create move tiles
        move_tiles = []
        current_pos = start_pos
        
        for i, letter in enumerate(word):
            # Check if position is empty
            if board.is_empty(current_pos.row, current_pos.col):
                # Find tile in rack
                tile = self._find_tile_in_rack(letter, rack, move_tiles)
                if not tile:
                    return None
                
                move_tiles.append(MoveTile(tile, current_pos, True))
            else:
                # Use existing tile
                existing_tile = board.get_tile(current_pos.row, current_pos.col)
                if existing_tile.letter != letter:
                    return None  # Letter doesn't match
                
                move_tiles.append(MoveTile(existing_tile, current_pos, False))
            
            # Move to next position
            if direction == Direction.ACROSS:
                current_pos = Position(current_pos.row, current_pos.col + 1)
            else:
                current_pos = Position(current_pos.row + 1, current_pos.col)
        
        # Calculate score
        score = self._calculate_move_score(move_tiles, word, direction)
        
        return Move(
            tiles=move_tiles,
            direction=direction,
            main_word=word,
            all_words=[word],  # Simplified for now
            total_score=score
        )
    
    def _can_form_word(self, word: str, rack: List[Tile]) -> bool:
        """Check if we can form a word with the given rack."""
        rack_letters = [tile.letter for tile in rack]
        
        for letter in word:
            if letter in rack_letters:
                rack_letters.remove(letter)
            else:
                return False
        
        return True
    
    def _find_tile_in_rack(self, letter: str, rack: List[Tile], 
                          used_tiles: List[MoveTile]) -> Optional[Tile]:
        """Find a tile in the rack that hasn't been used yet."""
        used_letters = [mt.tile.letter for mt in used_tiles]
        
        for tile in rack:
            if tile.letter == letter and tile.letter not in used_letters:
                return tile
        
        return None
    
    def _word_fits_on_board(self, word: str, start_pos: Position, 
                           direction: Direction) -> bool:
        """Check if a word fits on the board starting at the given position."""
        if direction == Direction.ACROSS:
            return start_pos.col + len(word) <= self.board_size
        else:
            return start_pos.row + len(word) <= self.board_size
    
    def _word_connects_to_board(self, board: Board, word: str, start_pos: Position,
                               direction: Direction) -> bool:
        """Check if a word connects to existing tiles on the board."""
        # For now, just check if at least one position is adjacent to existing tiles
        current_pos = start_pos
        
        for i, letter in enumerate(word):
            if board.is_empty(current_pos.row, current_pos.col):
                # Check if this position is adjacent to existing tiles
                if self._is_adjacent_to_existing_tiles(board, current_pos):
                    return True
            
            # Move to next position
            if direction == Direction.ACROSS:
                current_pos = Position(current_pos.row, current_pos.col + 1)
            else:
                current_pos = Position(current_pos.row + 1, current_pos.col)
        
        return False
    
    def _is_adjacent_to_existing_tiles(self, board: Board, pos: Position) -> bool:
        """Check if a position is adjacent to existing tiles."""
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = pos.row + di, pos.col + dj
            if (0 <= ni < self.board_size and 
                0 <= nj < self.board_size and 
                not board.is_empty(ni, nj)):
                return True
        
        return False
    
    def _calculate_move_score(self, move_tiles: List[MoveTile], 
                            main_word: str, direction: Direction) -> int:
        """Calculate the score for a move."""
        # Simplified scoring - just sum tile values
        score = 0
        
        for move_tile in move_tiles:
            if move_tile.is_new:  # Only count new tiles
                score += move_tile.tile.score
        
        return score


# Example usage and testing
if __name__ == "__main__":
    from ..rules.board import Board, Tile
    
    # Create a simple GADDAG
    from .gaddag import GADDAG
    gaddag = GADDAG()
    gaddag.add_word('CAT')
    gaddag.add_word('DOG')
    gaddag.add_word('HOUSE')
    
    # Create board and rack
    board = Board()
    rack = [Tile('C'), Tile('A'), Tile('T'), Tile('D'), Tile('O'), Tile('G')]
    
    # Create move generator
    generator = MoveGenerator(gaddag)
    
    # Generate moves
    moves = generator.generate_moves(board, rack)
    
    print(f"Generated {len(moves)} moves:")
    for move in moves:
        print(f"  {move.main_word} ({move.direction.value}) - {move.total_score} points")
