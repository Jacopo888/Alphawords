"""
Move generation using GADDAG/DAWG lexicons.

Generates all legal Scrabble moves for a given board state and rack.
"""

from typing import List, Set, Tuple, Optional
import itertools
from ..rules.board import Board, Move, Position, PlacedTile, PremiumType
from ..rules.tiles_en import Tile
from ..lexicon.gaddag_loader import GaddagLoader


class MoveGenerator:
    """Generates legal Scrabble moves using GADDAG/DAWG."""
    
    def __init__(self, gaddag_loader: GaddagLoader):
        """Initialize with GADDAG loader."""
        self.gaddag_loader = gaddag_loader
        self._anchor_cache = {}
    
    def generate_moves(self, board: Board, rack: List[Tile]) -> List[Move]:
        """Generate all legal moves for given board and rack."""
        if not self.gaddag_loader.is_loaded():
            raise RuntimeError("GADDAG loader not initialized")
        
        moves = []
        
        # If board is empty, first move must use center square
        if not board.is_center_occupied():
            moves.extend(self._generate_first_moves(board, rack))
        else:
            # Find anchor squares and generate moves from them
            anchors = self._find_anchor_squares(board)
            for anchor in anchors:
                moves.extend(self._generate_moves_from_anchor(board, rack, anchor))
        
        # Add pass move
        moves.append(self._create_pass_move())
        
        return moves
    
    def _generate_first_moves(self, board: Board, rack: List[Tile]) -> List[Move]:
        """Generate first moves (must use center square)."""
        moves = []
        center_row, center_col = 7, 7
        
        # Try all possible words that can fit on the board
        for word_length in range(2, min(len(rack) + 1, 8)):  # 2-7 letters
            for word in self._get_words_of_length(word_length):
                if self._can_form_word(rack, word):
                    # Try placing horizontally
                    if center_col + word_length <= 15:
                        move = self._create_move_from_word(
                            board, rack, word, center_row, center_col, 'across'
                        )
                        if move:
                            moves.append(move)
                    
                    # Try placing vertically
                    if center_row + word_length <= 15:
                        move = self._create_move_from_word(
                            board, rack, word, center_row, center_col, 'down'
                        )
                        if move:
                            moves.append(move)
        
        return moves
    
    def _find_anchor_squares(self, board: Board) -> List[Position]:
        """Find anchor squares (empty squares adjacent to occupied squares)."""
        anchors = set()
        
        for row in range(15):
            for col in range(15):
                if board.is_empty(row, col):
                    # Check if adjacent to occupied square
                    adjacent_occupied = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < 15 and 0 <= nc < 15 and not board.is_empty(nr, nc):
                            adjacent_occupied = True
                            break
                    
                    if adjacent_occupied:
                        anchors.add(Position(row, col))
        
        return list(anchors)
    
    def _generate_moves_from_anchor(self, board: Board, rack: List[Tile], 
                                   anchor: Position) -> List[Move]:
        """Generate moves from a specific anchor square."""
        moves = []
        
        # Try both directions
        for direction in ['across', 'down']:
            moves.extend(self._generate_moves_in_direction(
                board, rack, anchor, direction
            ))
        
        return moves
    
    def _generate_moves_in_direction(self, board: Board, rack: List[Tile],
                                   anchor: Position, direction: str) -> List[Move]:
        """Generate moves in a specific direction from anchor."""
        moves = []
        
        # Find the range of positions where we can place tiles
        if direction == 'across':
            start_col = self._find_word_start(board, anchor.row, anchor.col, 'across')
            end_col = self._find_word_end(board, anchor.row, anchor.col, 'across')
        else:
            start_row = self._find_word_start(board, anchor.row, anchor.col, 'down')
            end_row = self._find_word_end(board, anchor.row, anchor.col, 'down')
        
        # Generate all possible tile combinations
        for length in range(2, min(len(rack) + 1, 8)):
            for tile_combination in itertools.combinations(rack, length):
                for tile_permutation in itertools.permutations(tile_combination):
                    if direction == 'across':
                        move = self._try_place_tiles_across(
                            board, list(tile_permutation), anchor.row, 
                            start_col, end_col
                        )
                    else:
                        move = self._try_place_tiles_down(
                            board, list(tile_permutation), anchor.col,
                            start_row, end_row
                        )
                    
                    if move and self._validate_move(board, move):
                        moves.append(move)
        
        return moves
    
    def _find_word_start(self, board: Board, row: int, col: int, 
                        direction: str) -> int:
        """Find the start of a word containing the given position."""
        if direction == 'across':
            start = col
            while start > 0 and not board.is_empty(row, start - 1):
                start -= 1
            return start
        else:
            start = row
            while start > 0 and not board.is_empty(start - 1, col):
                start -= 1
            return start
    
    def _find_word_end(self, board: Board, row: int, col: int, 
                      direction: str) -> int:
        """Find the end of a word containing the given position."""
        if direction == 'across':
            end = col
            while end < 14 and not board.is_empty(row, end + 1):
                end += 1
            return end
        else:
            end = row
            while end < 14 and not board.is_empty(end + 1, col):
                end += 1
            return end
    
    def _try_place_tiles_across(self, board: Board, tiles: List[Tile], 
                               row: int, start_col: int, end_col: int) -> Optional[Move]:
        """Try to place tiles horizontally."""
        # This is a simplified implementation
        # In practice, you'd use GADDAG to find valid words
        if len(tiles) > end_col - start_col + 1:
            return None
        
        # Create a simple word from tiles
        word = ''.join(tile.letter for tile in tiles)
        if not self.gaddag_loader.is_word(word):
            return None
        
        # Create move
        placed_tiles = []
        for i, tile in enumerate(tiles):
            col = start_col + i
            if board.is_empty(row, col):
                placed_tiles.append(PlacedTile(tile, Position(row, col)))
        
        if not placed_tiles:
            return None
        
        move = Move(
            tiles=placed_tiles,
            direction='across',
            main_word=word
        )
        
        return move
    
    def _try_place_tiles_down(self, board: Board, tiles: List[Tile],
                             col: int, start_row: int, end_row: int) -> Optional[Move]:
        """Try to place tiles vertically."""
        if len(tiles) > end_row - start_row + 1:
            return None
        
        word = ''.join(tile.letter for tile in tiles)
        if not self.gaddag_loader.is_word(word):
            return None
        
        placed_tiles = []
        for i, tile in enumerate(tiles):
            row = start_row + i
            if board.is_empty(row, col):
                placed_tiles.append(PlacedTile(tile, Position(row, col)))
        
        if not placed_tiles:
            return None
        
        move = Move(
            tiles=placed_tiles,
            direction='down',
            main_word=word
        )
        
        return move
    
    def _get_words_of_length(self, length: int) -> List[str]:
        """Get all words of a specific length from lexicon."""
        # This is a placeholder - in practice you'd query the GADDAG
        # For now, return some common words
        common_words = {
            2: ['AT', 'BE', 'GO', 'IT', 'NO', 'ON', 'TO', 'UP'],
            3: ['CAT', 'DOG', 'RUN', 'SUN', 'BIG', 'RED', 'BLUE'],
            4: ['WORD', 'GAME', 'PLAY', 'MOVE', 'TILE', 'SCORE'],
            5: ['HELLO', 'WORLD', 'GAMES', 'BOARD', 'TILES'],
            6: ['SCRABBLE', 'PLAYER', 'SCORES', 'BOARDS'],
            7: ['PLAYING', 'SCORING', 'WINNING']
        }
        return common_words.get(length, [])
    
    def _can_form_word(self, rack: List[Tile], word: str) -> bool:
        """Check if rack can form the given word."""
        rack_letters = [tile.letter for tile in rack]
        word_letters = list(word)
        
        for letter in word_letters:
            if letter in rack_letters:
                rack_letters.remove(letter)
            elif '?' in rack_letters:  # Blank tile
                rack_letters.remove('?')
            else:
                return False
        
        return True
    
    def _create_move_from_word(self, board: Board, rack: List[Tile], 
                              word: str, row: int, col: int, 
                              direction: str) -> Optional[Move]:
        """Create a move from a word at a specific position."""
        if not self._can_form_word(rack, word):
            return None
        
        # Create placed tiles
        placed_tiles = []
        used_tiles = []
        
        for i, letter in enumerate(word):
            if direction == 'across':
                tile_row, tile_col = row, col + i
            else:
                tile_row, tile_col = row + i, col
            
            if board.is_empty(tile_row, tile_col):
                # Find tile in rack
                tile = None
                for rack_tile in rack:
                    if rack_tile.letter == letter and rack_tile not in used_tiles:
                        tile = rack_tile
                        used_tiles.append(tile)
                        break
                
                if not tile:
                    # Try blank tile
                    for rack_tile in rack:
                        if rack_tile.letter == '?' and rack_tile not in used_tiles:
                            tile = rack_tile
                            tile.set_blank_letter(letter)
                            used_tiles.append(tile)
                            break
                
                if tile:
                    placed_tiles.append(PlacedTile(tile, Position(tile_row, tile_col)))
        
        if not placed_tiles:
            return None
        
        move = Move(
            tiles=placed_tiles,
            direction=direction,
            main_word=word
        )
        
        return move
    
    def _validate_move(self, board: Board, move: Move) -> bool:
        """Validate that a move is legal."""
        # Check that all tiles are placed on empty squares
        for placed_tile in move.tiles:
            if not board.is_empty(placed_tile.position.row, placed_tile.position.col):
                return False
        
        # Check that the main word is valid
        if not self.gaddag_loader.is_word(move.main_word):
            return False
        
        # Check cross words
        for cross_word in move.cross_words:
            if not self.gaddag_loader.is_word(cross_word):
                return False
        
        return True
    
    def _create_pass_move(self) -> Move:
        """Create a pass move."""
        return Move(
            tiles=[],
            direction='',
            main_word='',
            score=0
        )
