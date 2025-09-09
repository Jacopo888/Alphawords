"""
Tests for Scrabble rules implementation.
"""

import pytest
import numpy as np
from alphascrabble.rules.board import Board, Move, Position, PlacedTile, PremiumType
from alphascrabble.rules.tiles_en import Tile, TileBag, TILE_SCORES, TILE_COUNTS
from alphascrabble.rules.bag import ScrabbleBag


class TestTile:
    """Test tile functionality."""
    
    def test_tile_creation(self):
        """Test tile creation and properties."""
        tile = Tile('A')
        assert tile.letter == 'A'
        assert tile.score == 1
        assert not tile.is_blank
        assert tile.display_letter == 'A'
    
    def test_blank_tile(self):
        """Test blank tile functionality."""
        blank = Tile('?')
        assert blank.is_blank
        assert blank.score == 0
        assert blank.display_letter == '?'
        
        # Set blank letter
        blank.set_blank_letter('Z')
        assert blank.blank_letter == 'Z'
        assert blank.display_letter == 'z'  # Lowercase for blanks
    
    def test_tile_scores(self):
        """Test tile scoring."""
        assert TILE_SCORES['A'] == 1
        assert TILE_SCORES['Q'] == 10
        assert TILE_SCORES['Z'] == 10
        assert TILE_SCORES['?'] == 0
    
    def test_tile_counts(self):
        """Test tile distribution."""
        assert sum(TILE_COUNTS.values()) == 100
        assert TILE_COUNTS['A'] == 9
        assert TILE_COUNTS['E'] == 12
        assert TILE_COUNTS['?'] == 2


class TestTileBag:
    """Test tile bag functionality."""
    
    def test_bag_initialization(self):
        """Test bag initialization."""
        bag = TileBag()
        assert bag.tiles_remaining() == 100
        assert not bag.is_empty()
    
    def test_draw_tiles(self):
        """Test drawing tiles."""
        bag = TileBag()
        drawn = bag.draw_tiles(7)
        assert len(drawn) == 7
        assert bag.tiles_remaining() == 93
    
    def test_return_tiles(self):
        """Test returning tiles."""
        bag = TileBag()
        drawn = bag.draw_tiles(7)
        bag.return_tiles(drawn)
        assert bag.tiles_remaining() == 100
    
    def test_empty_bag(self):
        """Test empty bag behavior."""
        bag = TileBag()
        # Draw all tiles
        drawn = bag.draw_tiles(100)
        assert bag.is_empty()
        assert bag.tiles_remaining() == 0
        
        # Try to draw from empty bag
        more_drawn = bag.draw_tiles(5)
        assert len(more_drawn) == 0


class TestBoard:
    """Test board functionality."""
    
    def test_board_initialization(self):
        """Test board initialization."""
        board = Board()
        assert board.grid is not None
        assert len(board.grid) == 15
        assert len(board.grid[0]) == 15
    
    def test_premium_squares(self):
        """Test premium square detection."""
        board = Board()
        
        # Test center square
        assert board.get_premium_type(7, 7) == PremiumType.CENTER
        
        # Test triple word score
        assert board.get_premium_type(0, 0) == PremiumType.TWS
        assert board.get_premium_type(0, 7) == PremiumType.TWS
        
        # Test double word score
        assert board.get_premium_type(1, 1) == PremiumType.DWS
        
        # Test triple letter score
        assert board.get_premium_type(1, 5) == PremiumType.TLS
        
        # Test double letter score
        assert board.get_premium_type(0, 3) == PremiumType.DLS
        
        # Test regular square
        assert board.get_premium_type(2, 3) == PremiumType.NONE
    
    def test_place_tile(self):
        """Test placing tiles on board."""
        board = Board()
        tile = Tile('A')
        
        # Place tile
        board.place_tile(tile, 7, 7)
        assert not board.is_empty(7, 7)
        assert board.get_tile(7, 7).tile.letter == 'A'
    
    def test_invalid_position(self):
        """Test invalid position handling."""
        board = Board()
        tile = Tile('A')
        
        with pytest.raises(ValueError):
            board.place_tile(tile, -1, 0)
        
        with pytest.raises(ValueError):
            board.place_tile(tile, 0, 15)
    
    def test_occupied_position(self):
        """Test placing tile on occupied position."""
        board = Board()
        tile1 = Tile('A')
        tile2 = Tile('B')
        
        board.place_tile(tile1, 7, 7)
        
        with pytest.raises(ValueError):
            board.place_tile(tile2, 7, 7)
    
    def test_word_formation(self):
        """Test word formation detection."""
        board = Board()
        
        # Place tiles to form "HELLO"
        tiles = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]
        for i, tile in enumerate(tiles):
            board.place_tile(tile, 7, 7 + i)
        
        # Test word detection
        word = board.get_word_at_position(7, 7, 'across')
        assert word == "HELLO"
        
        # Test single letter (should return None)
        word = board.get_word_at_position(7, 7, 'down')
        assert word is None
    
    def test_move_scoring(self):
        """Test move scoring."""
        board = Board()
        
        # Create a simple move
        tiles = [PlacedTile(Tile('H'), Position(7, 7)),
                PlacedTile(Tile('E'), Position(7, 8)),
                PlacedTile(Tile('L'), Position(7, 9)),
                PlacedTile(Tile('L'), Position(7, 10)),
                PlacedTile(Tile('O'), Position(7, 11))]
        
        move = Move(tiles=tiles, direction='across', main_word='HELLO')
        
        # Calculate score
        score = board.calculate_move_score(move)
        assert score > 0  # Should have positive score
    
    def test_bingo_bonus(self):
        """Test bingo bonus scoring."""
        board = Board()
        
        # Create bingo move (7 tiles)
        tiles = [PlacedTile(Tile('H'), Position(7, 7 + i)) for i in range(7)]
        move = Move(tiles=tiles, direction='across', main_word='HELLOXX')
        
        assert move.is_bingo
        assert move.bingo_bonus == 50


class TestGameState:
    """Test game state functionality."""
    
    def test_game_state_initialization(self):
        """Test game state initialization."""
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        
        game_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
        
        assert game_state.current_player == 0
        assert game_state.get_current_rack() == racks[0]
        assert game_state.get_current_score() == 0
        assert not game_state.game_over
    
    def test_player_switching(self):
        """Test player switching."""
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        
        game_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
        
        assert game_state.current_player == 0
        game_state.next_player()
        assert game_state.current_player == 1
        game_state.next_player()
        assert game_state.current_player == 0
    
    def test_score_management(self):
        """Test score management."""
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        
        game_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
        
        game_state.add_score(0, 10)
        assert game_state.scores[0] == 10
        assert game_state.get_current_score() == 10
        
        game_state.add_score(1, 15)
        assert game_state.scores[1] == 15
    
    def test_tile_management(self):
        """Test tile management."""
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        
        game_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
        
        initial_rack_size = len(game_state.get_current_rack())
        
        # Draw more tiles
        drawn = game_state.draw_tiles(3)
        assert len(drawn) == 3
        assert len(game_state.get_current_rack()) == initial_rack_size + 3
        
        # Remove tiles
        tiles_to_remove = game_state.get_current_rack()[:2]
        game_state.remove_tiles_from_rack(tiles_to_remove)
        assert len(game_state.get_current_rack()) == initial_rack_size + 1


class TestScrabbleBag:
    """Test enhanced Scrabble bag."""
    
    def test_bag_initialization(self):
        """Test bag initialization."""
        bag = ScrabbleBag()
        assert bag.tiles_remaining() == 100
        assert len(bag.get_drawn_tiles()) == 0
    
    def test_draw_and_track(self):
        """Test drawing and tracking tiles."""
        bag = ScrabbleBag()
        drawn = bag.draw_tiles(7)
        
        assert len(drawn) == 7
        assert bag.tiles_remaining() == 93
        assert len(bag.get_drawn_tiles()) == 7
    
    def test_return_tiles(self):
        """Test returning tiles."""
        bag = ScrabbleBag()
        drawn = bag.draw_tiles(7)
        bag.return_tiles(drawn)
        
        assert bag.tiles_remaining() == 100
        assert len(bag.get_drawn_tiles()) == 0
    
    def test_frequency_tracking(self):
        """Test frequency tracking."""
        bag = ScrabbleBag()
        drawn = bag.draw_tiles(10)
        
        drawn_freq = bag.get_drawn_frequency()
        remaining_freq = bag.get_tile_frequency()
        
        assert sum(drawn_freq.values()) == 10
        assert sum(remaining_freq.values()) == 90
        assert sum(drawn_freq.values()) + sum(remaining_freq.values()) == 100
