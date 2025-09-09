"""
Tests for move generation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from alphascrabble.engine.movegen import MoveGenerator
from alphascrabble.lexicon.gaddag_loader import GaddagLoader
from alphascrabble.rules.board import Board, Move
from alphascrabble.rules.tiles_en import Tile


class TestMoveGenerator:
    """Test move generation functionality."""
    
    @pytest.fixture
    def mock_gaddag_loader(self):
        """Create mock GADDAG loader."""
        loader = Mock(spec=GaddagLoader)
        loader.is_loaded.return_value = True
        loader.is_word.return_value = True  # Accept all words for testing
        return loader
    
    @pytest.fixture
    def move_generator(self, mock_gaddag_loader):
        """Create move generator with mock loader."""
        return MoveGenerator(mock_gaddag_loader)
    
    @pytest.fixture
    def empty_board(self):
        """Create empty board."""
        return Board()
    
    @pytest.fixture
    def sample_rack(self):
        """Create sample rack."""
        return [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O'), Tile('A'), Tile('T')]
    
    def test_move_generator_initialization(self, mock_gaddag_loader):
        """Test move generator initialization."""
        generator = MoveGenerator(mock_gaddag_loader)
        assert generator.gaddag_loader == mock_gaddag_loader
    
    def test_generate_moves_empty_board(self, move_generator, empty_board, sample_rack):
        """Test move generation on empty board."""
        moves = move_generator.generate_moves(empty_board, sample_rack)
        
        # Should include pass move
        assert len(moves) > 0
        pass_moves = [m for m in moves if not m.tiles]
        assert len(pass_moves) == 1
    
    def test_generate_moves_occupied_board(self, move_generator, sample_rack):
        """Test move generation on occupied board."""
        board = Board()
        
        # Place a tile in the center
        board.place_tile(Tile('H'), 7, 7)
        
        moves = move_generator.generate_moves(board, sample_rack)
        
        # Should still generate moves
        assert len(moves) > 0
    
    def test_find_anchor_squares(self, move_generator):
        """Test anchor square detection."""
        board = Board()
        
        # Place a tile
        board.place_tile(Tile('H'), 7, 7)
        
        anchors = move_generator._find_anchor_squares(board)
        
        # Should find adjacent empty squares
        assert len(anchors) > 0
        
        # Check that anchors are adjacent to occupied squares
        for anchor in anchors:
            adjacent_occupied = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = anchor.row + dr, anchor.col + dc
                if 0 <= nr < 15 and 0 <= nc < 15 and not board.is_empty(nr, nc):
                    adjacent_occupied = True
                    break
            assert adjacent_occupied
    
    def test_can_form_word(self, move_generator, sample_rack):
        """Test word formation capability."""
        # Test valid word
        assert move_generator._can_form_word(sample_rack, "HELLO")
        
        # Test invalid word (not enough letters)
        assert not move_generator._can_form_word(sample_rack, "HELLOXX")
        
        # Test word with blank
        rack_with_blank = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('?'), Tile('A'), Tile('T')]
        assert move_generator._can_form_word(rack_with_blank, "HELLOZ")
    
    def test_create_move_from_word(self, move_generator, empty_board, sample_rack):
        """Test move creation from word."""
        # Test valid move
        move = move_generator._create_move_from_word(
            empty_board, sample_rack, "HELLO", 7, 7, 'across'
        )
        
        if move:  # Move might be None if word can't be formed
            assert move.main_word == "HELLO"
            assert move.direction == 'across'
            assert len(move.tiles) == 5
    
    def test_validate_move(self, move_generator, empty_board, sample_rack):
        """Test move validation."""
        # Create a valid move
        tiles = [move_generator._create_placed_tile(Tile('H'), 7, 7),
                move_generator._create_placed_tile(Tile('E'), 7, 8)]
        
        move = Move(tiles=tiles, direction='across', main_word='HE')
        
        # Should be valid
        assert move_generator._validate_move(empty_board, move)
        
        # Test invalid move (occupied position)
        board = Board()
        board.place_tile(Tile('X'), 7, 7)
        
        assert not move_generator._validate_move(board, move)
    
    def test_create_pass_move(self, move_generator):
        """Test pass move creation."""
        pass_move = move_generator._create_pass_move()
        
        assert not pass_move.tiles
        assert pass_move.direction == ''
        assert pass_move.main_word == ''
        assert pass_move.score == 0


class TestGaddagLoader:
    """Test GADDAG loader functionality."""
    
    def test_gaddag_loader_initialization(self):
        """Test GADDAG loader initialization."""
        loader = GaddagLoader("dummy.dawg", "dummy.gaddag")
        assert loader.dawg_path == "dummy.dawg"
        assert loader.gaddag_path == "dummy.gaddag"
        assert not loader.is_loaded()
    
    def test_load_without_files(self):
        """Test loading without files."""
        loader = GaddagLoader("nonexistent.dawg", "nonexistent.gaddag")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_verify_lexica_without_files(self):
        """Test lexicon verification without files."""
        loader = GaddagLoader("nonexistent.dawg", "nonexistent.gaddag")
        
        valid, message = loader.verify_lexica()
        assert not valid
        assert "not found" in message
    
    def test_get_file_info(self):
        """Test file info retrieval."""
        loader = GaddagLoader("nonexistent.dawg", "nonexistent.gaddag")
        
        info = loader.get_file_info()
        assert 'DAWG' in info
        assert 'GADDAG' in info
        assert not info['DAWG']['exists']
        assert not info['GADDAG']['exists']


class TestMoveGenerationIntegration:
    """Integration tests for move generation."""
    
    def test_move_generation_workflow(self):
        """Test complete move generation workflow."""
        # This would require actual lexicon files
        # For now, we'll test the structure
        
        # Mock the lexicon loader
        mock_loader = Mock(spec=GaddagLoader)
        mock_loader.is_loaded.return_value = True
        mock_loader.is_word.return_value = True
        
        # Create move generator
        generator = MoveGenerator(mock_loader)
        
        # Create board and rack
        board = Board()
        rack = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]
        
        # Generate moves
        moves = generator.generate_moves(board, rack)
        
        # Should return list of moves
        assert isinstance(moves, list)
        assert len(moves) > 0
        
        # Should include pass move
        pass_moves = [m for m in moves if not m.tiles]
        assert len(pass_moves) == 1
    
    def test_move_scoring_integration(self):
        """Test move scoring integration."""
        board = Board()
        rack = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]
        
        # Create a simple move
        from alphascrabble.rules.board import PlacedTile, Position
        tiles = [
            PlacedTile(Tile('H'), Position(7, 7)),
            PlacedTile(Tile('E'), Position(7, 8)),
            PlacedTile(Tile('L'), Position(7, 9)),
            PlacedTile(Tile('L'), Position(7, 10)),
            PlacedTile(Tile('O'), Position(7, 11))
        ]
        
        move = Move(tiles=tiles, direction='across', main_word='HELLO')
        
        # Calculate score
        score = board.calculate_move_score(move)
        
        # Should have positive score
        assert score > 0
        
        # Check bingo bonus
        if len(tiles) == 7:
            assert move.is_bingo
            assert move.bingo_bonus == 50
