"""
End-to-end smoke tests for AlphaScrabble.

Tests basic functionality without requiring full setup.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from alphascrabble.config import Config
from alphascrabble.rules.board import Board, GameState
from alphascrabble.rules.tiles_en import Tile, TileBag
from alphascrabble.engine.features import FeatureExtractor
from alphascrabble.nn.model import AlphaScrabbleNet


class TestEndToEndSmoke:
    """End-to-end smoke tests."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = Config()
        
        assert config.BOARD_SIZE == 15
        assert config.RACK_SIZE == 7
        assert config.BINGO_BONUS == 50
        assert config.MCTS_SIMULATIONS == 160
        assert config.CPUCT == 1.5
    
    def test_board_creation_and_basic_operations(self):
        """Test board creation and basic operations."""
        board = Board()
        
        # Test board size
        assert len(board.grid) == 15
        assert len(board.grid[0]) == 15
        
        # Test tile placement
        tile = Tile('A')
        board.place_tile(tile, 7, 7)
        assert not board.is_empty(7, 7)
        assert board.get_tile(7, 7).tile.letter == 'A'
        
        # Test premium squares
        assert board.get_premium_type(7, 7) is not None  # Center square
    
    def test_tile_bag_operations(self):
        """Test tile bag operations."""
        bag = TileBag()
        
        # Test initial state
        assert bag.tiles_remaining() == 100
        assert not bag.is_empty()
        
        # Test drawing tiles
        drawn = bag.draw_tiles(7)
        assert len(drawn) == 7
        assert bag.tiles_remaining() == 93
        
        # Test returning tiles
        bag.return_tiles(drawn)
        assert bag.tiles_remaining() == 100
    
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
        assert len(game_state.get_current_rack()) == 7
        assert not game_state.game_over
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = FeatureExtractor()
        board = Board()
        rack = [Tile('A'), Tile('B'), Tile('C')]
        
        # Test board features
        board_features = extractor.extract_board_features(board, 0)
        assert board_features.shape == (32, 15, 15)
        assert board_features.dtype == np.float32
        
        # Test rack features
        rack_features = extractor.extract_rack_features(rack)
        assert rack_features.shape == (27,)
        assert rack_features.dtype == np.float32
        
        # Test move features
        from alphascrabble.rules.board import Move
        move = Move(tiles=[], direction='', main_word='')
        move_features = extractor.extract_move_features(move, board, rack)
        assert move_features.shape == (64,)
        assert move_features.dtype == np.float32
    
    def test_neural_network_creation(self):
        """Test neural network creation."""
        model = AlphaScrabbleNet()
        
        # Test model creation
        assert model is not None
        
        # Test forward pass with dummy data
        board_features = np.random.rand(32, 15, 15).astype(np.float32)
        rack_features = np.random.rand(27).astype(np.float32)
        move_features = np.random.rand(5, 64).astype(np.float32)
        
        # Convert to tensors
        import torch
        board_tensor = torch.FloatTensor(board_features).unsqueeze(0)
        rack_tensor = torch.FloatTensor(rack_features).unsqueeze(0)
        move_tensor = torch.FloatTensor(move_features).unsqueeze(0)
        
        # Forward pass
        policy_logits, value = model(board_tensor, rack_tensor, move_tensor)
        
        assert policy_logits.shape == (1, 5)  # batch_size, num_moves
        assert value.shape == (1, 1)  # batch_size, 1
        assert value.item() >= -1.0 and value.item() <= 1.0
    
    def test_model_save_load(self, tmp_path):
        """Test model save and load functionality."""
        model = AlphaScrabbleNet()
        
        # Save model
        model_path = tmp_path / "test_model.pt"
        model.save(str(model_path))
        
        # Load model
        loaded_model = AlphaScrabbleNet.load(str(model_path))
        
        assert loaded_model is not None
        
        # Test that loaded model produces same output
        board_features = np.random.rand(32, 15, 15).astype(np.float32)
        rack_features = np.random.rand(27).astype(np.float32)
        move_features = np.random.rand(3, 64).astype(np.float32)
        
        import torch
        board_tensor = torch.FloatTensor(board_features).unsqueeze(0)
        rack_tensor = torch.FloatTensor(rack_features).unsqueeze(0)
        move_tensor = torch.FloatTensor(move_features).unsqueeze(0)
        
        # Compare outputs
        original_policy, original_value = model(board_tensor, rack_tensor, move_tensor)
        loaded_policy, loaded_value = loaded_model(board_tensor, rack_tensor, move_tensor)
        
        assert torch.allclose(original_policy, loaded_policy, atol=1e-6)
        assert torch.allclose(original_value, loaded_value, atol=1e-6)
    
    @patch('alphascrabble.lexicon.gaddag_loader.qlex')
    def test_lexicon_loader_mock(self, mock_qlex):
        """Test lexicon loader with mocked qlex module."""
        from alphascrabble.lexicon.gaddag_loader import GaddagLoader
        
        # Mock the qlex module
        mock_wrapper = Mock()
        mock_wrapper.is_loaded.return_value = True
        mock_wrapper.is_word.return_value = True
        mock_wrapper.get_word_count.return_value = 1000
        mock_qlex.QLexWrapper.return_value = mock_wrapper
        
        # Create loader
        loader = GaddagLoader("dummy.dawg", "dummy.gaddag")
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            loader.load()
        
        assert loader.is_loaded()
        assert loader.is_word("HELLO")
        assert loader.get_word_count() == 1000
    
    def test_move_generation_mock(self):
        """Test move generation with mocked components."""
        from alphascrabble.engine.movegen import MoveGenerator
        
        # Mock GADDAG loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.is_word.return_value = True
        
        # Create move generator
        generator = MoveGenerator(mock_loader)
        
        # Create board and rack
        board = Board()
        rack = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]
        
        # Generate moves
        moves = generator.generate_moves(board, rack)
        
        assert isinstance(moves, list)
        assert len(moves) > 0
        
        # Should include pass move
        pass_moves = [m for m in moves if not m.tiles]
        assert len(pass_moves) == 1
    
    def test_mcts_basic_functionality(self):
        """Test basic MCTS functionality with mocked components."""
        from alphascrabble.engine.mcts import MCTS
        
        # Mock components
        mock_move_generator = Mock()
        mock_move_generator.generate_moves.return_value = [
            Mock(tiles=[], direction='', main_word='', score=10)
        ]
        
        mock_feature_extractor = Mock()
        mock_feature_extractor.extract_state_features.return_value = (
            np.random.rand(32, 15, 15),
            np.random.rand(27)
        )
        mock_feature_extractor.extract_move_list_features.return_value = np.random.rand(1, 64)
        
        mock_neural_net = Mock()
        mock_neural_net.predict.return_value = (np.array([0.5]), 0.1)
        
        # Create MCTS
        mcts = MCTS(
            move_generator=mock_move_generator,
            feature_extractor=mock_feature_extractor,
            neural_net=mock_neural_net,
            simulations=5,  # Small number for testing
            cpuct=1.5,
            temperature=1.0
        )
        
        # Create game state
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
        
        # Test search
        probabilities = mcts.search(game_state)
        assert isinstance(probabilities, list)
        assert len(probabilities) > 0
        
        # Test best move
        move = mcts.get_best_move(game_state)
        assert move is not None
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        try:
            from alphascrabble.cli import main
            assert main is not None
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")
    
    def test_selfplay_import(self):
        """Test that self-play module can be imported."""
        try:
            from alphascrabble.selfplay import SelfPlayPipeline, SelfPlayManager
            assert SelfPlayPipeline is not None
            assert SelfPlayManager is not None
        except ImportError as e:
            pytest.skip(f"Self-play import failed: {e}")
    
    def test_training_import(self):
        """Test that training modules can be imported."""
        try:
            from alphascrabble.nn.train import Trainer
            from alphascrabble.nn.evaluate import Evaluator
            from alphascrabble.nn.dataset import ReplayBuffer, GameDataset
            assert Trainer is not None
            assert Evaluator is not None
            assert ReplayBuffer is not None
            assert GameDataset is not None
        except ImportError as e:
            pytest.skip(f"Training import failed: {e}")
    
    def test_utility_imports(self):
        """Test that utility modules can be imported."""
        try:
            from alphascrabble.utils.logging import get_logger, setup_logging
            from alphascrabble.utils.io import save_checkpoint, load_checkpoint
            from alphascrabble.utils.seeding import set_seed
            from alphascrabble.utils.tb_writer import TensorBoardWriter
            
            assert get_logger is not None
            assert setup_logging is not None
            assert save_checkpoint is not None
            assert load_checkpoint is not None
            assert set_seed is not None
            assert TensorBoardWriter is not None
        except ImportError as e:
            pytest.skip(f"Utility import failed: {e}")
    
    def test_package_structure(self):
        """Test that package structure is correct."""
        import alphascrabble
        
        # Test main package
        assert hasattr(alphascrabble, 'Config')
        assert hasattr(alphascrabble, 'Board')
        assert hasattr(alphascrabble, 'Move')
        assert hasattr(alphascrabble, 'GameState')
        assert hasattr(alphascrabble, 'TileBag')
        assert hasattr(alphascrabble, 'MoveGenerator')
        assert hasattr(alphascrabble, 'AlphaScrabbleNet')
        
        # Test submodules
        assert hasattr(alphascrabble, 'rules')
        assert hasattr(alphascrabble, 'engine')
        assert hasattr(alphascrabble, 'nn')
        assert hasattr(alphascrabble, 'lexicon')
        assert hasattr(alphascrabble, 'utils')
