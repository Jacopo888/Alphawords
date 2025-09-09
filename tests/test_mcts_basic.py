"""
Basic tests for MCTS functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from alphascrabble.engine.mcts import MCTS, MCTSNode
from alphascrabble.engine.movegen import MoveGenerator
from alphascrabble.engine.features import FeatureExtractor
from alphascrabble.rules.board import Board, GameState, Move
from alphascrabble.rules.tiles_en import Tile, TileBag


class TestMCTSNode:
    """Test MCTS node functionality."""
    
    def test_node_initialization(self):
        """Test node initialization."""
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
        
        node = MCTSNode(game_state)
        
        assert node.state == game_state
        assert node.parent is None
        assert node.move is None
        assert node.children == []
        assert node.visits == 0
        assert node.total_value == 0.0
        assert node.prior_prob == 0.0
        assert not node.is_expanded
    
    def test_node_properties(self):
        """Test node properties."""
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
        
        node = MCTSNode(game_state)
        
        # Test initial value
        assert node.value == 0.0
        
        # Test leaf property
        assert node.is_leaf
        
        # Update node
        node.update(1.0)
        node.update(0.5)
        
        assert node.visits == 2
        assert node.total_value == 1.5
        assert node.value == 0.75
    
    def test_node_update(self):
        """Test node update functionality."""
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
        
        node = MCTSNode(game_state)
        
        # Update with positive value
        node.update(1.0)
        assert node.visits == 1
        assert node.total_value == 1.0
        assert node.value == 1.0
        
        # Update with negative value
        node.update(-0.5)
        assert node.visits == 2
        assert node.total_value == 0.5
        assert node.value == 0.25
    
    def test_ucb_score(self):
        """Test UCB score calculation."""
        # Create parent node
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        
        parent_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
        
        parent = MCTSNode(parent_state)
        parent.visits = 10
        
        # Create child node
        child = MCTSNode(parent_state, parent=parent)
        child.visits = 2
        child.total_value = 1.0
        child.prior_prob = 0.5
        
        # Test UCB score
        ucb_score = child.get_ucb_score(1.5)
        assert ucb_score > 0  # Should be positive
        
        # Test unvisited node
        unvisited = MCTSNode(parent_state, parent=parent)
        unvisited.visits = 0
        unvisited.prior_prob = 0.3
        
        ucb_score_unvisited = unvisited.get_ucb_score(1.5)
        assert ucb_score_unvisited == float('inf')


class TestMCTS:
    """Test MCTS functionality."""
    
    @pytest.fixture
    def mock_move_generator(self):
        """Create mock move generator."""
        generator = Mock(spec=MoveGenerator)
        generator.generate_moves.return_value = [
            Move(tiles=[], direction='', main_word='', score=10),
            Move(tiles=[], direction='', main_word='', score=5)
        ]
        return generator
    
    @pytest.fixture
    def mock_feature_extractor(self):
        """Create mock feature extractor."""
        extractor = Mock(spec=FeatureExtractor)
        extractor.extract_state_features.return_value = (
            np.random.rand(32, 15, 15),
            np.random.rand(27)
        )
        extractor.extract_move_list_features.return_value = np.random.rand(2, 64)
        return extractor
    
    @pytest.fixture
    def mock_neural_net(self):
        """Create mock neural network."""
        net = Mock()
        net.predict.return_value = (
            np.array([0.6, 0.4]),  # Policy logits
            0.2  # Value
        )
        return net
    
    @pytest.fixture
    def mcts(self, mock_move_generator, mock_feature_extractor, mock_neural_net):
        """Create MCTS instance."""
        return MCTS(
            move_generator=mock_move_generator,
            feature_extractor=mock_feature_extractor,
            neural_net=mock_neural_net,
            simulations=10,
            cpuct=1.5,
            temperature=1.0
        )
    
    @pytest.fixture
    def sample_game_state(self):
        """Create sample game state."""
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        
        return GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
    
    def test_mcts_initialization(self, mcts):
        """Test MCTS initialization."""
        assert mcts.simulations == 10
        assert mcts.cpuct == 1.5
        assert mcts.temperature == 1.0
    
    def test_search_basic(self, mcts, sample_game_state):
        """Test basic MCTS search."""
        probabilities = mcts.search(sample_game_state)
        
        assert isinstance(probabilities, list)
        assert len(probabilities) > 0
        assert all(p >= 0 for p in probabilities)
        assert abs(sum(probabilities) - 1.0) < 1e-6  # Should sum to 1
    
    def test_get_best_move(self, mcts, sample_game_state):
        """Test getting best move."""
        move = mcts.get_best_move(sample_game_state)
        
        assert isinstance(move, Move)
    
    def test_simulation_workflow(self, mcts, sample_game_state):
        """Test single simulation workflow."""
        # Create root node
        root = MCTSNode(sample_game_state)
        
        # Run simulation
        value = mcts._simulate(root)
        
        assert isinstance(value, (int, float))
        assert -1.0 <= value <= 1.0
    
    def test_node_expansion(self, mcts, sample_game_state):
        """Test node expansion."""
        node = MCTSNode(sample_game_state)
        
        # Expand node
        mcts._expand_node(node)
        
        assert node.is_expanded
        assert len(node.children) > 0
    
    def test_move_probabilities(self, mcts, sample_game_state):
        """Test move probability calculation."""
        # Create root with children
        root = MCTSNode(sample_game_state)
        mcts._expand_node(root)
        
        # Update children with different visit counts
        if len(root.children) >= 2:
            root.children[0].visits = 10
            root.children[1].visits = 5
        
        probabilities = mcts._get_move_probabilities(root)
        
        assert len(probabilities) == len(root.children)
        assert all(p >= 0 for p in probabilities)
        assert abs(sum(probabilities) - 1.0) < 1e-6
    
    def test_temperature_scaling(self, sample_game_state):
        """Test temperature scaling in move selection."""
        # Create MCTS with different temperatures
        mcts_greedy = MCTS(
            move_generator=Mock(),
            feature_extractor=Mock(),
            neural_net=Mock(),
            simulations=10,
            temperature=0.0  # Greedy
        )
        
        mcts_exploratory = MCTS(
            move_generator=Mock(),
            feature_extractor=Mock(),
            neural_net=Mock(),
            simulations=10,
            temperature=2.0  # More exploratory
        )
        
        # Both should return valid probabilities
        probs_greedy = mcts_greedy._get_move_probabilities(MCTSNode(sample_game_state))
        probs_exploratory = mcts_exploratory._get_move_probabilities(MCTSNode(sample_game_state))
        
        # Greedy should be more deterministic (one move with probability 1)
        if len(probs_greedy) > 1:
            assert max(probs_greedy) >= max(probs_exploratory)
    
    def test_terminal_state_handling(self, mcts):
        """Test terminal state handling."""
        # Create terminal game state
        board = Board()
        tile_bag = TileBag()
        players = ["Player1", "Player2"]
        scores = [100, 50]  # Game over
        racks = [[], []]  # Empty racks
        
        terminal_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag,
            game_over=True,
            winner=0
        )
        
        # Test terminal value calculation
        value = mcts._get_terminal_value(terminal_state)
        assert value in [-1.0, 0.0, 1.0]
        
        # Test terminal state detection
        assert mcts._is_terminal(terminal_state)
    
    def test_backpropagation(self, mcts, sample_game_state):
        """Test value backpropagation."""
        # Create a simple tree
        root = MCTSNode(sample_game_state)
        child1 = MCTSNode(sample_game_state, parent=root)
        child2 = MCTSNode(sample_game_state, parent=child1)
        
        root.add_child(child1)
        child1.add_child(child2)
        
        # Backpropagate value
        mcts._backpropagate(child2, 1.0)
        
        # Check that values propagated correctly
        assert child2.total_value == 1.0
        assert child2.visits == 1
        assert child1.total_value == -1.0  # Alternating perspective
        assert child1.visits == 1
        assert root.total_value == 1.0
        assert root.visits == 1
