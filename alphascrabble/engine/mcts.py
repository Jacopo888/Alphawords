"""
Monte Carlo Tree Search with PUCT algorithm.

Implements MCTS for Scrabble move selection using neural network guidance.
"""

import math
import random
from typing import List, Optional, Dict, Tuple
import numpy as np
from ..rules.board import Board, Move, GameState
from ..rules.tiles_en import Tile
from .features import FeatureExtractor


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Move] = None):
        """Initialize MCTS node."""
        self.state = state
        self.parent = parent
        self.move = move
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_value = 0.0
        self.prior_prob = 0.0
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """Get average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return not self.is_expanded or len(self.children) == 0
    
    def add_child(self, child: 'MCTSNode') -> None:
        """Add a child node."""
        self.children.append(child)
    
    def update(self, value: float) -> None:
        """Update node statistics."""
        self.visits += 1
        self.total_value += value
    
    def get_ucb_score(self, cpuct: float) -> float:
        """Calculate UCB score for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = cpuct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        return exploitation + exploration


class MCTS:
    """Monte Carlo Tree Search with PUCT algorithm."""
    
    def __init__(self, move_generator, feature_extractor, neural_net, 
                 simulations: int = 160, cpuct: float = 1.5, temperature: float = 1.0):
        """Initialize MCTS."""
        self.move_generator = move_generator
        self.feature_extractor = feature_extractor
        self.neural_net = neural_net
        self.simulations = simulations
        self.cpuct = cpuct
        self.temperature = temperature
        
    def search(self, state: GameState) -> List[float]:
        """Perform MCTS search and return move probabilities."""
        root = MCTSNode(state)
        
        # Expand root node
        self._expand_node(root)
        
        # Run simulations
        for _ in range(self.simulations):
            self._simulate(root)
        
        # Calculate move probabilities
        return self._get_move_probabilities(root)
    
    def _simulate(self, root: MCTSNode) -> float:
        """Run one simulation from root."""
        node = root
        
        # Selection phase
        while not node.is_leaf:
            node = self._select_child(node)
        
        # Expansion phase
        if not node.is_expanded and not self._is_terminal(node.state):
            self._expand_node(node)
        
        # Evaluation phase
        if node.is_leaf:
            value = self._evaluate_node(node)
        else:
            value = self._evaluate_node(node)
        
        # Backpropagation phase
        self._backpropagate(node, value)
        
        return value
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB."""
        if not node.children:
            return node
        
        best_child = None
        best_score = float('-inf')
        
        for child in node.children:
            score = child.get_ucb_score(self.cpuct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_node(self, node: MCTSNode) -> None:
        """Expand a node by generating all possible moves."""
        if node.is_expanded:
            return
        
        # Generate legal moves
        moves = self.move_generator.generate_moves(node.state.board, node.state.get_current_rack())
        
        if not moves:
            node.is_expanded = True
            return
        
        # Get neural network predictions
        board_features, rack_features = self.feature_extractor.extract_state_features(
            node.state.board, node.state.get_current_rack(), node.state.current_player
        )
        
        move_features = self.feature_extractor.extract_move_list_features(
            moves, node.state.board, node.state.get_current_rack()
        )
        
        # Get policy and value from neural network
        policy_logits, value = self.neural_net.predict(board_features, rack_features, move_features)
        
        # Create child nodes
        for i, move in enumerate(moves):
            # Create new state with move applied
            new_state = self._apply_move(node.state, move)
            child = MCTSNode(new_state, parent=node, move=move)
            child.prior_prob = math.exp(policy_logits[i]) if i < len(policy_logits) else 0.0
            node.add_child(child)
        
        node.is_expanded = True
    
    def _evaluate_node(self, node: MCTSNode) -> float:
        """Evaluate a node using neural network."""
        if self._is_terminal(node.state):
            return self._get_terminal_value(node.state)
        
        # Get neural network prediction
        board_features, rack_features = self.feature_extractor.extract_state_features(
            node.state.board, node.state.get_current_rack(), node.state.current_player
        )
        
        # For leaf nodes, we need to generate moves to get move features
        moves = self.move_generator.generate_moves(node.state.board, node.state.get_current_rack())
        move_features = self.feature_extractor.extract_move_list_features(
            moves, node.state.board, node.state.get_current_rack()
        )
        
        _, value = self.neural_net.predict(board_features, rack_features, move_features)
        
        return value
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.update(value)
            value = -value  # Alternate perspective for opponent
            current = current.parent
    
    def _apply_move(self, state: GameState, move: Move) -> GameState:
        """Apply a move to create new game state."""
        # Create deep copy of state
        new_state = GameState(
            board=state.board.copy(),
            players=state.players.copy(),
            scores=state.scores.copy(),
            racks=[rack.copy() for rack in state.racks],
            current_player=state.current_player,
            tile_bag=state.tile_bag,
            game_over=state.game_over,
            winner=state.winner,
            last_move=state.last_move
        )
        
        if move.tiles:  # Not a pass
            # Apply move to board
            new_state.board.apply_move(move)
            
            # Remove tiles from rack
            new_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
            
            # Add score
            new_state.add_score(new_state.current_player, move.total_score)
            
            # Draw new tiles
            new_state.draw_tiles(len(move.tiles))
        
        # Move to next player
        new_state.next_player()
        
        # Check for game over
        new_state.check_game_over()
        
        return new_state
    
    def _is_terminal(self, state: GameState) -> bool:
        """Check if state is terminal."""
        return state.game_over
    
    def _get_terminal_value(self, state: GameState) -> float:
        """Get value for terminal state."""
        if state.winner is None:
            return 0.0  # Draw
        
        # Return 1.0 for win, -1.0 for loss from current player's perspective
        if state.winner == state.current_player:
            return 1.0
        else:
            return -1.0
    
    def _get_move_probabilities(self, root: MCTSNode) -> List[float]:
        """Get move probabilities from root node."""
        if not root.children:
            return []
        
        # Get visit counts
        visit_counts = [child.visits for child in root.children]
        total_visits = sum(visit_counts)
        
        if total_visits == 0:
            return [1.0 / len(root.children)] * len(root.children)
        
        # Apply temperature
        if self.temperature == 0:
            # Greedy selection
            max_visits = max(visit_counts)
            return [1.0 if visits == max_visits else 0.0 for visits in visit_counts]
        else:
            # Temperature scaling
            scaled_visits = [visits ** (1.0 / self.temperature) for visits in visit_counts]
            total_scaled = sum(scaled_visits)
            return [visits / total_scaled for visits in scaled_visits]
    
    def get_best_move(self, state: GameState) -> Move:
        """Get the best move according to MCTS."""
        probabilities = self.search(state)
        
        if not probabilities:
            # Return pass move if no moves available
            return Move(tiles=[], direction='', main_word='')
        
        # Select move with highest probability
        best_idx = np.argmax(probabilities)
        
        # Generate moves to get the actual move
        moves = self.move_generator.generate_moves(state.board, state.get_current_rack())
        
        if best_idx < len(moves):
            return moves[best_idx]
        else:
            return Move(tiles=[], direction='', main_word='')
