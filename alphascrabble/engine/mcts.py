"""
Monte Carlo Tree Search (MCTS) implementation for AlphaScrabble.

This module implements MCTS with PUCT (Polynomial Upper Confidence Trees) algorithm
for move selection in Scrabble, guided by neural network predictions.
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import math
import random
import time
from collections import defaultdict

from ..rules.board import Board, Tile, GameState
from ..lexicon.move_generator import MoveGenerator, Move, Position, Direction
from ..nn.model import AlphaScrabbleNet
from .features import FeatureExtractor


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    
    # Game state
    game_state: GameState
    move: Optional[Move] = None
    parent: Optional['MCTSNode'] = None
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = None
    
    # Neural network predictions
    prior_probability: float = 0.0
    value_estimate: float = 0.0
    
    # Move generation
    untried_moves: List[Move] = None
    is_expanded: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_moves is None:
            self.untried_moves = []
    
    @property
    def average_value(self) -> float:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.game_state.game_over
    
    def add_child(self, child: 'MCTSNode') -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def update(self, value: float) -> None:
        """Update node statistics."""
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """Monte Carlo Tree Search with PUCT algorithm."""
    
    def __init__(self, 
                 move_generator: MoveGenerator,
                 neural_network: AlphaScrabbleNet,
                 feature_extractor: FeatureExtractor,
                 c_puct: float = 1.0,
                 num_simulations: int = 1000,
                 time_limit: float = 10.0):
        """
        Initialize MCTS.
        
        Args:
            move_generator: Generator for valid moves
            neural_network: Neural network for value/policy estimation
            feature_extractor: Feature extractor for neural network
            c_puct: Exploration constant for PUCT
            num_simulations: Maximum number of simulations
            time_limit: Maximum time in seconds
        """
        self.move_generator = move_generator
        self.neural_network = neural_network
        self.feature_extractor = feature_extractor
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.time_limit = time_limit
        
        # Statistics
        self.simulation_count = 0
        self.total_time = 0.0
    
    def search(self, game_state: GameState) -> Tuple[Move, Dict[str, float]]:
        """
        Search for the best move using MCTS.
        
        Args:
            game_state: Current game state
            
        Returns:
            Tuple of (best_move, move_probabilities)
        """
        start_time = time.time()
        
        # Create root node
        root = MCTSNode(game_state=game_state)
        
        # Expand root node
        self._expand_node(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            if time.time() - start_time > self.time_limit:
                break
            
            # Selection, expansion, simulation, backpropagation
            self._simulate(root)
            self.simulation_count += 1
        
        self.total_time = time.time() - start_time
        
        # Select best move
        best_move = self._select_best_move(root)
        
        # Calculate move probabilities
        move_probs = self._calculate_move_probabilities(root)
        
        return best_move, move_probs
    
    def _simulate(self, root: MCTSNode) -> None:
        """Run one simulation from the root."""
        # Selection: traverse tree to leaf
        node = self._select(root)
        
        # Expansion: expand leaf if not terminal
        if not node.is_terminal:
            self._expand_node(node)
            if node.children:
                node = random.choice(node.children)
        
        # Simulation: get value estimate
        value = self._evaluate(node)
        
        # Backpropagation: update all nodes on path
        self._backpropagate(node, value)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node using PUCT algorithm."""
        while not node.is_leaf and not node.is_terminal:
            if not node.is_expanded:
                self._expand_node(node)
            
            if node.children:
                node = self._select_child(node)
            else:
                break
        
        return node
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the best child using PUCT."""
        best_child = None
        best_score = float('-inf')
        
        for child in node.children:
            # PUCT formula
            exploitation = child.average_value
            exploration = (self.c_puct * child.prior_probability * 
                          math.sqrt(node.visit_count) / (1 + child.visit_count))
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand_node(self, node: MCTSNode) -> None:
        """Expand a node by generating children."""
        if node.is_expanded or node.is_terminal:
            return
        
        # Generate valid moves
        current_rack = node.game_state.get_current_rack()
        moves = self.move_generator.generate_moves(
            node.game_state.board, current_rack
        )
        
        if not moves:
            # No valid moves, pass turn
            node.is_expanded = True
            return
        
        # Get neural network predictions
        board_features = self.feature_extractor.extract_board_features(
            node.game_state.board, node.game_state.current_player
        )
        rack_features = self.feature_extractor.extract_rack_features(current_rack)
        
        # For each move, get policy and value estimates
        for move in moves:
            move_features = self.feature_extractor.extract_move_features(
                move, node.game_state.board, current_rack
            )
            
            # Get neural network prediction
            policy, value = self.neural_network.predict(
                board_features, rack_features, move_features
            )
            
            # Create child node
            child_state = self._apply_move(node.game_state, move)
            child = MCTSNode(
                game_state=child_state,
                move=move,
                parent=node,
                prior_probability=policy[0],  # Simplified
                value_estimate=value
            )
            
            node.add_child(child)
        
        node.is_expanded = True
    
    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate a node using neural network."""
        if node.is_terminal:
            # Game over, return final score
            return self._get_final_score(node.game_state)
        
        # Use neural network value estimate
        return node.value_estimate
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        current = node
        
        while current is not None:
            current.update(value)
            value = -value  # Alternate perspective for opponent
            current = current.parent
    
    def _select_best_move(self, root: MCTSNode) -> Move:
        """Select the best move from the root."""
        if not root.children:
            return None
        
        # Select move with highest visit count
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.move
    
    def _calculate_move_probabilities(self, root: MCTSNode) -> Dict[str, float]:
        """Calculate move probabilities from visit counts."""
        if not root.children:
            return {}
        
        total_visits = sum(child.visit_count for child in root.children)
        
        move_probs = {}
        for child in root.children:
            if child.move:
                move_key = f"{child.move.main_word}_{child.move.direction.value}"
                move_probs[move_key] = child.visit_count / total_visits
        
        return move_probs
    
    def _apply_move(self, game_state: GameState, move: Move) -> GameState:
        """Apply a move to create a new game state."""
        # Create a copy of the game state
        new_state = GameState(
            board=game_state.board,  # Simplified - would need deep copy
            players=game_state.players.copy(),
            scores=game_state.scores.copy(),
            racks=game_state.racks.copy(),
            current_player=game_state.current_player,
            tile_bag=game_state.tile_bag
        )
        
        # Apply the move (simplified)
        # In a real implementation, this would:
        # 1. Place tiles on board
        # 2. Calculate score
        # 3. Update player scores
        # 4. Remove used tiles from rack
        # 5. Draw new tiles
        # 6. Switch to next player
        
        return new_state
    
    def _get_final_score(self, game_state: GameState) -> float:
        """Get the final score for a terminal game state."""
        if not game_state.game_over:
            return 0.0
        
        # Return score difference from current player's perspective
        current_score = game_state.scores[game_state.current_player]
        opponent_score = game_state.scores[1 - game_state.current_player]
        
        return current_score - opponent_score
    
    def get_statistics(self) -> Dict[str, float]:
        """Get MCTS statistics."""
        return {
            'simulation_count': self.simulation_count,
            'total_time': self.total_time,
            'simulations_per_second': self.simulation_count / max(self.total_time, 0.001)
        }


class MCTSPlayer:
    """A player that uses MCTS for move selection."""
    
    def __init__(self, 
                 move_generator: MoveGenerator,
                 neural_network: AlphaScrabbleNet,
                 feature_extractor: FeatureExtractor,
                 **mcts_kwargs):
        """Initialize MCTS player."""
        self.mcts = MCTS(
            move_generator=move_generator,
            neural_network=neural_network,
            feature_extractor=feature_extractor,
            **mcts_kwargs
        )
    
    def get_move(self, game_state: GameState) -> Tuple[Move, Dict[str, float]]:
        """Get the best move using MCTS."""
        return self.mcts.search(game_state)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get player statistics."""
        return self.mcts.get_statistics()


# Example usage and testing
if __name__ == "__main__":
    from ..rules.board import Board, Tile, TileBag, GameState
    from ..lexicon.gaddag import GADDAG
    
    # Create components
    gaddag = GADDAG()
    gaddag.add_word('CAT')
    gaddag.add_word('DOG')
    
    move_generator = MoveGenerator(gaddag)
    neural_network = AlphaScrabbleNet()
    feature_extractor = FeatureExtractor()
    
    # Create MCTS player
    mcts_player = MCTSPlayer(
        move_generator=move_generator,
        neural_network=neural_network,
        feature_extractor=feature_extractor,
        num_simulations=100,
        time_limit=5.0
    )
    
    # Create game state
    board = Board()
    tile_bag = TileBag()
    game_state = GameState(
        board=board,
        players=["Player 1", "Player 2"],
        scores=[0, 0],
        racks=[tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)],
        current_player=0,
        tile_bag=tile_bag
    )
    
    # Get move
    move, probs = mcts_player.get_move(game_state)
    
    print(f"Best move: {move.main_word if move else 'Pass'}")
    print(f"Move probabilities: {probs}")
    print(f"Statistics: {mcts_player.get_statistics()}")