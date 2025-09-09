"""
Self-play training pipeline for AlphaScrabble.

This module implements the self-play training loop where the AI plays against itself
to generate training data and improve the neural network.
"""

from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass
import random
import time
import pickle
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..rules.board import Board, Tile, TileBag, GameState
from ..lexicon.move_generator import MoveGenerator, Move
from ..nn.model import AlphaScrabbleNet
from ..engine.features import FeatureExtractor
from ..engine.mcts import MCTSPlayer


@dataclass
class GameRecord:
    """Record of a complete game."""
    game_id: str
    moves: List[Dict]
    winner: int
    final_scores: List[int]
    game_length: int
    timestamp: float


@dataclass
class TrainingExample:
    """A single training example."""
    board_features: np.ndarray
    rack_features: np.ndarray
    move_features: np.ndarray
    policy_target: np.ndarray
    value_target: float
    game_id: str
    move_number: int


class SelfPlayGenerator:
    """Generates self-play games for training."""
    
    def __init__(self,
                 move_generator: MoveGenerator,
                 neural_network: AlphaScrabbleNet,
                 feature_extractor: FeatureExtractor,
                 num_simulations: int = 1000,
                 temperature: float = 1.0):
        """
        Initialize self-play generator.
        
        Args:
            move_generator: Generator for valid moves
            neural_network: Neural network for move selection
            feature_extractor: Feature extractor
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for move selection
        """
        self.move_generator = move_generator
        self.neural_network = neural_network
        self.feature_extractor = feature_extractor
        self.num_simulations = num_simulations
        self.temperature = temperature
        
        # Create MCTS players
        self.mcts_player = MCTSPlayer(
            move_generator=move_generator,
            neural_network=neural_network,
            feature_extractor=feature_extractor,
            num_simulations=num_simulations
        )
        
        # Statistics
        self.games_played = 0
        self.total_moves = 0
        self.training_examples = []
    
    def generate_game(self) -> Tuple[GameRecord, List[TrainingExample]]:
        """Generate a single self-play game."""
        game_id = f"game_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Initialize game
        board = Board()
        tile_bag = TileBag()
        players = ["AI Player 1", "AI Player 2"]
        scores = [0, 0]
        racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
        current_player = 0
        
        game_state = GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=current_player,
            tile_bag=tile_bag
        )
        
        # Game loop
        moves = []
        training_examples = []
        move_number = 0
        
        while not game_state.game_over and move_number < 100:  # Max 100 moves
            # Get move using MCTS
            move, move_probs = self.mcts_player.get_move(game_state)
            
            if move is None:
                # No valid moves, pass turn
                game_state.next_player()
                move_number += 1
                continue
            
            # Record move
            move_record = {
                'move_number': move_number,
                'player': current_player,
                'move': move,
                'move_probs': move_probs,
                'board_state': self._serialize_board(board),
                'rack': [tile.letter for tile in game_state.get_current_rack()],
                'scores': game_state.scores.copy()
            }
            moves.append(move_record)
            
            # Create training example
            training_example = self._create_training_example(
                game_state, move, move_probs, game_id, move_number
            )
            training_examples.append(training_example)
            
            # Apply move
            self._apply_move(game_state, move)
            
            # Check game over
            game_state.check_game_over()
            
            move_number += 1
        
        # Create game record
        game_record = GameRecord(
            game_id=game_id,
            moves=moves,
            winner=game_state.winner if game_state.game_over else -1,
            final_scores=game_state.scores,
            game_length=move_number,
            timestamp=time.time()
        )
        
        # Update statistics
        self.games_played += 1
        self.total_moves += move_number
        
        return game_record, training_examples
    
    def _create_training_example(self, game_state: GameState, move: Move,
                                move_probs: Dict[str, float], game_id: str,
                                move_number: int) -> TrainingExample:
        """Create a training example from a move."""
        # Extract features
        board_features = self.feature_extractor.extract_board_features(
            game_state.board, game_state.current_player
        )
        rack_features = self.feature_extractor.extract_rack_features(
            game_state.get_current_rack()
        )
        move_features = self.feature_extractor.extract_move_features(
            move, game_state.board, game_state.get_current_rack()
        )
        
        # Create policy target
        policy_target = np.zeros(64)  # Simplified - would need proper mapping
        move_key = f"{move.main_word}_{move.direction.value}"
        if move_key in move_probs:
            policy_target[0] = move_probs[move_key]
        
        # Create value target (simplified)
        value_target = 0.0  # Would be calculated based on game outcome
        
        return TrainingExample(
            board_features=board_features,
            rack_features=rack_features,
            move_features=move_features,
            policy_target=policy_target,
            value_target=value_target,
            game_id=game_id,
            move_number=move_number
        )
    
    def _apply_move(self, game_state: GameState, move: Move) -> None:
        """Apply a move to the game state."""
        # Place tiles on board
        for move_tile in move.tiles:
            if move_tile.is_new:
                game_state.board.place_tile(
                    move_tile.tile, move_tile.position.row, move_tile.position.col
                )
        
        # Update score
        game_state.add_score(game_state.current_player, move.total_score)
        
        # Remove used tiles from rack
        used_tiles = [mt.tile for mt in move.tiles if mt.is_new]
        for tile in used_tiles:
            if tile in game_state.get_current_rack():
                game_state.get_current_rack().remove(tile)
        
        # Draw new tiles
        tiles_needed = 7 - len(game_state.get_current_rack())
        if tiles_needed > 0:
            new_tiles = game_state.tile_bag.draw_tiles(tiles_needed)
            game_state.get_current_rack().extend(new_tiles)
        
        # Switch to next player
        game_state.next_player()
    
    def _serialize_board(self, board: Board) -> List[List[str]]:
        """Serialize board state."""
        serialized = []
        for i in range(15):
            row = []
            for j in range(15):
                if board.is_empty(i, j):
                    row.append('.')
                else:
                    tile = board.get_tile(i, j)
                    row.append(tile.letter if tile else '.')
            serialized.append(row)
        return serialized
    
    def generate_batch(self, num_games: int) -> List[TrainingExample]:
        """Generate a batch of training examples."""
        all_examples = []
        
        for _ in range(num_games):
            game_record, examples = self.generate_game()
            all_examples.extend(examples)
        
        return all_examples
    
    def get_statistics(self) -> Dict[str, int]:
        """Get self-play statistics."""
        return {
            'games_played': self.games_played,
            'total_moves': self.total_moves,
            'avg_moves_per_game': self.total_moves / max(self.games_played, 1)
        }


class TrainingPipeline:
    """Complete training pipeline for AlphaScrabble."""
    
    def __init__(self,
                 move_generator: MoveGenerator,
                 neural_network: AlphaScrabbleNet,
                 feature_extractor: FeatureExtractor,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_games_per_iteration: int = 100):
        """
        Initialize training pipeline.
        
        Args:
            move_generator: Generator for valid moves
            neural_network: Neural network to train
            feature_extractor: Feature extractor
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_games_per_iteration: Number of games per training iteration
        """
        self.move_generator = move_generator
        self.neural_network = neural_network
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_games_per_iteration = num_games_per_iteration
        
        # Create self-play generator
        self.selfplay_generator = SelfPlayGenerator(
            move_generator=move_generator,
            neural_network=neural_network,
            feature_extractor=feature_extractor
        )
        
        # Setup training
        self.optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Training history
        self.training_history = []
        self.best_model = None
        self.best_score = float('-inf')
    
    def train_iteration(self) -> Dict[str, float]:
        """Run one training iteration."""
        print(f"Starting training iteration...")
        
        # Generate self-play games
        print(f"Generating {self.num_games_per_iteration} self-play games...")
        training_examples = self.selfplay_generator.generate_batch(
            self.num_games_per_iteration
        )
        
        if not training_examples:
            print("No training examples generated!")
            return {}
        
        print(f"Generated {len(training_examples)} training examples")
        
        # Prepare training data
        board_features = np.array([ex.board_features for ex in training_examples])
        rack_features = np.array([ex.rack_features for ex in training_examples])
        move_features = np.array([ex.move_features for ex in training_examples])
        policy_targets = np.array([ex.policy_target for ex in training_examples])
        value_targets = np.array([ex.value_target for ex in training_examples])
        
        # Convert to tensors
        board_tensor = torch.FloatTensor(board_features)
        rack_tensor = torch.FloatTensor(rack_features)
        move_tensor = torch.FloatTensor(move_features)
        policy_target_tensor = torch.FloatTensor(policy_targets)
        value_target_tensor = torch.FloatTensor(value_targets)
        
        # Create data loader
        dataset = TensorDataset(
            board_tensor, rack_tensor, move_tensor,
            policy_target_tensor, value_target_tensor
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        self.neural_network.train()
        
        for batch in dataloader:
            board_batch, rack_batch, move_batch, policy_target_batch, value_target_batch = batch
            
            # Forward pass
            policy_pred, value_pred = self.neural_network(
                board_batch, rack_batch, move_batch
            )
            
            # Calculate losses
            policy_loss = self.policy_loss_fn(policy_pred, policy_target_batch)
            value_loss = self.value_loss_fn(value_pred.squeeze(), value_target_batch)
            total_loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        # Calculate average losses
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = avg_policy_loss + avg_value_loss
        
        # Record training metrics
        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss,
            'num_examples': len(training_examples),
            'num_batches': num_batches
        }
        
        self.training_history.append(metrics)
        
        print(f"Training completed:")
        print(f"  Policy Loss: {avg_policy_loss:.4f}")
        print(f"  Value Loss: {avg_value_loss:.4f}")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        
        return metrics
    
    def train(self, num_iterations: int, save_interval: int = 10) -> None:
        """Train the model for multiple iterations."""
        print(f"Starting training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # Train one iteration
            metrics = self.train_iteration()
            
            # Save model periodically
            if (iteration + 1) % save_interval == 0:
                self.save_model(f"model_iteration_{iteration + 1}.pth")
            
            # Update best model
            if metrics.get('total_loss', float('inf')) < self.best_score:
                self.best_score = metrics['total_loss']
                self.best_model = self.neural_network.state_dict().copy()
        
        print(f"\nTraining completed!")
        print(f"Best loss: {self.best_score:.4f}")
    
    def save_model(self, filepath: str) -> None:
        """Save the current model."""
        torch.save({
            'model_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        checkpoint = torch.load(filepath)
        self.neural_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    from ..lexicon.gaddag import GADDAG
    
    # Create components
    gaddag = GADDAG()
    gaddag.add_word('CAT')
    gaddag.add_word('DOG')
    gaddag.add_word('HOUSE')
    
    move_generator = MoveGenerator(gaddag)
    neural_network = AlphaScrabbleNet()
    feature_extractor = FeatureExtractor()
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        move_generator=move_generator,
        neural_network=neural_network,
        feature_extractor=feature_extractor,
        num_games_per_iteration=10
    )
    
    # Train for a few iterations
    pipeline.train(num_iterations=3)
    
    print("Training completed!")
