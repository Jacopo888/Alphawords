"""
Evaluation module for AlphaScrabble.

Implements evaluation against different opponents and performance metrics.
"""

import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from ..rules.board import Board, GameState, Move
from ..rules.tiles_en import Tile, TileBag
from ..engine.mcts import MCTS
from ..engine.movegen import MoveGenerator
from ..engine.features import FeatureExtractor
from ..utils.logging import get_logger


class Evaluator:
    """Evaluator for AlphaScrabble models."""
    
    def __init__(self, model, move_generator: MoveGenerator, 
                 feature_extractor: FeatureExtractor, config: dict):
        """Initialize evaluator."""
        self.model = model
        self.move_generator = move_generator
        self.feature_extractor = feature_extractor
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize MCTS for the model
        self.mcts = MCTS(
            move_generator=move_generator,
            feature_extractor=feature_extractor,
            neural_net=model,
            simulations=config.get('mcts_simulations', 160),
            cpuct=config.get('cpuct', 1.5),
            temperature=config.get('temperature', 1.0)
        )
    
    def evaluate_against_random(self, num_games: int = 50) -> Dict[str, float]:
        """Evaluate model against random opponent."""
        self.logger.info(f"Evaluating against random opponent ({num_games} games)")
        
        wins = 0
        total_games = 0
        
        for game_idx in range(num_games):
            result = self._play_game_against_random()
            if result['winner'] == 0:  # Model wins
                wins += 1
            total_games += 1
            
            if (game_idx + 1) % 10 == 0:
                self.logger.info(f"Game {game_idx + 1}/{num_games}, "
                               f"Win rate: {wins/total_games:.2%}")
        
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        self.logger.info(f"Final win rate against random: {win_rate:.2%}")
        
        return {
            'win_rate': win_rate,
            'wins': wins,
            'total_games': total_games
        }
    
    def evaluate_against_greedy(self, num_games: int = 50) -> Dict[str, float]:
        """Evaluate model against greedy opponent."""
        self.logger.info(f"Evaluating against greedy opponent ({num_games} games)")
        
        wins = 0
        total_games = 0
        
        for game_idx in range(num_games):
            result = self._play_game_against_greedy()
            if result['winner'] == 0:  # Model wins
                wins += 1
            total_games += 1
            
            if (game_idx + 1) % 10 == 0:
                self.logger.info(f"Game {game_idx + 1}/{num_games}, "
                               f"Win rate: {wins/total_games:.2%}")
        
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        self.logger.info(f"Final win rate against greedy: {win_rate:.2%}")
        
        return {
            'win_rate': win_rate,
            'wins': wins,
            'total_games': total_games
        }
    
    def evaluate_against_previous(self, previous_model, num_games: int = 50) -> Dict[str, float]:
        """Evaluate current model against previous model."""
        self.logger.info(f"Evaluating against previous model ({num_games} games)")
        
        # Create MCTS for previous model
        previous_mcts = MCTS(
            move_generator=self.move_generator,
            feature_extractor=self.feature_extractor,
            neural_net=previous_model,
            simulations=self.config.get('mcts_simulations', 160),
            cpuct=self.config.get('cpuct', 1.5),
            temperature=self.config.get('temperature', 1.0)
        )
        
        wins = 0
        total_games = 0
        
        for game_idx in range(num_games):
            result = self._play_game_against_model(previous_mcts)
            if result['winner'] == 0:  # Current model wins
                wins += 1
            total_games += 1
            
            if (game_idx + 1) % 10 == 0:
                self.logger.info(f"Game {game_idx + 1}/{num_games}, "
                               f"Win rate: {wins/total_games:.2%}")
        
        win_rate = wins / total_games if total_games > 0 else 0.0
        
        self.logger.info(f"Final win rate against previous model: {win_rate:.2%}")
        
        return {
            'win_rate': win_rate,
            'wins': wins,
            'total_games': total_games
        }
    
    def _play_game_against_random(self) -> Dict:
        """Play a game against random opponent."""
        # Initialize game
        game_state = self._initialize_game()
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not game_state.game_over and move_count < max_moves:
            if game_state.current_player == 0:  # Model's turn
                move = self.mcts.get_best_move(game_state)
            else:  # Random opponent's turn
                moves = self.move_generator.generate_moves(
                    game_state.board, game_state.get_current_rack()
                )
                if moves:
                    move = np.random.choice(moves)
                else:
                    move = Move(tiles=[], direction='', main_word='')
            
            # Apply move
            if move.tiles:  # Not a pass
                game_state.board.apply_move(move)
                game_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
                game_state.add_score(game_state.current_player, move.total_score)
                game_state.draw_tiles(len(move.tiles))
            
            game_state.next_player()
            game_state.check_game_over()
            move_count += 1
        
        # Calculate final scores
        game_state.calculate_final_scores()
        
        # Determine winner
        if game_state.scores[0] > game_state.scores[1]:
            winner = 0
        elif game_state.scores[1] > game_state.scores[0]:
            winner = 1
        else:
            winner = None  # Draw
        
        return {
            'winner': winner,
            'scores': game_state.scores,
            'moves': move_count
        }
    
    def _play_game_against_greedy(self) -> Dict:
        """Play a game against greedy opponent."""
        # Initialize game
        game_state = self._initialize_game()
        
        move_count = 0
        max_moves = 200
        
        while not game_state.game_over and move_count < max_moves:
            if game_state.current_player == 0:  # Model's turn
                move = self.mcts.get_best_move(game_state)
            else:  # Greedy opponent's turn
                moves = self.move_generator.generate_moves(
                    game_state.board, game_state.get_current_rack()
                )
                if moves:
                    # Choose move with highest score
                    move = max(moves, key=lambda m: m.total_score)
                else:
                    move = Move(tiles=[], direction='', main_word='')
            
            # Apply move
            if move.tiles:  # Not a pass
                game_state.board.apply_move(move)
                game_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
                game_state.add_score(game_state.current_player, move.total_score)
                game_state.draw_tiles(len(move.tiles))
            
            game_state.next_player()
            game_state.check_game_over()
            move_count += 1
        
        # Calculate final scores
        game_state.calculate_final_scores()
        
        # Determine winner
        if game_state.scores[0] > game_state.scores[1]:
            winner = 0
        elif game_state.scores[1] > game_state.scores[0]:
            winner = 1
        else:
            winner = None  # Draw
        
        return {
            'winner': winner,
            'scores': game_state.scores,
            'moves': move_count
        }
    
    def _play_game_against_model(self, opponent_mcts: MCTS) -> Dict:
        """Play a game against another model."""
        # Initialize game
        game_state = self._initialize_game()
        
        move_count = 0
        max_moves = 200
        
        while not game_state.game_over and move_count < max_moves:
            if game_state.current_player == 0:  # Current model's turn
                move = self.mcts.get_best_move(game_state)
            else:  # Opponent model's turn
                move = opponent_mcts.get_best_move(game_state)
            
            # Apply move
            if move.tiles:  # Not a pass
                game_state.board.apply_move(move)
                game_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
                game_state.add_score(game_state.current_player, move.total_score)
                game_state.draw_tiles(len(move.tiles))
            
            game_state.next_player()
            game_state.check_game_over()
            move_count += 1
        
        # Calculate final scores
        game_state.calculate_final_scores()
        
        # Determine winner
        if game_state.scores[0] > game_state.scores[1]:
            winner = 0
        elif game_state.scores[1] > game_state.scores[0]:
            winner = 1
        else:
            winner = None  # Draw
        
        return {
            'winner': winner,
            'scores': game_state.scores,
            'moves': move_count
        }
    
    def _initialize_game(self) -> GameState:
        """Initialize a new game."""
        board = Board()
        tile_bag = TileBag()
        
        # Create two players
        players = ["Model", "Opponent"]
        scores = [0, 0]
        
        # Draw initial racks
        rack1 = tile_bag.draw_tiles(7)
        rack2 = tile_bag.draw_tiles(7)
        racks = [rack1, rack2]
        
        return GameState(
            board=board,
            players=players,
            scores=scores,
            racks=racks,
            current_player=0,
            tile_bag=tile_bag
        )
    
    def evaluate_model_performance(self, test_games: List) -> Dict[str, float]:
        """Evaluate model performance on test games."""
        self.logger.info(f"Evaluating model performance on {len(test_games)} test games")
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for game in test_games:
            # This is a simplified evaluation
            # In practice, you'd evaluate on actual game positions
            pass
        
        return {
            'avg_loss': total_loss / len(test_games) if test_games else 0.0,
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }
