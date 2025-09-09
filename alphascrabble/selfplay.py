"""
Self-play pipeline for AlphaScrabble.

Generates training data by playing games against itself using MCTS.
"""

import os
import time
import uuid
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

from .config import Config
from .nn.model import AlphaScrabbleNet
from .nn.dataset import GameRecord, ReplayBuffer
from .engine.mcts import MCTS
from .engine.movegen import MoveGenerator
from .engine.features import FeatureExtractor
from .lexicon.gaddag_loader import GaddagLoader
from .rules.board import Board, GameState, Move
from .rules.tiles_en import TileBag
from .utils.logging import get_logger
from .utils.seeding import set_seed


class SelfPlayPipeline:
    """Self-play pipeline for generating training data."""
    
    def __init__(self, config: Config, model: AlphaScrabbleNet, 
                 gaddag_loader: GaddagLoader):
        """Initialize self-play pipeline."""
        self.config = config
        self.model = model
        self.gaddag_loader = gaddag_loader
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.move_generator = MoveGenerator(gaddag_loader)
        self.feature_extractor = FeatureExtractor()
        
        # Initialize MCTS
        self.mcts = MCTS(
            move_generator=self.move_generator,
            feature_extractor=self.feature_extractor,
            neural_net=model,
            simulations=config.MCTS_SIMULATIONS,
            cpuct=config.CPUCT,
            temperature=config.TEMPERATURE
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.SELFPLAY_GAMES * 2,  # Store more than needed
            data_dir=config.DATA_DIR
        )
    
    def run_selfplay(self, num_games: int, output_dir: str = None) -> List[GameRecord]:
        """Run self-play games and return game records."""
        if output_dir is None:
            output_dir = os.path.join(self.config.DATA_DIR, "selfplay")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Starting self-play with {num_games} games")
        
        game_records = []
        
        with tqdm(total=num_games, desc="Self-play games") as pbar:
            for game_idx in range(num_games):
                # Set seed for reproducibility (optional)
                if self.config.SEED is not None:
                    set_seed(self.config.SEED + game_idx)
                
                # Play a single game
                game_record = self._play_single_game(game_idx)
                game_records.append(game_record)
                
                # Add to replay buffer
                self.replay_buffer.add_game(game_record)
                
                # Save periodically
                if (game_idx + 1) % 10 == 0:
                    self._save_games_batch(game_records[-10:], output_dir, game_idx - 9)
                
                pbar.update(1)
                pbar.set_postfix({
                    'avg_moves': np.mean([g.game_length for g in game_records[-10:]]),
                    'buffer_size': self.replay_buffer.current_size
                })
        
        # Save all games
        self._save_games_batch(game_records, output_dir, "all")
        
        self.logger.info(f"Completed {num_games} self-play games")
        return game_records
    
    def _play_single_game(self, game_idx: int) -> GameRecord:
        """Play a single self-play game."""
        # Initialize game state
        game_state = self._initialize_game()
        
        moves = []
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        # Game loop
        while not game_state.game_over and move_count < max_moves:
            # Get move from MCTS
            move = self.mcts.get_best_move(game_state)
            moves.append(move)
            
            # Apply move
            if move.tiles:  # Not a pass
                game_state.board.apply_move(move)
                game_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
                game_state.add_score(game_state.current_player, move.total_score)
                game_state.draw_tiles(len(move.tiles))
            
            # Move to next player
            game_state.next_player()
            game_state.check_game_over()
            move_count += 1
        
        # Calculate final scores
        game_state.calculate_final_scores()
        
        # Create game record
        game_record = GameRecord(
            game_id=f"selfplay_{game_idx:06d}_{uuid.uuid4().hex[:8]}",
            moves=moves,
            scores=game_state.scores,
            winner=game_state.winner,
            game_length=move_count
        )
        
        return game_record
    
    def _initialize_game(self) -> GameState:
        """Initialize a new game state."""
        board = Board()
        tile_bag = TileBag()
        
        # Create two players
        players = ["Player1", "Player2"]
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
    
    def _save_games_batch(self, games: List[GameRecord], output_dir: str, 
                         batch_id: str) -> None:
        """Save a batch of games to disk."""
        filename = f"games_batch_{batch_id}.pkl"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to serializable format
        games_data = [game.to_dict() for game in games]
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(games_data, f)
        
        self.logger.debug(f"Saved {len(games)} games to {filepath}")
    
    def get_training_data(self) -> List[GameRecord]:
        """Get all games from replay buffer for training."""
        return self.replay_buffer.get_games()
    
    def clear_buffer(self) -> None:
        """Clear the replay buffer."""
        self.replay_buffer.clear()


class SelfPlayManager:
    """Manages self-play sessions and data collection."""
    
    def __init__(self, config: Config):
        """Initialize self-play manager."""
        self.config = config
        self.logger = get_logger(__name__)
    
    def run_selfplay_session(self, model_path: Optional[str] = None, 
                           num_games: int = None) -> List[GameRecord]:
        """Run a complete self-play session."""
        if num_games is None:
            num_games = self.config.SELFPLAY_GAMES
        
        # Load lexicon
        gaddag_loader = GaddagLoader(self.config.dawg_path, self.config.gaddag_path)
        if not gaddag_loader.load():
            raise RuntimeError("Failed to load lexicon files")
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            model = AlphaScrabbleNet.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        else:
            model = AlphaScrabbleNet()
            self.logger.info("Using untrained model")
        
        # Initialize self-play pipeline
        pipeline = SelfPlayPipeline(self.config, model, gaddag_loader)
        
        # Run self-play
        game_records = pipeline.run_selfplay(num_games)
        
        # Save final data
        self._save_session_data(game_records)
        
        return game_records
    
    def _save_session_data(self, game_records: List[GameRecord]) -> None:
        """Save session data and statistics."""
        # Save to replay buffer
        replay_buffer = ReplayBuffer(data_dir=self.config.DATA_DIR)
        replay_buffer.add_games(game_records)
        replay_buffer.save("latest_selfplay.pkl")
        
        # Calculate and save statistics
        stats = self._calculate_game_statistics(game_records)
        self._save_statistics(stats)
        
        self.logger.info(f"Saved {len(game_records)} games and statistics")
    
    def _calculate_game_statistics(self, game_records: List[GameRecord]) -> Dict:
        """Calculate statistics from game records."""
        if not game_records:
            return {}
        
        game_lengths = [game.game_length for game in game_records]
        scores = [game.scores for game in game_records]
        
        stats = {
            'total_games': len(game_records),
            'avg_game_length': np.mean(game_lengths),
            'std_game_length': np.std(game_lengths),
            'min_game_length': np.min(game_lengths),
            'max_game_length': np.max(game_lengths),
            'avg_score_player1': np.mean([s[0] for s in scores]),
            'avg_score_player2': np.mean([s[1] for s in scores]),
            'winner_distribution': {
                'player1': sum(1 for g in game_records if g.winner == 0),
                'player2': sum(1 for g in game_records if g.winner == 1),
                'draw': sum(1 for g in game_records if g.winner is None)
            }
        }
        
        return stats
    
    def _save_statistics(self, stats: Dict) -> None:
        """Save statistics to file."""
        import json
        
        stats_file = os.path.join(self.config.DATA_DIR, "selfplay_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Saved statistics to {stats_file}")
