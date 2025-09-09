"""
Feature extraction for neural network training.

Extracts features from board state and moves for policy and value networks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ..rules.board import Board, Move, Position, PremiumType
from ..rules.tiles_en import Tile, TILE_SCORES


class FeatureExtractor:
    """Extracts features for neural network training."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.board_channels = 32  # Number of input channels for board
        self.rack_embedding_dim = 128
        self.move_feature_dim = 64
    
    def extract_board_features(self, board: Board, current_player: int) -> np.ndarray:
        """Extract board features as multi-channel tensor."""
        features = np.zeros((self.board_channels, 15, 15), dtype=np.float32)
        
        # Channel 0-25: One-hot encoding of letters (A-Z)
        for row in range(15):
            for col in range(15):
                tile = board.get_tile(row, col)
                if tile:
                    letter_idx = ord(tile.display_letter.upper()) - ord('A')
                    if 0 <= letter_idx < 26:
                        features[letter_idx, row, col] = 1.0
        
        # Channel 26: Blank tiles
        for row in range(15):
            for col in range(15):
                tile = board.get_tile(row, col)
                if tile and tile.is_blank:
                    features[26, row, col] = 1.0
        
        # Channel 27: Current player turn mask
        features[27, :, :] = current_player
        
        # Channel 28-31: Premium squares
        for row in range(15):
            for col in range(15):
                premium = board.get_premium_type(row, col)
                if premium == PremiumType.DLS:
                    features[28, row, col] = 1.0
                elif premium == PremiumType.TLS:
                    features[29, row, col] = 1.0
                elif premium == PremiumType.DWS:
                    features[30, row, col] = 1.0
                elif premium == PremiumType.TWS:
                    features[31, row, col] = 1.0
        
        return features
    
    def extract_rack_features(self, rack: List[Tile]) -> np.ndarray:
        """Extract rack features as embedding."""
        # One-hot encoding of letters with counts
        letter_counts = np.zeros(27, dtype=np.float32)  # A-Z + blank
        
        for tile in rack:
            if tile.is_blank:
                letter_counts[26] += 1
            else:
                letter_idx = ord(tile.letter.upper()) - ord('A')
                if 0 <= letter_idx < 26:
                    letter_counts[letter_idx] += 1
        
        # Normalize by rack size
        if len(rack) > 0:
            letter_counts = letter_counts / len(rack)
        
        return letter_counts
    
    def extract_move_features(self, move: Move, board: Board, rack: List[Tile]) -> np.ndarray:
        """Extract features for a specific move."""
        features = np.zeros(self.move_feature_dim, dtype=np.float32)
        
        if not move.tiles:  # Pass move
            return features
        
        idx = 0
        
        # Basic move properties
        features[idx] = len(move.tiles) / 7.0  # Normalized tile count
        idx += 1
        
        features[idx] = move.score / 100.0  # Normalized score
        idx += 1
        
        features[idx] = 1.0 if move.is_bingo else 0.0  # Bingo flag
        idx += 1
        
        features[idx] = len(move.cross_words) / 5.0  # Normalized cross word count
        idx += 1
        
        # Word length and position
        features[idx] = len(move.main_word) / 15.0  # Normalized word length
        idx += 1
        
        if move.tiles:
            start_pos = move.tiles[0].position
            features[idx] = start_pos.row / 14.0  # Normalized row
            idx += 1
            features[idx] = start_pos.col / 14.0  # Normalized col
            idx += 1
        
        # Direction
        features[idx] = 1.0 if move.direction == 'across' else 0.0
        idx += 1
        
        # Premium square usage
        premium_count = 0
        for placed_tile in move.tiles:
            premium = board.get_premium_type(
                placed_tile.position.row, placed_tile.position.col
            )
            if premium != PremiumType.NONE:
                premium_count += 1
        
        features[idx] = premium_count / len(move.tiles) if move.tiles else 0.0
        idx += 1
        
        # Blank usage
        blank_count = sum(1 for tile in move.tiles if tile.tile.is_blank)
        features[idx] = blank_count / len(move.tiles) if move.tiles else 0.0
        idx += 1
        
        # High-value letter usage
        high_value_letters = ['J', 'Q', 'X', 'Z']
        high_value_count = 0
        for tile in move.tiles:
            if tile.tile.letter in high_value_letters:
                high_value_count += 1
        
        features[idx] = high_value_count / len(move.tiles) if move.tiles else 0.0
        idx += 1
        
        # Leave analysis (remaining rack after move)
        if move.tiles:
            used_letters = [tile.tile.letter for tile in move.tiles]
            remaining_rack = [tile for tile in rack if tile.letter not in used_letters]
            
            # Vowel/consonant ratio in leave
            vowels = 'AEIOU'
            vowel_count = sum(1 for tile in remaining_rack if tile.letter in vowels)
            consonant_count = len(remaining_rack) - vowel_count
            
            if len(remaining_rack) > 0:
                features[idx] = vowel_count / len(remaining_rack)
                idx += 1
                features[idx] = consonant_count / len(remaining_rack)
                idx += 1
            else:
                features[idx] = 0.0
                idx += 1
                features[idx] = 0.0
                idx += 1
        
        # Fill remaining features with zeros
        while idx < self.move_feature_dim:
            features[idx] = 0.0
            idx += 1
        
        return features
    
    def extract_state_features(self, board: Board, rack: List[Tile], 
                             current_player: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract both board and rack features."""
        board_features = self.extract_board_features(board, current_player)
        rack_features = self.extract_rack_features(rack)
        
        return board_features, rack_features
    
    def extract_move_list_features(self, moves: List[Move], board: Board, 
                                 rack: List[Tile]) -> np.ndarray:
        """Extract features for a list of moves."""
        if not moves:
            return np.zeros((0, self.move_feature_dim), dtype=np.float32)
        
        move_features = []
        for move in moves:
            features = self.extract_move_features(move, board, rack)
            move_features.append(features)
        
        return np.array(move_features, dtype=np.float32)
    
    def get_feature_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shapes of different feature types."""
        return {
            'board': (self.board_channels, 15, 15),
            'rack': (27,),
            'move': (self.move_feature_dim,),
            'move_list': (None, self.move_feature_dim)
        }
