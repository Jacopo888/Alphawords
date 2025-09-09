"""
Flask web application for AlphaScrabble.

Provides a web interface for playing Scrabble against the AI.
"""

from flask import Flask, render_template, request, jsonify
import json
import time
import random
from typing import Dict, List, Optional

# Import AlphaScrabble components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from alphascrabble.rules.board import Board, Tile, TileBag, GameState
from alphascrabble.lexicon.gaddag import GADDAG
from alphascrabble.lexicon.move_generator import MoveGenerator
from alphascrabble.nn.model import AlphaScrabbleNet
from alphascrabble.engine.features import FeatureExtractor
from alphascrabble.engine.mcts import MCTSPlayer


app = Flask(__name__)

# Global game state
current_game = None
ai_player = None


class WebGameState:
    """Game state for web interface."""
    
    def __init__(self):
        self.board = Board()
        self.tile_bag = TileBag()
        self.players = ["Human Player", "AI Player"]
        self.scores = [0, 0]
        self.racks = [self.tile_bag.draw_tiles(7), self.tile_bag.draw_tiles(7)]
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.current_move = []
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'board': self._serialize_board(),
            'scores': self.scores,
            'current_player': self.current_player,
            'racks': [self._serialize_rack(rack) for rack in self.racks],
            'game_over': self.game_over,
            'winner': self.winner,
            'move_count': self.move_count,
            'tiles_left': self.tile_bag.tiles_remaining()
        }
    
    def _serialize_board(self):
        """Serialize board state."""
        board_state = []
        for i in range(15):
            row = []
            for j in range(15):
                if self.board.is_empty(i, j):
                    row.append(None)
                else:
                    tile = self.board.get_tile(i, j)
                    row.append({
                        'letter': tile.letter,
                        'score': tile.score,
                        'is_blank': tile.is_blank
                    })
            board_state.append(row)
        return board_state
    
    def _serialize_rack(self, rack):
        """Serialize rack state."""
        return [{'letter': tile.letter, 'score': tile.score} for tile in rack]
    
    def get_current_rack(self):
        """Get current player's rack."""
        return self.racks[self.current_player]
    
    def next_player(self):
        """Switch to next player."""
        self.current_player = (self.current_player + 1) % len(self.players)
    
    def add_score(self, player, points):
        """Add points to player's score."""
        self.scores[player] += points
    
    def check_game_over(self):
        """Check if game is over."""
        if self.tile_bag.is_empty():
            self.game_over = True
            # Find winner
            max_score = max(self.scores)
            winners = [i for i, score in enumerate(self.scores) if score == max_score]
            if len(winners) == 1:
                self.winner = winners[0]


def initialize_ai():
    """Initialize AI components."""
    global ai_player
    
    try:
        # Create GADDAG with sample words
        gaddag = GADDAG()
        sample_words = [
            'CAT', 'DOG', 'HOUSE', 'COMPUTER', 'SCRABBLE', 'WORD', 'GAME',
            'PLAY', 'BOARD', 'TILE', 'SCORE', 'POINT', 'LETTER', 'ALPHABET',
            'DICTIONARY', 'QUICK', 'BROWN', 'FOX', 'JUMPED', 'OVER', 'LAZY',
            'MOUSE', 'KEYBOARD', 'MONITOR', 'SCREEN', 'WINDOW', 'DOOR',
            'CHAIR', 'TABLE', 'BOOK', 'PEN', 'PAPER', 'NOTE', 'MUSIC',
            'SONG', 'DANCE', 'SING', 'READ', 'WRITE', 'DRAW', 'PAINT',
            'COLOR', 'RED', 'BLUE', 'GREEN', 'YELLOW', 'BLACK', 'WHITE'
        ]
        
        for word in sample_words:
            gaddag.add_word(word)
        
        # Create move generator
        move_generator = MoveGenerator(gaddag)
        
        # Create neural network
        neural_network = AlphaScrabbleNet()
        
        # Create feature extractor
        feature_extractor = FeatureExtractor()
        
        # Create MCTS player
        ai_player = MCTSPlayer(
            move_generator=move_generator,
            neural_network=neural_network,
            feature_extractor=feature_extractor,
            num_simulations=100,  # Reduced for web interface
            time_limit=5.0
        )
        
        print("AI initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing AI: {e}")
        ai_player = None


@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    global current_game
    
    current_game = WebGameState()
    
    return jsonify({
        'success': True,
        'game_state': current_game.to_dict()
    })


@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    """Get current game state."""
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    return jsonify({
        'success': True,
        'game_state': current_game.to_dict()
    })


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a move."""
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    data = request.get_json()
    move = data.get('move')
    
    if not move:
        return jsonify({'error': 'No move provided'}), 400
    
    try:
        # Validate move
        if not _validate_move(move):
            return jsonify({'error': 'Invalid move'}), 400
        
        # Apply move
        score = _apply_move(move)
        
        # Update game state
        current_game.add_score(current_game.current_player, score)
        current_game.move_count += 1
        
        # Check if game is over
        current_game.check_game_over()
        
        # Switch to next player
        current_game.next_player()
        
        return jsonify({
            'success': True,
            'score': score,
            'game_state': current_game.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """Get AI move."""
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    if current_game.current_player != 1:
        return jsonify({'error': 'Not AI player turn'}), 400
    
    if ai_player is None:
        return jsonify({'error': 'AI not initialized'}), 500
    
    try:
        # Create game state for AI
        game_state = GameState(
            board=current_game.board,
            players=current_game.players,
            scores=current_game.scores,
            racks=current_game.racks,
            current_player=current_game.current_player,
            tile_bag=current_game.tile_bag
        )
        
        # Get AI move
        move, move_probs = ai_player.get_move(game_state)
        
        if move is None:
            # No valid moves, pass turn
            current_game.next_player()
            return jsonify({
                'success': True,
                'move': None,
                'message': 'AI passes turn',
                'game_state': current_game.to_dict()
            })
        
        # Convert move to web format
        web_move = _convert_move_to_web(move)
        
        # Apply move
        score = _apply_move(web_move)
        
        # Update game state
        current_game.add_score(current_game.current_player, score)
        current_game.move_count += 1
        
        # Check if game is over
        current_game.check_game_over()
        
        # Switch to next player
        current_game.next_player()
        
        return jsonify({
            'success': True,
            'move': web_move,
            'score': score,
            'game_state': current_game.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get move suggestions."""
    if current_game is None:
        return jsonify({'error': 'No game in progress'}), 400
    
    try:
        # Generate simple suggestions based on current rack
        current_rack = current_game.get_current_rack()
        letters = ''.join(tile.letter for tile in current_rack)
        
        # Simple word suggestions
        suggestions = _generate_simple_suggestions(letters)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _validate_move(move):
    """Validate a move."""
    if not move or 'tiles' not in move:
        return False
    
    # Check if all tiles are valid
    for tile_placement in move['tiles']:
        if 'letter' not in tile_placement or 'row' not in tile_placement or 'col' not in tile_placement:
            return False
        
        row, col = tile_placement['row'], tile_placement['col']
        if not (0 <= row < 15 and 0 <= col < 15):
            return False
        
        if not current_game.board.is_empty(row, col):
            return False
    
    return True


def _apply_move(move):
    """Apply a move to the game state."""
    score = 0
    
    for tile_placement in move['tiles']:
        letter = tile_placement['letter']
        row = tile_placement['row']
        col = tile_placement['col']
        
        # Create tile
        tile = Tile(letter)
        
        # Place on board
        current_game.board.place_tile(tile, row, col)
        
        # Add to score
        score += tile.score
        
        # Remove from rack
        current_rack = current_game.get_current_rack()
        for i, rack_tile in enumerate(current_rack):
            if rack_tile.letter == letter:
                current_rack.pop(i)
                break
        
        # Draw new tile
        if current_game.tile_bag.tiles_remaining() > 0:
            new_tile = current_game.tile_bag.draw_tiles(1)[0]
            current_rack.append(new_tile)
    
    return score


def _convert_move_to_web(move):
    """Convert AI move to web format."""
    web_move = {
        'tiles': [],
        'word': move.main_word,
        'direction': move.direction.value,
        'score': move.total_score
    }
    
    for move_tile in move.tiles:
        if move_tile.is_new:
            web_move['tiles'].append({
                'letter': move_tile.tile.letter,
                'row': move_tile.position.row,
                'col': move_tile.position.col
            })
    
    return web_move


def _generate_simple_suggestions(letters):
    """Generate simple move suggestions."""
    # Simple word suggestions based on common patterns
    suggestions = []
    
    # Try to form simple words
    word_patterns = [
        'CAT', 'DOG', 'HOUSE', 'WORD', 'GAME', 'PLAY', 'BOARD',
        'TILE', 'SCORE', 'POINT', 'LETTER', 'QUICK', 'BROWN',
        'FOX', 'MOUSE', 'BOOK', 'PEN', 'PAPER', 'NOTE', 'MUSIC'
    ]
    
    for word in word_patterns:
        if _can_form_word(letters, word):
            score = sum(_get_letter_score(letter) for letter in word)
            suggestions.append({
                'word': word,
                'score': score,
                'letters_used': word
            })
    
    # Sort by score
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    
    return suggestions[:5]  # Return top 5 suggestions


def _can_form_word(letters, word):
    """Check if a word can be formed with given letters."""
    available_letters = list(letters)
    
    for letter in word:
        if letter in available_letters:
            available_letters.remove(letter)
        else:
            return False
    
    return True


def _get_letter_score(letter):
    """Get score for a letter."""
    scores = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
        'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
        'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
        'Y': 4, 'Z': 10
    }
    return scores.get(letter, 0)


if __name__ == '__main__':
    # Initialize AI
    initialize_ai()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
