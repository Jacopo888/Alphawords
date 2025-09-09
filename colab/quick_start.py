#!/usr/bin/env python3
"""
Quick start script for AlphaScrabble in Google Colab.

This script provides a quick way to get started with AlphaScrabble
without going through the full setup process.
"""

import sys
import os
sys.path.append('alphascrabble')

def quick_start():
    """Quick start demonstration."""
    print("🚀 AlphaScrabble Quick Start")
    print("=" * 50)
    
    try:
        # Import core components
        from alphascrabble.rules.board import Board, GameState
        from alphascrabble.rules.tiles_en import Tile, TileBag
        from alphascrabble.nn.model import AlphaScrabbleNet
        from alphascrabble.engine.features import FeatureExtractor
        
        print("✅ Core components imported successfully!")
        
        # Create a simple game
        print("\n🎮 Creating a simple game...")
        board = Board()
        tile_bag = TileBag()
        players = ["Human", "Bot"]
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
        
        print(f"✅ Game created!")
        print(f"👤 Human rack: {' '.join(tile.display_letter for tile in racks[0])}")
        print(f"🤖 Bot rack: {' '.join(tile.display_letter for tile in racks[1])}")
        
        # Create neural network
        print("\n🧠 Creating neural network...")
        model = AlphaScrabbleNet()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Neural network created with {param_count:,} parameters")
        
        # Test feature extraction
        print("\n🔍 Testing feature extraction...")
        extractor = FeatureExtractor()
        board_features = extractor.extract_board_features(board, 0)
        rack_features = extractor.extract_rack_features(racks[0])
        
        print(f"✅ Board features: {board_features.shape}")
        print(f"✅ Rack features: {rack_features.shape}")
        
        # Simulate a move
        print("\n🎯 Simulating a move...")
        if racks[0]:
            tile = racks[0][0]
            board.place_tile(tile, 7, 7)  # Center square
            game_state.remove_tiles_from_rack([tile])
            game_state.add_score(0, tile.score)
            game_state.draw_tiles(1)
            
            print(f"✅ Placed '{tile.display_letter}' at center (score: {tile.score})")
            print(f"📊 Updated scores: {game_state.scores}")
        
        print("\n🎉 Quick start complete!")
        print("\nNext steps:")
        print("1. Run 'alphascrabble selfplay --games 10' to generate training data")
        print("2. Run 'alphascrabble train --data data/selfplay --epochs 5' to train the model")
        print("3. Run 'alphascrabble eval --net-a checkpoints/best_model.pt --opponent random' to evaluate")
        print("4. Run 'alphascrabble play --net checkpoints/best_model.pt' to play interactively")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run 'pip install -e .' to install AlphaScrabble")
        print("2. Check that all dependencies are installed")
        print("3. Try running the demo script: python colab/demo.py")
        return False

if __name__ == "__main__":
    success = quick_start()
    sys.exit(0 if success else 1)
