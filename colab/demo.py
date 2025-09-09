#!/usr/bin/env python3
"""
Demo script for AlphaScrabble in Google Colab.

This script demonstrates the key features of AlphaScrabble
without requiring the full setup.
"""

import sys
import os
sys.path.append('alphascrabble')

def demo_basic_components():
    """Demonstrate basic components."""
    print("🎯 AlphaScrabble Demo")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from alphascrabble.rules.board import Board, Tile
        from alphascrabble.rules.tiles_en import TileBag
        from alphascrabble.nn.model import AlphaScrabbleNet
        from alphascrabble.engine.features import FeatureExtractor
        print("✅ All imports successful!")
        
        # Test board creation
        print("\n🏗️  Testing board creation...")
        board = Board()
        print(f"✅ Board created: {len(board.grid)}x{len(board.grid[0])}")
        
        # Test tile bag
        print("\n🎲 Testing tile bag...")
        tile_bag = TileBag()
        print(f"✅ Tile bag created: {tile_bag.tiles_remaining()} tiles")
        
        # Test neural network
        print("\n🧠 Testing neural network...")
        model = AlphaScrabbleNet()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Neural network created: {param_count:,} parameters")
        
        # Test feature extraction
        print("\n🔍 Testing feature extraction...")
        extractor = FeatureExtractor()
        rack = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]
        rack_features = extractor.extract_rack_features(rack)
        print(f"✅ Feature extraction: {rack_features.shape}")
        
        print("\n🎉 All components working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def demo_game_play():
    """Demonstrate basic gameplay."""
    print("\n🎮 Gameplay Demo")
    print("=" * 50)
    
    try:
        from alphascrabble.rules.board import Board, GameState
        from alphascrabble.rules.tiles_en import Tile, TileBag
        
        # Create game
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
        
        print(f"✅ Game created with {len(players)} players")
        print(f"📊 Player 1 rack: {' '.join(tile.display_letter for tile in racks[0])}")
        print(f"📊 Player 2 rack: {' '.join(tile.display_letter for tile in racks[1])}")
        
        # Place a tile
        if racks[0]:
            tile = racks[0][0]
            board.place_tile(tile, 7, 7)  # Center square
            game_state.remove_tiles_from_rack([tile])
            game_state.add_score(0, tile.score)
            game_state.draw_tiles(1)
            
            print(f"✅ Placed '{tile.display_letter}' at center (score: {tile.score})")
            print(f"📊 Updated scores: {game_state.scores}")
        
        print("✅ Gameplay demo complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def demo_neural_network():
    """Demonstrate neural network functionality."""
    print("\n🧠 Neural Network Demo")
    print("=" * 50)
    
    try:
        import torch
        import numpy as np
        from alphascrabble.nn.model import AlphaScrabbleNet
        from alphascrabble.engine.features import FeatureExtractor
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        # Create model
        model = AlphaScrabbleNet().to(device)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create dummy data
        board_features = np.random.rand(32, 15, 15).astype(np.float32)
        rack_features = np.random.rand(27).astype(np.float32)
        move_features = np.random.rand(5, 64).astype(np.float32)
        
        # Convert to tensors
        board_tensor = torch.FloatTensor(board_features).unsqueeze(0).to(device)
        rack_tensor = torch.FloatTensor(rack_features).unsqueeze(0).to(device)
        move_tensor = torch.FloatTensor(move_features).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = model(board_tensor, rack_tensor, move_tensor)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Policy logits: {policy_logits.shape}")
        print(f"📊 Value: {value.item():.3f}")
        
        # Test prediction method
        policy_pred, value_pred = model.predict(board_features, rack_features, move_features)
        print(f"✅ Prediction method: policy {policy_pred.shape}, value {value_pred:.3f}")
        
        print("✅ Neural network demo complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def main():
    """Run all demos."""
    print("🚀 Starting AlphaScrabble Demo")
    print("=" * 60)
    
    demos = [
        ("Basic Components", demo_basic_components),
        ("Gameplay", demo_game_play),
        ("Neural Network", demo_neural_network),
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Demo Summary")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} demos passed")
    
    if total_passed == len(results):
        print("🎉 All demos passed! AlphaScrabble is working correctly.")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    return total_passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
