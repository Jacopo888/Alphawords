#!/usr/bin/env python3
"""
Quick test to verify real English dictionaries are working.
NO MORE PLACEHOLDERS!
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_real_english_dictionaries():
    """Test that we're using real English dictionaries."""
    
    print("🧪 Testing Real English Dictionaries")
    print("=" * 50)
    print("❌ NO MORE PLACEHOLDERS")
    print("✅ REAL ENGLISH WORDS ONLY")
    print("")
    
    try:
        # Test 1: Check if real dictionary files exist
        print("1. Checking real dictionary files...")
        dawg_file = Path("lexica_cache/enable1.dawg")
        gaddag_file = Path("lexica_cache/enable1.gaddag")
        
        if not dawg_file.exists():
            print("❌ Real DAWG file not found!")
            print("   Run: ./scripts/lexicon_setup/setup_english_lexicon.sh")
            return False
        
        if not gaddag_file.exists():
            print("❌ Real GADDAG file not found!")
            print("   Run: ./scripts/lexicon_setup/setup_english_lexicon.sh")
            return False
        
        print("✅ Real dictionary files found")
        
        # Test 2: Try to import and load real lexicon
        print("\n2. Loading real lexicon...")
        try:
            from alphascrabble.lexicon.gaddag_loader import GaddagLoader
            
            loader = GaddagLoader(str(dawg_file), str(gaddag_file))
            if not loader.load():
                print("❌ Failed to load real lexicon!")
                return False
            
            word_count = loader.get_word_count()
            print(f"✅ Real lexicon loaded: {word_count} words")
            
            if word_count < 100000:
                print("❌ Word count too low - likely placeholder!")
                return False
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Run: pip install -e .")
            return False
        
        # Test 3: Test real English words
        print("\n3. Testing real English words...")
        real_words = ['HELLO', 'WORLD', 'SCRABBLE', 'QUACKLE', 'PYTHON', 'COMPUTER']
        
        for word in real_words:
            if loader.is_word(word):
                print(f"✅ Real word found: {word}")
            else:
                print(f"❌ Real word missing: {word}")
                return False
        
        # Test 4: Test fake words are rejected
        print("\n4. Testing fake words are rejected...")
        fake_words = ['XYZ123', 'FAKE', 'PLACEHOLDER', 'TEST', 'DUMMY']
        
        for word in fake_words:
            if not loader.is_word(word):
                print(f"✅ Fake word correctly rejected: {word}")
            else:
                print(f"❌ Fake word incorrectly accepted: {word}")
                return False
        
        # Test 5: Test move generation
        print("\n5. Testing move generation with real words...")
        try:
            from alphascrabble.lexicon.move_generator import MoveGenerator
            from alphascrabble.rules.board import Board, TileBag
            
            move_generator = MoveGenerator(loader)
            board = Board()
            tile_bag = TileBag()
            rack = tile_bag.draw_tiles(7)
            
            moves = move_generator.generate_moves(board, rack)
            
            if not moves:
                print("❌ No moves generated!")
                return False
            
            print(f"✅ Generated {len(moves)} moves")
            
            # Check first few moves use real words
            for i, move in enumerate(moves[:5]):
                if loader.is_word(move.main_word):
                    print(f"✅ Move {i+1}: {move.main_word} (real word)")
                else:
                    print(f"❌ Move {i+1}: {move.main_word} (fake word!)")
                    return False
        
        except Exception as e:
            print(f"❌ Move generation error: {e}")
            return False
        
        # All tests passed!
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Real English dictionaries working")
        print("✅ NO placeholders or fake dictionaries")
        print("✅ Ready for 100% accurate simulations")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_real_english_dictionaries()
    sys.exit(0 if success else 1)
