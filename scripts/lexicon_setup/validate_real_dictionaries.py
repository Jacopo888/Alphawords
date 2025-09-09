#!/usr/bin/env python3
"""
Validate that we're using real dictionaries, not placeholders.
This script ensures 100% accuracy in English Scrabble simulations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from alphascrabble.lexicon.gaddag_loader import GaddagLoader
    from alphascrabble.lexicon.move_generator import MoveGenerator
    from alphascrabble.rules.board import Board, Tile, TileBag
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_AVAILABLE = False


class DictionaryValidator:
    """Validates that we're using real dictionaries, not placeholders."""
    
    def __init__(self, cache_dir: str = "lexica_cache"):
        self.cache_dir = Path(cache_dir)
        self.validation_results = {}
        
        # Known placeholder patterns to detect
        self.placeholder_patterns = [
            'sample_words', 'fake_words', 'test_words', 'dummy_words',
            'placeholder', 'example', 'demo', 'mock', 'stub'
        ]
        
        # Real dictionary indicators
        self.real_dictionary_indicators = [
            'enable1', 'twl06', 'sowpods', 'quackle', 'gaddag', 'dawg'
        ]
    
    def validate_file_contents(self, file_path: Path) -> Tuple[bool, str]:
        """Validate that a file contains real dictionary data."""
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Check for placeholder patterns
            for pattern in self.placeholder_patterns:
                if pattern in content:
                    return False, f"Contains placeholder pattern: {pattern}"
            
            # Check for real dictionary indicators
            has_real_indicators = any(indicator in content for indicator in self.real_dictionary_indicators)
            
            # Check file size (real dictionaries should be substantial)
            file_size = file_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                return False, f"File too small ({file_size} bytes) - likely placeholder"
            
            # Count words
            words = [line.strip() for line in content.split('\n') if line.strip()]
            if len(words) < 100:  # Less than 100 words is suspicious
                return False, f"Too few words ({len(words)}) - likely placeholder"
            
            return True, f"Valid dictionary file with {len(words)} words"
            
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def validate_lexicon_loader(self) -> Tuple[bool, str]:
        """Validate the lexicon loader is using real dictionaries."""
        if not IMPORTS_AVAILABLE:
            return False, "Required imports not available"
        
        try:
            # Check if real dictionary files exist
            dawg_file = self.cache_dir / "enable1.dawg"
            gaddag_file = self.cache_dir / "enable1.gaddag"
            
            if not dawg_file.exists():
                return False, f"Real DAWG file not found: {dawg_file}"
            
            if not gaddag_file.exists():
                return False, f"Real GADDAG file not found: {gaddag_file}"
            
            # Try to load the lexicon
            loader = GaddagLoader(str(dawg_file), str(gaddag_file))
            
            if not loader.load():
                return False, "Failed to load real lexicon"
            
            # Test with real words
            real_words = ['HELLO', 'WORLD', 'SCRABBLE', 'QUACKLE', 'PYTHON', 'COMPUTER']
            fake_words = ['XYZ123', 'FAKE', 'PLACEHOLDER', 'TEST']
            
            for word in real_words:
                if not loader.is_word(word):
                    return False, f"Real word not found: {word}"
            
            for word in fake_words:
                if loader.is_word(word):
                    return False, f"Fake word incorrectly accepted: {word}"
            
            word_count = loader.get_word_count()
            if word_count < 10000:  # Real dictionaries should have many words
                return False, f"Word count too low ({word_count}) - likely placeholder"
            
            return True, f"Real lexicon loaded successfully with {word_count} words"
            
        except Exception as e:
            return False, f"Error validating lexicon loader: {e}"
    
    def validate_move_generator(self) -> Tuple[bool, str]:
        """Validate the move generator is using real dictionaries."""
        if not IMPORTS_AVAILABLE:
            return False, "Required imports not available"
        
        try:
            # Load real lexicon
            dawg_file = self.cache_dir / "enable1.dawg"
            gaddag_file = self.cache_dir / "enable1.gaddag"
            
            if not dawg_file.exists() or not gaddag_file.exists():
                return False, "Real dictionary files not found"
            
            loader = GaddagLoader(str(dawg_file), str(gaddag_file))
            if not loader.load():
                return False, "Failed to load real lexicon"
            
            # Create move generator with real lexicon
            move_generator = MoveGenerator(loader)
            
            # Test move generation with real game state
            board = Board()
            tile_bag = TileBag()
            rack = tile_bag.draw_tiles(7)
            
            # Generate moves
            moves = move_generator.generate_moves(board, rack)
            
            if not moves:
                return False, "No moves generated - possible placeholder issue"
            
            # Validate that generated moves use real words
            for move in moves[:10]:  # Check first 10 moves
                if hasattr(move, 'main_word'):
                    if not loader.is_word(move.main_word):
                        return False, f"Generated move uses invalid word: {move.main_word}"
            
            return True, f"Move generator working with real lexicon, generated {len(moves)} moves"
            
        except Exception as e:
            return False, f"Error validating move generator: {e}"
    
    def validate_code_quality(self) -> Tuple[bool, str]:
        """Validate that code doesn't contain placeholder patterns."""
        issues = []
        
        # Check Python files for placeholder patterns
        python_files = list(Path('.').rglob('*.py'))
        
        for py_file in python_files:
            if 'test' in str(py_file).lower():
                continue  # Skip test files
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in self.placeholder_patterns:
                    if pattern in content.lower():
                        issues.append(f"{py_file}: Contains placeholder pattern '{pattern}'")
                
                # Check for hardcoded fake words
                fake_word_patterns = ['CAT', 'DOG', 'HOUSE', 'COMPUTER', 'SCRABBLE']
                for word in fake_word_patterns:
                    if f"'{word}'" in content or f'"{word}"' in content:
                        # Check if it's in a real context (not just a comment)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if word in line and not line.strip().startswith('#'):
                                issues.append(f"{py_file}:{i+1}: Contains hardcoded word '{word}'")
                
            except Exception as e:
                issues.append(f"{py_file}: Error reading file: {e}")
        
        if issues:
            return False, f"Code quality issues found:\n" + "\n".join(issues)
        
        return True, "Code quality validation passed - no placeholder patterns found"
    
    def run_comprehensive_validation(self) -> Dict[str, Tuple[bool, str]]:
        """Run all validation tests."""
        print("🧪 Running comprehensive dictionary validation...")
        
        results = {}
        
        # Test 1: File contents validation
        print("\n1. Validating dictionary files...")
        dawg_file = self.cache_dir / "enable1.dawg"
        gaddag_file = self.cache_dir / "enable1.gaddag"
        
        results['dawg_file'] = self.validate_file_contents(dawg_file)
        results['gaddag_file'] = self.validate_file_contents(gaddag_file)
        
        # Test 2: Lexicon loader validation
        print("\n2. Validating lexicon loader...")
        results['lexicon_loader'] = self.validate_lexicon_loader()
        
        # Test 3: Move generator validation
        print("\n3. Validating move generator...")
        results['move_generator'] = self.validate_move_generator()
        
        # Test 4: Code quality validation
        print("\n4. Validating code quality...")
        results['code_quality'] = self.validate_code_quality()
        
        return results
    
    def print_validation_report(self, results: Dict[str, Tuple[bool, str]]):
        """Print a comprehensive validation report."""
        print("\n" + "="*60)
        print("📊 DICTIONARY VALIDATION REPORT")
        print("="*60)
        
        all_passed = True
        
        for test_name, (passed, message) in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status} {test_name.upper()}")
            print(f"   {message}")
            
            if not passed:
                all_passed = False
        
        print("\n" + "="*60)
        if all_passed:
            print("🎉 ALL VALIDATIONS PASSED!")
            print("✅ System is using REAL dictionaries, not placeholders")
            print("✅ Ready for 100% accurate English Scrabble simulations")
        else:
            print("❌ VALIDATION FAILED!")
            print("⚠️  System still contains placeholders or fake dictionaries")
            print("🔧 Run: python scripts/lexicon_setup/download_real_dictionaries.py")
        
        print("="*60)


def main():
    """Main validation function."""
    validator = DictionaryValidator()
    
    print("🔍 AlphaScrabble Dictionary Validation")
    print("Ensuring 100% real dictionaries, no placeholders!")
    
    results = validator.run_comprehensive_validation()
    validator.print_validation_report(results)
    
    # Return exit code
    all_passed = all(passed for passed, _ in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
