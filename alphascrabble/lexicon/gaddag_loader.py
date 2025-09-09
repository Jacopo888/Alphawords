"""
GADDAG loader and move generation interface.

Provides Python interface to Quackle's C++ lexicon implementation.
"""

import os
import hashlib
from typing import List, Optional, Tuple
import numpy as np

try:
    import qlex
    QLEX_AVAILABLE = True
except ImportError:
    QLEX_AVAILABLE = False
    print("Warning: qlex module not available. Install with: pip install -e .")


class GaddagLoader:
    """Interface to Quackle's GADDAG and DAWG lexicons."""
    
    def __init__(self, dawg_path: str, gaddag_path: str):
        """Initialize with paths to lexicon files."""
        self.dawg_path = dawg_path
        self.gaddag_path = gaddag_path
        self.qlex_wrapper = None
        self._loaded = False
        
        if QLEX_AVAILABLE:
            self.qlex_wrapper = qlex.QLexWrapper()
    
    def load(self) -> bool:
        """Load the lexicon files."""
        if not QLEX_AVAILABLE:
            raise RuntimeError("qlex module not available. Please install with: pip install -e .")
        
        if not os.path.exists(self.dawg_path):
            raise FileNotFoundError(f"DAWG file not found: {self.dawg_path}")
        
        if not os.path.exists(self.gaddag_path):
            raise FileNotFoundError(f"GADDAG file not found: {self.gaddag_path}")
        
        try:
            self.qlex_wrapper.load_lexica(self.dawg_path, self.gaddag_path)
            self._loaded = True
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load lexica: {e}")
    
    def is_loaded(self) -> bool:
        """Check if lexicon is loaded."""
        return self._loaded and self.qlex_wrapper is not None and self.qlex_wrapper.is_loaded()
    
    def is_word(self, word: str) -> bool:
        """Check if a word is valid."""
        if not self.is_loaded():
            raise RuntimeError("Lexicon not loaded")
        
        return self.qlex_wrapper.is_word(word.upper())
    
    def generate_moves(self, board_state: str, rack: str) -> List[str]:
        """Generate legal moves for given board state and rack."""
        if not self.is_loaded():
            raise RuntimeError("Lexicon not loaded")
        
        return self.qlex_wrapper.generate_moves(board_state, rack.upper())
    
    def get_word_count(self) -> int:
        """Get total number of words in lexicon."""
        if not self.is_loaded():
            return 0
        
        return self.qlex_wrapper.get_word_count()
    
    def verify_lexica(self) -> Tuple[bool, str]:
        """Verify lexicon files are valid."""
        if not os.path.exists(self.dawg_path):
            return False, f"DAWG file not found: {self.dawg_path}"
        
        if not os.path.exists(self.gaddag_path):
            return False, f"GADDAG file not found: {self.gaddag_path}"
        
        try:
            if not self.is_loaded():
                self.load()
            
            word_count = self.get_word_count()
            if word_count == 0:
                return False, "Lexicon appears to be empty"
            
            # Test a few common words
            test_words = ["HELLO", "WORLD", "SCRABBLE", "QUACKLE"]
            for word in test_words:
                if not self.is_word(word):
                    return False, f"Common word '{word}' not found in lexicon"
            
            return True, f"Lexicon verified with {word_count} words"
        
        except Exception as e:
            return False, f"Error verifying lexicon: {e}"
    
    def get_file_info(self) -> dict:
        """Get information about lexicon files."""
        info = {}
        
        for path, name in [(self.dawg_path, "DAWG"), (self.gaddag_path, "GADDAG")]:
            if os.path.exists(path):
                stat = os.stat(path)
                with open(path, 'rb') as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                
                info[name] = {
                    "path": path,
                    "size": stat.st_size,
                    "md5": md5_hash,
                    "exists": True
                }
            else:
                info[name] = {
                    "path": path,
                    "exists": False
                }
        
        return info
