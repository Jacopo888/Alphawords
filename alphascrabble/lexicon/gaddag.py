"""
GADDAG (Generalized Augmented Directed Acyclic Graph) implementation for Scrabble move generation.

GADDAG is a data structure that efficiently represents all possible words and their
substrings, allowing for fast move generation in Scrabble.
"""

from typing import List, Dict, Set, Tuple, Optional, Iterator
from collections import defaultdict
import pickle
import os


class GADDAGNode:
    """Node in the GADDAG structure."""
    
    def __init__(self):
        self.children: Dict[str, 'GADDAGNode'] = {}
        self.is_end_of_word: bool = False
        self.word_endings: Set[str] = set()


class GADDAG:
    """GADDAG data structure for efficient Scrabble move generation."""
    
    def __init__(self):
        self.root = GADDAGNode()
        self.word_count = 0
        self.max_word_length = 0
    
    def add_word(self, word: str) -> None:
        """Add a word to the GADDAG."""
        if not word or not word.isalpha():
            return
        
        word = word.upper()
        self.word_count += 1
        self.max_word_length = max(self.max_word_length, len(word))
        
        # Add all possible prefixes and suffixes
        for i in range(len(word) + 1):
            # Create the GADDAG path for this position
            if i == 0:
                # Full word
                path = word
            else:
                # Split word and reverse first part
                prefix = word[:i][::-1]  # Reverse prefix
                suffix = word[i:]
                path = prefix + '#' + suffix
            
            self._add_path(path, word)
    
    def _add_path(self, path: str, original_word: str) -> None:
        """Add a path to the GADDAG."""
        node = self.root
        
        for char in path:
            if char not in node.children:
                node.children[char] = GADDAGNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.word_endings.add(original_word)
    
    def contains_word(self, word: str) -> bool:
        """Check if a word exists in the GADDAG."""
        if not word or not word.isalpha():
            return False
        
        word = word.upper()
        
        # Try full word first
        if self._search_path(word):
            return True
        
        # Try all possible splits
        for i in range(1, len(word)):
            prefix = word[:i][::-1]
            suffix = word[i:]
            path = prefix + '#' + suffix
            
            if self._search_path(path):
                return True
        
        return False
    
    def _search_path(self, path: str) -> bool:
        """Search for a specific path in the GADDAG."""
        node = self.root
        
        for char in path:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end_of_word
    
    def get_anagrams(self, letters: str) -> List[str]:
        """Get all anagrams of the given letters."""
        if not letters:
            return []
        
        letters = letters.upper()
        anagrams = set()
        
        # Try all possible combinations
        self._find_anagrams(self.root, letters, "", anagrams)
        
        return sorted(list(anagrams))
    
    def _find_anagrams(self, node: GADDAGNode, remaining_letters: str, 
                      current_path: str, anagrams: Set[str]) -> None:
        """Recursively find anagrams."""
        if node.is_end_of_word and current_path:
            anagrams.add(current_path)
        
        if not remaining_letters:
            return
        
        # Try each remaining letter
        for i, letter in enumerate(remaining_letters):
            if letter in node.children:
                new_remaining = remaining_letters[:i] + remaining_letters[i+1:]
                new_path = current_path + letter
                self._find_anagrams(node.children[letter], new_remaining, new_path, anagrams)
    
    def get_words_with_letters(self, letters: str, min_length: int = 2) -> List[str]:
        """Get all words that can be formed with the given letters."""
        if not letters:
            return []
        
        letters = letters.upper()
        words = set()
        
        # Try all possible combinations
        self._find_words_with_letters(self.root, letters, "", words, min_length)
        
        return sorted([w for w in words if len(w) >= min_length])
    
    def _find_words_with_letters(self, node: GADDAGNode, remaining_letters: str,
                                current_path: str, words: Set[str], min_length: int) -> None:
        """Recursively find words that can be formed with given letters."""
        if node.is_end_of_word and len(current_path) >= min_length:
            words.add(current_path)
        
        if not remaining_letters:
            return
        
        # Try each remaining letter
        for i, letter in enumerate(remaining_letters):
            if letter in node.children:
                new_remaining = remaining_letters[:i] + remaining_letters[i+1:]
                new_path = current_path + letter
                self._find_words_with_letters(node.children[letter], new_remaining, 
                                            new_path, words, min_length)
    
    def get_extensions(self, word: str) -> List[str]:
        """Get all words that can be formed by extending the given word."""
        if not word or not word.isalpha():
            return []
        
        word = word.upper()
        extensions = set()
        
        # Try extending from the end
        self._find_extensions(self.root, word, "", extensions)
        
        return sorted(list(extensions))
    
    def _find_extensions(self, node: GADDAGNode, word: str, 
                        current_path: str, extensions: Set[str]) -> None:
        """Recursively find word extensions."""
        if not word:
            # We've consumed the word, now find all possible extensions
            if node.is_end_of_word and current_path:
                extensions.add(current_path)
            
            # Continue extending
            for char, child in node.children.items():
                if char != '#':
                    self._find_extensions(child, "", current_path + char, extensions)
            return
        
        # Try to match the current character
        if word[0] in node.children:
            self._find_extensions(node.children[word[0]], word[1:], 
                                current_path + word[0], extensions)
    
    def save(self, filepath: str) -> None:
        """Save the GADDAG to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'GADDAG':
        """Load a GADDAG from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the GADDAG."""
        return {
            'word_count': self.word_count,
            'max_word_length': self.max_word_length,
            'total_nodes': self._count_nodes(self.root)
        }
    
    def _count_nodes(self, node: GADDAGNode) -> int:
        """Count total nodes in the GADDAG."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


class WordListLoader:
    """Utility class for loading word lists into GADDAG."""
    
    @staticmethod
    def load_from_file(filepath: str) -> GADDAG:
        """Load words from a text file into a GADDAG."""
        gaddag = GADDAG()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word.isalpha():
                    gaddag.add_word(word)
        
        return gaddag
    
    @staticmethod
    def load_from_list(words: List[str]) -> GADDAG:
        """Load words from a list into a GADDAG."""
        gaddag = GADDAG()
        
        for word in words:
            if word and word.isalpha():
                gaddag.add_word(word)
        
        return gaddag
    
    @staticmethod
    def load_enable1() -> GADDAG:
        """Load the ENABLE1 word list."""
        # This would typically load from a file
        # For now, we'll create a small sample
        sample_words = [
            'CAT', 'DOG', 'HOUSE', 'COMPUTER', 'SCRABBLE',
            'WORD', 'GAME', 'PLAY', 'BOARD', 'TILE',
            'SCORE', 'POINT', 'LETTER', 'ALPHABET', 'DICTIONARY'
        ]
        
        return WordListLoader.load_from_list(sample_words)


# Example usage and testing
if __name__ == "__main__":
    # Create a GADDAG
    gaddag = GADDAG()
    
    # Add some words
    words = ['CAT', 'DOG', 'HOUSE', 'COMPUTER', 'SCRABBLE']
    for word in words:
        gaddag.add_word(word)
    
    # Test word lookup
    print("Testing word lookup:")
    print(f"CAT exists: {gaddag.contains_word('CAT')}")
    print(f"DOG exists: {gaddag.contains_word('DOG')}")
    print(f"XYZ exists: {gaddag.contains_word('XYZ')}")
    
    # Test anagrams
    print("\nTesting anagrams:")
    anagrams = gaddag.get_anagrams('TAC')
    print(f"Anagrams of 'TAC': {anagrams}")
    
    # Test words with letters
    print("\nTesting words with letters:")
    words_with_letters = gaddag.get_words_with_letters('CATDOG')
    print(f"Words with 'CATDOG': {words_with_letters}")
    
    # Test extensions
    print("\nTesting extensions:")
    extensions = gaddag.get_extensions('CAT')
    print(f"Extensions of 'CAT': {extensions}")
    
    # Print stats
    print(f"\nGADDAG stats: {gaddag.get_stats()}")
