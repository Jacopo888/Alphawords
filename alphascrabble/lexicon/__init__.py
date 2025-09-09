"""
Lexicon module for AlphaScrabble.

Provides access to Quackle's DAWG and GADDAG lexicons for move generation.
"""

from .gaddag_loader import GaddagLoader

__all__ = ["GaddagLoader"]
