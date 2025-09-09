"""
Training module for AlphaScrabble.

Contains self-play training pipeline and related utilities.
"""

from .selfplay import SelfPlayGenerator, TrainingPipeline, GameRecord, TrainingExample

__all__ = [
    'SelfPlayGenerator', 'TrainingPipeline', 'GameRecord', 'TrainingExample'
]
