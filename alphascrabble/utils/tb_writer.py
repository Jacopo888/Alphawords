"""
TensorBoard writer for AlphaScrabble.
"""

import os
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """TensorBoard writer for logging training metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize TensorBoard writer."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log an image."""
        self.writer.add_image(tag, image, step)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def close(self) -> None:
        """Close the writer."""
        self.writer.close()
