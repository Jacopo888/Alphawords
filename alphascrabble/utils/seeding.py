"""
Seeding utilities for reproducible results.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_state() -> Dict[str, Any]:
    """Get current random state."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def set_random_state(state: Dict[str, Any]) -> None:
    """Set random state."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available() and state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda'])
