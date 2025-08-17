"""
A minimal neural network framework with automatic differentiation.
Built from scratch for educational purposes.
"""

from .object import Value
from .nn_scratch import Neural_Network
from .training import training

__version__ = "0.1.0"
__author__ = "Pritiyax Shukla"
__email__ = "spritiyax@gmail.com"

__all__ = [
    "Value",
    "Neural_Network", 
    "train_network"
]


