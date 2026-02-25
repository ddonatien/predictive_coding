"""
Predictive Coding Package

This package provides tools for predictive coding applications.
"""

__version__ = "0.1.0"
__author__ = "ddonatien"

# Import key modules here as the package grows
from .core import hello, add
from .utils import format_result
from .layers import BaseLayer, NeuronLayer, PredictiveCodingNetwork