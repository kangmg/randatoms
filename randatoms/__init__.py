"""
Random Atoms Package
====================

A package for handling datasets of atomic structures with tools for conversion, merging, and loading.

High-level API for easy access to random structure generation.
"""

from .dataloader import DataLoader

# High-level API for direct access
randomatoms = DataLoader('default', data_dir=None).get_random_structures

__version__ = '0.1.0'
__all__ = ['DataLoader', 'randomatoms']
