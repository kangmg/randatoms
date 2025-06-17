"""
Random Atoms Package
====================

A package for handling datasets of atomic structures with tools for conversion, merging, and loading.

High-level API for easy access to random structure generation.
"""

from .dataloader import DataLoader

# High-level API for direct access
_loader = None

def randomatoms(*args, **kwargs):
    global _loader
    if _loader is None:
        _loader = DataLoader('default', data_dir=None)
    return _loader.get_random_structures(*args, **kwargs)