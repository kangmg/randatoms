"""
Utility functions for the Random Atoms package.
"""

from typing import List, Dict, Any, Tuple
from ase import Atoms


def extract_metadata_worker(args_tuple: Tuple[int, Atoms, str]) -> Dict[str, Any]:
    """Global worker function for multiprocessing - extracts metadata from single atoms object"""
    idx, atoms, dataset_name = args_tuple
    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    
    from .constants import METALS
    
    return {
        'index': idx,
        'dataset': dataset_name,
        'key': f"{dataset_name}_{idx:06d}",
        'molecular_weight': float(atoms.get_masses().sum()),
        'elements': unique_elements,
        'n_atoms': len(atoms),
        'formula': atoms.get_chemical_formula(),
        'is_periodic': any(atoms.pbc),
        'has_metals': any(e in METALS for e in unique_elements),
    }


def extract_metadata_batch_worker(args_tuple: Tuple[List[Tuple[int, Atoms, str]]]) -> List[Dict[str, Any]]:
    """Batch worker for processing multiple structures in one process"""
    batch_args = args_tuple[0]  # Unpack the single argument
    results = []
    for args in batch_args:
        results.append(extract_metadata_worker(args))
    return results
