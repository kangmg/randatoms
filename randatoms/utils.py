"""
Utility functions for the Random Atoms package.
"""

from typing import List, Dict, Any, Tuple
from ase import Atoms
import os
import tarfile
import pickle

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


def available_datasets(data_dir=None):
    """
    Lists available datasets and their summary statistics.

    Scans the dataset directory for .tar archives and prints a summary
    of each dataset, including the number of structures and file size.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'dataset')

    if not os.path.exists(data_dir):
        print(f"Dataset directory not found at: {data_dir}")
        return

    dataset_files = [f for f in os.listdir(data_dir) if f.endswith('.tar')]

    if not dataset_files:
        print("No datasets found.")
        return

    print("\033[1;34mAvailable Datasets\033[0m")
    print(f"Dataset found : {os.path.abspath(data_dir)}")

    summaries = []

    for tar_name in sorted(dataset_files):
        dataset_name = tar_name.replace('.tar', '')
        tar_path = os.path.join(data_dir, tar_name)

        try:
            file_size_bytes = os.path.getsize(tar_path)
            file_size_mb = file_size_bytes / (1024 * 1024) # Convert bytes to MB

            with tarfile.open(tar_path, 'r') as tar:
                pkl_member = next((m for m in tar.getmembers() if m.name.endswith('.pkl')), None)

                if pkl_member:
                    with tar.extractfile(pkl_member) as f:
                        metadata = pickle.load(f)
                        df = metadata['dataframe']
                        num_structures = len(df)

                        summaries.append({
                            "name": dataset_name,
                            "structures": num_structures,
                            "mw_range": (
                                df['molecular_weight'].min(),
                                df['molecular_weight'].max()
                            ),
                            "atoms_range": (
                                df['n_atoms'].min(),
                                df['n_atoms'].max()
                            ),
                            "size_mb": file_size_mb
                        })
                else:
                    summaries.append({
                        "name": dataset_name,
                        "structures": "N/A (no metadata)",
                        "mw_range": "N/A",
                        "atoms_range": "N/A",
                        "size_mb": file_size_mb
                    })

        except Exception as e:
            summaries.append({
                "name": dataset_name,
                "structures": f"Error: {e}",
                "mw_range": "N/A",
                "atoms_range": "N/A",
                "size_mb": "N/A"
            })

    if summaries:
        name_width     = 18
        struct_width   = 12
        mw_width       = 20
        atoms_width    = 18
        size_width     = 10

        # Header
        print("=" * (name_width + struct_width + mw_width + atoms_width + size_width + 11))
        print(f"{'Dataset Name':^{name_width}} | {'Structures':^{struct_width}} | {'MW Range':^{mw_width}} | {'Atoms Range':^{atoms_width}} | {'Size (MB)':^{size_width}}")
        print("-" * (name_width + struct_width + mw_width + atoms_width + size_width + 11))

        # Rows
        for s in summaries:
            size_str = f"{s['size_mb']:.2f}" if isinstance(s['size_mb'], (int, float)) else "N/A"
            if isinstance(s['structures'], int):
                mw_range_str = f"({s['mw_range'][0]:.1f}, {s['mw_range'][1]:.1f})"
                atoms_range_str = f"({s['atoms_range'][0]}, {s['atoms_range'][1]})"
                print(f"{s['name'][:name_width]:^{name_width}} | "
                      f"{s['structures']:^{struct_width},} | "
                      f"{mw_range_str:^{mw_width}} | "
                      f"{atoms_range_str:^{atoms_width}} | "
                      f"{size_str:^{size_width}}")
            else:
                print(f"{s['name'][:name_width]:^{name_width}} | "
                      f"{s['structures']:^{struct_width}} | "
                      f"{'N/A':^{mw_width}} | "
                      f"{'N/A':^{atoms_width}} | "
                      f"{size_str:^{size_width}}")
        print("=" * (name_width + struct_width + mw_width + atoms_width + size_width + 11))

