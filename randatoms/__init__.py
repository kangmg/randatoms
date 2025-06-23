"""
Random Atoms Package
====================

A package for handling datasets of atomic structures with tools for conversion, merging, and loading.

High-level API for easy access to random structure generation.
"""

from .dataloader import DataLoader
import os
import tarfile
import pickle
import pandas as pd

# High-level API for direct access
_loader = None

def randomatoms(*args, filename='default', data_dir=None, **kwargs):
    global _loader
    if _loader is None:
        _loader = DataLoader(filename=filename, data_dir=data_dir)
    return _loader.get_random_structures(*args, **kwargs)

def show_datasets(data_dir=None):
    """
    Lists available datasets and their summary statistics.

    Scans the dataset directory for .tar archives and prints a summary
    of each dataset, including the number of structures.
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
    print("=======================================================")
    
    summaries = []
    for tar_name in sorted(dataset_files):
        dataset_name = tar_name.replace('.tar', '')
        tar_path = os.path.join(data_dir, tar_name)
        try:
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
                            "mw_range": (df['molecular_weight'].min(), df['molecular_weight'].max()),
                            "atoms_range": (df['n_atoms'].min(), df['n_atoms'].max())
                        })
                else:
                    summaries.append({"name": dataset_name, "structures": "N/A (no metadata)", "mw_range": "N/A", "atoms_range": "N/A"})
        except Exception as e:
            summaries.append({"name": dataset_name, "structures": f"Error: {e}", "mw_range": "N/A", "atoms_range": "N/A"})

    if summaries:
        max_name_len = max(len(s['name']) for s in summaries)
        print(f"{'Dataset Name':<{max_name_len}} | {'Structures':>12} | {'MW Range':>18} | {'Atoms Range':>15}")
        print("-" * (max_name_len + 55))
        for s in summaries:
            if isinstance(s['structures'], int):
                mw_range_str = f"({s['mw_range'][0]:.1f}, {s['mw_range'][1]:.1f})"
                atoms_range_str = f"({s['atoms_range'][0]}, {s['atoms_range'][1]})"
                print(f"{s['name']:<{max_name_len}} | {s['structures']:>12,} | {mw_range_str:>18} | {atoms_range_str:>15}")
            else:
                print(f"{s['name']:<{max_name_len}} | {s['structures']:>12} | {'N/A':>18} | {'N/A':>15}")

    print("=======================================================")
