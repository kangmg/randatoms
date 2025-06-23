"""
DataLoader module for loading and filtering HDF5 molecular datasets.
"""

import h5py
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional, Tuple
from ase import Atoms
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import multiprocessing as mp
import importlib.resources as resources
import tarfile
import io


class DataLoader:
    """DatasetLoader for TAR-archived molecular datasets with filtering and indexing."""
    
    def __init__(self, filename: str = 'default', data_dir: str = None, n_workers: int = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        
        self.filename = filename
        self.tar_path = os.path.join(data_dir, f"{filename}.tar")
        self.n_workers = n_workers if n_workers else max(2, mp.cpu_count())
        self.h5_buffer = None

        print(f"Loading dataset from TAR archive: {self.tar_path}")
        
        try:
            with tarfile.open(self.tar_path, 'r') as tar:
                # Load metadata from .pkl file
                pkl_member = next((m for m in tar.getmembers() if m.name.endswith('.pkl')), None)
                if not pkl_member:
                    raise FileNotFoundError("No .pkl file found in the TAR archive.")
                
                with tar.extractfile(pkl_member) as f:
                    self.metadata_dict = pickle.load(f)

                # Load .h5 file into an in-memory buffer
                h5_member = next((m for m in tar.getmembers() if m.name.endswith('.h5')), None)
                if not h5_member:
                    raise FileNotFoundError("No .h5 file found in the TAR archive.")

                with tar.extractfile(h5_member) as f:
                    self.h5_buffer = io.BytesIO(f.read())

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset archive not found at {self.tar_path}")
        except tarfile.ReadError:
            raise IOError(f"Could not read TAR archive at {self.tar_path}")

        self.df = self.metadata_dict['dataframe']
        self.element_index = self.metadata_dict.get('element_index', {})
        self.mw_sorted_indices = self.metadata_dict.get('mw_sorted_indices', [])
        self.statistics = self.metadata_dict.get('statistics', {})
        
        self.element_index = {k: set(v) for k, v in self.element_index.items()}
        
        print(f"Loaded dataset with {len(self.df)} structures")

    def filter_indices(self, include_elements: Optional[List[str]] = None,
                      exclude_elements: Optional[List[str]] = None,
                      mw_range: Optional[Tuple[float, float]] = None,
                      max_atoms: Optional[int] = None,
                      min_atoms: Optional[int] = None,
                      is_periodic: Optional[bool] = None,
                      has_metals: Optional[bool] = None,
                      include_datasets: Optional[List[str]] = None) -> List[int]:
        """Fast index filtering using precomputed indices with additional filters"""
        valid_indices = set(self.df.index)

        # Element filters (most selective first)
        if include_elements:
            include_set = set()
            for element in include_elements:
                if element in self.element_index:
                    if not include_set:
                        include_set = self.element_index[element].copy()
                    else:
                        include_set &= self.element_index[element]
                else:
                    return []  # Element not found, no matches possible
            valid_indices &= include_set

        if exclude_elements:
            for element in exclude_elements:
                if element in self.element_index:
                    valid_indices -= self.element_index[element]

        # Apply remaining filters using pandas boolean indexing for efficiency
        if valid_indices:  # Only if we still have candidates
            df_subset = self.df.loc[list(valid_indices)]
            
            masks = []
            if mw_range:
                min_mw, max_mw = mw_range
                masks.append((df_subset['molecular_weight'] >= min_mw) & 
                            (df_subset['molecular_weight'] <= max_mw))
            
            if max_atoms:
                masks.append(df_subset['n_atoms'] <= max_atoms)
                
            if min_atoms:
                masks.append(df_subset['n_atoms'] >= min_atoms)
            
            if is_periodic is not None:
                masks.append(df_subset['is_periodic'] == is_periodic)
            
            if has_metals is not None:
                masks.append(df_subset['has_metals'] == has_metals)

            if include_datasets:
                if 'dataset' in df_subset.columns:
                    masks.append(df_subset['dataset'].isin(include_datasets))
                else:
                    raise ValueError('`dataset` keyword is missing in metadata')
            # Apply all masks efficiently
            if masks:
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask &= mask
                valid_indices = set(df_subset[combined_mask].index)

        return sorted(list(valid_indices))

    def load_structures(self, indices: List[int], show_progress: bool = True) -> List[Atoms]:
        """Load structures by indices with optimized parallel processing from in-memory HDF5."""
        if not indices:
            return []
        
        if len(indices) == 1:
            return [self._load_single(indices[0])]

        batch_size = max(1, min(50, len(indices) // self.n_workers))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        def load_batch(batch_indices):
            structures = []
            with h5py.File(self.h5_buffer, 'r') as f:
                for idx in batch_indices:
                    structures.append(self._load_from_h5(f, idx))
            return structures

        iterator = batches
        if show_progress and len(batches) > 1:
            iterator = tqdm(batches, desc=f"Loading {len(indices)} structures")

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            if show_progress and len(batches) > 1:
                batch_results = []
                for batch in iterator:
                    batch_results.append(executor.submit(load_batch, batch))
                batch_results = [future.result() for future in batch_results]
            else:
                batch_results = list(executor.map(load_batch, batches))

        # Flatten results
        return [structure for batch in batch_results for structure in batch]

    def _load_single(self, idx: int) -> Atoms:
        """Load a single structure from the in-memory HDF5 buffer."""
        with h5py.File(self.h5_buffer, 'r') as f:
            return self._load_from_h5(f, idx)

    def _load_from_h5(self, h5_file, idx: int) -> Atoms:
        """Load Atoms object from HDF5 group"""
        key = self.df.iloc[idx]['key']
        group = h5_file[key]

        positions = group['positions'][:]
        symbols = [s.decode('utf-8') for s in group['symbols'][:]]
        
        cell = group['cell'][:] if 'cell' in group else None
        pbc = group['pbc'][:] if 'pbc' in group else False

        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    def get_random_structures(self, n_structures: int = 1, seed: int = None, **filter_kwargs) -> List[Atoms]:
        """
        Randomly select one or more molecular structures that match specified filter criteria.

        This method allows for reproducible and filtered selection of molecular structures
        from a dataset. The selection is performed using a random seed, and only structures
        that match the filter criteria will be considered.

        Parameters
        ----------
        n_structures : int, optional (default=1)
            The number of random structures to return. If more structures are requested than
            available after filtering, all valid structures will be returned.

        seed : int, optional
            A random seed for reproducibility. If None, the random selection will be non-deterministic.

        **filter_kwargs
            - include_elements : list of str, optional
            - exclude_elements : list of str, optional
            - mw_range : tuple of float, optional
                A tuple specifying the allowed range of molecular weight (min, max). Only structures within this range are retained.
            - max_atoms : int, optional
                Maximum number of atoms allowed in a structure.
            - min_atoms : int, optional
                Minimum number of atoms required in a structure.
            - is_periodic : bool, optional
            - has_metals : bool, optional
            - include_datasets : list of str, optional
                A list of dataset names to include. Only structures from these datasets will be considered.

        Returns
        -------
        List[Atoms] or Atoms
        """
        if seed is not None:
            np.random.seed(seed)
            
        valid_indices = self.filter_indices(**filter_kwargs)
        
        if not valid_indices:
            raise ValueError("No structures match the criteria")

        n_structures = min(n_structures, len(valid_indices))
        selected_indices = np.random.choice(valid_indices, n_structures, replace=False)
        
        structures = self.load_structures(selected_indices.tolist())
        return structures if n_structures > 1 else structures[0]

    def get_filtered_statistics(self, **filter_kwargs) -> Dict[str, Any]:
        """Get statistics for filtered dataset"""
        valid_indices = self.filter_indices(**filter_kwargs)
        if not valid_indices:
            return {'count': 0}

        filtered_df = self.df.iloc[valid_indices]
        
        stats = {
            'count': len(valid_indices),
            'percentage': len(valid_indices) / len(self.df) * 100,
            'mw_range': (float(filtered_df['molecular_weight'].min()), 
                        float(filtered_df['molecular_weight'].max())),
            'avg_atoms': float(filtered_df['n_atoms'].mean()),
            'atoms_range': (int(filtered_df['n_atoms'].min()), 
                           int(filtered_df['n_atoms'].max())),
            'periodic_ratio': float(filtered_df['is_periodic'].mean()),
            'has_metals_ratio': float(filtered_df['has_metals'].mean()),
        }
        
        # Add element composition
        element_counts = {}
        for idx in valid_indices:
            # Use set to count each element only once per structure
            for element in set(self.df.iloc[idx]['elements']):
                element_counts[element] = element_counts.get(element, 0) + 1
        
        stats['element_coverage'] = dict(sorted(element_counts.items(), 
                                                   key=lambda x: x[1], reverse=True))
        
        return stats

    def print_statistics(self, **filter_kwargs):
        """Print formatted statistics"""

        def format_range(val1, val2, precision=1):
            return f"({val1:.{precision}f}, {val2:.{precision}f})"

        # if not filter_kwargs:
        #     raise ValueError("At least one filter keyword argument must be provided.")

        stats = self.get_filtered_statistics(**filter_kwargs)

        if stats['count'] == 0:
            print("No structures match the criteria")
            return
        labels = [
            "Total structures",
            "Percentage of dataset",
            "Molecular weight range",
            "Average atoms per structure",
            "Num. of atoms range",
            "Periodic structures",
            "Structures with metals"
        ]

        max_label_len = max(len(label) for label in labels) + 1  # +1 for colon
        gap = 2  

        value_width = 20

        count_str = f"{stats['count']:,}"
        percentage_str = f"{stats['percentage']:.1f} %"
        mw_range_str = format_range(stats['mw_range'][0], stats['mw_range'][1])
        avg_atoms_str = f"{stats['avg_atoms']:.1f}"
        atoms_range_str = format_range(stats['atoms_range'][0], stats['atoms_range'][1], 0)
        periodic_str = f"{stats['periodic_ratio'] * 100:.1f} %"
        metals_str = f"{stats['has_metals_ratio'] * 100:.1f} %"

        contents = [
            f"\n\033[1;34mDataset Statistics\033[0m",
            f"=======================================================",
            f"* {labels[0] + ':':<{max_label_len}}{' ' * gap}{count_str:>{value_width}}",
            f"* {labels[1] + ':':<{max_label_len}}{' ' * gap}{percentage_str:>{value_width}}",
            f"* {labels[2] + ':':<{max_label_len}}{' ' * gap}{mw_range_str:>{value_width}}",
            f"* {labels[3] + ':':<{max_label_len}}{' ' * gap}{avg_atoms_str:>{value_width}}",
            f"* {labels[4] + ':':<{max_label_len}}{' ' * gap}{atoms_range_str:>{value_width}}",
            f"* {labels[5] + ':':<{max_label_len}}{' ' * gap}{periodic_str:>{value_width}}",
            f"* {labels[6] + ':':<{max_label_len}}{' ' * gap}{metals_str:>{value_width}}",
            f"======================================================="
        ]


        print('\n'.join(contents))

                
        if 'element_coverage' in stats:
            print("\n\033[1;34mElemental Coverage in Dataset\033[0m")
            print("=======================================================")

            total_count = stats['count']
            bar_length = 10

            elements = list(stats['element_coverage'].items())

            max_elem_len = max(len(elem) for elem, _ in elements)
            max_count_len = max(len(f"{count:,}") for _, count in elements)
            max_percent_len = 5 

            for elem, count in elements:
                percentage = count / total_count * 100
                filled_length = int(round(bar_length * percentage / 100))
                bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'

                elem_fmt = f"{elem:<{max_elem_len}}"
                count_fmt = f"{count:>{max_count_len},}"
                percent_fmt = f"{percentage:>4.1f} %"

                print(f"{elem_fmt}: {count_fmt} structures {bar} {percent_fmt}")
            print("=======================================================")
