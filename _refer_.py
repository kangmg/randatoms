import h5py
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional, Tuple
from ase import Atoms
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from tqdm import tqdm
import multiprocessing as mp


METALS = {'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr',
          'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs',
          'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
          'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
          'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th',
          'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
          'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'}


def extract_metadata_worker(args_tuple: Tuple[int, Atoms, str]) -> Dict[str, Any]:
    """Global worker function for multiprocessing - extracts metadata from single atoms object"""
    idx, atoms, dataset_name = args_tuple
    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    
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


class ASEtoHDF5Converter:
    """Convert ASE Atoms objects to HDF5 format with metadata indexing"""
    
    def __init__(self, chunk_size: int = 1000, n_workers: int = None, batch_size: int = 100):
        self.chunk_size = chunk_size
        self.n_workers = n_workers or min(4, mp.cpu_count())
        self.batch_size = batch_size  # Process multiple structures per worker

    def convert_atoms_list(self, atoms_list: List[Atoms], filename: str, 
                          data_dir: str = None, dataset_name: str = None, compress: bool = True):
        """Convert list of atoms to HDF5 with metadata"""
        if data_dir is None:
            data_dir = 'package_internal'
        if dataset_name is None:
            dataset_name = 'default'
            
        output_h5 = os.path.join(data_dir, f"{filename}.h5")
        metadata_path = os.path.join(data_dir, f"{filename}.pkl")

        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        print(f"Converting {len(atoms_list)} structures...")
        
        # Extract metadata with optimized parallel processing
        metadata_list = self._extract_metadata(atoms_list, dataset_name)
        
        self._write_hdf5_optimized(atoms_list, output_h5, dataset_name, compress)
        
        print("Saving metadata...")
        self._save_metadata(metadata_list, metadata_path)
        
        print(f"\033[1;34mConversion complete! Files saved as {filename}.h5 and {filename}.pkl\033[0m")

    def _extract_metadata(self, atoms_list: List[Atoms], dataset_name: str) -> List[Dict]:
        """Extract metadata using optimized parallel processing"""
        n_structures = len(atoms_list)
        
        # For small datasets, use sequential processing
        if n_structures < 100:
            return [extract_metadata_worker((i, atoms, dataset_name)) 
                   for i, atoms in enumerate(tqdm(atoms_list, desc="Extracting metadata"))]

        # For larger datasets, use batched multiprocessing
        print(f"Using {self.n_workers} workers for metadata extraction...")
        
        # Create batches for better load balancing
        batches = []
        for i in range(0, n_structures, self.batch_size):
            batch = [(j, atoms_list[j], dataset_name) 
                    for j in range(i, min(i + self.batch_size, n_structures))]
            batches.append((batch,))  # Wrap in tuple for multiprocessing
        
        # Process batches in parallel
        print()
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            batch_results = list(tqdm(
                executor.map(extract_metadata_batch_worker, batches),
                total=len(batches),
                desc="Processing batches"
            ))
        
        # Flatten results and sort by index to maintain order
        metadata_list = []
        for batch_result in batch_results:
            metadata_list.extend(batch_result)
        
        # Sort by index to ensure correct order
        metadata_list.sort(key=lambda x: x['index'])
        return metadata_list

    def _write_hdf5_optimized(self, atoms_list: List[Atoms], output_path: str, 
                             dataset_name: str, compress: bool):
        """Write atoms to HDF5 file with optimized chunking"""
        compression = 'gzip' if compress else None
        compression_opts = 6 if compress else None  # Good balance of speed/compression

        with h5py.File(output_path, 'w') as f:
            f.attrs.update({
                'dataset_name': dataset_name,
                'n_structures': len(atoms_list),
                'creation_date': pd.Timestamp.now().isoformat(),
            })

            # Process in chunks for better memory management
            for chunk_start in tqdm(range(0, len(atoms_list), self.chunk_size), 
                                  desc="Writing HDF5"):
                chunk_end = min(chunk_start + self.chunk_size, len(atoms_list))
                
                for i in range(chunk_start, chunk_end):
                    atoms = atoms_list[i]
                    group = f.create_group(f"{dataset_name}_{i:06d}")
                    
                    # Core data with optimized chunking
                    positions_shape = atoms.positions.shape
                    chunk_shape = (min(1000, positions_shape[0]), positions_shape[1])
                    
                    group.create_dataset(
                        'positions', 
                        data=atoms.positions, 
                        compression=compression,
                        compression_opts=compression_opts,
                        chunks=chunk_shape if positions_shape[0] > 100 else None
                    )
                    
                    symbols_encoded = [s.encode('utf-8') for s in atoms.get_chemical_symbols()]
                    group.create_dataset(
                        'symbols', 
                        data=symbols_encoded, 
                        compression=compression,
                        compression_opts=compression_opts
                    )

                    # Cell data
                    if atoms.cell is not None and not np.allclose(atoms.cell.array, 0):
                        group.create_dataset(
                            'cell', 
                            data=atoms.cell.array, 
                            compression=compression,
                            compression_opts=compression_opts
                        )
                        group.create_dataset('pbc', data=atoms.pbc)

    def _save_metadata(self, metadata_list: List[Dict], metadata_path: str):
        """Save metadata with indices and statistics - optimized for large datasets"""
        df = pd.DataFrame(metadata_list)
        
        # Build element index efficiently using vectorized operations
        print("Building element index...")
        element_index = {}
        
        # Vectorized approach for building element index
        for i, metadata in enumerate(tqdm(metadata_list, desc="Processing elements")):
            for element in metadata['elements']:
                if element not in element_index:
                    element_index[element] = set()
                element_index[element].add(i)
        
        # Convert sets to sorted lists for better serialization
        element_index = {k: sorted(list(v)) for k, v in element_index.items()}
        
        # Calculate statistics
        stats = {
            'total_structures': len(df),
            'mw_range': (float(df['molecular_weight'].min()), float(df['molecular_weight'].max())),
            'avg_atoms': float(df['n_atoms'].mean()),
            'max_atoms': int(df['n_atoms'].max()),
            'min_atoms': int(df['n_atoms'].min()),
            'periodic_ratio': float(df['is_periodic'].mean()),
            'has_metals_ratio': float(df['has_metals'].mean()),
            'unique_elements': sorted(element_index.keys()),
            'element_counts': {elem: len(indices) for elem, indices in element_index.items()}
        }

        metadata_dict = {
            'dataframe': df,
            'element_index': element_index,
            'mw_sorted_indices': df['molecular_weight'].argsort().values,
            'statistics': stats,
            'version': '2.0',  # Version for compatibility tracking
        }
        
        # Use highest protocol for better performance
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


class DatasetMerger:
    """Merge multiple HDF5 datasets into one with optimized performance"""
    
    def merge_datasets(self, data_dir: str, merge_name_list: List[str], output_name: str):
        """Merge multiple datasets into single HDF5 file with progress tracking"""
        output_h5 = os.path.join(data_dir, f"{output_name}.h5")
        output_metadata = os.path.join(data_dir, f"{output_name}.pkl")
        
        all_metadata = []
        current_index = 0
        total_structures = 0
        
        # First pass: count total structures
        print("Counting structures...")
        for name in merge_name_list:
            metadata_file = os.path.join(data_dir, f"{name}.pkl")
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                total_structures += len(metadata['dataframe'])
        
        print(f"Merging {total_structures} structures from {len(merge_name_list)} datasets...")

        with h5py.File(output_h5, 'w') as out_f:
            out_f.attrs['merged_datasets'] = merge_name_list
            out_f.attrs['total_structures'] = total_structures
            
            with tqdm(total=total_structures, desc="Merging structures") as pbar:
                for name in merge_name_list:
                    h5_file = os.path.join(data_dir, f"{name}.h5")
                    metadata_file = os.path.join(data_dir, f"{name}.pkl")
                    
                    # Load metadata
                    with open(metadata_file, 'rb') as f:
                        metadata_dict = pickle.load(f)
                        df = metadata_dict['dataframe']

                    # Copy HDF5 data with batch processing
                    with h5py.File(h5_file, 'r') as in_f:
                        for old_key in in_f.keys():
                            new_key = f"merged_{current_index:06d}"
                            
                            # Use HDF5's efficient copy
                            in_f.copy(old_key, out_f, new_key)

                            # Update metadata
                            old_idx = int(old_key.split('_')[-1])
                            row = df[df['index'] == old_idx].iloc[0].copy()
                            row['index'] = current_index
                            row['key'] = new_key
                            row['dataset'] = 'merged'
                            all_metadata.append(row.to_dict())
                            
                            current_index += 1
                            pbar.update(1)

        # Save merged metadata
        print("Saving merged metadata...")
        converter = ASEtoHDF5Converter()
        converter._save_metadata(all_metadata, output_metadata)
        print(f"\033[1;34mMerge complete! Output saved as {output_name}.h5 and {output_name}.pkl\033[0m")


class DatasetLoader:
    """DatasetLoader for HDF5 molecular datasets with filtering and indexing - optimized"""
    
    def __init__(self, filename: str, data_dir: str, n_workers: int = None):
        self.h5_path = os.path.join(data_dir, f"{filename}.h5")
        self.n_workers = n_workers or min(4, mp.cpu_count())
        
        # Load metadata
        metadata_path = os.path.join(data_dir, f"{filename}.pkl")
        print(f"Loading metadata from {metadata_path}...")
        
        with open(metadata_path, 'rb') as f:
            self.metadata_dict = pickle.load(f)

        self.df = self.metadata_dict['dataframe']
        self.element_index = self.metadata_dict['element_index']
        self.mw_sorted_indices = self.metadata_dict['mw_sorted_indices']
        self.statistics = self.metadata_dict['statistics']
        
        # Convert element index back to sets for faster operations
        self.element_index = {k: set(v) for k, v in self.element_index.items()}
        
        print(f"Loaded dataset with {len(self.df)} structures")

    def filter_indices(self, include_elements: Optional[List[str]] = None,
                      exclude_elements: Optional[List[str]] = None,
                      mw_range: Optional[Tuple[float, float]] = None,
                      max_atoms: Optional[int] = None,
                      min_atoms: Optional[int] = None,
                      is_periodic: Optional[bool] = None,
                      has_metals: Optional[bool] = None,
                      dataset_name: Optional[str] = None) -> List[int]:
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
            
            if dataset_name is not None:
                masks.append(df_subset['dataset'] == dataset_name)

            # Apply all masks efficiently
            if masks:
                combined_mask = masks[0]
                for mask in masks[1:]:
                    combined_mask &= mask
                valid_indices = set(df_subset[combined_mask].index)

        return sorted(list(valid_indices))

    def load_structures(self, indices: List[int], show_progress: bool = True) -> List[Atoms]:
        """Load structures by indices with optimized parallel processing"""
        if not indices:
            return []
        
        if len(indices) == 1:
            return [self._load_single(indices[0])]

        # Optimized batch loading with better load balancing
        batch_size = max(1, min(50, len(indices) // self.n_workers))
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        def load_batch(batch_indices):
            structures = []
            with h5py.File(self.h5_path, 'r') as f:
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
        """Load single structure"""
        with h5py.File(self.h5_path, 'r') as f:
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
        
        # Add element distribution
        element_counts = {}
        for idx in valid_indices:
            for element in self.df.iloc[idx]['elements']:
                element_counts[element] = element_counts.get(element, 0) + 1
        
        stats['element_distribution'] = dict(sorted(element_counts.items(), 
                                                   key=lambda x: x[1], reverse=True))
        
        return stats

    def print_statistics(self, **filter_kwargs):
        """Print formatted statistics"""

        def format_range(val1, val2, precision=1):
            return f"({val1:.{precision}f}, {val2:.{precision}f})"

        if not filter_kwargs:
            raise ValueError("At least one filter keyword argument must be provided.")

        stats = self.get_filtered_statistics(**filter_kwargs)

        if stats['count'] == 0:
            print("No structures match the criteria")
            return
        labels = [
            "Total structures",
            "Percentage of dataset",
            "Molecular weight range",
            "Average atoms per structure",
            "Atoms range",
            "Periodic structures",
            "Structures with metals"
        ]

        max_label_len = max(len(label) for label in labels) + 1  # +1 for colon
        gap = 2  

        value_width = 20

        count_str = f"{stats['count']:,}"
        percentage_str = f"{stats['percentage']:.1f}%"
        mw_range_str = format_range(stats['mw_range'][0], stats['mw_range'][1])
        avg_atoms_str = f"{stats['avg_atoms']:.1f}"
        atoms_range_str = format_range(stats['atoms_range'][0], stats['atoms_range'][1], 0)
        periodic_str = f"{stats['periodic_ratio'] * 100:.1f}%"
        metals_str = f"{stats['has_metals_ratio'] * 100:.1f}%"

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

                
        if 'element_distribution' in stats:
            print("\n\033[1;34mElement Distribution\033[0m")
            print("=======================================================")

            total_count = stats['count']
            bar_length = 10

            elements = list(stats['element_distribution'].items())

            max_elem_len = max(len(elem) for elem, _ in elements)
            max_count_len = max(len(f"{count:,}") for _, count in elements)
            max_percent_len = 5 

            for elem, count in elements:
                percentage = count / total_count * 100
                filled_length = int(round(bar_length * percentage / 100))
                bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'

                elem_fmt = f"{elem:<{max_elem_len}}"
                count_fmt = f"{count:>{max_count_len},}"
                percent_fmt = f"{percentage:>4.1f}%"

                print(f"{elem_fmt}: {count_fmt} structures {bar} {percent_fmt}")
            print("=======================================================")
