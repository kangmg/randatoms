"""
Converter module for transforming ASE Atoms objects to HDF5 format with metadata indexing.
"""

import h5py
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any
from ase import Atoms
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
import multiprocessing as mp
import pkgutil
import importlib.resources as resources


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
            # Use package resources to locate the dataset directory
            with resources.path('randatoms.dataset', 'default.pkl') as default_path:
                data_dir = os.path.dirname(default_path)
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
        from .utils import extract_metadata_worker, extract_metadata_batch_worker
        
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
