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
import tarfile
import io


class ASEtoHDF5Converter:
    """Convert ASE Atoms objects to HDF5 format with metadata indexing"""
    
    def __init__(self, chunk_size: int = 1000, n_workers: int = None, batch_size: int = 100):
        self.chunk_size = chunk_size
        self.n_workers = n_workers if n_workers else max(2, mp.cpu_count())
        self.batch_size = batch_size  # Process multiple structures per worker

    def convert_atoms_list(self, atoms_list: List[Atoms], filename: str,
                           data_dir: str = None, dataset_name: str = None, compress: bool = True):
        """Convert list of atoms to a single TAR file containing HDF5 and metadata."""
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        if dataset_name is None:
            dataset_name = filename

        output_tar_path = os.path.join(data_dir, f"{filename}.tar")

        print(f"Converting {len(atoms_list)} structures...")

        # Extract metadata
        metadata_list = self._extract_metadata(atoms_list, dataset_name)

        # In-memory buffers
        h5_buffer = io.BytesIO()
        pkl_buffer = io.BytesIO()

        # Write HDF5 data to buffer
        self._write_hdf5_optimized(atoms_list, h5_buffer, dataset_name, compress)

        # Save metadata to buffer
        print("Saving metadata...")
        self._save_metadata(metadata_list, pkl_buffer)

        # Create TAR archive
        print(f"Creating TAR archive at {output_tar_path}...")
        with tarfile.open(output_tar_path, 'w') as tar:
            # Add HDF5 data
            h5_buffer.seek(0)
            h5_info = tarfile.TarInfo(name=f"{filename}.h5")
            h5_info.size = len(h5_buffer.getvalue())
            tar.addfile(h5_info, h5_buffer)

            # Add pickle data
            pkl_buffer.seek(0)
            pkl_info = tarfile.TarInfo(name=f"{filename}.pkl")
            pkl_info.size = len(pkl_buffer.getvalue())
            tar.addfile(pkl_info, pkl_buffer)

        print(f"\033[1;34m\nConversion complete! File saved as {filename}.tar\033[0m")

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

    def _write_hdf5_optimized(self, atoms_list: List[Atoms], h5_buffer: io.BytesIO,
                             dataset_name: str, compress: bool):
        """Write atoms to an in-memory HDF5 buffer with optimized chunking."""
        compression = 'gzip' if compress else None
        compression_opts = 6 if compress else None

        with h5py.File(h5_buffer, 'w') as f:
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

    def _save_metadata(self, metadata_list: List[Dict], pkl_buffer: io.BytesIO):
        """Save metadata to an in-memory pickle buffer."""
        df = pd.DataFrame(metadata_list)

        # Efficiently build element index
        print("Building element index...")
        element_index = {}
        for i, metadata in enumerate(tqdm(metadata_list, desc="Processing elements")):
            for element in metadata['elements']:
                element_index.setdefault(element, set()).add(i)
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
        }

        pickle.dump(metadata_dict, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
