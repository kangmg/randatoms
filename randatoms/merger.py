"""
Merger module for combining multiple HDF5 datasets into a single dataset.
"""

import h5py
import os
import pickle
from typing import List
from tqdm import tqdm
import importlib.resources as resources
from .converter import ASEtoHDF5Converter


class DatasetMerger:
    """Merge multiple HDF5 datasets into one with optimized performance"""
    
    def merge_datasets(self, data_dir: str = None, merge_name_list: List[str] = None, output_name: str = 'merged'):
        """Merge multiple datasets into single HDF5 file with progress tracking"""
        if data_dir is None:
            # Use package resources to locate the dataset directory
            with resources.path('randatoms.dataset', 'default.pkl') as default_path:
                data_dir = os.path.dirname(default_path)
        if merge_name_list is None:
            merge_name_list = ['default']
            
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
