"""
Merger module for combining multiple HDF5 datasets into a single dataset.
"""

import h5py
import os
import pickle
import pandas as pd
from typing import List
from tqdm import tqdm
import importlib.resources as resources
from .converter import ASEtoHDF5Converter
import warnings
import tarfile
import io

class DatasetMerger:
    """Merge multiple HDF5 datasets into one with optimized performance"""

    def __init__(self, merge_name_list: List[str], output_name: str, data_dir: str = None):
        self.merge_name_list = merge_name_list
        self.output_name = output_name        
        if not data_dir:
            # internal dataset path
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        else:
            self.data_dir = data_dir        
        

    def merge(self):
        """Merge multiple datasets into a single TAR file with progress tracking"""
        output_tar = os.path.join(self.data_dir, f"{self.output_name}.tar")

        all_metadata = []
        current_index = 0
        total_structures = 0

        # First pass: count total structures
        print("Counting structures...")
        for name in self.merge_name_list:
            tar_path = os.path.join(self.data_dir, f"{name}.tar")
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    pkl_member = next((m for m in tar.getmembers() if m.name.endswith('.pkl')), None)
                    if pkl_member:
                        with tar.extractfile(pkl_member) as f:
                            metadata = pickle.load(f)
                            total_structures += len(metadata['dataframe'])
                    else:
                        warnings.warn(f"Warning: No .pkl file found in {tar_path}. Skipping.")
            except FileNotFoundError:
                warnings.warn(f"Warning: Dataset archive not found at {tar_path}. Skipping.")

        print(f"Merging {total_structures} structures from {len(self.merge_name_list)} datasets...")

        keys_in_output = set()
        h5_buffer = io.BytesIO()
        with h5py.File(h5_buffer, 'w') as out_f:
            out_f.attrs['merged_datasets'] = self.merge_name_list
            out_f.attrs['total_structures'] = total_structures

            with tqdm(total=total_structures, desc="Merging structures") as pbar:
                for name in self.merge_name_list:
                    tar_path = os.path.join(self.data_dir, f"{name}.tar")
                    if not os.path.exists(tar_path):
                        continue

                    with tarfile.open(tar_path, 'r') as tar:
                        # Load metadata
                        pkl_member = next((m for m in tar.getmembers() if m.name.endswith('.pkl')), None)
                        if not pkl_member:
                            continue
                        with tar.extractfile(pkl_member) as f:
                            metadata_dict = pickle.load(f)
                            df = metadata_dict['dataframe']

                        # Load HDF5 data
                        h5_member = next((m for m in tar.getmembers() if m.name.endswith('.h5')), None)
                        if not h5_member:
                            continue
                        h5_file = tar.extractfile(h5_member)
                        h5_content = h5_file.read()

                        with h5py.File(io.BytesIO(h5_content), 'r') as in_f:
                            for old_key in in_f.keys():
                                if old_key in keys_in_output:
                                    new_key = f"{old_key}@{name}"
                                    warnings.warn(f"Warning: Duplicate key '{old_key}' from dataset '{name}'. Renaming to '{new_key}'.")
                                else:
                                    new_key = old_key

                                keys_in_output.add(new_key)
                                in_f.copy(old_key, out_f, new_key)

                                # Update metadata
                                row = df[df['key'] == old_key].iloc[0].copy()
                                row['index'] = current_index
                                row['key'] = new_key
                                row['dataset'] = name
                                all_metadata.append(row.to_dict())

                                current_index += 1
                                pbar.update(1)

        # Prepare metadata for saving
        merged_metadata_df = pd.DataFrame(all_metadata)
        
        print("Building merged element index...")
        element_index = {}
        for metadata_row in tqdm(all_metadata, desc="Building index"):
            idx = metadata_row['index']
            for element in metadata_row['elements']:
                element_index.setdefault(element, set()).add(idx)
        element_index = {k: sorted(list(v)) for k, v in element_index.items()}

        print("Calculating merged statistics...")
        stats = {
            'total_structures': len(merged_metadata_df),
            'mw_range': (float(merged_metadata_df['molecular_weight'].min()), float(merged_metadata_df['molecular_weight'].max())),
            'avg_atoms': float(merged_metadata_df['n_atoms'].mean()),
            'max_atoms': int(merged_metadata_df['n_atoms'].max()),
            'min_atoms': int(merged_metadata_df['n_atoms'].min()),
            'periodic_ratio': float(merged_metadata_df['is_periodic'].mean()),
            'has_metals_ratio': float(merged_metadata_df['has_metals'].mean()),
            'unique_elements': sorted(element_index.keys()),
            'element_counts': {elem: len(indices) for elem, indices in element_index.items()}
        }
        
        metadata_to_save = {
            'dataframe': merged_metadata_df,
            'element_index': element_index,
            'mw_sorted_indices': merged_metadata_df['molecular_weight'].argsort().values,
            'statistics': stats,
        }
        
        pkl_buffer = io.BytesIO()
        pickle.dump(metadata_to_save, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)

        # Write to a single TAR file
        print("Saving merged data to TAR archive...")
        with tarfile.open(output_tar, 'w') as tar:
            # Add HDF5 data
            h5_buffer.seek(0)
            h5_info = tarfile.TarInfo(name=f"{self.output_name}.h5")
            h5_info.size = len(h5_buffer.getvalue())
            tar.addfile(h5_info, h5_buffer)

            # Add pickle data
            pkl_buffer.seek(0)
            pkl_info = tarfile.TarInfo(name=f"{self.output_name}.pkl")
            pkl_info.size = len(pkl_buffer.getvalue())
            tar.addfile(pkl_info, pkl_buffer)

        print(f"\033[1;34mMerge complete! Output saved as {self.output_name}.tar\033[0m")

    def merge_preview(self):
        """Preview the result of merging without actually merging."""
        
        total_structures = 0
        all_datasets_data = []
        
        for name in self.merge_name_list:
            tar_path = os.path.join(self.data_dir, f"{name}.tar")
            if not os.path.exists(tar_path):
                warnings.warn(f"Warning: Dataset archive not found at {tar_path}. Skipping.")
                continue
            
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    pkl_member = next((m for m in tar.getmembers() if m.name.endswith('.pkl')), None)
                    if pkl_member:
                        with tar.extractfile(pkl_member) as f:
                            metadata = pickle.load(f)
                            df = metadata['dataframe']
                            all_datasets_data.append((name, df))
                            total_structures += len(df)
                    else:
                        warnings.warn(f"Warning: No .pkl file found in {tar_path} for '{name}'. Skipping.")
            except tarfile.ReadError:
                warnings.warn(f"Warning: Could not read TAR archive at {tar_path}. Skipping.")

        if not all_datasets_data:
            print("No valid datasets to preview.")
            return

        all_dfs = [df for _, df in all_datasets_data]
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # Calculate combined statistics
        mw_min = merged_df['molecular_weight'].min()
        mw_max = merged_df['molecular_weight'].max()
        atoms_min = merged_df['n_atoms'].min()
        atoms_max = merged_df['n_atoms'].max()
        
        element_counts = {}
        for elements_list in merged_df['elements']:
            for element in set(elements_list):
                element_counts[element] = element_counts.get(element, 0) + 1
        
        sorted_element_counts = dict(sorted(element_counts.items(), key=lambda item: item[1], reverse=True))

        # Print preview
        print("\n\033[1;34mMerged Dataset Preview\033[0m")
        print("=======================================================")
        print(f"Datasets to merge: {self.merge_name_list}")
        print(f"Total structures: {total_structures:,}")
        print(f"Molecular weight range: ({mw_min:.1f}, {mw_max:.1f})")
        print(f"Num. of atoms range: ({atoms_min}, {atoms_max})")
        print("=======================================================")
        
        print("\n\033[1;34mCombined Elemental Composition\033[0m")
        print("=======================================================")
        
        bar_length = 10
        if sorted_element_counts:
            max_elem_len = max(len(elem) for elem in sorted_element_counts.keys())
            max_count_len = max(len(f"{count:,}") for count in sorted_element_counts.values())

            for elem, count in sorted_element_counts.items():
                percentage = count / total_structures * 100
                filled_length = int(round(bar_length * percentage / 100))
                bar = '[' + '=' * filled_length + ' ' * (bar_length - filled_length) + ']'

                elem_fmt = f"{elem:<{max_elem_len}}"
                count_fmt = f"{count:>{max_count_len},}"
                percent_fmt = f"{percentage:>4.1f}%"

                print(f"{elem_fmt}: {count_fmt} structures {bar} {percent_fmt}")
        print("=======================================================")

        # Print individual dataset info
        print("\n\033[1;32mDatasets to merge\033[0m")
        print("=======================================================")
        for i, (name, df) in enumerate(all_datasets_data):
            mw_min_ind = df['molecular_weight'].min()
            mw_max_ind = df['molecular_weight'].max()
            atoms_min_ind = df['n_atoms'].min()
            atoms_max_ind = df['n_atoms'].max()
            
            print(f"  Dataset: \033[1m{name}\033[0m")
            print(f"  - Structures: {len(df):,}")
            print(f"  - Molecular weight range: ({mw_min_ind:.1f}, {mw_max_ind:.1f})")
            print(f"  - Num. of atoms range: ({atoms_min_ind}, {atoms_max_ind})")
            if i < len(all_datasets_data) - 1:
                 print("-------------------------------------------------------")
        print("=======================================================")
