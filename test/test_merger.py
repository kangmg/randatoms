"""
Tests for the DatasetMerger class.

This test suite verifies that:
- Multiple dataset .tar archives can be merged into a single archive.
- The merged dataset contains all structures from the source datasets.
- The metadata in the merged dataset is consistent and correct.
"""
import os
import shutil
import unittest
from ase.build import molecule
from randatoms.converter import ASEtoHDF5Converter
from randatoms.merger import DatasetMerger
from randatoms.dataloader import DataLoader

class TestDatasetMerger(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.converter = ASEtoHDF5Converter()
        
        # Create dataset 1
        self.molecules1 = [molecule("H2O"), molecule("NH3")]
        self.filename1 = "dataset1"
        self.converter.convert_atoms_list(
            self.molecules1,
            filename=self.filename1,
            data_dir=self.test_dir
        )
        
        # Create dataset 2
        self.molecules2 = [molecule("CH4"), molecule("C6H6")]
        self.filename2 = "dataset2"
        self.converter.convert_atoms_list(
            self.molecules2,
            filename=self.filename2,
            data_dir=self.test_dir
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_merge(self):
        print("\nRunning test: test_merge (Merger)")
        output_name = "merged_dataset"
        merger = DatasetMerger(
            merge_name_list=[self.filename1, self.filename2],
            output_name=output_name,
            data_dir=self.test_dir
        )
        merger.merge_preview()
        merger.merge()
        
        output_path = os.path.join(self.test_dir, f"{output_name}.tar")
        self.assertTrue(os.path.exists(output_path))
        
        # Verify the merged dataset
        loader = DataLoader(filename=output_name, data_dir=self.test_dir)
        total_molecules = len(self.molecules1) + len(self.molecules2)
        self.assertEqual(len(loader.df), total_molecules)
        
        loaded_structures = loader.load_structures(indices=list(range(total_molecules)))
        self.assertEqual(len(loaded_structures), total_molecules)
        
        original_symbols = sorted([str(m.symbols) for m in self.molecules1 + self.molecules2])
        loaded_symbols = sorted([str(s.symbols) for s in loaded_structures])
        self.assertEqual(original_symbols, loaded_symbols)

if __name__ == '__main__':
    unittest.main()
