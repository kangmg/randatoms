"""
Tests for the DataLoader class.

This test suite verifies that:
- A dataset can be loaded correctly from a .tar archive.
- Structures can be retrieved from the dataset by their indices.
- Filtering by chemical elements works as expected.
- Random structures can be sampled from the dataset with optional filters.
"""
import os
import shutil
import unittest
from ase.build import molecule
from randatoms.converter import ASEtoHDF5Converter
from randatoms.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(cls.test_dir, exist_ok=True)
        
        cls.molecules = [
            molecule("H2O"), molecule("NH3"), molecule("CH3OH"),
            molecule("CH4"), molecule("C6H6"), molecule("H2")
        ]
        
        cls.filename = "test_loader_molecules"
        converter = ASEtoHDF5Converter()
        converter.convert_atoms_list(
            cls.molecules,
            filename=cls.filename,
            data_dir=cls.test_dir
        )
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.loader = DataLoader(filename=self.filename, data_dir=self.test_dir)

    def test_load_all_structures(self):
        print("\nRunning test: test_load_all_structures (DataLoader)")
        structures = self.loader.load_structures(indices=list(range(len(self.molecules))))
        self.assertEqual(len(structures), len(self.molecules))

    def test_filter_by_elements(self):
        print("\nRunning test: test_filter_by_elements (DataLoader)")
        # Include C and H
        indices = self.loader.filter_indices(include_elements=['C', 'H'])
        structures = self.loader.load_structures(indices)
        # CH3OH, CH4, C6H6
        self.assertEqual(len(structures), 3)
        for s in structures:
            self.assertTrue('C' in s.get_chemical_symbols() and 'H' in s.get_chemical_symbols())

    def test_get_random_structures(self):
        print("\nRunning test: test_get_random_structures (DataLoader)")
        # Get 2 random structures containing only C and H
        structures = self.loader.get_random_structures(
            n_structures=2,
            include_elements=['C', 'H'],
            seed=42
        )
        self.assertEqual(len(structures), 2)
        for s in structures:
            self.assertTrue('C' in s.get_chemical_symbols() and 'H' in s.get_chemical_symbols())

if __name__ == '__main__':
    unittest.main()
