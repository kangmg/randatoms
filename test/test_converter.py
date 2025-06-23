"""
Tests for the ASEtoHDF5Converter class.

This test suite verifies that:
- A list of ASE Atoms objects can be successfully converted into a .tar archive
  containing an HDF5 file and a metadata pickle file.
- The converted data can be loaded back correctly using the DataLoader.
"""
import os
import shutil
import unittest
from ase.build import molecule
from randatoms.converter import ASEtoHDF5Converter
from randatoms.dataloader import DataLoader

class TestASEtoHDF5Converter(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        self.converter = ASEtoHDF5Converter()
        self.molecules = [
            molecule("H2O"),
            molecule("NH3"),
            molecule("CH3OH"),
            molecule("CH4"),
            molecule("C6H6"),
            molecule("H2"),
        ]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_convert_atoms_list(self):
        print("\nRunning test: test_convert_atoms_list (Converter)")
        filename = "test_molecules"
        self.converter.convert_atoms_list(
            self.molecules,
            filename=filename,
            data_dir=self.test_dir
        )
        
        output_path = os.path.join(self.test_dir, f"{filename}.tar")
        self.assertTrue(os.path.exists(output_path))

        # Verify the contents
        loader = DataLoader(filename=filename, data_dir=self.test_dir)
        self.assertEqual(len(loader.df), len(self.molecules))
        
        loaded_structures = loader.load_structures(indices=list(range(len(self.molecules))))
        self.assertEqual(len(loaded_structures), len(self.molecules))
        
        # Check if symbols match
        original_symbols = sorted([str(m.symbols) for m in self.molecules])
        loaded_symbols = sorted([str(s.symbols) for s in loaded_structures])
        self.assertEqual(original_symbols, loaded_symbols)

if __name__ == '__main__':
    unittest.main()
