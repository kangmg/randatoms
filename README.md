# RandAtoms

A random atoms package for atomistic scientists: easily sample random structures from existing datasets, filter, and manage atomic datasets.

## Overview

`randatoms` provides tools for sampling random atomic structures from pre-existing datasets, as well as utilities for filtering, merging, and loading these structures. The package is designed to help researchers in computational chemistry and materials science efficiently retrieve random structures, apply various filters, and manage large collections of atomic data.

## Installation

You can install randatoms using pip:

```bash
pip install randatoms
```

## Usage

### Loading Random Structures

```python
from randatoms import randomatoms

# Get a single random structure
structure = randomatoms()

# Get multiple random structures with filters
structures = randomatoms(5, seed=42, include_elements=['C', 'H'], max_atoms=50)
```

### Advanced Data Loading

```python
from randatoms import DataLoader

# Initialize loader
loader = DataLoader('default')

# Filter structures
indices = loader.filter_indices(
    include_elements=['C', 'H', 'O'],
    mw_range=(100, 500),
    max_atoms=100
)

# Load filtered structures
structures = loader.load_structures(indices)

# View statistics
loader.print_statistics(include_elements=['C', 'H', 'O'])
```

## Features

- **Conversion**: Transform ASE Atoms objects to HDF5 format with metadata indexing.
- **Merging**: Combine multiple HDF5 datasets into a single dataset.
- **Loading**: Efficiently load and filter structures based on various criteria like elements, molecular weight, and atom count.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
