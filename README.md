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
atoms = randomatoms()

# Get multiple random structures with filters
atoms_list = randomatoms(5, seed=42, include_elements=['C', 'H'], max_atoms=50)
```

### Advanced Data Loading

```python
from randatoms import DataLoader

# Initialize loader
loader = DataLoader('default')

# filter query
query = dict(
    include_elements=['C', 'H', 'O'],
    has_metals=True,
    is_periodic=True
    )

# Get random structures
atoms = loader.get_random_structures(**query)

# View statistics
loader.print_statistics(**query)
```


## Unit test
```shell
python3 -m unittest discover test -v
```
