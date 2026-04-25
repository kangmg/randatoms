# Quick Start

This page walks you through the most common `randatoms` workflows.

---

## 1. Get a random structure

The `randomatoms` convenience function is the simplest entry point.

```python
from randatoms import randomatoms

# Single random structure (ASE Atoms object)
atoms = randomatoms()
print(atoms)
```

---

## 2. Get multiple structures with a seed

Pass `n_structures` and `seed` for reproducible batches:

```python
atoms_list = randomatoms(10, seed=42)
print(len(atoms_list))  # 10
```

---

## 3. Apply filters

Keyword arguments are forwarded directly to the filter engine:

```python
# Carbon/hydrogen molecules with at most 50 atoms
atoms_list = randomatoms(
    5,
    seed=0,
    include_elements=["C", "H"],
    max_atoms=50,
)
```

See the [Filtering Guide](user-guide/filtering.md) for all available parameters.

---

## 4. Explore the dataset with DataLoader

For more control, use the `DataLoader` class directly.

```python
from randatoms import DataLoader

loader = DataLoader()

# Print statistics for the whole dataset
loader.print_statistics()
```

Example output:

```
Dataset Statistics
=======================================================
* Total structures:                          1,000,000
* Percentage of dataset:                        100.0 %
* Molecular weight range:          (2.0, 100000.0)
* Average atoms per structure:                    12.3
* Num. of atoms range:                        (1, 512)
* Periodic structures:                          32.1 %
* Structures with metals:                       45.6 %
=======================================================
```

---

## 5. Filter and sample

```python
filter_kwargs = dict(
    include_elements=["C", "H", "O"],
    is_periodic=False,
    max_atoms=30,
)

# Print statistics for the filtered subset
loader.print_statistics(**filter_kwargs)

# Sample 5 structures from the filtered subset
atoms_list = loader.get_random_structures(n_structures=5, seed=1, **filter_kwargs)
```

---

## 6. Load structures by index

If you already have indices (e.g., from `filter_indices`), load them directly:

```python
indices = loader.filter_indices(include_elements=["Fe", "O"], is_periodic=True)
print(f"{len(indices)} structures match")

structures = loader.load_structures(indices[:20])
```

---

## 7. Use a custom dataset

Point `DataLoader` to your own `.tar` dataset file:

```python
loader = DataLoader(filename="my_dataset", data_dir="/path/to/my/datasets")
atoms = loader.get_random_structures(n_structures=3, seed=99)
```

Or use the high-level helper:

```python
randomatoms(3, seed=99, filename="my_dataset", data_dir="/path/to/my/datasets")
```

---

## Next steps

- [Filtering Guide](user-guide/filtering.md) — detailed explanation of every filter parameter
- [Dataset Management](user-guide/dataset-management.md) — convert and merge your own datasets
- [API Reference](api/index.md) — full class and method documentation
