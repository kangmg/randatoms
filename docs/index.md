# randatoms

**randatoms** is a Python package for atomistic scientists that makes it easy to randomly sample atomic structures from existing datasets, apply chemical filters, and manage large collections of molecular and materials data.

---

## Why randatoms?

Working with large atomistic datasets often involves repetitive boilerplate: loading files, filtering by element or size, handling different formats. `randatoms` removes that friction and lets you focus on your science.

- **Random sampling** — draw random structures from any dataset in one line
- **Flexible filtering** — filter by elements, molecular weight, atom count, periodicity, and more
- **Dataset management** — convert, merge, and organise datasets in a standardised TAR/HDF5 format
- **Fast access** — pre-computed indices and in-memory HDF5 loading for quick queries

---

## At a glance

```python
from randatoms import randomatoms

# Single random structure
atoms = randomatoms()

# Five carbon/hydrogen structures with at most 50 atoms (reproducible)
atoms_list = randomatoms(5, seed=42, include_elements=["C", "H"], max_atoms=50)
```

For more advanced usage, use the `DataLoader` class directly:

```python
from randatoms import DataLoader

loader = DataLoader()
loader.print_statistics(include_elements=["C", "H", "O"], is_periodic=False)

atoms = loader.get_random_structures(
    n_structures=10,
    seed=0,
    include_elements=["C", "H", "O"],
    max_atoms=30,
)
```

---

## Quick links

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**  
  Install via pip in seconds.

- :material-rocket-launch: **[Quick Start](quickstart.md)**  
  Up and running with code examples.

- :material-filter: **[Filtering Guide](user-guide/filtering.md)**  
  All available filter parameters explained.

- :material-database: **[Dataset Management](user-guide/dataset-management.md)**  
  Convert, add, and merge datasets.

- :material-book-open-variant: **[API Reference](api/index.md)**  
  Full class and function documentation.

- :material-database-search: **[Datasets](datasets.md)**  
  Supported datasets and citations.

</div>

---

## Resources

| Resource | Link |
|---|---|
| Source code | [github.com/kangmg/randatoms](https://github.com/kangmg/randatoms) |
| PyPI | `pip install randatoms` |
| Tutorial notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kangmg/randatoms/blob/main/notebooks/randatoms_tutorial.ipynb) |
| Conversational docs | [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kangmg/randatoms) |
