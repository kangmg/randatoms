# Data Format

This page describes the internal storage format used by `randatoms`. Understanding it can be useful if you want to inspect raw files or build custom tooling.

---

## TAR archive layout

Every dataset is stored as a single uncompressed `.tar` archive containing exactly two files:

```
my_dataset.tar
├── my_dataset.h5     # HDF5 file — atomic coordinates and species
└── my_dataset.pkl    # Pickle file — metadata and search indices
```

---

## HDF5 file (`.h5`)

The HDF5 file stores the raw structural data. Each atomic structure occupies a separate HDF5 group named `{dataset_name}_{index:06d}` (e.g., `my_dataset_000042`).

### Group attributes

Each group contains:

| Dataset | Shape | dtype | Description |
|---------|-------|-------|-------------|
| `positions` | `(N, 3)` | `float64` | Cartesian coordinates in Å |
| `symbols` | `(N,)` | `bytes` | Chemical symbols (UTF-8 encoded) |
| `cell` | `(3, 3)` | `float64` | Unit cell vectors (only for periodic structures) |
| `pbc` | `(3,)` | `bool` | Periodic boundary condition flags |

`cell` and `pbc` are absent for non-periodic (molecular) structures.

### Root attributes

The root of the HDF5 file stores global metadata:

| Attribute | Description |
|-----------|-------------|
| `dataset_name` | Name of the dataset |
| `n_structures` | Total number of structures |
| `creation_date` | ISO 8601 timestamp |

### Compression

By default, `ASEtoHDF5Converter` applies **gzip level 6** compression to all datasets. Large `positions` arrays with more than 100 atoms are also chunked in memory for efficient access.

---

## Pickle file (`.pkl`)

The pickle file contains a dictionary with four keys:

### `dataframe`

A `pandas.DataFrame` with one row per structure:

| Column | dtype | Description |
|--------|-------|-------------|
| `index` | `int` | Zero-based position in the dataset |
| `dataset` | `str` | Source dataset name |
| `key` | `str` | HDF5 group name |
| `molecular_weight` | `float` | Total mass in atomic mass units |
| `elements` | `list[str]` | Unique elements, sorted alphabetically |
| `n_atoms` | `int` | Total atom count |
| `formula` | `str` | Chemical formula (ASE format) |
| `is_periodic` | `bool` | Whether any PBC is set |
| `has_metals` | `bool` | Whether any element is a metal |

### `element_index`

```python
{
    "C": [0, 3, 7, 12, ...],   # indices of structures containing carbon
    "H": [0, 1, 2, 3, ...],
    "Fe": [5, 9, 23, ...],
    ...
}
```

This inverted index enables O(1) element lookups during filtering — no full DataFrame scan is needed.

### `mw_sorted_indices`

A `numpy.ndarray` of integer indices sorted by ascending molecular weight. Reserved for range-based MW queries.

### `statistics`

A summary dictionary:

```python
{
    "total_structures": 500000,
    "mw_range": (2.0, 99999.9),
    "avg_atoms": 12.4,
    "max_atoms": 512,
    "min_atoms": 1,
    "periodic_ratio": 0.32,
    "has_metals_ratio": 0.45,
    "unique_elements": ["Ag", "Al", "C", "H", ...],
    "element_counts": {"H": 320000, "C": 280000, ...},
}
```

---

## In-memory loading

When a `DataLoader` is initialised, the **entire HDF5 file is read into a `io.BytesIO` buffer in RAM**. This avoids repeated disk I/O for random-access workloads. The trade-off is higher memory usage proportional to the dataset size.

```
DataLoader.__init__
    └── tarfile.open(tar_path)
        ├── extractfile(.pkl)  →  self.metadata_dict
        └── extractfile(.h5)   →  self.h5_buffer  (BytesIO)
```

Subsequent `load_structures` calls open the in-memory buffer with `h5py.File(self.h5_buffer, 'r')` using a `ThreadPoolExecutor` for parallel batch loading.

---

## Creating a dataset manually

If you need to create a compatible `.tar` file from scratch (e.g., from a non-ASE source), you must produce:

1. An HDF5 file matching the group schema above.
2. A pickle file with the four keys: `dataframe`, `element_index`, `mw_sorted_indices`, `statistics`.
3. Bundle them into an uncompressed `.tar` archive.

Using `ASEtoHDF5Converter` is the recommended approach as it handles all of this automatically.
