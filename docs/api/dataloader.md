# DataLoader

The `DataLoader` class is the core of `randatoms`. It loads a dataset from a TAR archive into memory and exposes methods for filtering, sampling, and statistics.

---

## High-level helpers

These module-level functions wrap `DataLoader` for quick one-liners.

### `randomatoms`

```python
from randatoms import randomatoms

randomatoms(n_structures=1, seed=None, filename='default', data_dir=None, **filters)
```

Sample random structures from the default (or named) dataset.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_structures` | `int` | `1` | Number of structures to return |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `filename` | `str` | `'default'` | Dataset name (without `.tar` extension) |
| `data_dir` | `str \| None` | `None` | Directory containing datasets; defaults to internal package directory |
| `**filters` | | | Any [filter parameters](../user-guide/filtering.md) |

**Returns** `Atoms` when `n_structures=1`, otherwise `list[Atoms]`.

**Example**

```python
from randatoms import randomatoms

atoms = randomatoms()
atoms_list = randomatoms(5, seed=42, include_elements=["C", "H"], max_atoms=50)
```

---

### `set_default_dataset`

```python
from randatoms import set_default_dataset

set_default_dataset(source_path)
```

Move a `.tar` file to the internal dataset directory and rename it to `default.tar`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_path` | `str` | Path to the `.tar` file to promote |

!!! warning
    The file is **moved**, not copied.

---

### `add_dataset`

```python
from randatoms import add_dataset

add_dataset(source_path)
```

Move a `.tar` file to the internal dataset directory, keeping its original filename.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_path` | `str` | Path to the `.tar` file to add |

---

## DataLoader class

::: randatoms.dataloader.DataLoader
    options:
      show_source: false
      members:
        - __init__
        - filter_indices
        - load_structures
        - get_random_structures
        - get_filtered_statistics
        - print_statistics
