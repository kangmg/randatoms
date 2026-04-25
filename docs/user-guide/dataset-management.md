# Dataset Management

This guide explains how to add, convert, and merge datasets so you can use your own atomistic data with `randatoms`.

---

## Dataset directory

By default, `randatoms` looks for `.tar` dataset files in its internal `dataset/` directory (inside the installed package). You can also supply a custom `data_dir` path to any loading class or function.

### List available datasets

```python
from randatoms import available_datasets

# Internal dataset directory
available_datasets()

# Custom directory
available_datasets(data_dir="/path/to/my/datasets")
```

---

## Setting the default dataset

The `default` dataset is what `randomatoms()` loads when no `filename` is specified. Use `set_default_dataset` to promote any `.tar` file to the default:

```python
from randatoms import set_default_dataset

set_default_dataset("/path/to/my_dataset.tar")
# The file is moved to the internal dataset directory as 'default.tar'
```

!!! warning
    This **moves** the file (not copies it). Make sure you have a backup if needed.

---

## Adding a dataset

To add a dataset without making it the default, use `add_dataset`:

```python
from randatoms import add_dataset

add_dataset("/path/to/another_dataset.tar")
# The file is moved to the internal dataset directory, keeping its filename
```

After adding, you can load it by name:

```python
from randatoms import DataLoader

loader = DataLoader(filename="another_dataset")
```

---

## Converting your own data

If you have a list of [ASE `Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) objects, `ASEtoHDF5Converter` converts them to the `.tar` format used by `randatoms`.

```python
from ase.build import molecule
from randatoms import ASEtoHDF5Converter

# Example: build a list of molecules
atoms_list = [molecule("H2O"), molecule("CO2"), molecule("CH4")]

converter = ASEtoHDF5Converter()
converter.convert_atoms_list(
    atoms_list=atoms_list,
    filename="my_molecules",
    data_dir="/path/to/output",
    dataset_name="my_molecules",   # label stored in metadata
    compress=True,                 # gzip compression (recommended)
)
# Creates /path/to/output/my_molecules.tar
```

### Constructor options

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Number of structures per HDF5 write chunk |
| `n_workers` | `cpu_count` | Worker processes for metadata extraction |
| `batch_size` | `100` | Structures per worker batch |

After conversion, add the file to randatoms:

```python
from randatoms import add_dataset

add_dataset("/path/to/output/my_molecules.tar")
```

---

## Merging datasets

`DatasetMerger` combines multiple `.tar` datasets into one, preserving all metadata and updating indices.

### Preview before merging

```python
from randatoms import DatasetMerger

merger = DatasetMerger(
    merge_name_list=["dataset_a", "dataset_b"],
    output_name="combined",
    data_dir="/path/to/datasets",   # omit to use internal directory
)

merger.merge_preview()
```

The preview prints per-dataset statistics and element coverage without writing anything.

### Run the merge

```python
merger.merge()
# Creates /path/to/datasets/combined.tar
```

### Load the merged dataset

```python
from randatoms import DataLoader

loader = DataLoader(filename="combined", data_dir="/path/to/datasets")

# Filter by source dataset within the merged file
atoms = loader.get_random_structures(
    n_structures=5,
    include_datasets=["dataset_a"],
)
```

!!! note "Duplicate keys"
    If two source datasets contain a structure with the same internal key, the duplicate is automatically renamed to `original_key@dataset_name` and a warning is printed.

---

## Workflow example: end-to-end custom dataset

```python
from ase.io import read
from randatoms import ASEtoHDF5Converter, add_dataset, DataLoader

# 1. Load your structures
atoms_list = read("my_structures.xyz", index=":")

# 2. Convert to randatoms format
converter = ASEtoHDF5Converter()
converter.convert_atoms_list(atoms_list, filename="my_data", data_dir=".")

# 3. Register with randatoms
add_dataset("./my_data.tar")

# 4. Use it
loader = DataLoader(filename="my_data")
sample = loader.get_random_structures(n_structures=10, seed=0)
```
