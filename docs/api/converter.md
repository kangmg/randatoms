# ASEtoHDF5Converter

`ASEtoHDF5Converter` converts a Python list of [ASE `Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) objects into the TAR/HDF5 format used by `randatoms`.

---

## Quick example

```python
from ase.build import molecule
from randatoms.converter import ASEtoHDF5Converter

atoms_list = [molecule("H2O"), molecule("CO2"), molecule("CH4")]

converter = ASEtoHDF5Converter()
converter.convert_atoms_list(
    atoms_list=atoms_list,
    filename="small_molecules",
    data_dir="./datasets",
)
# → ./datasets/small_molecules.tar
```

---

## Output format

`convert_atoms_list` produces a single `.tar` file containing:

- `{filename}.h5` — HDF5 file with positions, symbols, cell, and PBC for every structure
- `{filename}.pkl` — Pickle file with metadata DataFrame, element index, and statistics

See the [Data Format](../user-guide/data-format.md) guide for details.

---

## Performance

For datasets with fewer than 100 structures, metadata extraction runs sequentially. For larger datasets, a `ProcessPoolExecutor` distributes the work across `n_workers` processes in `batch_size`-sized batches.

---

## ASEtoHDF5Converter class

::: randatoms.converter.ASEtoHDF5Converter
    options:
      show_source: false
      members:
        - __init__
        - convert_atoms_list
