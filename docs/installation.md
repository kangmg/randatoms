# Installation

## Requirements

- Python 3.8 or later
- [ASE](https://wiki.fysik.dtu.dk/ase/) 3.22+

## Install from PyPI

```bash
pip install randatoms
```

This installs `randatoms` along with all required dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `ase` | ≥ 3.22.0 | Atomic simulation environment — `Atoms` objects |
| `h5py` | ≥ 3.0.0 | HDF5 file I/O |
| `numpy` | ≥ 1.20.0 | Numerical operations |
| `pandas` | ≥ 1.2.0 | Metadata indexing |
| `tqdm` | ≥ 4.60.0 | Progress bars |

## Install from source

```bash
git clone https://github.com/kangmg/randatoms.git
cd randatoms
pip install -e .
```

## Verify installation

```python
import randatoms
print(randatoms.__version__)
```

## Building the docs

To build this documentation locally you need a few extra packages:

```bash
pip install mkdocs-material mkdocstrings[python]
```

Then serve the docs with live-reload:

```bash
mkdocs serve
```

Or build static HTML to the `site/` directory:

```bash
mkdocs build
```
