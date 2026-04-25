# Datasets

`randatoms` is designed to work with any dataset that follows the [TAR/HDF5 format](user-guide/data-format.md). The following publicly available datasets are referenced in the project.

---

## Supported datasets

### OMOL25 — Open Molecules 2025

A large-scale dataset of molecular structures covering a broad range of organic and inorganic chemistry.

- **Structures:** millions of molecular conformations
- **Domain:** computational chemistry, molecular property prediction
- **Reference:** Levine, D.S. *et al.* (2025). *The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models*. *arXiv preprint* [arXiv:2505.08762](https://arxiv.org/abs/2505.08762)

---

### OMAT24 — Open Materials 2024

Inorganic materials dataset covering a wide chemical space of crystalline solids.

- **Structures:** periodic bulk and surface structures
- **Domain:** materials science, density functional theory
- **Reference:** Barroso-Luque, L. *et al.* (2024). *Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models*. *arXiv preprint* [arXiv:2410.12771](https://arxiv.org/abs/2410.12771)

---

### Peptide set

A benchmark dataset of small peptide conformations for intermolecular interaction studies.

- **Structures:** peptide dimers and clusters
- **Domain:** biomolecular simulation, non-covalent interactions
- **Reference:** Řezáč, J. *et al.* (2018). *Journal of Chemical Theory and Computation*, **14**(3), 1254–1266. [DOI: 10.1021/acs.jctc.7b01074](https://doi.org/10.1021/acs.jctc.7b01074)

---

### X23b set

Organic molecular crystals benchmark dataset used for crystal structure prediction and lattice energy calculations.

- **Structures:** 23 organic crystal unit cells with multiple polymorphs
- **Domain:** crystal engineering, solid-state chemistry
- **Reference:** Zhugayevych, A. *et al.* (2023). *Journal of Chemical Theory and Computation*, **19**(22), 8481–8490. [DOI: 10.1021/acs.jctc.3c00861](https://doi.org/10.1021/acs.jctc.3c00861)

---

### ODAC23 — Open DAC 2023

Dataset targeting direct air capture of CO₂ — metal-organic frameworks and porous materials.

- **Structures:** MOF adsorbate configurations
- **Domain:** carbon capture, porous materials, catalysis
- **Reference:** Sriram, A. *et al.* (2024). *ACS Central Science*, **10**(5), 923–941. [DOI: 10.1021/acscentsci.3c01629](https://doi.org/10.1021/acscentsci.3c01629)

---

## Using your own data

`randatoms` is not limited to the datasets above. Any collection of [ASE `Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) objects can be converted to the required format:

```python
from ase.io import read
from randatoms.converter import ASEtoHDF5Converter
from randatoms import add_dataset

atoms_list = read("my_structures.xyz", index=":")

converter = ASEtoHDF5Converter()
converter.convert_atoms_list(atoms_list, filename="my_data", data_dir=".")

add_dataset("./my_data.tar")
```

See the [Dataset Management](user-guide/dataset-management.md) guide for the full workflow.
