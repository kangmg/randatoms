# randatoms

A random atoms package for atomistic scientists: easily sample random structures from existing datasets, filter, and manage atomic datasets.

## Overview

`randatoms` provides tools for sampling random atomic structures from pre-existing datasets, as well as utilities for filtering, merging, and loading these structures. The package is designed to help researchers in computational chemistry and materials science efficiently retrieve random structures, apply various filters, and manage large collections of atomic data.

## Installation

You can install randatoms using pip:

```bash
pip install randatoms
```

## Usage

<table>
  <tr>
    <th>💬 Conversational AI Docs </th>
    <th>🔗 Try it on Google Colab</th>
  </tr>
  <tr>
    <td>
      <a href="https://deepwiki.com/kangmg/randatoms">
        <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
      </a>
    </td>
    <td>
      <a href="https://colab.research.google.com/github/kangmg/randatoms/blob/main/notebooks/randatoms_tutorial.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
      </a>
    </td>
  </tr>
</table>


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
loader = DataLoader()

# filter query
filter = dict(
    include_elements=['C', 'H', 'O'],
    has_metals=True,
    is_periodic=True
    )

# Get random structures
atoms = loader.get_random_structures(**filter)

# View statistics
loader.print_statistics(**filter)
```


## Unit test
```shell
wget -O randatoms-main.zip https://github.com/kangmg/randatoms/archive/refs/heads/main.zip

unzip randatoms-main.zip "randatoms-main/test/*" && mv randatoms-main/test ./test && rm -rf randatoms-main randatoms-main.zip


python3 -m unittest discover test -v
```

## Dataset References

- [**OMOL25 set**]  
  Levine, D.S. *et al.* (2025). *The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models*.  
  *arXiv preprint* [arXiv:2505.08762](https://arxiv.org/abs/2505.08762)

- [**OMAT24 set**]  
  Barroso-Luque, L. *et al.* (2024). *Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models*.  
  *arXiv preprint* [arXiv:2410.12771](https://arxiv.org/abs/2410.12771)

- [**peptide set**]  
  Řezáč, J. *et al.* (2018). *Journal of Chemical Theory and Computation*, **14**(3), 1254–1266.  
  [DOI: 10.1021/acs.jctc.7b01074](https://doi.org/10.1021/acs.jctc.7b01074)

- [**X23b set**]  
  Zhugayevych, A. *et al.* (2023). *Journal of Chemical Theory and Computation*, **19**(22), 8481–8490.  
  [DOI: 10.1021/acs.jctc.3c00861](https://doi.org/10.1021/acs.jctc.3c00861)

- [**ODAC23 set**]  
  Sriram, A. *et al.* (2024). *ACS Central Science*, **10**(5), 923–941.  
  [DOI: 10.1021/acscentsci.3c01629](https://doi.org/10.1021/acscentsci.3c01629)
