# Utilities

Utility functions used internally and exposed for user convenience.

---

## `available_datasets`

::: randatoms.utils.available_datasets
    options:
      show_source: false

**Example**

```python
from randatoms import available_datasets

available_datasets()
```

Example output:

```
Available Datasets
Dataset found : /usr/local/lib/python3.10/site-packages/randatoms/dataset
==========================================================================
   Dataset Name    |  Structures  |      MW Range        |   Atoms Range    | Size (MB)
----------------------------------------------------------------------------------
     default       |  1,000,000   |  (2.0, 99999.9)      |    (1, 512)      |  512.34
     my_molecules  |       3      |  (18.0, 44.0)        |    (3, 5)        |    0.01
==========================================================================
```

---

## Internal worker functions

These functions are used by `ASEtoHDF5Converter` for parallel metadata extraction. They are part of the public API but are not normally called directly.

::: randatoms.utils.extract_metadata_worker
    options:
      show_source: false

::: randatoms.utils.extract_metadata_batch_worker
    options:
      show_source: false
