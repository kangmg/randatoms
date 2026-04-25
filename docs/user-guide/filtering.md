# Filtering Guide

`randatoms` provides a set of composable filter parameters that can be passed to any loading or statistics method. All filters are combined with a logical **AND** — only structures that satisfy every specified criterion are returned.

---

## Where filters are accepted

The same keyword arguments work across all of these interfaces:

| Interface | Method / Function |
|---|---|
| High-level | `randomatoms(..., **filters)` |
| `DataLoader` | `get_random_structures(**filters)` |
| `DataLoader` | `filter_indices(**filters)` |
| `DataLoader` | `get_filtered_statistics(**filters)` |
| `DataLoader` | `print_statistics(**filters)` |

---

## Filter parameters

### `include_elements`

**Type:** `list[str]`

Keeps only structures that contain **all** of the specified elements. Elements are matched by standard atomic symbol (case-sensitive).

```python
# Only structures that contain both C and H
loader.get_random_structures(include_elements=["C", "H"])

# Must contain Fe, O, and Mn
loader.filter_indices(include_elements=["Fe", "O", "Mn"])
```

!!! note
    A structure must contain *every* element in the list. Specifying `["C", "H"]` will not match a structure that has only C.

---

### `exclude_elements`

**Type:** `list[str]`

Removes structures that contain **any** of the specified elements.

```python
# No fluorine or chlorine
loader.get_random_structures(exclude_elements=["F", "Cl"])

# Organic-only: exclude all metals implicitly via has_metals instead
loader.get_random_structures(exclude_elements=["Na", "K", "Li"])
```

---

### `mw_range`

**Type:** `tuple[float, float]` — `(min_mw, max_mw)`

Filters by **molecular weight** (in atomic mass units, u). Both bounds are inclusive.

```python
# Molecular weight between 100 and 500 u
loader.get_random_structures(mw_range=(100.0, 500.0))
```

---

### `min_atoms`

**Type:** `int`

Minimum number of atoms in a structure (inclusive).

```python
loader.get_random_structures(min_atoms=10)
```

---

### `max_atoms`

**Type:** `int`

Maximum number of atoms in a structure (inclusive).

```python
loader.get_random_structures(max_atoms=50)

# Combined: structures with 10–50 atoms
loader.get_random_structures(min_atoms=10, max_atoms=50)
```

---

### `is_periodic`

**Type:** `bool`

- `True` — keep only **periodic** structures (at least one periodic boundary condition is set).
- `False` — keep only **non-periodic** (finite/molecular) structures.

```python
# Bulk / surface / slab structures only
loader.get_random_structures(is_periodic=True)

# Molecules only
loader.get_random_structures(is_periodic=False)
```

---

### `has_metals`

**Type:** `bool`

- `True` — keep only structures that contain at least one metal element.
- `False` — keep only structures with no metal elements.

The set of metals used for this check is defined in `randatoms.constants.METALS` and covers all standard metallic elements on the periodic table.

```python
# Organometallic / metal-containing structures
loader.get_random_structures(has_metals=True)

# Purely organic / non-metal structures
loader.get_random_structures(has_metals=False)
```

---

### `include_datasets`

**Type:** `list[str]`

When working with a **merged** dataset, restricts sampling to structures that originated from specific source datasets.

```python
# Only from the OMOL25 and ODAC23 sub-datasets
loader.get_random_structures(include_datasets=["omol25", "odac23"])
```

!!! warning
    This filter requires a `dataset` column in the metadata, which is present only in merged datasets created with `DatasetMerger`. Using it on a single-source dataset will raise a `ValueError`.

---

## Combining filters

All parameters can be combined freely:

```python
atoms_list = loader.get_random_structures(
    n_structures=20,
    seed=42,
    include_elements=["C", "H", "O", "N"],
    exclude_elements=["S"],
    mw_range=(50.0, 300.0),
    min_atoms=5,
    max_atoms=40,
    is_periodic=False,
    has_metals=False,
)
```

---

## Checking available structures

Before sampling, use `print_statistics` to see how many structures match a given filter:

```python
loader.print_statistics(
    include_elements=["C", "H", "O"],
    is_periodic=False,
    max_atoms=30,
)
```

Or retrieve the statistics as a dictionary:

```python
stats = loader.get_filtered_statistics(include_elements=["Fe"], is_periodic=True)
print(stats["count"], "structures match")
print(stats["element_coverage"])
```
