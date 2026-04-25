# API Reference

This section documents every public class and function in `randatoms`.

---

## High-level functions

These are the primary entry points for most users.

| Function | Description |
|----------|-------------|
| [`randomatoms`](dataloader.md#randomatoms) | Sample one or more random structures |
| [`set_default_dataset`](dataloader.md#set_default_dataset) | Promote a `.tar` file to the default dataset |
| [`add_dataset`](dataloader.md#add_dataset) | Register a `.tar` file with randatoms |
| [`available_datasets`](utils.md#available_datasets) | List all registered datasets |

---

## Classes

| Class | Module | Description |
|-------|--------|-------------|
| [`DataLoader`](dataloader.md) | `randatoms.dataloader` | Load and filter structures from a dataset |
| [`ASEtoHDF5Converter`](converter.md) | `randatoms.converter` | Convert ASE Atoms lists to `.tar` datasets |
| [`DatasetMerger`](merger.md) | `randatoms.merger` | Merge multiple `.tar` datasets into one |

---

## Importing

```python
# High-level API
from randatoms import randomatoms, set_default_dataset, add_dataset

# Classes
from randatoms import DataLoader
from randatoms.converter import ASEtoHDF5Converter
from randatoms.merger import DatasetMerger

# Utilities
from randatoms import available_datasets
```
