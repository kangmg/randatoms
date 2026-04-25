# DatasetMerger

`DatasetMerger` combines multiple `.tar` dataset files into a single TAR archive. The merged file can be used with `DataLoader` like any other dataset, with the added ability to filter by source dataset via `include_datasets`.

---

## Quick example

```python
from randatoms.merger import DatasetMerger

merger = DatasetMerger(
    merge_name_list=["dataset_a", "dataset_b", "dataset_c"],
    output_name="combined",
    data_dir="/path/to/datasets",
)

# Preview statistics without writing
merger.merge_preview()

# Run the merge
merger.merge()
# → /path/to/datasets/combined.tar
```

---

## Using the merged dataset

```python
from randatoms import DataLoader

loader = DataLoader(filename="combined", data_dir="/path/to/datasets")

# Sample only from dataset_a
atoms = loader.get_random_structures(n_structures=5, include_datasets=["dataset_a"])
```

---

## Duplicate key handling

If two source datasets share a structure key, the duplicate is renamed to `original_key@source_dataset_name` and a `UserWarning` is emitted. No data is lost.

---

## DatasetMerger class

::: randatoms.merger.DatasetMerger
    options:
      show_source: false
      members:
        - __init__
        - merge_preview
        - merge
