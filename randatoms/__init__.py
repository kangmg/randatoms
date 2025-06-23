from .dataloader import DataLoader
from .utils import available_datasets

# High-level API for direct access
_loader = None

def randomatoms(*args, filename='default', data_dir=None, **kwargs):
    global _loader
    if _loader is None:
        _loader = DataLoader(filename=filename, data_dir=data_dir)
    return _loader.get_random_structures(*args, **kwargs)

