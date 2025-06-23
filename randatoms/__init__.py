from .dataloader import DataLoader
from .utils import available_datasets

# High-level API for direct access
_loader_dict = {}

def randomatoms(*args, filename='default', data_dir=None, **kwargs):
    if not _loader_dict.get(filename, None):
        _loader_dict[filename] = DataLoader(filename=filename, data_dir=data_dir)
        
    return _loader_dict[filename].get_random_structures(*args, **kwargs)

