from .dataloader import DataLoader
from .utils import available_datasets
import os
import shutil

# High-level API for direct access
_loader_dict = {}

def randomatoms(*args, filename='default', data_dir=None, **kwargs):
    if not _loader_dict.get(filename, None):
        _loader_dict[filename] = DataLoader(filename=filename, data_dir=data_dir)
        
    return _loader_dict[filename].get_random_structures(*args, **kwargs)


def set_default_dataset(source_path: str):
    """
    Moves a dataset file from the source path to become the default dataset.
    The file will be renamed to 'default.tar'.

    Parameters
    ----------
    source_path : str
        The path to the dataset file to be moved.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found at: {source_path}")

    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    destination_path = os.path.join(dataset_dir, 'default.tar')

    shutil.move(source_path, destination_path)
    print(f"Successfully moved and set '{os.path.basename(source_path)}' as the default dataset.")
    print(f"Location: {destination_path}")
    available_datasets()


def add_dataset(source_path: str):
    """
    Moves a dataset file from the source path to the package's dataset directory,
    keeping its original filename.

    Parameters
    ----------
    source_path : str
        The path to the dataset file to be moved.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found at: {source_path}")

    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    filename = os.path.basename(source_path)
    destination_path = os.path.join(dataset_dir, filename)

    shutil.move(source_path, destination_path)
    print(f"Successfully added dataset '{filename}'.")
    print(f"Location: {destination_path}")
    available_datasets()
