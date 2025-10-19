from pathlib import Path
import re
import os
import numpy as np
import healpy as hp

def create_dir(directory: str):
    """
    Create a directory if it does not exist.

    Parameters:
        dir (str): The path of the directory to create.

    Returns:
        None
    """
    if not os.path.exists(directory):
        print("Creating directory:", directory)
        os.makedirs(directory)
    else:
        pass

def np_save_and_load(data, filename: str):
    """
    Save data to a file and then load it back.

    Parameters:
        data: The data to save.
        filename (str): The name of the file to save the data to.
    
    Returns:
        The loaded data.
    """
    np.save(filename, data)
    return np.load(filename)

def save_map(filepath, hp_map, overwrite = False):
    """Save the processed map to the specified filepath."""
    if os.path.exists(filepath) and overwrite == False:
        print(f"File {filepath} already exists. Skipping saving.")
    else:
        hp.write_map(filepath, hp_map, overwrite = True)

def normalize_targets(extra_comp):
    """Return (list_of_names, tag_string) for filenames."""
    if isinstance(extra_comp, (list, tuple, np.ndarray)):
        names = [str(n) for n in extra_comp]
    elif extra_comp is None:
        names = []
    else:
        names = [str(extra_comp)]
    tag = "+".join(names) if names else "none"
    return names, tag

def _save_file(task):
    out_path, arr = task
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    return out_path


def save_array(task):
    """
    General-purpose array saver for use with concurrent.futures executors.

    Parameters
    ----------
    task : tuple
        (out_path, arr) where
          - out_path (str): full file path to save the array
          - arr (np.ndarray): array to be saved

    Returns
    -------
    str
        The output path that was saved.
    """
    out_path, arr = task
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    return out_path

@staticmethod
def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)