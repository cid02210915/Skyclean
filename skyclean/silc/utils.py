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
