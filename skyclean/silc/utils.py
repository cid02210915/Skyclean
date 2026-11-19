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

@staticmethod
def top_scale_index(L, lam):
    """
    Compute J = floor(log_{lam}(L-1)).
    """
    L = int(L); lam = float(lam)
    return int(np.floor(np.log(L - 1) / np.log(lam)))

@staticmethod
def admissibility(Phi_l0, Psi_j_l0, ells, tol=1e-6):
    """
    Compute S_ell = (4π/(2ℓ+1)) ( |Φ_{ℓ0}|^2 + Σ_j |Ψ_{j;ℓ0}|^2 ) for ℓ=0..L-1
    and check admissibility: |S_ell - 1| < tol for all ℓ ≥ 1.

    Returns
    -------
    S : np.ndarray, shape (L,)
        The admissibility sum over ℓ.
    ok : bool
        True if admissibility holds (excluding ℓ=0), else False.
    """
    S = np.abs(Phi_l0)**2
    for W in Psi_j_l0.values():
        S = S + np.abs(W)**2
    S = (4.0*np.pi) / (2.0*ells + 1.0) * S

    ok = np.all(np.abs(S[1:] - 1.0) < tol)
    return S, bool(ok)
