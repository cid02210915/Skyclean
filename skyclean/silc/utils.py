from pathlib import Path
import re
import os
import time
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tmp_path = f"{filepath}.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            hp.write_map(tmp_path, hp_map, overwrite=True)
            os.replace(tmp_path, filepath)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

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


DEPROJECTABLE_COMPONENTS = ("tsz", "cib")


def normalise_deproject(deproject, extract_comp: str | None = None) -> list:
    """
    Canonical list of components deprojected by the constrained ILC. The cILC preserves extract_comp
    (w^T a_extract = 1) and nulls every component in this list (w^T a_comp = 0); the pcILC (pcilc_eps given)
    preserves extract_comp and bounds the single listed component, |w^T b| <= eps.

    Parameters:
        deproject: None or [] for the plain ILC, otherwise a component name or a list of names (e.g. ["tsz"]).
        extract_comp (str): Component preserved by the ILC; it cannot also be deprojected.

    Returns:
        list: Lower-case, de-duplicated, sorted component names (empty for the plain ILC).
    """
    if deproject is None:
        return []
    if isinstance(deproject, str):
        deproject = [deproject]
    names = sorted({str(c).strip().lower() for c in deproject if str(c).strip()})
    unsupported = [c for c in names if c not in DEPROJECTABLE_COMPONENTS]
    if unsupported:
        raise ValueError(
            f"Cannot deproject {unsupported}: only {list(DEPROJECTABLE_COMPONENTS)} have a trusted spectral "
            "response in mixing_matrix_constraint.build_F_theory."
        )
    if extract_comp is not None and str(extract_comp).strip().lower() in names:
        raise ValueError(f"Cannot deproject the extracted component '{extract_comp}'.")
    return names


def ilc_mode(deproject=None, pcilc_eps: float | None = None) -> str:
    """
    Canonical ILC mode used in output filenames.

    This is the single definition of the ``{mode}`` field shared by the SILC
    outputs (ilc_synth, trimmed_maps, ...) and the ML products derived from
    them (foreground_estimate, ilc_residual, ilc_ml, ...). Keep the SILC
    writer and every ML reader on this function so the two cannot drift apart.

    Parameters:
        deproject: Components deprojected by the constrained ILC (see normalise_deproject); None/[] for the plain ILC.
        pcilc_eps (float): pcILC tolerance |w^T b| <= eps on the single deprojected component; None for the cILC
            (exact null). Must be > 0: eps = 0 is the cILC.

    Returns:
        str: "ilc", "cilc-dp-<comp>[-<comp>...]" (e.g. "cilc-dp-cib-tsz") or "pcilc-dp-<comp>-eps<eps>"
            (e.g. "pcilc-dp-tsz-eps0.1").
    """
    deproject = normalise_deproject(deproject)
    if pcilc_eps is not None:
        eps = float(pcilc_eps)
        if not deproject:
            raise ValueError("pcilc_eps (pcILC) needs the component to partially deproject, e.g. --deproject tsz.")
        if len(deproject) != 1:
            raise ValueError(f"The pcILC bounds a single component; got deproject={deproject}.")
        if eps < 0.0:
            raise ValueError(f"pcilc_eps must be > 0, got {eps}.")
        if eps == 0.0:
            raise ValueError("pcilc_eps=0 is the cILC: drop --pcilc-eps to null the component exactly.")
        return f"pcilc-dp-{deproject[0]}-eps{eps:.6g}"
    if deproject:
        return "cilc-dp-" + "-".join(deproject)
    return "ilc"


def ml_tag(masked: bool = False) -> str:
    """
    ML stage tag placed after the ILC mode in the ML-cleaned map names ({mode}_{ml}_...):
    "ml", or "ml-masked" when the model was trained with the masked loss.
    """
    return "ml-masked" if masked else "ml"
