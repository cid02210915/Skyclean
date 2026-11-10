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


PYSM_MODELS = ("a1", "a2", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12",
               "f1", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7")


def pysm_model(comp) -> str | None:
    """
    Model code of a PySM foreground component key: "pysm_d1" -> "d1". None for any other component. Raises ValueError
    for pysm_<model> with a model not in PYSM_MODELS. PySM maps are pre-generated (pysm3) in K_CMB, one per model
    and band, so they take no unit conversion in the process stage.
    """
    comp = str(comp)
    if not comp.startswith("pysm_"):
        return None
    model = comp[len("pysm_"):]
    if model not in PYSM_MODELS:
        raise ValueError(f"Unknown PySM model '{model}' in component '{comp}'; choose from {list(PYSM_MODELS)}.")
    return model


# Foreground mixture of the CFN-type products. The plain "cfn" name stands for the full FFP10 set below (the job
# scripts' --components list minus cmb/noise, also what --components all expands to); any other mixture is tagged with
# its sorted foreground tokens (see foreground_tag).
FFP10_FOREGROUNDS = ("sync", "dust", "tsz", "strongirps", "strongradiops", "faintirps", "faintradiops", "clusterirps",
                     "co", "freefree", "ksz", "spindust", "stronguchii", "cib")
FFP10_TOKENS = {"freefree": "ff"}   # filename tokens that differ from the component name
FFP10_FULL_TOKEN = "ffp10"          # stands for the whole FFP10_FOREGROUNDS set in a tag
NO_FOREGROUND_TOKEN = "none"        # cmb + noise only
NON_FOREGROUND_COMPONENTS = ("cmb", "noise", "extra_feature", "all", "real", "total", "mask")
FOREGROUND_TOKENS = PYSM_MODELS + tuple(FFP10_TOKENS.get(c, c) for c in FFP10_FOREGROUNDS) + (FFP10_FULL_TOKEN, NO_FOREGROUND_TOKEN)


def foreground_token(comp) -> str | None:
    """
    Filename token of a foreground component: the PySM model code (pysm_d3 -> d3) or the FFP10 token (freefree -> ff,
    dust -> dust). None for cmb, noise, extra_feature, all, real, total and mask. Raises ValueError for any other name,
    so a typo in --components fails before any map is processed.
    """
    comp = str(comp)
    if comp in NON_FOREGROUND_COMPONENTS:
        return None
    model = pysm_model(comp)
    if model is not None:
        return model
    if comp in FFP10_FOREGROUNDS:
        return FFP10_TOKENS.get(comp, comp)
    raise ValueError(
        f"Unknown component '{comp}': expected cmb, noise, extra_feature, all, pysm_<model> or one of {list(FFP10_FOREGROUNDS)}."
    )


def foreground_tag(components) -> str:
    """
    Mixture tag appended to the CFN-type product name: the sorted foreground tokens of ``components``, e.g.
    "-a1-d3-f1-s1" (PySM only), "-dust-ff-spindust-sync" (an FFP10 subset), "-d3-ffp10" (full FFP10 plus a PySM
    model), "-none" (cmb + noise only). The full FFP10_FOREGROUNDS set ('all' counts as full) collapses to "ffp10" and
    on its own gives "", so the plain "cfn" of the existing full-FFP10 runs is unchanged. cmb and noise are never
    listed: the base name (cfn, fn, ...) already says they are there.
    """
    comps = [str(c) for c in components]
    if "all" in comps:
        comps += list(FFP10_FOREGROUNDS)
    tokens = {t for t in (foreground_token(c) for c in comps) if t is not None}
    ffp10 = {c for c in comps if c in FFP10_FOREGROUNDS}
    if ffp10 == set(FFP10_FOREGROUNDS):
        tokens = (tokens - {FFP10_TOKENS.get(c, c) for c in ffp10}) | {FFP10_FULL_TOKEN}
    if tokens == {FFP10_FULL_TOKEN}:
        return ""
    if not tokens:
        tokens = {NO_FOREGROUND_TOKEN}
    return "".join(f"-{t}" for t in sorted(tokens))


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


def b_tol(N_directions) -> float:
    """
    Covariance-smoothing tolerance b_tol used by the SILC covariance (ilc.py) as a function of the number of
    wavelet directions: 0.01 * N (N=1 -> 0.01, N=2 -> 0.02, N=4 -> 0.04). Single definition shared by the
    covariance computation and the output filenames (see b_tol_tag), so the two cannot drift apart.
    """
    return round(0.01 * int(N_directions), 6)


def b_tol_tag(N_directions) -> str:
    """Filename field for b_tol, e.g. "btol0.04"; fills {btol} in the b_tol-dependent FileTemplates entries."""
    return f"btol{b_tol(N_directions):g}"


def ml_tag(masked: bool = False) -> str:
    """
    ML stage tag placed after the ILC mode in the ML-cleaned map names ({mode}_{ml}_...):
    "ml", or "ml-masked" when the model was trained with the masked loss.
    """
    return "ml-masked" if masked else "ml"
