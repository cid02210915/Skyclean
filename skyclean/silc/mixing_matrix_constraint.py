import os, glob, re
import numpy as np
import healpy as hp

from .utils import _norm

class SpectralVector:
    # ---- THEORETICAL: CMB (flat), tSZ, synchrotron ----
    @staticmethod
    def build_F_theory(beta_s: float = -3.1, nu0: float = 30e9,
                       frequencies: list[str] = None):
        # channels
        if frequencies is None:
            frequencies = ['030','044','070','100','143','217','353','545','857']
        nu = np.array([float(f) for f in frequencies]) * 1e9

        # basic physics
        h = 6.62607015e-34; k = 1.380649e-23; T = 2.7255
        x = h * nu / (k * T)
        g_nu = (x**2 * np.exp(x)) / (np.exp(x) - 1.0)**2

        # vectors (thermodynamic units)
        cmb  = np.ones_like(nu)
        tsz  = x * ((np.exp(x) + 1.0) / (np.exp(x) - 1.0)) - 4.0
        sync = (nu / nu0) ** beta_s / g_nu

        reference_vectors = {"cmb": _norm(cmb), 
                             "tsz": _norm(tsz), 
                             "sync": _norm(sync)}
        
        F_cols = ["cmb", "tsz", "sync"]
        F = np.column_stack([reference_vectors[n] for n in F_cols])
        return F, F_cols, reference_vectors, frequencies

    # ---- EMPIRICAL: masked mean per channel from saved maps ----
    @staticmethod
    def build_F_empirical(
        base_dir: str,
        file_templates: dict[str, str],
        frequencies: list[str],
        realization: int = 0,
        mask_path: str = "",
        normalize: bool = True,
    ):

        # masked mean helper
        def _mean(path, mask):
            M = hp.read_map(path, verbose=False)
            if mask is not None:
                m = hp.ud_grade(mask, nside_out=hp.get_nside(M), power=0) > 0
                vals = M[np.isfinite(M) & m]
            else:
                vals = M[np.isfinite(M)]
            return vals.mean() if vals.size else 0.0

        mask = hp.read_map(mask_path, verbose=False) if mask_path else None
        comps = ["cmb", "tsz", "sync"]

        k = {}
        for comp in comps:
            tmpl = file_templates.get(comp)
            if tmpl is None:
                continue
            vec = np.zeros(len(frequencies), dtype=float)
            for i, f in enumerate(frequencies):
                path = os.path.join(base_dir, tmpl.format(frequency=f, realisation=realization))
                if os.path.exists(path):
                    vec[i] = _mean(path, mask)
            k[comp] = _norm(vec) if normalize else vec

        F_cols = [c for c in comps if c in k]
        reference_vectors = {c: k[c] for c in F_cols}
        F = np.column_stack([reference_vectors[c] for c in F_cols])
        return F, F_cols, reference_vectors, list(frequencies)

    # ---- SWITCH ----
    @staticmethod
    def get_F(source: str = "theory", **kwargs):
        return SpectralVector.build_F_theory(**kwargs) if source == "theory" else SpectralVector.build_F_empirical(**kwargs)


class ILCConstraints:
    def __init__(self, F_cols, F):
        self.F_cols = F_cols
        self.F = F

    def build_f(self, component_name: str) -> np.ndarray:
        if isinstance(component_name, (list, tuple)):
            raise ValueError("build_f expects a single name; pass one of: " + ", ".join(self.F_cols))
        try:
            j = self.F_cols.index(component_name.lower())
        except ValueError:
            raise ValueError(f"Component '{component_name}' not in F columns {self.F_cols}")
        f = np.zeros(self.F.shape[1], dtype=float)
        f[j] = 1.0
        return f
    
    @staticmethod
    def find_f_from_extract_comp(F, extract_comp_or_comps, reference_vectors, allow_sign_flip=False, atol=1e-8):
        if isinstance(extract_comp_or_comps, (str, bytes)):
            names = [extract_comp_or_comps]
        else:
            names = list(extract_comp_or_comps)

        N_comp = F.shape[1]
        f = np.zeros(N_comp, dtype=float)

        for name in names:
            key = name.lower()
            if key not in reference_vectors:
                raise ValueError(f"Reference vector for '{name}' not set in reference_vectors.")
            target_vec = reference_vectors[key]
            t = target_vec / np.linalg.norm(target_vec)

            matched = False
            for j in range(N_comp):
                col = F[:, j] / np.linalg.norm(F[:, j])
                if np.allclose(col, t, atol=atol) or (allow_sign_flip and np.allclose(col, -t, atol=atol)):
                    f[j] = 1.0          # always +1
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Component '{name}' not found in F.")
        return f


