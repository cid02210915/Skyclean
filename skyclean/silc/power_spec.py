import os
import numpy as np
import healpy as hp
import jax.numpy as jnp
import s2fft   # MW forward SHT
import matplotlib.pyplot as plt 

from typing import Optional, Union, Iterable, Dict, Any

class MapAlmConverter:
    def __init__(self, file_templates: Dict[str, str]):
        self.file_templates = file_templates

    def to_alm(
        self,
        component: str,
        *,
        source: str = "downloaded",         # "downloaded" | "processed" | "ilc_synth"
        processed: Optional[bool] = None,
        frequency: Optional[Union[int, str]] = None,
        realisation: Optional[int] = None,
        lmax: Optional[int] = None,
        field: int = 0,
        # ilc_synth-only args:
        extract_comp: Optional[str] = None,
        frequencies: Optional[Union[str, int, Iterable[int]]] = None,
        lam: Optional[Union[int, float, str]] = None,
        nsamp: float | None = None, 
        constraint: bool | None = None, 
    ) -> Dict[str, Any]:

        if processed is not None:
            source = "processed" if processed else "downloaded"

        path = self._format_path(
            component=component, source=source,
            frequency=frequency, realisation=realisation, lmax=lmax,
            extract_comp=extract_comp, frequencies=frequencies, lam=lam, 
            nsamp=nsamp, constraint=constraint,
        )

        # 2) Load the map from disk
        arr = self._load_map(path, field=field)
    
        # 3) Decide sampling by file extension first; fallback to shape
        pl = path.lower()
        if pl.endswith((".fits", ".fit")):
            sampling = "healpix"
        else:
            sampling, _ = self._detect_sampling(arr)  # returns ("healpix"/"mw", info)

        # 4) Transform to alm
        if sampling == "healpix":
            if lmax is None:
                raise ValueError("HEALPix input: please pass lmax explicitly.")
            lmax_used = int(lmax)
            alm_hp = hp.map2alm(arr, lmax=lmax_used, pol=False, iter=0).astype(np.complex128, copy=False)
            return {"alm": alm_hp, "format": "healpy", "lmax": lmax_used, "path": path}
    
        # sampling == 'mw'
        L = arr.shape[0]
        lmax_used = L - 1
        arr = np.asarray(np.real(np.squeeze(arr)), dtype=np.float64, order="C")
        alm_mw = s2fft.forward(arr, L=L)
        return {"alm": alm_mw, "format": "mw", "lmax": lmax_used, "path": path}

    # ---------- internals ----------
    @staticmethod
    def _fmt_has(fmt: str, name: str) -> bool:
        """True if template contains '{name}' (handles '{name:04d}', etc.)."""
        return ("{" + name) in fmt

    def _format_path(
        self,
        *,
        component: str,
        source: str,
        frequency: Optional[Union[int, str]],
        realisation: Optional[int],
        lmax: Optional[int],
        extract_comp: Optional[str],
        frequencies: Optional[Union[str, int, Iterable[int]]],
        lam: Optional[Union[int, float, str]],
        nsamp: Optional[Union[int, float]],  
        constraint: Optional[bool] = None,
    ) -> str:
        
        if source == "downloaded":
            key = component
            fmt = self.file_templates[key]
            kw = {}
            if self._fmt_has(fmt, "frequency"):
                if frequency is None: raise ValueError(f"{key} needs frequency")
                kw["frequency"] = str(frequency)
            if self._fmt_has(fmt, "realisation"):
                if realisation is None: raise ValueError(f"{key} needs realisation")
                kw["realisation"] = int(realisation)
            if self._fmt_has(fmt, "lmax"):
                if lmax is None: raise ValueError(f"{key} needs lmax")
                kw["lmax"] = int(lmax)
            return fmt.format(**kw)

        if source == "processed":
            key = f"processed_{component}"
            fmt = self.file_templates[key]
            kw = {}
            if self._fmt_has(fmt, "frequency"):
                if frequency is None: raise ValueError(f"{key} needs frequency")
                kw["frequency"] = str(frequency)
            if self._fmt_has(fmt, "realisation"):
                if realisation is None: raise ValueError(f"{key} needs realisation")
                kw["realisation"] = int(realisation)
            if self._fmt_has(fmt, "lmax"):
                if lmax is None: raise ValueError(f"{key} needs lmax")
                kw["lmax"] = int(lmax)
            return fmt.format(**kw)

        if source == "ilc_synth":
            fmt = self.file_templates["ilc_synth"]

            # frequency tag: underscores, preserve provided order
            if isinstance(frequencies, (list, tuple)):
                freq_str = "_".join(str(x) for x in frequencies)
            else:
                freq_str = str(frequencies) if frequencies is not None else ""

            if extract_comp is None or realisation is None or lmax is None or lam is None:
                raise ValueError("ilc_synth needs extract_comp, realisation, lmax, lam, and frequencies")

            lam_str = lam if isinstance(lam, str) else f"{float(lam):.1f}"

            if nsamp is None:
                nsamp = 1200  
            nsamp_str = str(int(nsamp))

            mode = "con" if constraint else "uncon"

            kw = dict(
                mode=mode,
                extract_comp=extract_comp,   # target, e.g. 'cmb'
                component=component,         # source mixture, e.g. 'cfn'
                frequencies=freq_str,
                realisation=int(realisation),
                lmax=int(lmax),
                lam=lam_str,
                nsamp=nsamp_str,
            )
            return fmt.format(**kw)

        raise ValueError("source must be 'downloaded', 'processed', or 'ilc_synth'")
    
    def _load_map(self, path: str, field: int) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Map file not found: {path}")
        pl = path.lower()

        # If it’s actually FITS content (even misnamed), read with healpy
        with open(path, "rb") as f:
            head = f.read(10)
        if head.startswith(b"SIMPLE  =") or head.startswith(b"\x53\x49\x4D\x50\x4C\x45\x20\x20\x3D"):
            arr = hp.read_map(path, field=field, dtype=np.float64)
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 3:
                arr = arr[0]
            return np.asarray(arr, dtype=np.float64)

        if pl.endswith((".fits", ".fit")):
            arr = hp.read_map(path, field=field, dtype=np.float64)
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 3:
                arr = arr[0]
            return np.asarray(arr, dtype=np.float64)

        if pl.endswith(".npy"):
            try:
                arr = np.load(path)  # allow_pickle=False by default
            except ValueError as e:
                if "pickled data" not in str(e).lower():
                    raise
                obj = np.load(path, allow_pickle=True)
                if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
                    obj = obj.item()
                if isinstance(obj, dict):
                    for k in ("map", "MW_Pix", "data", "arr", "array"):
                        if k in obj:
                            arr = obj[k]
                            break
                    else:
                        raise ValueError(f"Pickled file lacks a known numeric map key. Keys: {list(obj.keys())}")
                else:
                    arr = obj
            return np.asarray(arr, dtype=np.float64)

        raise ValueError(f"Unsupported file type: {path}")

    def _detect_sampling(self, arr: np.ndarray):
        # HEALPix: 1D with Npix = 12 * nside^2
        if arr.ndim == 1:
            try:
                nside = hp.npix2nside(arr.size)
                return "healpix", {"nside": int(nside)}
            except Exception:
                pass
        # MW: 2D with (L, 2L-1) only (by your spec)
        if arr.ndim == 2:
            n0, n1 = arr.shape
            if n1 == 2 * n0 - 1:
                return "mw", {"L": int(n0)}
        raise ValueError("Could not determine sampling (neither HEALPix nor MW).")

class PowerSpectrumTT:
    '''
    Minimal utilities to compute TT angular power spectra.

    What it does
    ------------
    - `from_mw_alm`          : (MW a_{ℓm})   -> (ℓ, C_ℓ)
    - `from_healpy_alm`      : (healpy alm)  -> (ℓ, C_ℓ)
    - `cl_to_Dl`             : C_ℓ           -> D_ℓ = ℓ(ℓ+1)C_ℓ / (2π)
    '''
    @staticmethod
    def from_mw_alm(alm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        MW alm shape (L, 2L-1), columns m = -(L-1)..(L-1) -> (ell, C_ell)
        """
        alm = np.asarray(alm)
        L = alm.shape[0]
        m0 = L - 1                       # column index of m=0
        cl = np.empty(L, dtype=np.float64)
        for ell in range(L):
            a = alm[ell, m0-ell : m0+ell+1]           # m in [-ell, ell]
            cl[ell] = (a.real*a.real + a.imag*a.imag).sum() / (2*ell + 1)
        return np.arange(L), cl

    @staticmethod
    def from_healpy_alm(alm_hp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """healpy packed alm -> (ell, C_ell)"""
        cl = hp.alm2cl(alm_hp)
        return np.arange(cl.size), cl
    
    
    @staticmethod
    def cl_to_Dl(ell: np.ndarray, cl: np.ndarray, input_unit: str = "K") -> np.ndarray:
        """
        Convert C_ell -> D_ell = ell(ell+1) C_ell / (2π), returning D_ell in µK^2.
    
        input_unit:
          - "K"  : C_ell provided in K^2  -> multiply by 1e12
          - "uK" : C_ell provided in µK^2 -> multiply by 1
        """
        Dl = ell * (ell + 1) * cl / (2.0 * np.pi)
        factor = 1e12 if str(input_unit).lower() in ("k", "kelvin") else 1.0
        return Dl * factor


    @staticmethod
    def plot_Dl_series(curves, save_path=None, show=True):
        """
        curves can be:
          - tuple: (ell, Dl) or (ell, Dl, label) or (ell, Dl, label, style)
          - dict : {"ell":..., "Dl":..., "label":..., "source": "downloaded|processed|ilc_synth", "style": optional}
    
        Auto styles (if 'source' given in dict):
          ilc_synth -> "-"
          processed -> "--"
          downloaded -> "-."
        """
    
        # normalize input to a list
        if isinstance(curves, (dict, tuple)):
            curves = [curves]
    
        style_by_source = {"ilc_synth": "-", "processed": "--", "downloaded": "-."}
    
        plt.figure(figsize=(7, 4))
        any_label = False
        for it in curves:
            if isinstance(it, dict):
                ell = np.asarray(it["ell"]); Dl = np.asarray(it["Dl"])
                label = it.get("label")
                style = it.get("style") or style_by_source.get(it.get("source"), "-")
            else:
                ell = np.asarray(it[0]); Dl = np.asarray(it[1])
                label = it[2] if len(it) > 2 else None
                style = it[3] if len(it) > 3 else "-"
            plt.plot(ell, Dl, style, label=label)
            any_label |= bool(label)
    
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$D_\ell\ [\mu\mathrm{K}^2]$")
        plt.grid(True, alpha=0.5)
        if any_label: plt.legend()
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        if show: plt.show()
    
    @staticmethod
    def load_planck_Dl(directory: str):
        """Load Planck 2018 TT theory from <directory>/cmb_spectrum.txt (cols: ell, Dl[µK^2], ...)."""
        path = os.path.join(directory, "cmb_spectrum.txt")
        if not os.path.exists(path):
            return None
        data = np.loadtxt(path)
        ell = data[:, 0].astype(int)
        Dl  = data[:, 1].astype(np.float64)   # already µK^2
        return ell, Dl
    
    
class PowerSpectrumCrossTT:
    """
    Minimal cross-spectrum utilities (TT):

    - from_mw_alms(ax, ay)        : MW a_{ℓm} (L, 2L-1) -> (ℓ, C_ℓ^{XY})
    - from_healpy_alms(ax, ay)    : healpy-packed alms   -> (ℓ, C_ℓ^{XY})
    - cl_to_Dl(ell, cl, unit)     : C_ℓ -> D_ℓ = ℓ(ℓ+1)C_ℓ/(2π), in µK² if unit="K"
    """

    @staticmethod
    def from_mw_alm(alm_X: np.ndarray, alm_Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cross C_ell from MW layout alms shaped (L, 2L-1); columns m = -(L-1)..+(L-1)."""
        aX = np.asarray(alm_X)
        aY = np.asarray(alm_Y)
        if aX.shape != aY.shape or aX.ndim != 2 or aX.shape[1] != 2 * aX.shape[0] - 1:
            raise ValueError("Expected matching MW alms of shape (L, 2L-1).")

        L = aX.shape[0]
        m0 = L - 1
        Cl = np.empty(L, dtype=np.float64)
        for l in range(L):
            sl = slice(m0 - l, m0 + l + 1)  # m = -l..+l
            Cl[l] = np.real(np.dot(aX[l, sl], np.conj(aY[l, sl]))) / (2 * l + 1)
        return np.arange(L, dtype=int), Cl

    @staticmethod
    def from_healpy_alm(
        alm_X: np.ndarray,
        alm_Y: np.ndarray,
        lmax: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cross C_ell from healpy-packed alms; lmax inferred if not provided."""
        aX = np.asarray(alm_X)
        aY = np.asarray(alm_Y)
        if lmax is None:
            lmax = min(hp.Alm.getlmax(aX.size), hp.Alm.getlmax(aY.size))
        Cl = hp.alm2cl(aX, aY, lmax=int(lmax))
        return np.arange(Cl.size, dtype=int), Cl

    # --------------------------
    # Direct MW -> healpy converter
    # --------------------------
    @staticmethod
    def mw_to_healpy_alm(alm_mw: np.ndarray, lmax: int | None = None) -> np.ndarray:
        """
        Convert MW alm layout (L, 2L-1) with m=-(L-1)..+(L-1)
        to healpy packed alm (m>=0 only).

        We only need m>=0 because healpy stores that convention.
        """
        a = np.asarray(alm_mw)
        if a.ndim != 2 or a.shape[1] != 2 * a.shape[0] - 1:
            raise ValueError("Expected MW alm shape (L, 2L-1).")

        L = int(a.shape[0])
        lmax_used = (L - 1) if lmax is None else int(lmax)
        if lmax_used > L - 1:
            raise ValueError(f"Requested lmax={lmax_used} exceeds MW L-1={L-1}.")

        out = np.empty(hp.Alm.getsize(lmax_used), dtype=np.complex128)
        m0 = L - 1  # column for m=0 in MW layout

        for l in range(lmax_used + 1):
            for m in range(l + 1):
                out[hp.Alm.getidx(lmax_used, l, m)] = a[l, m0 + m]  # pick m>=0 column

        return out

    # --------------------------
    # Direct dispatcher: works for MW×MW, healpy×healpy, MW×healpy
    # --------------------------
    @staticmethod
    def from_any_alms(
        alm_X: np.ndarray,
        fmt_X: str,
        alm_Y: np.ndarray,
        fmt_Y: str,
        *,
        lmax: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cross C_ell^{XY} given alms + format tags.

        fmt_X/fmt_Y must be either:
          - "mw"
          - "healpy"  (or "hp")

        If mixed, convert MW -> healpy packed and use healpy cross.
        """
        fx = str(fmt_X).lower()
        fy = str(fmt_Y).lower()
        if fx == "hp":
            fx = "healpy"
        if fy == "hp":
            fy = "healpy"

        if fx == "mw" and fy == "mw":
            return PowerSpectrumCrossTT.from_mw_alm(alm_X, alm_Y)

        # mixed or both healpy -> do healpy cross (convert MW if needed)
        if fx == "mw":
            ax = PowerSpectrumCrossTT.mw_to_healpy_alm(alm_X, lmax=lmax)
        elif fx == "healpy":
            ax = np.asarray(alm_X)
        else:
            raise ValueError(f"Unknown fmt_X='{fmt_X}' (expected 'mw' or 'healpy').")

        if fy == "mw":
            ay = PowerSpectrumCrossTT.mw_to_healpy_alm(alm_Y, lmax=lmax)
        elif fy == "healpy":
            ay = np.asarray(alm_Y)
        else:
            raise ValueError(f"Unknown fmt_Y='{fmt_Y}' (expected 'mw' or 'healpy').")

        # infer lmax if needed
        if lmax is None:
            lmax = min(hp.Alm.getlmax(ax.size), hp.Alm.getlmax(ay.size))

        return PowerSpectrumCrossTT.from_healpy_alm(ax, ay, lmax=int(lmax))

    @staticmethod
    def cl_to_Dl(ell: np.ndarray, cl: np.ndarray, input_unit: str = "K") -> np.ndarray:
        """D_ell = ell(ell+1) C_ell / (2π); returns µK² if input_unit=='K'."""
        ell = np.asarray(ell)
        cl = np.asarray(cl)
        Dl = ell * (ell + 1) * cl / (2.0 * np.pi)
        return Dl * (1e12 if str(input_unit).lower() in ("k", "kelvin") else 1.0)

    @staticmethod
    def r_ell(Cl_xy: np.ndarray, Cl_xx: np.ndarray, Cl_yy: np.ndarray) -> np.ndarray:
        """r_ell = C_ell^{xy} / sqrt(C_ell^{xx} C_ell^{yy})"""
        Cl_xy = np.asarray(Cl_xy)
        Cl_xx = np.asarray(Cl_xx)
        Cl_yy = np.asarray(Cl_yy)
        denom = np.sqrt(Cl_xx * Cl_yy)
        return Cl_xy / denom

    @staticmethod
    def plot_r_ell(
        ell: np.ndarray,
        Cl_xy: np.ndarray,
        Cl_xx: np.ndarray,
        Cl_yy: np.ndarray,
        *,
        save_path: str | None = None,
        show: bool = True,
        label: str | None = None,
        style: str = "-"
    ) -> np.ndarray:
        """Plot per-ℓ r_ell and return it."""
        ell = np.asarray(ell)
        r = PowerSpectrumCrossTT.r_ell(Cl_xy, Cl_xx, Cl_yy)
        m = np.isfinite(r)

        plt.figure(figsize=(7, 4))
        plt.plot(ell[m], r[m], style, label=label)
        plt.axhline(0.0, ls=":", lw=1)
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$r_\ell = C_\ell^{xy}/\sqrt{C_\ell^{xx} C_\ell^{yy}}$")
        if label:
            plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        return r