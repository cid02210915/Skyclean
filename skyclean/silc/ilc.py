from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import os
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2wav
import math 
import healpy as hp
from s2wav import filters
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import s2wav.filters as filters

from .map_tools import *
from .utils import *
from .custom_s2wav_bandlimits import *
from .file_templates import FileTemplates
from .utils import normalize_targets   
from .utils import save_array
from .mixing_matrix_constraint import ILCConstraints 
import concurrent.futures
import time
from .harmonic_response import AxisymmetricGenerators, SimpleHarmonicWindows
from .custom_s2wav_bandlimits import j_max_silc

lam_list = [2, 2, 2, 2,
            2, 2, 2, 2,
            2, 2, 2, 2]
ell_peak = np.array([2, 5, 10, 19, 38, 77, 153, 307, 614, 1227, 2454, 3600], dtype=int)


class SILCTools():
    '''Tools for Scale-discretised, directional wavelet ILC (SILC).'''

    @staticmethod
    def Single_Map_doubleworker(mw_map: np.ndarray, method: str):

        def double_one_map(m):
            alm = s2fft.forward(
                m,
                L=m.shape[0],
                method=method,
                spmd=False,
                reality=True,
            )

            L = alm.shape[0]
            H = 2 * L - 1
            W = 2 * H - 1

            padded = np.zeros((H, W), dtype=np.complex128)

            mid_in = alm.shape[1] // 2
            mid_out = W // 2
            start = mid_out - mid_in

            padded[:L, start:start + alm.shape[1]] = alm

            x2 = np.real(
                s2fft.inverse(
                    padded,
                    L=H,
                    method=method,
                    spmd=False,
                    reality=True,
                )
            )

            return x2

        mw_map = np.asarray(mw_map)

        if mw_map.ndim == 2:
            return double_one_map(mw_map)

        if mw_map.ndim == 3:
            doubled_dirs = [
                double_one_map(mw_map[d])
                for d in range(mw_map.shape[0])
            ]
            return np.stack(doubled_dirs, axis=0)

        raise ValueError(
            f"Single_Map_doubleworker expects 2D or 3D input, got shape {mw_map.shape}"
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def _cached_gauss_beam(L: int, fwhm_rad: float) -> np.ndarray:
        """B_ell for ell=0..L-1, Gaussian beam with FWHM in radians."""
        return hp.gauss_beam(float(fwhm_rad), lmax=int(L) - 1)  # (L,)
    
    # ------------------------------------------------------------
    # Paper helpers: sigma^2, f_sky, conversions (Eqs 44 + text)
    # ------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=256)
    def sigma2_from_fwhm(fwhm_rad: float) -> float:
        """Eq (44) rearranged: sigma^2 = FWHM^2 / (8 ln 2)."""
        fwhm_rad = float(fwhm_rad)
        if fwhm_rad <= 0.0:
            return 0.0
        return (fwhm_rad * fwhm_rad) / (8.0 * math.log(2.0))
    
    @staticmethod
    @lru_cache(maxsize=256)
    def f_sky_paper_from_fwhm(fwhm_rad: float) -> float:
        """Paper approximation: f_sky ≈ sigma^2 / 2."""
        return 0.5 * SILCTools.sigma2_from_fwhm(float(fwhm_rad))
    
    @staticmethod
    def fwhm_from_sigma2(sigma2: float) -> float:
        """Eq (44): FWHM = sqrt(8 ln2 * sigma^2)."""
        sigma2 = float(sigma2)
        if sigma2 <= 0.0:
            return 0.0
        return float(np.sqrt(8.0 * math.log(2.0) * sigma2))
    
    # ============================================================
    # (B) Gaussian real-space window -> effective pixel count
    #     (DIAGNOSTIC ONLY; NOT USED BY PAPER LOGIC)
    # ============================================================

    @staticmethod
    @lru_cache(maxsize=64)
    def n_pix_eff_gaussian_fullsky(fwhm_rad: float, nside: int) -> float:
        """
        Diagnostic only:
            N_pix,eff = (ΣW)^2 / Σ(W^2)
        """
        fwhm_rad = float(fwhm_rad)
        nside = int(nside)
        if fwhm_rad <= 0.0:
            return 0.0
        sigma = fwhm_rad / math.sqrt(8.0 * math.log(2.0))
        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))  # centered at pole
        W = np.exp(-0.5 * (theta / sigma) ** 2)
        s1 = float(np.sum(W))
        s2 = float(np.sum(W * W))
        return 0.0 if s2 == 0.0 else (s1 * s1) / s2
    
    @staticmethod
    @lru_cache(maxsize=256)
    def gaussian_pixel_counts_fullsky(fwhm_rad: float, nside: int, k_sigma: float = 3.0):
        """
        Diagnostic helper:
          N_sphere_hp : total HEALPix pixels on sphere
          N_in_gauss  : pixels with theta <= k_sigma*sigma (hard cut; not paper)
          N_pix_eff   : (ΣW)^2/Σ(W^2) (not paper)
          sigma       : sigma in radians
          theta_cut   : cutoff radius in radians
          f_sky_paper : sigma^2/2 from paper
          f_sky_eff_hp: N_pix_eff / N_pix_fullsky(HP)
        """
        fwhm_rad = float(fwhm_rad)
        nside = int(nside)
        k_sigma = float(k_sigma)
        sigma = fwhm_rad / math.sqrt(8.0 * math.log(2.0))
        theta_cut = min(k_sigma * sigma, math.pi)
        npix = hp.nside2npix(nside)
        theta, _ = hp.pix2ang(nside, np.arange(npix))
        N_in_gauss = int(np.count_nonzero(theta <= theta_cut))
        W = np.exp(-0.5 * (theta / sigma) ** 2)
        s1 = float(np.sum(W))
        s2 = float(np.sum(W * W))
        N_pix_eff = 0.0 if s2 == 0.0 else (s1 * s1) / s2
        f_sky_paper = SILCTools.f_sky_paper_from_fwhm(fwhm_rad)
        f_sky_eff_hp = float(N_pix_eff) / float(npix) if npix > 0 else float("nan")
        return (
            int(npix),
            int(N_in_gauss),
            float(N_pix_eff),
            float(sigma),
            float(theta_cut),
            float(f_sky_paper),
            float(f_sky_eff_hp),
        )
    
    # ============================================================
    # (C) Paper-style per-band FWHM (Eqs 42–44) and N_modes
    # ============================================================
    @staticmethod
    def fwhm_rad_wavelet(
        L: int,
        lam_list: list[float],
        ell_peak: list[float] | np.ndarray,  
        j: int,
        Nfreq: int,
        lmax: int, 
        Ndeproj: int = 0,
        N_directions: int = 1,
        direction_index: int | None = None,
        b_tol: float = 0.02,

    ) -> float:
        """
        Paper Eqs (42)-(44): per-band real-space Gaussian FWHM (radians) for WAVELET band j.

        Fix: use the smooth band window h_l^j = kappa_l^j, with per-band lambda.
        """
        L = int(L)
        j = int(j)

        if j < 0 or j >= len(lam_list):
            raise ValueError(f"j={j} out of range for lam_list (len={len(lam_list)})")

        ells = np.arange(L, dtype=float)

        ell_min = int(L0_j_silc(j))
        ell_max = int(wav_j_bandlimit_silc(L, j, multiresolution=True))

        lmax = int(lmax)
        ell_data_max = min(L - 1, lmax)

        ell_min = max(0, min(ell_min, ell_data_max))
        ell_max = max(0, min(ell_max, ell_data_max))
        if ell_min > ell_max:
            return 0.0

        lam_j = float(lam_list[j])
        gen = AxisymmetricGenerators(lam_j)

        # Evaluate kappa on a dimensionless argument; then hard-mask to the bank support.
        ell_peak_j = float(ell_peak[j])
        x = ells / ell_peak_j if ell_peak_j > 0 else ells
        h = gen.kappa(x)

        mask = (ells >= ell_min) & (ells <= ell_max)
        #print(f"[fwhm_wavelet] L={L} j={j} ell_min={ell_min} ell_max={ell_max} ")
        h = h * mask

        # Eq (42)
        weights = (2.0 * ells + 1.0) * (h * h)
        '''
        if int(N_directions) > 1 and direction_index is not None:
            s_elm = np.asarray(filters.tiling_direction(L, int(N_directions)))
            S = float(np.sum(weights * np.abs(s_elm[:, int(direction_index)])**2))
        else:
            S = float(np.sum(weights))
        
        print(
        f"[DBG] j={j} d={direction_index} "
        f"ell=[{ell_min},{ell_max}] "
        f"hmax={np.max(np.abs(h)):.3e} "
        f"S={S:.3e}"
        )
        '''
        
        S = float(np.sum(weights))

        width = ell_max - ell_min + 1
        if S > 0:
            A = abs(1 + int(Ndeproj) - int(Nfreq))
            sigma2_dbg = 2.0 * (A / (float(b_tol) * S))
            fwhm_arcmin_dbg = SILCTools.fwhm_from_sigma2(sigma2_dbg) * (180.0/np.pi) * 60.0
        else:
            fwhm_arcmin_dbg = float("inf")
        '''
        print(
            f"[band diag] L={L} j={j} "
            f"ell_min={ell_min} ell_max={ell_max} width={width} "
            f"S={S:.3e} fwhm~{fwhm_arcmin_dbg:.2f} arcmin"
        )
        '''
        # Eq (43)
        A = abs(1 + int(Ndeproj) - int(Nfreq))
        sigma2 = 0.0 if S <= 0.0 else 2.0 * (A / (float(b_tol) * S))

        # Eq (44)
        return SILCTools.fwhm_from_sigma2(sigma2)

    @staticmethod
    def fwhm_rad_scaling(
        L: int,
        lam: float,
        J: int,
        Nfreq: int,
        Ndeproj: int = 0,
        b_tol: float = 0.02,
        scal_ell_cut: float = 64.0,
        scal_ell_max: int = 64,
    ) -> float:
        """
        Paper Eqs (42)-(44) for SCALING band, consistent with Table-1 scal: ell=0..64.

        Uses: h_ell = eta(ell / scal_ell_cut), then truncates to ell<=scal_ell_max.
        """
        L = int(L)
        gen = AxisymmetricGenerators(float(lam))
        ells = np.arange(L, dtype=float)

        # scaling window shape (consistent with SimpleHarmonicWindows.scaling_raw)
        t = ells / float(scal_ell_cut)
        h = gen.eta(t)

        # Table-1 truncation: ell in [0, 64]
        ell_max_clip = min(int(scal_ell_max), L - 1)
        h *= (ells <= float(ell_max_clip))

        # Eq (42)
        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))

        # Eq (43)
        A = abs(1 + int(Ndeproj) - int(Nfreq))
        sigma2 = 0.0 if S <= 0.0 else 2.0 * (A / (float(b_tol) * S))

        # Eq (44)
        return SILCTools.fwhm_from_sigma2(sigma2)


    @staticmethod
    def n_modes_scaling_band(
        L: int,
        lam: float,
        J: int,
        fwhm_rad: float,
        scal_ell_cut: float = 64.0,
        scal_ell_max: int = 64,
    ) -> float:
        """
        Paper local modes for scaling band, consistent with Table-1 scal: ell=0..64.

          N_modes(local) = f_sky(paper) * Σ_l (2l+1) h_l^2
        with h_l = eta(l / scal_ell_cut), truncated to l<=scal_ell_max.
        """
        L = int(L)
        gen = AxisymmetricGenerators(float(lam))
        ells = np.arange(L, dtype=float)

        t = ells / float(scal_ell_cut)
        h = gen.eta(t)

        ell_max_clip = min(int(scal_ell_max), L - 1)
        h *= (ells <= float(ell_max_clip))

        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))  # Eq (42)
        f_sky = SILCTools.f_sky_paper_from_fwhm(float(fwhm_rad))
        return f_sky * S

    
    @staticmethod
    def n_modes_wavelet_band(
        L: int,
        lam_list: list[float],
        band_j: int,
        fwhm_rad: float,
        lmax: int,
        nside_nmodes: int = 2048
    ) -> float:
        """
        PAPER local modes for wavelet band j:
            N_modes(local) = f_sky(paper) * Σ_l (2l+1) (h_l^j)^2
        with h_l^j = kappa_l^j.
        """
        L = int(L)
        j = int(band_j)

        if j < 0 or j >= len(lam_list):
            raise ValueError(f"band_j={j} out of range for lam_list (len={len(lam_list)})")

        ells = np.arange(L, dtype=float)
        ell_data_max = min(L - 1, int(lmax))

        ell_min_bank = int(L0_j_silc(j))
        ell_max_bank = int(wav_j_bandlimit_silc(L, j, multiresolution=True))

        if ell_min_bank > ell_data_max:
            return 0.0

        ell_min = max(0, ell_min_bank)
        ell_max = min(ell_max_bank, ell_data_max)
        if ell_min > ell_max:
            return 0.0

        lam_j = float(lam_list[j])
        gen = AxisymmetricGenerators(lam_j)

        # Dimensionless argument places the bump in the band; then we hard-mask to bank support.
        x = ells / float(ell_peak[j]) if ell_peak[j] > 0 else ells
        h = gen.kappa(x)

        mask = (ells >= ell_min) & (ells <= ell_max)
        #print(f"[n_modes_wavelet_band] L={L} j={j} ell_min={ell_min} ell_max={ell_max} ")
        h = h * mask

        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))
        f_sky = SILCTools.f_sky_paper_from_fwhm(float(fwhm_rad))
        return f_sky * S

    # ============================================================
    # (D)  smoothed_covariance (pipeline-safe)
    #     + optional info return (does NOT change default behaviour)
    # ============================================================

    @staticmethod
    def smoothed_covariance(
        MW_Map1: np.ndarray,
        MW_Map2: np.ndarray,
        method: str = "jax_cuda",
        nsamp: float = 1200,  # kept for API compatibility; ignored in paper-only usage
        *,
        return_info: bool = False,
        use_paper_fwhm: bool = False,  # kept for API compatibility; forced True below
        lam: float | None = None,
        lmax: int,
        N_directions: int,
        direction_index: int | None = None,
        band_j: int | None = None,
        J_scal: int | None = None,   # kept for API compatibility, ignored for SILC
        scaling: bool = False,
        Nfreq: int | None = None,
        Ndeproj: int = 0,
        b_tol: float = 0.02,
        # For diagnostics:
        nside_nmodes: int = 2048,
        k_sigma: float = 3.0,
    ):
        """
        PAPER-ONLY behaviour:
          - compute per-band FWHM via Eqs (42)-(44)
          - beam = hp.gauss_beam(FWHM)
          - smooth real-space product

        nsamp/use_paper_fwhm kept only so upstream pipeline calls don't break.
        """
        # ---- force paper path ----
        use_paper_fwhm = True

        if lam is None or Nfreq is None:
            raise ValueError("Paper mode requires lam and Nfreq.")

        L = int(MW_Map1.shape[0])
        lmax = int(lmax)

        # --- ensure MW sampling width=2L-1 only if needed ---
        exp = 2 * L - 1

        def _ensure_mw(a: np.ndarray) -> np.ndarray:
            nphi = a.shape[1]
            if nphi == exp:
                return a
            if nphi > exp:
                return a[:, :exp]
            reps = (exp + nphi - 1) // nphi
            return np.tile(a, reps)[:, :exp]

        map1 = _ensure_mw(np.real(MW_Map1))
        map2 = _ensure_mw(np.real(MW_Map2))

        if L < 2 or map1.shape[1] != exp:
            out0 = np.real(map1 * map2)
            return (out0, {}) if return_info else out0

        # ----- forward product -----
        Rpix = map1 * map2
        Ralm = s2fft.forward(Rpix, L=L, method=method, spmd=False, reality=True)

        # ----- paper FWHM -----
        info: dict = {}

        if scaling:
            fwhm_rad = SILCTools.fwhm_rad_scaling(
                L, float(lam), 0, int(Nfreq), int(Ndeproj), float(b_tol)
            )
            info["band"] = "scaling"
        else:
            if band_j is None:
                raise ValueError("Paper wavelet mode requires band_j.")
            fwhm_rad = SILCTools.fwhm_rad_wavelet(
                L,
                lam_list,
                ell_peak,
                int(band_j),
                int(Nfreq),
                lmax=int(lmax),
                N_directions=int(N_directions),
                direction_index=direction_index,
                Ndeproj=int(Ndeproj),
                b_tol=float(b_tol),
            )

            info["band"] = "wavelet"
            info["band_j"] = int(band_j)

        beam = SILCTools._cached_gauss_beam(int(Ralm.shape[0]), float(fwhm_rad))

        # ---- info ----
        info.update({
            "beam_source": "paper",
            "fwhm_rad": float(fwhm_rad),
            "lam": float(lam),
            "Nfreq": int(Nfreq),
            "Ndeproj": int(Ndeproj),
            "b_tol": float(b_tol),
            # paper: f_sky = sigma^2/2, sigma^2 = FWHM^2/(8 ln2)
            "f_sky_paper": float(SILCTools.f_sky_paper_from_fwhm(float(fwhm_rad))),
        })

        # optional diagnostics: HEALPix-consistent effective sky fraction
        if hasattr(SILCTools, "gaussian_pixel_counts_fullsky"):
            try:
                # (npix_hp, N_in_gauss, N_pix_eff, sigma, theta_cut, f_sky_paper, f_sky_eff_hp)
                out = SILCTools.gaussian_pixel_counts_fullsky(float(fwhm_rad), int(nside_nmodes), float(k_sigma))
                if len(out) == 7:
                    npix_hp, N_in_gauss, N_pix_eff, sigma, theta_cut, fsky_p, fsky_eff_hp = out
                    info.update({
                        "nside_nmodes": int(nside_nmodes),
                        "k_sigma": float(k_sigma),
                        "sigma_rad": float(sigma),
                        "theta_cut_rad": float(theta_cut),
                        "N_sphere_hp": int(npix_hp),
                        "N_in_gauss": int(N_in_gauss),
                        "N_pix_eff": float(N_pix_eff),
                        "f_sky_eff_hp": float(fsky_eff_hp),
                    })
                else:
                    # fallback for older 5-tuple version
                    npix_hp, N_in_gauss, N_pix_eff, sigma, theta_cut = out
                    info.update({
                        "nside_nmodes": int(nside_nmodes),
                        "k_sigma": float(k_sigma),
                        "sigma_rad": float(sigma),
                        "theta_cut_rad": float(theta_cut),
                        "N_sphere_hp": int(npix_hp),
                        "N_in_gauss": int(N_in_gauss),
                        "N_pix_eff": float(N_pix_eff),
                        "f_sky_eff_hp": float(N_pix_eff) / float(npix_hp) if npix_hp > 0 else float("nan"),
                    })
            except Exception:
                pass

        # ----- apply beam + inverse -----
        if method == "numpy":
            convolved = Ralm * beam[:, None].astype(Ralm.dtype, copy=False)
            out = s2fft.inverse(convolved, L=L, method="numpy", spmd=False, reality=True)
            smoothed = np.real(out)
        else:
            import jax.numpy as jnp
            jbeam = jnp.asarray(beam, dtype=Ralm.dtype)
            convolved = Ralm * jbeam[:, None]
            out = s2fft.inverse(convolved, L=L, method=method, spmd=False, reality=True)
            smoothed = np.asarray(jnp.real(out))

        return (smoothed, info) if return_info else smoothed


    @staticmethod
    def compute_covariance(task):
        """
        PAPER-ONLY:
          task: (i, fq, frequencies, scale, doubled_MW_wav_c_j,
                 method, lam, lmax, Ndeproj, b_tol, N_directions)
        """
        i, fq, frequencies, scale, doubled_MW_wav_c_j, method, lam, lmax, Ndeproj, b_tol, N_directions = task

        key_i  = (frequencies[i], scale)
        key_fq = (frequencies[fq], scale)

        if key_i not in doubled_MW_wav_c_j or key_fq not in doubled_MW_wav_c_j:
            raise KeyError(f"Missing data for keys {key_i} or {key_fq}.")

        map_i = doubled_MW_wav_c_j[key_i]
        map_fq = doubled_MW_wav_c_j[key_fq]

        L = int(map_i.shape[-2])

        smoothed_list = []
        info_list = []

        for d in range(map_i.shape[0]):

            smoothed_d, info_d = SILCTools.smoothed_covariance(
                map_i[d],
                map_fq[d],
                method=method,
                nsamp=0.0,
                return_info=True,
                use_paper_fwhm=True,
                lam=float(lam),
                lmax=int(lmax),
                N_directions=int(N_directions),
                direction_index=d,
                Nfreq=len(frequencies),
                Ndeproj=int(Ndeproj),
                b_tol=float(b_tol),
                scaling=(int(scale) == 0),
                band_j=(int(scale) - 1) if int(scale) > 0 else None,
            )

            smoothed_list.append(np.asarray(smoothed_d))
            info_list.append(info_d)

        smoothed = np.stack(smoothed_list, axis=0)

        # ---- print once per scale, for every direction ----
        if (i == 0) and (fq == 0):
        
            for d, info_d in enumerate(info_list):
            
                if info_d is None or info_d.get("beam_source") != "paper":
                    continue
                
                fwhm_rad = float(info_d["fwhm_rad"])
                fwhm_arcmin = fwhm_rad * (180.0 / np.pi) * 60.0
        
                f_sky_paper = float(info_d.get("f_sky_paper", float("nan")))
                f_sky_eff_hp = float(info_d.get("f_sky_eff_hp", float("nan")))
        
                print(
                    f"[locality] "
                    f"scale={scale} "
                    f"d={d} "
                    f"FWHM={fwhm_arcmin:.3f} arcmin "
                    f"f_sky(paper)={f_sky_paper:.3e} "
                    f"f_sky(eff_hp,diag)={f_sky_eff_hp:.3e}"
                )
        return i, fq, smoothed


    @staticmethod
    def calculate_covariance_matrix(frequencies: list, doubled_MW_wav_c_j: dict, scale: int,
                                    realisation: int, method: str, path_template: str, *,
                                    component: str = "cfn", lmax: int, lam: float | None = None,
                                    N_directions: int = 1,
                                    nsamp: float = 1200, overwrite: bool = False,
                                    Ndeproj: int = 0, b_tol: float = 0.02,):
        
        #print("calculate_covariance_matrix", flush = True)

        """
        Calculates the covariance matrices for given frequencies and saves them to disk,
        accommodating any size of the input data arrays.
        """

        if not frequencies:
                raise ValueError("Frequency list is empty.")

        # --- normalization to fit current pipeline ---
        norm_freqs = [str(f).zfill(3) for f in frequencies]
        scale_i = int(scale)
    
        f_str = "_".join(norm_freqs)
        lam_str = str(lam)
        save_path = path_template.format(
            component=component,
            frequencies=f_str,
            scale=scale_i,
            N_directions=int(N_directions),
            realisation=int(realisation),
            lmax=int(lmax),
            lam=lam_str,
            nsamp=nsamp,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        # early exit: reuse existing matrix
        if (not overwrite) and os.path.exists(save_path):
            return np.load(save_path)
    
        # --- only now do the heavy work ---
        sample_data = doubled_MW_wav_c_j[(norm_freqs[0], scale_i)]
        total_frequency = len(norm_freqs)

        if sample_data.ndim == 2:
            n_rows, n_cols = sample_data.shape
            full_array = np.zeros(
                (total_frequency, total_frequency, n_rows, n_cols)
            )

        elif sample_data.ndim == 3:
            n_dir, n_rows, n_cols = sample_data.shape
            full_array = np.zeros(
                (total_frequency, total_frequency, n_dir, n_rows, n_cols)
            )

        else:
            raise ValueError(
                f"Unexpected sample_data shape {sample_data.shape}; "
                "expected 2D or 3D."
            )

        tasks = [
            (i, fq, norm_freqs, scale_i, doubled_MW_wav_c_j, method, lam, lmax, int(Ndeproj), float(b_tol), int(N_directions))
            for i in range(total_frequency)
            for fq in range(i, total_frequency)
        ]
    
        if method != "numpy":
            for t in tasks:
                i, fq, cov = SILCTools.compute_covariance(t)
                full_array[i, fq] = cov
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(mp_context=ctx) as executor:
                futures = [executor.submit(SILCTools.compute_covariance, t) for t in tasks]
                for fut in as_completed(futures):
                    i, fq, covariance_matrix = fut.result()
                    full_array[i, fq] = covariance_matrix
    
        for l1 in range(1, total_frequency):
            for l2 in range(l1):
                full_array[l1, l2] = full_array[l2, l1]
    
        np.save(save_path, full_array)
        return full_array
    

    @staticmethod
    def double_and_save_wavelet_maps(
        original_wavelet_c_j, frequencies, scales, realisation,
        component, path_template, *, lmax=64, lam: float | None = None,
        N_directions: int = 1,
        method="jax_cuda", nsamp: float | None = None, overwrite: bool = False,
    ):
        """Compute + save doubled maps serially (no MP), respecting overwrite."""
        doubled_MW_wav_c_j = {}

        for f in frequencies:
            for s in scales:
                freq_tag = str(f).zfill(3)   # <-- force "090" everywhere
                s = int(s)

                out_path = path_template.format(
                    component=component,
                    frequency=freq_tag,
                    scale=s,
                    realisation=int(realisation),
                    lmax=int(lmax),
                    lam=str(lam),
                    nsamp=nsamp,
                    N_directions=N_directions
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                if (not overwrite) and os.path.exists(out_path):
                    doubled = np.load(out_path)
                else:
                    src = original_wavelet_c_j.get((freq_tag, s))
                    if src is None:
                        src = original_wavelet_c_j[(f, s)] 

                    doubled = SILCTools.Single_Map_doubleworker(src, method)
                    np.save(out_path, np.asarray(doubled))

                doubled_MW_wav_c_j[(freq_tag, s)] = doubled

        return doubled_MW_wav_c_j


    @staticmethod
    def compute_weight_vector(R: np.ndarray, scale: int, realisation: int):
        #print("compute_weight_vector", flush = True)
        """
        Processes the given 4D matrix R by computing and saving the weight vectors for each matrix in the first two dimensions.
        Also stores results in memory as arrays and saves them to disk. Adjusts the size of the identity vector based on sub-matrix size.

        Parameters:
            R (np.ndarray): A 4D matrix with dimensions suitable for swapping and inverting.
            scale (int): The scale.
            realisation (int): The realisation.
        Returns:
            inverses: (np.ndarray): An Array containing the inverse matrices
            weight_vectors (np.ndarray): A 3D Array containing the weight vector.
            The size of the first two dimensions of the weight vector is the size of the wavelet coefficient map at the given scale.
            The third dimension is the weight vector (The contribution from each frequency).
            Each element of the weight vector is a 1D array.
            singular_matrices_location (list): The locations of singular matrices.
        """
        print('R_shape:', R.shape)
        # Swap the axes to get R_Pix
        R_Pix = np.swapaxes(np.swapaxes(R, 0, 2), 1, 3) #(pix,pix,freq,freq)
        # Get dimensions for looping and size of sub-matrices
        dim1, dim2, subdim1, subdim2 = R_Pix.shape
        # print(dim1, dim2, subdim1, subdim2)
        # Create arrays to store inverses and weight vectors
        inverses = np.zeros((dim1, dim2, subdim1, subdim2))
        weight_vectors = np.zeros((dim1, dim2, subdim1)) # weight vector at each pixel (dim1,dim2) and channel
        # Realistion 6 has a singular matrix
        # Adjust identity vector size based on sub-matrix dimensions
        identity_vector = np.ones(subdim2, dtype=float)
        singular_matrices_location = []
        singular_matrices = []
        
        for i in range(dim1):
            for j in range(dim2):
                det = np.linalg.det(R_Pix[i, j] ) 

                if det == 0:
                    zeros = np.zeros((subdim1))
                    singular_matrices_location.append((i,j))
                    singular_matrices.append(R_Pix[i, j])
                    weight_vectors[i, j] = zeros
                else:
                    inverses[i, j] = np.linalg.inv(R_Pix[i, j])
                    numerator = np.dot(inverses[i, j], identity_vector)
                    denominator = np.dot(np.dot(inverses[i, j], identity_vector),identity_vector)
                    weight_vectors[i, j] = numerator / denominator
        if len(singular_matrices_location) > 0:
            print("Discovered ", len(singular_matrices_location), "singular matrices at scale", scale, "realisation", realisation)
        return weight_vectors
    
    
    def compute_weights_generalised(
        R,
        scale,
        realisation,
        weight_vector_matrix_template,
        comp,
        L_max,
        extract_comp,
        *,
        N_directions: int = 1,
        constraint=False,
        F=None,
        f=None,
        reference_vectors=None,
        lam: float | None = None,
        nsamp: float | None = None,
        overwrite: bool = False,
        # ---- pcILC (inequality / boundary) ----
        pcilc: bool = False,
        pcilc_component: str | None = None,      # name to auto-lookup in reference_vectors if pcilc_b is None
        pcilc_b=None,                             # explicit b-vector (Nf,)
        pcilc_eps: float | None = None,          # epsilon
        pcilc_pick: str = "minvar",              # "minvar" or "sign"
    ):
        """
        Computes weight vectors from a covariance matrix R using:
          - standard ILC (constraint=False),
          - hard constrained ILC / cILC (constraint=True),
          - pcILC (pcilc=True): |w^T b| <= eps enforced by boundary solutions.

        Args:
            R: (Nf,Nf) or (H,W,Nf,Nf)
            scale: wavelet scale index
            realisation: realisation id (int-like)
            weight_vector_matrix_template: save path template
            comp: component tag for filenames (e.g. 'cfn')
            L_max: L_max = lmax + 1
            extract_comp: target component name string (e.g. 'cmb' or 'tsz')
            constraint: if True, use cILC with (F,f)
            F: (Nf,Nc) spectral response matrix for constraints
            f: (Nc,) constraint target vector
            reference_vectors: dict with named SED vectors (e.g. {"tsz": ...})
            lam, nsamp, overwrite: bookkeeping / naming

            pcilc: if True, use pcILC (paper inequality formulation)
            pcilc_component: name of b-vector to auto lookup in reference_vectors (e.g. "tsz")
            pcilc_b: explicit b-vector, overrides auto lookup
            pcilc_eps: epsilon tolerance
            pcilc_pick: choose boundary solution ("minvar" or "sign")
        """
        import os
        import numpy as np

        # -----------------------------
        # Basic checks / shapes
        # -----------------------------
        if pcilc and constraint:
            raise ValueError("Choose either constraint=True (cILC) OR pcilc=True (pcILC), not both.")
        ''' 
        print("\n=== compute_weights_generalised ===")
        print("scale =", scale)
        print("R.shape =", R.shape)
        print("R.ndim  =", R.ndim)
        print("=================================\n")
        '''
        if R.ndim == 2:
            # Treat as a single 'pixel'
            R_Pix = R.reshape((1, 1, R.shape[0], R.shape[1]))
        elif R.ndim == 4:
            # original layout expects (Nf,Nf,H,W) or (H,W,Nf,Nf)?
            # do: swapaxes twice to get (H,W,Nf,Nf). Keep that behaviour.
            R_Pix = np.swapaxes(np.swapaxes(R, 0, 2), 1, 3)
        elif R.ndim == 5:
            # (Nf, Nf, D, H, W) -> (D, H, W, Nf, Nf)
            R_Pix = np.moveaxis(R, (0, 1), (-2, -1))
        else:
            raise ValueError(f"R must have ndim 2, 4, or 5. Got ndim={R.ndim} with shape {R.shape}")
        #print("R_Pix.shape =", R_Pix.shape)

        spatial_shape = R_Pix.shape[:-2]
        subdim1, subdim2 = R_Pix.shape[-2:]
        if subdim1 != subdim2:
            raise ValueError(f"Covariance sub-matrix must be square. Got {subdim1}x{subdim2}")
        
        #print("Covariance sub-matrix shape:", (subdim1, subdim2))

        N_freq = subdim2

        inverses = np.zeros((*spatial_shape, N_freq, N_freq))
        weight_vectors = np.zeros((*spatial_shape, N_freq), dtype=float)

        singular_matrices_location = []
        singular_matrices = []

        # ---------------------------------------
        # Standard ILC identity_vector (non-constraint)
        # ---------------------------------------
        identity_vector = np.ones(N_freq, dtype=float)

        # ---------------------------------------
        # pcILC configuration (a,b,eps)
        # Here we implement CMB-preserving pcILC:
        #   w^T a = 1 with a = ones (K_CMB convention)
        #   |w^T b| <= eps
        # ---------------------------------------
        eps_str = "None"
        if pcilc:
            if pcilc_eps is None:
                raise ValueError("pcilc=True requires pcilc_eps (epsilon).")
            eps = float(pcilc_eps)
            if eps < 0.0:
                raise ValueError("pcilc_eps must be >= 0.")
            eps_str = f"{eps:.6g}"

            # preserve CMB by default in pcILC
            if reference_vectors is None or "cmb" not in reference_vectors:
                raise ValueError("pcilc=True needs reference_vectors['cmb'] to preserve CMB.")
            a_vec = np.asarray(reference_vectors["cmb"], dtype=float).reshape((-1,))
            if a_vec.shape != (N_freq,):
                raise ValueError(f"reference_vectors['cmb'] must have shape ({N_freq},), got {a_vec.shape}")

            # preserve tSZ by default in pcILC
            if reference_vectors is None or "tsz" not in reference_vectors:
                raise ValueError("pcilc=True needs reference_vectors['tsz'] to preserve tSZ.")
            a_vec = np.asarray(reference_vectors["tsz"], dtype=float).reshape((-1,))
            if a_vec.shape != (N_freq,):
                raise ValueError(f"reference_vectors['tsz'] must have shape ({N_freq},), got {a_vec.shape}")
            
            # partially constrain CMB by default in pcILC
            b_vec = np.ones(N_freq, dtype=float)

            #print ('pcilc b_vec',b_vec)
            if b_vec.shape != (N_freq,):
                raise ValueError(f"pcilc_b must have shape ({N_freq},), got {b_vec.shape}")

            if pcilc_pick not in ("minvar", "sign"):
                raise ValueError("pcilc_pick must be 'minvar' or 'sign'.")

        # ---------------------------------------
        # cILC configuration (F,f)
        # ---------------------------------------
        if constraint:
            if F is None:
                raise ValueError("F must be provided when constraint=True")
            F = np.asarray(F, dtype=float)
            Nf_F, N_comp = F.shape
            if Nf_F != N_freq:
                raise ValueError(f"F has {Nf_F} rows but R has {N_freq} channels")

            if f is None and extract_comp is not None:
                f = ILCConstraints.find_f_from_extract_comp(F, extract_comp, reference_vectors)

            if f is None:
                raise ValueError("Constraint vector f must be provided (or inferable) when constraint=True")
            f = np.asarray(f, dtype=float).reshape((-1,))
            if f.shape != (N_comp,):
                raise ValueError(f"Constraint vector f must have shape ({N_comp},), got {f.shape}")

        # --------------------------------------------------
        # Non-constraint, non-pcILC: decide identity_vector
        # --------------------------------------------------
        if (not constraint) and (not pcilc):
            key = str(extract_comp).lower()
            if key == "cmb":
                identity_vector = np.ones(N_freq, dtype=float)
            elif key == "tsz":
                if reference_vectors is None or "tsz" not in reference_vectors:
                    raise ValueError("constraint=False + extract_comp='tsz' needs reference_vectors['tsz'] (tSZ SED).")
                identity_vector = np.asarray(reference_vectors["tsz"], dtype=float).reshape((-1,))
                if identity_vector.shape != (N_freq,):
                    raise ValueError(f"reference_vectors['tsz'] must have shape ({N_freq},), got {identity_vector.shape}")
            else:
                raise ValueError(f"constraint=False only supports extract_comp='cmb' or 'tsz' for now (got '{extract_comp}').")

        # ---------------------------------------
        # Naming / saving
        # ---------------------------------------
        if pcilc:
            name = f"pcilc_{extract_comp}_eps{eps_str}"
        elif constraint:
            name = f"cilc_{extract_comp}"
        else:
            name = "weight_vector"

        fmt = dict(
            component=comp,
            comp=comp,
            type=name,
            extract_comp=extract_comp,
            scale=int(scale),
            realisation=int(realisation),
            lmax=int(L_max - 1),
            N_directions=int(N_directions),
            lam=str(lam),
            nsamp=nsamp,
            eps=eps_str,
        )

        weight_path = weight_vector_matrix_template.format(**fmt)

        if (not overwrite) and os.path.exists(weight_path):
            weight_vectors = np.load(weight_path)
            return None, weight_vectors, [], extract_comp

        # Track constraint singularities separately
        singular_constraints_location = []
        singular_constraints = []

        # ---------------------------------------
        # Main loop
        # ---------------------------------------
        tol = 1e-14

        for pix in np.ndindex(spatial_shape):
                Rij = R_Pix[pix]

                # ---- pcILC branch ----
                if pcilc:
                    try:
                        Rinva = np.linalg.solve(Rij, a_vec)
                    except np.linalg.LinAlgError:
                        zeros = np.zeros((N_freq,), dtype=float)
                        singular_matrices_location.append(pix)
                        singular_matrices.append(Rij)
                        weight_vectors[pix] = zeros
                        continue

                    den_a = float(np.dot(a_vec, Rinva))
                    if (not np.isfinite(den_a)) or abs(den_a) < tol:
                        zeros = np.zeros((N_freq,), dtype=float)
                        singular_constraints_location.append(pix)
                        singular_constraints.append(np.array([[den_a]]))
                        weight_vectors[pix] = zeros
                        continue
                    
                    w_ilc = (Rinva / den_a).ravel()
                    resp = float(np.dot(w_ilc, b_vec))               # w^T b
                    '''
                    if pcilc and abs(eps) < tol and i == 0 and j == 0:
                        print("abs(resp) =", abs(resp), "eps =", eps)
                    '''
                    if abs(resp) <= eps:
                        w = w_ilc
                    else:
                        try:
                            Rinvb = np.linalg.solve(Rij, b_vec)      # R^{-1} b
                        except np.linalg.LinAlgError:
                            zeros = np.zeros((N_freq,), dtype=float)
                            singular_matrices_location.append((pix))
                            singular_matrices.append(Rij)
                            weight_vectors[pix] = zeros
                            continue
                        
                        Ka  = den_a
                        Kb  = float(np.dot(b_vec, Rinvb))
                        Kab = float(np.dot(a_vec, Rinvb))

                        Delta = Ka * Kb - Kab * Kab
                        if (not np.isfinite(Delta)) or abs(Delta) < tol:
                            zeros = np.zeros((N_freq,), dtype=float)
                            singular_constraints_location.append((pix))
                            singular_constraints.append(np.array([[Ka, Kab], [Kab, Kb]]))
                            weight_vectors[pix] = zeros
                            continue
                        
                        w_plus  = (Rinva * (Kb - eps * Kab) + Rinvb * ( eps * Ka - Kab)) / Delta
                        w_minus = (Rinva * (Kb + eps * Kab) + Rinvb * (-eps * Ka - Kab)) / Delta

                        if pcilc_pick == "minvar":
                            var_plus  = float(np.dot(w_plus,  np.dot(Rij, w_plus)))
                            var_minus = float(np.dot(w_minus, np.dot(Rij, w_minus)))
                            w = w_plus if var_plus <= var_minus else w_minus
                        else:
                            w = w_plus if resp > 0.0 else w_minus
                        '''
                        if pcilc and abs(eps) < tol and i == 0 and j == 0:
                            F_eq = np.column_stack([a_vec, b_vec])
                            f_eq = np.array([1.0, 0.0], dtype=float)

                            RinvF = np.linalg.solve(Rij, F_eq)
                            M = np.dot(F_eq.T, RinvF) 
                            temp = np.linalg.solve(M, f_eq)
                            w_cilc_eq = np.dot(RinvF, temp).ravel() 

                            print("---- local eps=0 check ----")
                            print("scale =", scale, "realisation =", realisation, "i =", i, "j =", j)
                            print("Rij (pcilc eps=0) =\n", Rij)
                            print("max|w_pcilc - w_cilc| =", np.max(np.abs(w - w_cilc_eq)))
                            print("max|w_plus - w_cilc|  =", np.max(np.abs(w_plus - w_cilc_eq)))
                            print("max|w_minus - w_cilc| =", np.max(np.abs(w_minus - w_cilc_eq)))
                            print("w_plus[:5]  =", w_plus[:5])
                            print("w_minus[:5] =", w_minus[:5])
                            print("w_cilc[:5]  =", w_cilc_eq[:5])
                        '''
                    weight_vectors[pix] = np.asarray(w, dtype=float).ravel()
                    continue
                
                # ---- cILC branch ----
                if constraint:
                    #print('cilc')
                    try:
                        RinvF = np.linalg.solve(Rij, F)              # (Nf, Nc)
                    except np.linalg.LinAlgError:
                        zeros = np.zeros((N_freq,), dtype=float)
                        singular_matrices_location.append((pix))
                        singular_matrices.append(Rij)
                        weight_vectors[pix] = zeros
                        continue
                    
                    constraint_matrix = np.dot(F.T, RinvF)           # (Nc, Nc)

                    try:
                        temp = np.linalg.solve(constraint_matrix, f) # (Nc,)
                    except np.linalg.LinAlgError:
                        zeros = np.zeros((N_freq,), dtype=float)
                        singular_constraints_location.append((pix))
                        singular_constraints.append(constraint_matrix)
                        weight_vectors[pix] = zeros
                        continue
                    
                    w = np.dot(RinvF, temp).ravel()                  # (Nf,)
                    '''
                    if i == 0 and j == 0:
                        print("---- local cILC check ----")
                        print("scale =", scale, "realisation =", realisation, "i =", i, "j =", j)
                        print("Rij (cilc) =\n", Rij)
                        print("F =\n", F)
                        print("f =", f)
                        print("w_cilc[:5] =", w[:5])
                    '''
                    weight_vectors[pix] = w
                    continue
                
                # ---- standard ILC branch ----
                try:
                    #print('ilc')
                    num = np.linalg.solve(Rij, identity_vector)      # R^{-1} a
                except np.linalg.LinAlgError:
                    zeros = np.zeros((N_freq,), dtype=float)
                    singular_matrices_location.append((pix))
                    singular_matrices.append(Rij)
                    weight_vectors[pix] = zeros
                    continue
                
                den = float(np.dot(identity_vector, num))
                if (not np.isfinite(den)) or abs(den) < tol:
                    zeros = np.zeros((N_freq,), dtype=float)
                    singular_constraints_location.append((pix))
                    singular_constraints.append(np.array([[den]]))
                    weight_vectors[pix] = zeros
                    continue
                
                w = (num / den).ravel()
                weight_vectors[pix] = w

        if len(singular_matrices_location) > 0:
            print("Discovered", len(singular_matrices_location),
                  "singular covariance matrices at scale", scale, "realisation", realisation)
        if len(singular_constraints_location) > 0:
            print("Discovered", len(singular_constraints_location),
                  "constraint singularities at scale", scale, "realisation", realisation)

        if overwrite or not os.path.exists(weight_path):
            np.save(weight_path, weight_vectors)
            W = np.load(weight_path)
            #print("max|w| =", np.max(np.abs(W)))

        return inverses, weight_vectors, singular_matrices_location, extract_comp


    @staticmethod
    def create_doubled_ILC_map(frequencies, scale, weight_vector_load, doubled_MW_wav_c_j, *_, **__):
        cube = np.stack(
            [doubled_MW_wav_c_j[(f, scale)] for f in frequencies],
            axis=-1,
        )

        W = np.asarray(weight_vector_load)
        '''
        print(
            f"[create ILC] scale={scale} "
            f"cube.shape={cube.shape} W.shape={W.shape}"
        )
        '''
        if W.ndim == 1:
            return np.tensordot(cube, W, axes=([-1], [0]))

        if W.shape == cube.shape:
            return np.einsum("...c,...c->...", cube, W)

        raise ValueError(
            f"Unexpected weight_vector shape {W.shape}; expected (F,) or {cube.shape}."
        )

    @staticmethod
    def trim_to_original(MW_Doubled_Map: np.ndarray, scale: int, realisation: int, method: str, *,
                         path_template: str, component: str, extract_comp: str, lmax: int,
                         lam: float | None = None, nsamp: float | None = None, N_directions: int = 1,
                         overwrite: bool = False, mode: str = "ilc"):

        MW_Doubled_Map = np.asarray(MW_Doubled_Map)

        # ---------- directional case: (D, L2, 2L2-1) ----------
        if MW_Doubled_Map.ndim == 3:
            D, L2, W2 = MW_Doubled_Map.shape

            if W2 != 2 * L2 - 1:
                raise ValueError(
                    f"[DEBUG] Not MW grid: shape={D}x{L2}x{W2}, expected {D}x{L2}x{2*L2-1}"
                )

            inner_v = (L2 + 1) // 2
            inner_h = 2 * inner_v - 1
            outer_mid = W2 // 2
            start_col = outer_mid - (inner_h // 2)
            end_col = start_col + inner_h

            trimmed_dirs = []

            for d in range(D):
                alm_doubled = s2fft.forward(
                    MW_Doubled_Map[d],
                    L=L2,
                    method=method,
                    spmd=False,
                    reality=True,
                )

                trimmed_alm = alm_doubled[:inner_v, start_col:end_col]

                pix = s2fft.inverse(
                    trimmed_alm,
                    L=inner_v,
                    method=method,
                    spmd=False,
                    reality=True,
                )

                trimmed_dirs.append(pix)

            mw_map_original = np.stack(trimmed_dirs, axis=0)
            '''
            print(
                f"[trim] scale={scale} input={MW_Doubled_Map.shape} "
                f"output={mw_map_original.shape}"
            )
            '''

        # ---------- original 2D case: (L2, 2L2-1) ----------
        elif MW_Doubled_Map.ndim == 2:
            L2, W2 = MW_Doubled_Map.shape

            if W2 != 2 * L2 - 1:
                raise ValueError(
                    f"[DEBUG] Not MW grid: shape={L2}x{W2}, expected {L2}x{2*L2-1}"
                )

            inner_v = (L2 + 1) // 2
            inner_h = 2 * inner_v - 1
            outer_mid = W2 // 2
            start_col = outer_mid - (inner_h // 2)
            end_col = start_col + inner_h

            alm_doubled = s2fft.forward(
                MW_Doubled_Map,
                L=L2,
                method=method,
                spmd=False,
                reality=True,
            )

            trimmed_alm = alm_doubled[:inner_v, start_col:end_col]

            pix = s2fft.inverse(
                trimmed_alm,
                L=inner_v,
                method=method,
                spmd=False,
                reality=True,
            )

            mw_map_original = pix[np.newaxis, ...]
            '''
            print(
                f"[trim] scale={scale} input={MW_Doubled_Map.shape} "
                f"output={mw_map_original.shape}"
            )
            '''
        else:
            raise ValueError(f"[DEBUG] Not 2D or 3D: got {MW_Doubled_Map.shape}")

        if path_template and component and extract_comp:
            lmax = inner_v - 1
            save_path = path_template.format(
                mode=mode,
                component=component,
                extract_comp=extract_comp,
                scale=scale,
                realisation=int(realisation),
                lmax=int(lmax),
                lam=str(lam),
                nsamp=nsamp,
                N_directions=N_directions
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if overwrite or not os.path.exists(save_path):
                np.save(save_path, mw_map_original)

        return int(scale), mw_map_original

    @staticmethod
    def load_frequency_data(file_template: str, frequencies: list, scales: list, comp: str, lmax: int, N_directions: int, 
                            *, realisation: int, lam: float | None = None, nsamp: float | None = None):
        #print("load_frequency_data", flush = True)
        """
        Load NumPy arrays from dynamically generated file paths for each frequency and scale.
        
        Parameters:
            file_template (str): The template for the file names, with placeholders for frequency and scale.
            frequencies (list): A list of frequency names.
            scales (list): A list of scales.
            comp (str): The component name (e.g., "cfn").
            realisation (int): The realisation index.
            lmax (int): The maximum multipole.
            lam (float): The lambda parameter for the wavelet transform.

        Returns:
            dict: A dictionary where keys are tuples of (frequency, scale) and values are loaded NumPy arrays.
        """
        realisation = int(realisation)

        frequency_data = {}
        for frequency in frequencies:
            for scale in scales:
                filename = file_template.format(
                    comp=comp,
                    component=comp,        
                    frequency=frequency,
                    scale=scale,
                    realisation=realisation,
                    lmax=lmax,
                    lam=lam,
                    nsamp=nsamp, 
                    N_directions=N_directions
                )
                try:
                    arr = np.load(filename)
                    if N_directions == 1 and arr.ndim == 2:
                        arr = arr[np.newaxis, ...]

                    frequency_data[(frequency, scale)] = arr

                except Exception as e:
                    print(f"Error loading {filename} for frequency {frequency} and scale {scale}: {e}.")
        return frequency_data
    
    
    @staticmethod
    def visualize_MW_Pix_map(MW_Pix_Map, title, coord=["G"], unit=r"K", is_MW_alm=False):

        """
        Visualize an MW pixel map by converting to HEALPix and drawing a mollview.

        Args:
            MW_Pix_Map (np.ndarray): MW pixel map (spatial) or MW alm if is_MW_alm=True.
            title (str): Plot title.
            coord (list): Coordinate transform for mollview, e.g., ["G"].
            unit (str): Unit string for the colorbar.
            is_MW_alm (bool): If True, MW_Pix_Map is already MW alm.
        """

        # NOTE: relies on s2fft, healpy as hp, matplotlib.pyplot as plt,
        # and mw_alm_2_hp_alm being importable
        if not is_MW_alm:
            # Detect L from map shape
            if MW_Pix_Map.ndim == 3:
                L_max = MW_Pix_Map.shape[1]
            else:
                L_max = MW_Pix_Map.shape[0]
            original_map_alm = s2fft.forward(MW_Pix_Map, L=L_max)
            print("MW alm shape:", original_map_alm.shape)
        else:
            original_map_alm = MW_Pix_Map
            L_max = original_map_alm.shape[0]

        original_map_hp_alm = SamplingConverters.mw_alm_2_hp_alm(original_map_alm, L_max - 1)
        nside = (L_max - 1) // 2
        original_hp_map = hp.alm2map(original_map_hp_alm, nside=nside)

        hp.mollview(
            original_hp_map,
            coord=coord,
            title=title,
            unit=unit,
        )
        plt.show()

    @staticmethod
    def synthesize_ILC_maps_generalised(
        trimmed_maps, realisation, file_templates, lmax, N_directions,lam, component=None, 
        extract_comp=None, visualise=False, constraint=None, frequencies=None, F=None, f=None, 
        reference_vectors=None, nsamp=None, overwrite: bool = False, mode: str = "ilc",
    ):
        #print("synthesize_ILC_maps_generalised", flush=True)
        """
        Synthesizes full-sky ILC or cILC map from trimmed wavelet coefficient maps.
        ...
        """
        # 1) normalise realisation formatting
        if isinstance(realisation, int):
            realisation_str = f"{realisation:04d}"
        else:
            realisation_str = str(realisation).zfill(4)

        # 2) use the passed parameters
        file_tmpl = file_templates 

        # build frequencies tag for filename
        if isinstance(frequencies, (list, tuple)):
            freq_tag = "_".join(map(str, frequencies))
            freq0 = str(frequencies[0])
        elif frequencies is None:
            freq_tag, freq0 = "unknown", None
        else:
            freq_tag = str(frequencies)
            freq0 = freq_tag.split("_")[0]

        # 3) build filters and synthesise
        L = int(lmax) + 1
        
        print(
            "[synthesis]",
            "N_directions =", N_directions,
            "trimmed shapes =", [np.asarray(w).shape for w in trimmed_maps],
        )
        MW_Pix = MWTools.inverse_wavelet_transform(trimmed_maps, L, N_directions=int(N_directions), lam=float(lam))

        # 4) Save
        out_path = file_tmpl["ilc_synth"].format(
            mode=mode,
            extract_comp=extract_comp,
            component=component,
            frequencies=freq_tag,
            realisation=int(realisation_str),
            N_directions=int(N_directions),
            lmax=int(lmax),
            lam=str(lam),
            nsamp=nsamp,
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if overwrite or not os.path.exists(out_path):
            np.save(out_path, MW_Pix)
    
        # 5) Visualise
        if visualise:
            try:
                if str(mode).startswith("pcilc"):
                    prefix = f"pcILC ε={str(mode).replace('pcilc_epsa', '')}"
                elif str(mode) == "cilc":
                    prefix = "cILC"
                else:
                    prefix = "ILC"
        
                name = extract_comp.upper() if extract_comp else ""
                title = f"{prefix} {name} | r={realisation_str}, lmax={int(lmax)}, N={N_directions}, λ={lam}".strip()
                SILCTools.visualize_MW_Pix_map(MW_Pix, title)
            except Exception:
                pass
            
        return MW_Pix
    
    ELL_MIN = np.array([1, 2, 5, 9, 19, 38, 76, 153, 307, 613, 1227, 1800], dtype=int)
    ELL_PEAK = np.array([2, 5, 10, 19, 38, 77, 153, 307, 614, 1227, 2454, 3600], dtype=int)
    ELL_MAX = np.array([4, 10, 20, 38, 76, 154, 306, 614, 1228, 2454, 4908, 7200], dtype=int)

    @staticmethod
    def wavelet_js_custom(L: int, lam: float = 2.0) -> list[int]:
        """
        Wavelet indices to build, consistent with j_max_silc:
        js = [0, 1, ..., j_max_silc(L)].
        """
        jmax = j_max_silc(int(L), lam=lam)
        return list(range(jmax + 1))


class ProduceSILC():
    """Perform  Scale-discretised, directional wavelet ILC (SILC)."""
    def __init__(self, 
                 ilc_components: list, 
                 frequencies: list, 
                 realisations: int, 
                 start_realisation: int,
                 lmax: int, 
                 lam: float,
                 N_directions: int = 1,  
                 synthesise = True, 
                 directory: str = "data/", 
                 method: str = "jax_cuda", 
                 overwrite: bool = False):
        """
        Parameters:
            ilc_components (list): List of components to produce an ILC for.
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            start_realisation (int): Starting realisation number for processing.
            lmax (int): Maximum multipole for the wavelet transform.
            synthesise (bool): Whether to synthesise the ILC map from the wavelet transforms.
            directory (str): Directory where data is stored / saved to.
            method (str): s2fft method to use for forward/inverse.
            overwrite (bool): Whether to overwrite existing files.
        """
        self.ilc_components = ilc_components
        self.frequencies = frequencies
        self.realisations = realisations
        self.start_realisation = start_realisation
        self.lmax = lmax
        self.N_directions = N_directions  
        self.lam = lam
        self.synthesise = synthesise
        self.directory = directory
        self.method = method
        self.overwrite = overwrite

        files = FileTemplates(self.directory)
        self.file_templates = files.file_templates

        L = self.lmax + 1  # convert lmax -> bandlimit
        js = SILCTools.wavelet_js_custom(L)
        self.scales = range(len(js) + 1)  # +1 scaling
        self._js = js  


    def ILC_wav_coeff_maps_MP(file_template, frequencies, scales, realisations, output_templates, L_max, lam,
                              N_directions, comp, constraint=False, F=None, extract_comp=None,
                             reference_vectors=None, nsamp=None, overwrite: bool = False, 
                             pcilc: bool = False, pcilc_component: str = "tsz", 
                             pcilc_eps: float | None = None, pcilc_pick: str = "minvar",):
    
        L = int(L_max)
        js = SILCTools.wavelet_js_custom(L)
        scales = range(len(js) + 1)
        lmax = L - 1 

        print(f"[DEBUG] passed L_max={L_max} -> L={int(L_max)} lmax={int(L_max)-1}")
        print(f"[DEBUG] wavelet_js_custom(L)={SILCTools.wavelet_js_custom(int(L_max))}")

        # --- determine output mode for naming ---
        if pcilc:
            if pcilc_eps is None:
                raise ValueError("pcilc=True requires pcilc_eps")
            eps_str = f"{float(pcilc_eps):.6g}"
            mode = f"pcilc_eps{eps_str}"
        elif constraint:
            mode = "cilc"
        else:
            mode = "ilc"
 
        def _check_against_F(W, F, f, tol=1e-6):
            W = np.asarray(W)
            if W.ndim == 2 and 1 in W.shape:   # (1,Nf) or (Nf,1) -> (Nf,)
                W = W.reshape(-1)
            resp = np.tensordot(W, F, axes=([-1], [0]))  # (..., N_comp)
            ok = np.allclose(resp, f, atol=tol, rtol=0.0)
            #print('w:', W)
            #print("F^T w =", resp)
            print("FINAL CHECK  F^T w == f  ->", ok)
            if not ok:
                print("|F^T w - f| =", float(np.max(np.abs(resp - f))))
            return ok
        
        timings = {   # store timings per step
            "double_and_save": [],
            "covariance": [],
            "weights": [],
            "create_ilc_maps": [],
            "trim": [],
        }
        # --- Prepare constraint vector / tags ---
        if constraint:
            if F is None or extract_comp is None:
                raise ValueError("Must provide F and extract_comp if constraint=True")
            target_names, extract_comp = normalize_targets(extract_comp)
            if len(target_names) == 0:
                raise ValueError("Provide at least one target component name when constraint=True")
            f = ILCConstraints.find_f_from_extract_comp(F, target_names, reference_vectors)
        else:
            if isinstance(extract_comp, (list, tuple, np.ndarray)):
                raise ValueError("For unconstrained ILC, pass a single extract_comp (e.g., 'cmb').")
            _, extract_comp = normalize_targets(extract_comp)
            f = None  # not used in unconstrained mode
        # ---- derive Ndeproj automatically ----
        if constraint:
            N_constraints = int(F.shape[1])      # columns = number of constraint spectra
            Ndeproj = max(N_constraints - 1, 0)  # deproject all except the preserved one
        else:
            Ndeproj = 0
        
        Ndeproj = 0
        
        print(f"Derived Ndeproj={Ndeproj}")
        b_tol = 0.02
            
        synthesized_map = []
        
        for realisation in realisations:
            realisation_str = str(realisation).zfill(4)
            print(f"Processing realisation {realisation_str} for component {comp}")

            # 1) Load original wavelet maps
            original_wavelet_c_j = SILCTools.load_frequency_data(
                file_template=file_template,
                frequencies=frequencies,
                scales=scales,
                comp=comp,
                realisation=int(realisation),   # int
                N_directions=int(N_directions),
                lmax=L_max-1,
                lam=lam,
                nsamp=nsamp, 
            )

            # 2) Double resolution and save (single call)
            t0 = time.perf_counter()
            SILCTools.double_and_save_wavelet_maps(
                original_wavelet_c_j,
                frequencies,
                scales,
                realisation,
                method="jax_cuda",  # or "jax"
                path_template=output_templates['doubled_maps'],
                component=comp,
                lmax=L_max-1,
                N_directions=int(N_directions),
                lam=lam,
                nsamp=nsamp, 
                overwrite=overwrite,  
            )
            dt = time.perf_counter() - t0
            print(f'Doubled and saved wavelet maps in {dt:.2f} seconds')
            timings["double_and_save"].append(dt)

            # 3) Load doubled resolution wavelet maps from disk
            doubled_MW_wav_c_j = SILCTools.load_frequency_data(
                file_template=output_templates['doubled_maps'],
                frequencies=frequencies,
                scales=scales,
                comp=comp,
                realisation=int(realisation),   # int
                lmax=L_max-1,
                N_directions=int(N_directions),
                lam=lam,
                nsamp=nsamp,
            )

            # 4) Compute covariance matrices (serial, JAX backend)
            t0 = time.perf_counter()

            for scale in scales:
                SILCTools.calculate_covariance_matrix(
                    frequencies=frequencies,
                    doubled_MW_wav_c_j=doubled_MW_wav_c_j,
                    scale=int(scale),
                    realisation=int(realisation),
                    method= "jax_cuda"  ,                  
                    path_template=output_templates["covariance_matrices"],
                    component=comp,
                    lmax=L_max - 1,
                    N_directions=int(N_directions),
                    lam=lam,
                    nsamp=nsamp,
                    overwrite=overwrite,   
                    Ndeproj=Ndeproj,
                    b_tol=b_tol,
                )

            dt = time.perf_counter() - t0
            print(f"Calculated covariance matrices in {dt:.2f} seconds")
            timings["covariance"].append(dt)

            # 5) Load covariance matrices (per scale)
            F_str = '_'.join(frequencies)
            R_covariance = [
                np.load(
                    output_templates['covariance_matrices'].format(
                        component=comp,
                        frequencies=F_str,
                        scale=scale,
                        realisation=int(realisation),
                        lmax=L_max-1,
                        N_directions=int(N_directions),
                        lam = str(lam), 
                        nsamp=nsamp,
                    )
                )
                for scale in scales
            ]

            # 6) Compute weight vectors (serial; no MP to avoid pickling big arrays)
            t0 = time.perf_counter()
            for idx, scale in enumerate(scales):
                SILCTools.compute_weights_generalised(
                    R=R_covariance[idx],
                    scale=scale,
                    realisation=int(realisation),
                    weight_vector_matrix_template=output_templates['weight_vector_matrices'],
                    comp=comp,
                    L_max=L_max,
                    extract_comp=extract_comp,
                    constraint=constraint,
                    F=F,
                    f=f,
                    reference_vectors=reference_vectors,
                    lam=str(lam),
                    nsamp=nsamp,
                    N_directions=int(N_directions),
                    overwrite=overwrite,
                    pcilc=pcilc,
                    pcilc_component=pcilc_component,
                    pcilc_eps=pcilc_eps,
                    pcilc_pick=pcilc_pick,
                )

            dt = time.perf_counter() - t0
            print(f'Calculated weight vector matrices in {dt:.2f} seconds')
            timings["weights"].append(dt)

            # Load weights (in order)
            weight_vector_load = []
            W_for_final_check = None

            if pcilc:
                eps_str = f"{float(pcilc_eps):.6g}"
                name = f"pcilc_{extract_comp}_eps{eps_str}"
            elif constraint:
                name = f"cilc_{extract_comp}"
            else:
                name = "weight_vector"

            for scale in scales:
                weight_vector_path = output_templates['weight_vector_matrices'].format(
                    component=comp,
                    extract_comp=extract_comp,
                    type=name,
                    scale=scale,
                    realisation=int(realisation),
                    lmax=L_max-1,
                    lam=str(lam),
                    nsamp=nsamp,
                    N_directions=int(N_directions), 
                )
                W = np.load(weight_vector_path)  # mmap_mode='r' optional
                print("→ loading:", weight_vector_path)
                if W.ndim == 2 and 1 in W.shape:
                    W = W.reshape(-1)            # handle saved (1,F) or (F,1)
                weight_vector_load.append(W)
                W_for_final_check = W

            # 7) Create doubled ILC maps (serial; when noted MP here is slower)
            t0 = time.perf_counter()
            doubled_maps = []
            for i, scale in enumerate(scales):
                map_ = SILCTools.create_doubled_ILC_map(
                    frequencies,
                    scale,
                    weight_vector_load[i],
                    doubled_MW_wav_c_j,
                    realisation=int(realisation),
                    component=comp,
                    constraint=constraint,
                    extract_comp=extract_comp
                )
                doubled_maps.append(map_)
                ilc_path = output_templates['ilc_maps'].format(
                    mode=mode,
                    component=comp,
                    extract_comp=extract_comp,
                    scale=scale,
                    realisation=int(realisation),
                    lmax=lmax,
                    lam=str(lam),
                    nsamp=nsamp,
                    N_directions=int(N_directions), 
                )
                if overwrite or not os.path.exists(ilc_path):
                    np.save(ilc_path, map_)

            dt = time.perf_counter() - t0
            print(f'Created ILC maps in {dt:.2f} seconds')
            timings["create_ilc_maps"].append(dt)

            # 8) Trim to original resolution (serial, in-order)
            t0 = time.perf_counter()

            trimmed_maps = []
            for i, sc in enumerate(scales):
                save_path = output_templates['trimmed_maps'].format(
                    mode=mode,
                    component=comp,
                    extract_comp=extract_comp,
                    scale=int(sc),
                    realisation=int(realisation),
                    lmax=int(lmax),
                    lam=str(lam),
                    nsamp=nsamp,
                    N_directions=int(N_directions), 
                )
                #print(f"[trim] scale={sc} save_path={save_path}")

                if (not overwrite) and os.path.exists(save_path):
                    tm = np.load(save_path)
                    trimmed_maps.append(tm)
                    continue
                
                # Optional tiny guard: skip trimming if not truly doubled (e.g., j=0)
                arr = np.asarray(doubled_maps[i])
                #print(f"[trim] scale={sc} doubled shape={arr.shape}")
                L2, W2 = arr.shape[-2:]

                Lorig = (L2 + 1) // 2
                if W2 != 2 * L2 - 1 or L2 != 2 * Lorig - 1 or L2 <= 1:
                    #print(f"[trim] scale={sc} PASS-THROUGH (no trim) shape={np.shape(arr)}")
                    tm = arr  # pass-through
                else:
                    _, tm = SILCTools.trim_to_original(
                        MW_Doubled_Map=arr,
                        scale=int(sc),
                        realisation=int(realisation),
                        method='jax_cuda',
                        path_template=output_templates["trimmed_maps"],
                        component=comp,
                        extract_comp=extract_comp,
                        lmax=int(lmax),
                        lam=str(lam),
                        nsamp=nsamp,
                        N_directions=int(N_directions),
                        overwrite=overwrite, 
                        mode=mode,
                    )

                # Ensure saved on disk (trim_to_original already saved when path_template provided,
                # but save again if we took the pass-through branch)
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, tm)

                trimmed_maps.append(tm)

            dt = time.perf_counter() - t0
            print(f'Trimmed maps to original resolution in {dt:.2f} seconds')
            timings["trim"].append(dt)
            
            # 9) Synthesize final map (serial)        
            synth_map = SILCTools.synthesize_ILC_maps_generalised(
             trimmed_maps=trimmed_maps,
             realisation=int(realisation_str),
             file_templates=output_templates,
             lmax=lmax,             
             N_directions=N_directions,
             lam=lam,
             component=comp,
             extract_comp=extract_comp,
             frequencies=frequencies,
             visualise=True,
             constraint=constraint,
             F=F, 
             f=f, 
             reference_vectors=reference_vectors,
             nsamp=nsamp,
             overwrite=overwrite, 
             mode=mode,
            )         
            synthesized_map.append(np.asarray(synth_map))

            # 10) One-time verification per realisation (use mid scale)
            if constraint and weight_vector_load:
                mid_idx = len(weight_vector_load) // 2
                W_for_final_check = weight_vector_load[mid_idx]
                _check_against_F(W_for_final_check, F, f)

        return synthesized_map, timings

