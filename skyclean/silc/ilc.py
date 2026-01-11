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
from .harmonic_response import AxisymmetricGenerators
from .custom_s2wav_bandlimits import j_max_silc

class SILCTools():
    '''Tools for Scale-discretised, directional wavelet ILC (SILC).'''

    @staticmethod
    def Single_Map_doubleworker(mw_map: np.ndarray, method: str):
        """
        Doubles the resolution of a single MW pixel map using s2fft.

        Params:
            mw_map (np.ndarray): MW pixel map at original resolution.
            method (str): s2fft method to use for doubling the resolution.
        
        Returns:
            mw_map_doubled: The MW pixel map with increased resolution.
        """
        # use jax/numpy 
        #print('Single_Map_doubleworker', flush = True)

        alm = s2fft.forward(mw_map, L=mw_map.shape[0], method = method, spmd = False, reality = True)

        L = alm.shape[0]
        H = 2*L - 1
        W = 2*H - 1
        padded = np.zeros((H, W), dtype=np.complex128)
        mid_in = alm.shape[1]//2
        mid_out = W//2
        start = mid_out - mid_in
        padded[:L, start:start+alm.shape[1]] = alm

        x2 = np.real(s2fft.inverse(padded, L=H, method = method, spmd = False, reality = True))

        return x2 

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
        lam: float,
        j: int,
        Nfreq: int,
        Ndeproj: int = 0,
        b_tol: float = 0.01
    ) -> float:
        """
        Paper Eqs (42)-(44): per-band real-space Gaussian FWHM (radians) for WAVELET band j.
        SILC FIX: use fixed band edges from the bank.
        """
        L = int(L)
        ells = np.arange(L, dtype=int)
        ell_min = int(L0_j_silc(int(j)))
        ell_max = int(wav_j_bandlimit_silc(L, int(j), multiresolution=True))
        # Clip to available harmonic support 0..L-1
        ell_min = max(0, min(ell_min, L - 1))
        ell_max = max(0, min(ell_max, L - 1))
        if ell_min > ell_max:
            return 0.0
        h = ((ells >= ell_min) & (ells <= ell_max)).astype(float)
        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))  # Eq (42)
        A = abs(1 + int(Ndeproj) - int(Nfreq))
        # Eq (43)
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
        b_tol: float = 0.01
    ) -> float:
        """
        Paper Eqs (42)-(44): per-band real-space Gaussian FWHM (radians) for SCALING band (eta).
        SILC FIX: scaling window fixed low-ℓ; J unused (API compatibility).
        """
        gen = AxisymmetricGenerators(float(lam))
        ells = np.arange(int(L), dtype=float)
        h = gen.eta(ells)
        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))  # Eq (42)
        A = abs(1 + int(Ndeproj) - int(Nfreq))
        sigma2 = 0.0 if S <= 0.0 else 2.0 * (A / (float(b_tol) * S))  # Eq (43)
        return SILCTools.fwhm_from_sigma2(sigma2)  # Eq (44)
    
    @staticmethod
    def n_modes_scaling_band(
        L: int,
        lam: float,
        J: int,
        fwhm_rad: float,
        nside_nmodes: int = 2048
    ) -> float:
        """
        PAPER local modes for scaling band:
            N_modes(local) = f_sky(paper) * Σ(2l+1) (h_l)^2
        """
        gen = AxisymmetricGenerators(float(lam))
        ells = np.arange(int(L), dtype=float)
        h = gen.eta(ells)
        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))  # Eq (42)
        f_sky = SILCTools.f_sky_paper_from_fwhm(float(fwhm_rad))
        return f_sky * S
    
    @staticmethod
    def n_modes_needlet_band(
        L: int,
        lam: float,
        band_j: int,
        fwhm_rad: float,
        nside_nmodes: int = 2048
    ) -> float:
        """
        PAPER local modes for wavelet band j:
            N_modes(local) = f_sky(paper) * Σ(2l+1) (h_l^j)^2
        """
        L = int(L)
        ells = np.arange(L, dtype=int)
        ell_data_max = L - 1
        ell_min_bank = int(L0_j_silc(int(band_j)))
        ell_max_bank = int(wav_j_bandlimit_silc(L, int(band_j), multiresolution=True))
        if ell_min_bank > ell_data_max:
            return 0.0
        ell_min = max(0, ell_min_bank)
        ell_max = min(ell_max_bank, ell_data_max)
        if ell_min > ell_max:
            return 0.0
        h = ((ells >= ell_min) & (ells <= ell_max)).astype(float)
        S = float(np.sum((2.0 * ells + 1.0) * (h * h)))  # Eq (42)
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
        band_j: int | None = None,
        J_scal: int | None = None,   # kept for API compatibility, ignored for SILC
        scaling: bool = False,
        Nfreq: int | None = None,
        Ndeproj: int = 0,
        b_tol: float = 0.01,
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
                L, float(lam), int(band_j), int(Nfreq), int(Ndeproj), float(b_tol)
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
                # Your updated gaussian_pixel_counts_fullsky returns:
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
          task: (i, fq, frequencies, scale, doubled_MW_wav_c_j, method, lam, Ndeproj, b_tol)
        """
        i, fq, frequencies, scale, doubled_MW_wav_c_j, method, lam, Ndeproj, b_tol = task

        key_i  = (frequencies[i], scale)
        key_fq = (frequencies[fq], scale)
        if key_i not in doubled_MW_wav_c_j or key_fq not in doubled_MW_wav_c_j:
            raise KeyError(f"Missing data for keys {key_i} or {key_fq}.")

        L = int(doubled_MW_wav_c_j[key_i].shape[0])

        smoothed, info = SILCTools.smoothed_covariance(
            doubled_MW_wav_c_j[key_i],
            doubled_MW_wav_c_j[key_fq],
            method=method,
            # nsamp kept for API compatibility, ignored:
            nsamp=0.0,
            return_info=True,
            use_paper_fwhm=True,
            lam=float(lam),
            Nfreq=len(frequencies),
            Ndeproj=int(Ndeproj),
            b_tol=float(b_tol),
            scaling=(int(scale) == 0),
            band_j=(int(scale) - 1) if int(scale) > 0 else None,
        )

        # ---- print once per scale ----
        if (i == 0) and (fq == 0) and info.get("beam_source") == "paper":
            fwhm_rad = float(info["fwhm_rad"])
            fwhm_arcmin = fwhm_rad * (180.0 / math.pi) * 60.0

            f_sky_paper = float(info.get("f_sky_paper", float("nan")))
            f_sky_eff_hp = float(info.get("f_sky_eff_hp", float("nan")))

            if int(scale) == 0:
                Ls = scal_bandlimit_silc(L, multiresolution=True)
                tag = f"scal(ell=0..{Ls-1})"

                gen = AxisymmetricGenerators(float(lam))
                ells = np.arange(int(L), dtype=float)
                h = gen.eta(ells)
                N_modes_full = float(np.sum((2.0 * ells + 1.0) * (h * h)))

                N_modes_local = SILCTools.n_modes_scaling_band(L, float(lam), 0, fwhm_rad)
            else:
                j = int(scale) - 1
                ell_min = int(L0_j_silc(j))
                ell_max = int(wav_j_bandlimit_silc(L, j, multiresolution=True))
                tag = f"wav(j={j}, ell={ell_min}..{ell_max})"

                ells = np.arange(int(L), dtype=int)
                h = ((ells >= max(0, min(ell_min, L - 1))) & (ells <= max(0, min(ell_max, L - 1)))).astype(float)
                N_modes_full = float(np.sum((2.0 * ells + 1.0) * (h * h)))

                N_modes_local = SILCTools.n_modes_needlet_band(L, float(lam), j, fwhm_rad)

            print(
                f"[locality] {tag}  "
                f"FWHM={fwhm_arcmin:.3f} arcmin  "
                f"f_sky(paper)={f_sky_paper:.3e}  "
                f"f_sky(eff_hp,diag)={f_sky_eff_hp:.3e}  "
                f"N_modes(full)={N_modes_full:.3e}  "
                f"N_modes(local,paper)={N_modes_local:.3e}"
            )

        return i, fq, smoothed


    @staticmethod
    def calculate_covariance_matrix(frequencies: list, doubled_MW_wav_c_j: dict, scale: int,
                                    realisation: int, method: str, path_template: str, *,
                                    component: str = "cfn", lmax: int = 64, lam: float | None = None,
                                    nsamp: float = 1200, overwrite: bool = False,
                                    Ndeproj: int = 0, b_tol: float = 0.01,):
        
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
        n_rows, n_cols = sample_data.shape
        total_frequency = len(norm_freqs)
        full_array = np.zeros((total_frequency, total_frequency, n_rows, n_cols))

        tasks = [
            (i, fq, norm_freqs, scale_i, doubled_MW_wav_c_j, method, lam, int(Ndeproj), float(b_tol))
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
        # print(R.shape)
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
                det = np.linalg.det(R_Pix[i, j])
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
    
    
    def compute_weights_generalised(R, scale, realisation, weight_vector_matrix_template, comp, L_max, 
                                    extract_comp, *, constraint=False, F=None, f=None, 
                                    reference_vectors=None, lam: float | None = None,
                                    nsamp: float | None = None, overwrite: bool = False,):
        #print("compute_weights_generalised", flush=True)
        """
        Computes weight vectors from a covariance matrix R using either standard or generalized ILC.

        Args:
            R (np.ndarray): (Nf,Nf) or (H,W,Nf,Nf)
            scale (int): Wavelet scale index.
            realisation (str): realisation id string.
            weight_vector_matrix_template (str): Save path template.
            comp (str): Component tag for filenames.
            extract_comp (str or None): Target component name (for constrained ILC).
            constraint (bool): Use constrained ILC if True.
            F (np.ndarray): Spectral response, shape (Nf, Nc) when constraint=True.
            f (np.ndarray): Constraint vector, shape (Nc,).
            reference_vectors (dict): Dict of named reference spectra (for auto-f).

        Returns:
            inverses, weight_vectors, singular_matrices_location, extract_comp
        """
        #print('R:', R.shape)
 
        # Swap the axes to get R_Pix
        R_Pix = np.swapaxes(np.swapaxes(R, 0, 2), 1, 3) #(pix,pix,freq,freq)
        # Get dimensions for looping and size of sub-matrices
        dim1, dim2, subdim1, subdim2 = R_Pix.shape
        #print(dim1, dim2, subdim1, subdim2)
        # Create arrays to store inverses and weight vectors
        inverses = np.zeros((dim1, dim2, subdim1, subdim2))
        weight_vectors = np.zeros((dim1, dim2, subdim1)) # weight vector at each pixel (dim1,dim2) and channel
        # Realiztion 6 has a singular matrix
        # Adjust identity vector size based on sub-matrix dimensions
        identity_vector = np.ones(subdim2, dtype=float)
        singular_matrices_location = []
        singular_matrices = []

        N_freq = subdim2 

        # --- branch config ---
        if constraint:
            if F is None:
                raise ValueError("F must be provided when constraint=True")
            #print ('F:',F.shape)
            Nf_F, N_comp = F.shape
            if Nf_F != N_freq:
                raise ValueError(f"F has {Nf_F} rows but R has {N_freq} channels")

            # Automatically set f from extract_comp if given
            if f is None and extract_comp is not None:
                f = ILCConstraints.find_f_from_extract_comp(F, extract_comp, reference_vectors)
            #print ('f:', f.shape)
            if f is None:
                raise ValueError("Constraint vector f must be provided (or inferable) when constraint=True")
            if f.shape != (N_comp,):
                raise ValueError(f"Constraint vector f must have shape ({N_comp},)")
        else:
            # Unconstrained ILC uses the all-ones vector; no F/extract_comp needed
            identity_vector = np.ones(N_freq, dtype=float)
            print (identity_vector)   

            # ----------------------------------------------------------

        # Common filename fields (provide both 'component' and 'comp'; pass ints so {:04d} works)
        name = f"cilc_{extract_comp}" if constraint else "weight_vector"
        fmt = dict(
            component=comp,
            comp=comp,
            type=name,
            extract_comp=extract_comp,
            scale=int(scale),
            realisation=int(realisation),
            lmax=int(L_max-1),
            lam=str(lam),
            nsamp=nsamp, 
        )
        weight_path = weight_vector_matrix_template.format(**fmt)

        if (not overwrite) and os.path.exists(weight_path):
            weight_vectors = np.load(weight_path)
            # return dummy inverses + existing weights
            return None, weight_vectors, [], extract_comp

        # Track constraint singularities (mirrors R singular tracking)
        singular_constraints_location = []
        singular_constraints = []

        for i in range(dim1):
            for j in range(dim2):

                det = np.linalg.det(R_Pix[i, j])
                if det == 0:
                    zeros = np.zeros((subdim1))
                    singular_matrices_location.append((i,j))
                    singular_matrices.append(R_Pix[i, j])
                    weight_vectors[i, j] = zeros
                else:
                    # compute inverse
                    R_inv = np.linalg.inv(R_Pix[i, j])
                    #print ('R_inv:', R_inv.shape)

                    if constraint:
                        # Step 1: Fᵗ R⁻¹
                        FT_Rinv = np.dot(F.T, R_inv)                 # (Nc, Nf)
                        #print ('FT_Rinv:', FT_Rinv.shape)

                        # Step 2: Fᵗ R⁻¹ F
                        constraint_matrix = np.dot(FT_Rinv, F)       # (Nc, Nc)
                        #print ('constraint_matrix:', constraint_matrix.shape)

                        # singularity check for (Fᵗ R⁻¹ F)
                        G_det = np.linalg.det(constraint_matrix)
                        if G_det == 0:
                            zeros = np.zeros((subdim1))
                            singular_constraints_location.append((i, j))
                            singular_constraints.append(constraint_matrix)
                            weight_vectors[i, j] = zeros
                            continue  # skip the rest for this pixel

                        # Step 3: (Fᵗ R⁻¹ F)⁻¹
                        constraint_matrix_inv = np.linalg.inv(constraint_matrix)
                        #print ('constraint_matrix_inv:', constraint_matrix_inv.shape)

                        # Step 4: build temp = (Fᵗ R⁻¹ F)⁻¹ f
                        temp = np.dot(constraint_matrix_inv, f)      # (Nc,)
                        #print ('temp:', temp.shape)

                        # Step 5: F temp
                        F_temp = np.dot(F, temp)                     # (Nf,)
                        #print ('F_temp:', F_temp.shape)

                        # Step 6: w = R⁻¹ F (Fᵗ R⁻¹ F)⁻¹ f
                        w = np.dot(R_inv, F_temp)                    # (Nf,)
                        w = np.asarray(w).ravel()
                        #print('w:', w.shape)
                    else:
                        num = np.dot(R_inv, identity_vector)         # (Nf,)
                        den = float(np.dot(num, identity_vector))    # scalar
                        w = (num / den).ravel()                      # (Nf,)

                    if R.ndim == 4:
                        inverses[i, j] = R_inv
                        weight_vectors[i, j] = w
                        #print('weight_vectors[i,j]:', weight_vectors[i,j].shape)
                    else:
                        inverses[i, j] = np.linalg.inv(R_Pix[i, j])
                        numerator = np.dot(inverses[i, j], identity_vector)
                        denominator = np.dot(np.dot(inverses[i, j], identity_vector),identity_vector)
                        weight_vectors[i, j] = numerator / denominator

        if len(singular_matrices_location) > 0:
            print("Discovered ", len(singular_matrices_location), "singular matrices at scale", scale, "realisation", realisation)
        if len(singular_constraints_location) > 0:
            print("Discovered ", len(singular_constraints_location),
              "constraint singularities (F^T R^{-1} F) at scale", scale, "realisation", realisation)

        # save final weight vector matrix
        if overwrite or not os.path.exists(weight_path):
            np.save(weight_path, weight_vectors)

        return inverses, weight_vectors, singular_matrices_location, extract_comp
    

    @staticmethod
    def create_doubled_ILC_map(frequencies, scale, weight_vector_load, doubled_MW_wav_c_j, *_, **__):
        """
        weight_vector_load is expected to be the 1D weight vector for THIS scale (shape (F,))
        """
        cube = np.stack([doubled_MW_wav_c_j[(f, scale)] for f in frequencies], axis=-1)

        W = np.asarray(weight_vector_load)

        if W.ndim == 1:
            H, Wd, F = cube.shape
            return np.dot(cube.reshape(-1, F), W).reshape(H, Wd)
        
        if W.ndim == 3:
            # Elementwise multiply then sum over freq
            return np.einsum('ijc,ijc->ij', cube, W)

        raise ValueError(f"Unexpected weight_vector shape {W.shape}; expected (F,) or (H,W,F).")
    

    @staticmethod
    def trim_to_original(MW_Doubled_Map: np.ndarray, scale: int, realisation: int, method: str, *, 
                         path_template:str, component: str, extract_comp: str, lmax:int, 
                         lam: float | None = None, nsamp: float | None = None, overwrite: bool = False,):

        #print("trim_to_original", flush=True)
        """
        Trim a doubled‐resolution MW Pixel map back to its original resolution.

        Parameters:
            MW_Doubled_Map (ndarray): MW pixel map at doubled resolution (L2, 2L2-1).
            scale (int): wavelet scale (for filename only).
            realisation (int): realisation index (for filename only).

        Returns:
            mw_map_original (ndarray or None): trimmed pixel map at original resolution. 
        """
        '''
        for d in jax.devices():
            ms = d.memory_stats()
            print(f"Device {d.id}:",
                f"bytes_in_use={ms['bytes_in_use']}",
                f"peak_bytes_in_use={ms['peak_bytes_in_use']}",
                f"bytes_limit={ms['bytes_limit']}",
                f"largest_free_chunk={ms.get('largest_free_chunk', 'n/a')}",
                f"num_allocs={ms.get('num_allocs', 'n/a')}")
        '''

        #print(f"[DEBUG] scale={scale}, realisation={realisation}, shape={MW_Doubled_Map.shape}", flush=True)
        if MW_Doubled_Map.ndim != 2:
            raise ValueError(f"[DEBUG] Not 2D: got {MW_Doubled_Map.shape}")

        L2, W2 = MW_Doubled_Map.shape
        if W2 != 2 * L2 - 1:
            raise ValueError(f"[DEBUG] Not MW grid: shape={L2}x{W2}, expected {L2}x{2*L2-1}")
        #print('L2:',L2)
     
        L2, W2 = MW_Doubled_Map.shape
        #print(f"[trim pid={os.getpid()}] scale={scale} r={realisation} L2={L2} shape={MW_Doubled_Map.shape}",flush=True)
        # compute trim indices once
        inner_v = (L2 + 1) // 2           # original bandlimit L (rows) after trimming
        inner_h = 2 * inner_v - 1         # original width 2L-1
        outer_mid = W2 // 2
        start_col = outer_mid - (inner_h // 2)
        end_col = start_col + inner_h
        
        # forward (numpy path)
        alm_doubled = s2fft.forward(MW_Doubled_Map, L=L2, method=method, spmd=False, reality=True)
   
        # trim in harmonic space
        trimmed_alm = alm_doubled[:inner_v, start_col:end_col]

        # inverse back to pixels
        pix = s2fft.inverse(trimmed_alm, L=inner_v, method=method, spmd=False, reality=True)

        mw_map_original = pix[np.newaxis, ...]

        # ---- only if a template and tags are provided ----
        if path_template and component and extract_comp:
            lmax = inner_v - 1
            lam_str = str(lam)
            save_path = path_template.format(
                component=component,
                extract_comp=extract_comp,
                scale=scale,
                realisation=int(realisation),
                lmax=int(lmax),
                lam=lam_str,
                nsamp=nsamp, 
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if overwrite or not os.path.exists(save_path):
                np.save(save_path, mw_map_original)
            #print(f"[SAVE] Trimmed map -> {save_path}")

        return int(scale), mw_map_original


    @staticmethod
    def load_frequency_data(file_template: str, frequencies: list, scales: list, comp: str, lmax: int, *,
                        realisation: int, lam: float | None = None, nsamp: float | None = None):
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
                )
                try:
                    frequency_data[(frequency, scale)] = np.load(filename)
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
        reference_vectors=None, nsamp=None, overwrite: bool = False, 
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
        
        print([w.shape for w in trimmed_maps])
        MW_Pix = MWTools.inverse_wavelet_transform(trimmed_maps, L, N_directions=int(1), lam=float(lam))

        # 4) Save
        mode = "con" if constraint else "uncon"
        out_path = file_tmpl["ilc_synth"].format(
            mode=mode,
            extract_comp=extract_comp,
            component=component,
            frequencies=freq_tag,
            realisation=int(realisation_str),
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
                prefix = "cILC" if extract_comp else "ILC"
                name = extract_comp.upper() if extract_comp else ""
                title = f"{prefix} {name} | r={realisation_str}, lmax={int(lmax)}, N={N_directions}, λ={lam}".strip()
                SILCTools.visualize_MW_Pix_map(MW_Pix, title)
            except Exception:
                pass

        return MW_Pix
    
    ELL_MIN = np.array([32, 64, 128, 256, 542, 705, 916, 1192, 1550, 2115, 2539, 3046], dtype=int)
    ELL_PEAK = np.array([64, 128, 256, 512, 705, 917, 1192, 1550, 2015, 2539, 3047, 3600],dtype=int,)
    ELL_MAX = np.array([128, 256, 512, 705, 916, 1192, 1549, 2015, 2539, 3047, 3656, 4253], dtype=int)

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
        self._js = js  # store mapping if you need it


    def ILC_wav_coeff_maps_MP(file_template, frequencies, scales, realisations, output_templates, L_max, lam,
                              N_directions, comp, constraint=False, F=None, extract_comp=None,
                             reference_vectors=None, nsamp=None, overwrite: bool = False,):
    
        L = int(L_max)
        js = SILCTools.wavelet_js_custom(L)
        scales = range(len(js) + 1)
        lmax = L - 1 

        print(f"[DEBUG] passed L_max={L_max} -> L={int(L_max)} lmax={int(L_max)-1}")
        print(f"[DEBUG] wavelet_js_custom(L)={SILCTools.wavelet_js_custom(int(L_max))}")
 
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

        b_tol = 0.01
            
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
                    overwrite=overwrite,  
                )

            dt = time.perf_counter() - t0
            print(f'Calculated weight vector matrices in {dt:.2f} seconds')
            timings["weights"].append(dt)

            # Load weights (in order)
            weight_vector_load = []
            W_for_final_check = None
            name = f"cilc_{extract_comp}" if constraint else "weight_vector"

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
                    component=comp,
                    extract_comp=extract_comp,
                    scale=scale,
                    realisation=int(realisation),
                    lmax=lmax,
                    lam=str(lam),
                    nsamp=nsamp,
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
                    component=comp,
                    extract_comp=extract_comp,
                    scale=int(sc),
                    realisation=int(realisation),
                    lmax=int(lmax),
                    lam=str(lam),
                    nsamp=nsamp,
                )
                print(f"[trim] scale={sc} save_path={save_path}")

                if (not overwrite) and os.path.exists(save_path):
                    tm = np.load(save_path)
                    trimmed_maps.append(tm)
                    continue
                
                # Optional tiny guard: skip trimming if not truly doubled (e.g., j=0)
                arr = np.asarray(doubled_maps[i])
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr2d = arr[0]
                    print(f"[trim] scale={sc} doubled arr2d shape={arr2d.shape}") 
                else:
                    arr2d = arr
                L2, W2 = arr2d.shape
                Lorig = (L2 + 1) // 2
                if W2 != 2 * L2 - 1 or L2 != 2 * Lorig - 1 or L2 <= 1:
                    print(f"[trim] scale={sc} PASS-THROUGH (no trim) shape={np.shape(arr)}")
                    tm = arr  # pass-through
                else:
                    _, tm = SILCTools.trim_to_original(
                        MW_Doubled_Map=arr2d,
                        scale=int(sc),
                        realisation=int(realisation),
                        method='jax_cuda',
                        path_template=output_templates["trimmed_maps"],
                        component=comp,
                        extract_comp=extract_comp,
                        lmax=int(lmax),
                        lam=str(lam),
                        nsamp=nsamp,
                        overwrite=overwrite, 
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
            )         
            synthesized_map.append(np.asarray(synth_map))

            # 10) One-time verification per realisation (use mid scale)
            if constraint and weight_vector_load:
                mid_idx = len(weight_vector_load) // 2
                W_for_final_check = weight_vector_load[mid_idx]
                _check_against_F(W_for_final_check, F, f)

            return synthesized_map, timings
