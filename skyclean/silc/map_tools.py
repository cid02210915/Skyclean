import os 

import jax
jax.config.update("jax_enable_x64", True)
import healpy as hp
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
import matplotlib.pyplot as plt
import s2fft
import s2wav
import s2wav.filters as filters
from .utils import *
from .harmonic_response import build_axisym_filter_bank
from .harmonic_response import SimpleHarmonicWindows
from .power_spec import PowerSpectrumTT


class HPTools():
    """Tools to process healpy/MW maps"""
    @staticmethod
    def reduce_hp_map_resolution(hp_map: np.ndarray, lmax: int, nside: int):
        """
        Processes a Healpix map by converting it to spherical harmonics and back,
        and reducing the resolution.
        
        Parameters:
            map_data (numpy.ndarray): Input map data.
            lmax (int): Maximum multipole moment for spherical harmonics.
            nside (int): Desired nside resolution for the output map.
            
        Returns:
            numpy.ndarray: Processed map data.
        """
        hp_alm = hp.map2alm(hp_map, lmax=lmax )
        processed_map = hp.alm2map(hp_alm, nside=nside)
        return processed_map, hp_alm
    

    def pixwin_deconvolve(hp_map: np.ndarray, lmax: int):
        """
        Converts healpix map to alm space, deconvolves the pixel window function,
        and converts back to map space.
    
        Parameters:
            hp_map (numpy.ndarray): Input healpix map.
            lmax (int): Maximum multipole moment for spherical harmonics.
    
        Returns:
            numpy.ndarray: Reconstructed healpix map after pixel-window deconvolution.
        """
        nside = hp.get_nside(hp_map)
        alm = hp.map2alm(hp_map, lmax=lmax )
        # Pixel window function
        pixwin = hp.sphtfunc.pixwin(nside, lmax=lmax, pol=False)
        # Divide out pixel window function
        alm_deconv = hp.almxfl(alm, (1/pixwin))
        # Convert back to map
        hp_map_deconv = hp.alm2map(alm_deconv, nside=nside)
        return hp_map_deconv
    

    def beam_deconvolve_and_convolve(
        hp_map,
        lmax,
        frequency,
        standard_fwhm_rad,
        LFI_beam_fwhm={"030": 32.33, "044": 27.01, "070": 13.25},
        **kwargs
    ):
        """
        Minimal beam deconvolution:
        - Convert to alm
        - Divide by beam transfer function
        - Convert back to map
        """
        # external beam path for HFI maps
        beam_path = kwargs.get("beam_path")
    
        # Convert map → alm
        nside = hp.get_nside(hp_map)
        alm = hp.map2alm(hp_map, lmax=lmax )
    
        # LFI: Gaussian beam
        if frequency in {"030", "044", "070"}:
            fwhm_rad = np.radians(LFI_beam_fwhm[frequency] / 60.0)
            bl = hp.sphtfunc.gauss_beam(
                fwhm_rad, lmax=lmax, pol=False
            )
    
        # HFI: Load beam from FITS
        else:
            if beam_path is None:
                raise ValueError(
                    "beam_path required for HFI beams "
                    "(use FileTemplates.hfi_beam_path(frequency))"
                )
            hfi = fits.open(beam_path)
            bl = hfi[1].data["TEMPERATURE"]

        # --- Planck-style threshold: B_ell^c = max(B_ell^c, 0.001) ---
        bl = np.maximum(bl, 1e-3)
    
        # Apply the deconvolution
        alm_deconv = hp.almxfl(alm, 1.0 / bl)
    
        # Convolve with standard beam with the desired FWHM
        Standard_bl = hp.sphtfunc.gauss_beam(
            standard_fwhm_rad, lmax=lmax, pol=False
        )
        alm_conv = hp.almxfl(alm_deconv, Standard_bl)
    
        # Convert back to map
        hp_map_conv = hp.alm2map(alm_conv, nside=nside)
    
        return hp_map_conv
    

    def fermi_taper(L, ell_taper, ell_max=None):
        """
        Fermi (logistic) taper in multipole l.

        w(l) = 1                       for l ≤ l_taper
             = 1 / (1 + exp((l-l0)/Δℓ)) for l_taper < l < l_max
             = 0                       for l ≥ l_max

        with l0 = (l_taper + l_max)/2 and Δℓ = (l_max - l_taper)/4.
        """
        if ell_max is None:
            ell_max = L - 1

        ells = np.arange(L, dtype=float)
        w = np.ones(L, dtype=float)

        ell0 = 0.5 * (ell_taper + ell_max)
        delta = (ell_max - ell_taper) / 4.0

        mid = (ells > ell_taper) & (ells < ell_max)

        w[ells >= ell_max] = 0.0
        w[mid] = 1.0 / (1.0 + np.exp((ells[mid] - ell0) / delta))
        w[ells <= ell_taper] = 1.0

        return w


    def beam_convolve(hp_map: np.ndarray, lmax: int, standard_fwhm_rad: float, ell_taper_frac: float = 0.97):
        """
        Converts healpix map to alm space, applies a tapered standard beam,
        and converts back to map space.

        Beam = Gaussian(FWHM=standard_fwhm_rad) * Fermi taper in l,
        where taper starts at ell_taper_frac * lmax and goes to 0 at lmax.
        """
        nside = hp.get_nside(hp_map)

        # Map -> alm
        alm = hp.map2alm(hp_map, lmax=lmax )

        # Gaussian beam
        bl = hp.sphtfunc.gauss_beam(standard_fwhm_rad, lmax=lmax, pol=False)

        # Fermi taper in ℓ
        ell_taper = int(ell_taper_frac * lmax)
        taper = HPTools.fermi_taper(lmax + 1, ell_taper, ell_max=lmax)

        # Tapered effective beam
        #bl *= taper

        # Apply tapered beam
        alm_conv = hp.almxfl(alm, bl)

        # alm -> map
        hp_map_conv = hp.alm2map(alm_conv, nside=nside)

        return hp_map_conv

    
    def fermi_taper(L, ell_taper, ell_max=None):
        """
        Fermi (logistic) taper in multipole l.

        Define the window w(l) by
            w(l) = 1
                for  l ≤ l_taper

            w(l) = 1 / [ 1 + exp( (l - l0) / Δl ) ]
                for  l_taper < l < l_max

            w(l) = 0
                for  l ≥ l_max

        where
            l0  = (l_taper + l_max) / 2
            Δl  = (l_max - l_taper) / 4

        so that the transition from 1 → 0 is smooth between l_taper and l_max,
        with midpoint at l0 and a width of order (l_max - l_taper)/4.

        Parameters
        ----------
        L : int
            Band-limit (l runs from 0 to L-1).
        ell_taper : int
            Multipole at which tapering starts.
        ell_max : int, optional
            Multipole at which the window reaches zero. Defaults to L-1.

        Returns
        -------
        w : ndarray, shape (L,)
            Taper window w(l).
        """

        if ell_max is None:
            ell_max = L - 1

        ells = np.arange(L, dtype=float)
        w = np.ones(L, dtype=float)

        # Fermi parameters
        ell0 = 0.5 * (ell_taper + ell_max)
        delta = (ell_max - ell_taper) / 4.0

        mid = (ells > ell_taper) & (ells < ell_max)

        # Hard 0 above ell_max
        w[ells >= ell_max] = 0.0
        # Smooth transition in (ell_taper, ell_max)
        w[mid] = 1.0 / (1.0 + np.exp((ells[mid] - ell0) / delta))
        # Explicitly enforce w=1 below/at ell_taper
        w[ells <= ell_taper] = 1.0

        return w

        
    @staticmethod
    def unit_convert(hp_map: np.ndarray, frequency: str):
        """
        Convert the units of the given healpix map based on the frequency.

        Parameters:
            hp_map (numpy.ndarray): The input healpix map.
            frequency (str): The frequency of the map, used to determine the unit conversion. 

        Returns:
            numpy.ndarray: The healpix map with converted units.    
        """
        if frequency == "545":
            unit_conversion = 57.117072864249856
            hp_map /= unit_conversion
        if frequency == "857":
            unit_conversion = 1.4357233820474276
            hp_map /= unit_conversion
        else:
            hp_map = hp_map  # No conversion for other frequencies
        return hp_map
    
    
    @staticmethod
    def unit_convert_cib(hp_map: np.ndarray, frequency: str) -> np.ndarray:
        """
        Convert CIB GNILC maps from MJy/sr to K_CMB for 353, 545, 857 GHz.
        NRAS 466, 286-319 (2017)  
        """
        factors = {
            "353": 287.45,   # MJy/sr per K_CMB at 353 GHz
            "545": 58.0356,  # MJy/sr per K_CMB at 545 GHz
            "857": 2.2681,   # MJy/sr per K_CMB at 857 GHz
        }
        if frequency in factors:
            hp_map = hp_map / factors[frequency]
        return hp_map


    # For synch/dust/tSZ (have pixel window): coonv standard beam -> deconv Pℓ -> reduce
    @staticmethod
    def convolve_and_reduce(hp_map: np.ndarray, lmax: int, nside: int, standard_fwhm_rad: float) -> np.ndarray:
        """
        Deconvolve pixel window, convolve with standard beam, then reduce resolution.
        Use for components with pixel window (skyinbands foregrounds, noise).
        """
        
        hp_map_beamed = HPTools.beam_convolve(hp_map, lmax=lmax, standard_fwhm_rad=standard_fwhm_rad)
        #hp_map_noP = HPTools.pixwin_deconvolve(hp_map_beamed, lmax=lmax)
        hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(hp_map_beamed, lmax=lmax, nside=nside)
        return hp_map_reduced
    
    @staticmethod
    def deconvolve_and_convolve_and_reduce(
        hp_map: np.ndarray,
        lmax: int,
        nside: int,
        frequency: str,
        standard_fwhm_rad: float,
        **kwargs
    ) -> np.ndarray:
        """
        Noise processing:
          1) Deconvolve native beam (LFI: Gaussian; HFI: beam_path) and reconvolve to standard beam
          2) Deconvolve pixel window
          3) Reduce to target NSIDE

        Accepts optional beam_path=... for HFI.
        """
        beam_path = kwargs.get("beam_path")

        # Step 1: beam deconv + standard reconv (returns map at input NSIDE)
        hp_map_beamed = HPTools.beam_deconvolve_and_convolve(
            hp_map,
            lmax=lmax,
            frequency=frequency,
            standard_fwhm_rad=standard_fwhm_rad,
            beam_path=beam_path
        )

        # Step 2: pixel-window deconvolution
        #hp_map_noP = HPTools.pixwin_deconvolve(hp_map_beamed, lmax=lmax)

        # Step 3: reduce to target NSIDE for CFN accumulation
        hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(
            hp_map_beamed,
            lmax=lmax,
            nside=nside
        )
        return hp_map_reduced
    

    # For CMB synfast (no pixel window): beam -> reduce
    @staticmethod
    def convolve_and_reduce_cmb(hp_map: np.ndarray, lmax: int, nside: int, standard_fwhm_rad: float) -> np.ndarray:
        """
        Convolve with standard beam, then reduce resolution.
        Use for CMB synfast maps (no pixel window to deconvolve).
        """
        hp_map_beamed = HPTools.beam_convolve(hp_map, lmax=lmax, standard_fwhm_rad=standard_fwhm_rad)
        hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(hp_map_beamed, lmax=lmax, nside=nside)
        return hp_map_reduced
    

    @staticmethod
    def get_nside_from_lmax(lmax: int):
        """
        Get the nside value corresponding to the given lmax.

        Parameters:
            lmax (int): Maximum multipole moment for spherical harmonics.
        
        Returns:
            int: The nside value corresponding to the given lmax.
        """
        # Calculate nside as the largest power of 2 that satisfies lmax >= 2*nside

        min_nside = lmax / 2
        nside = 2 ** np.ceil(np.log2(min_nside)) # find the smallest power of 2 greater than or equal to min_nside
        return int(nside)
    

class MWTools():
    """Tools to process maps in MW (McEwen & Wiaux) sampling"""

    '''
    @staticmethod
    def wavelet_transform_from_map(mw_map: jnp.ndarray, L: int, N_directions: int, lam: float):
        """
        Performs a wavelet transform on MW map using the S2WAV library.   

        Parameters: 
            mw_map (jnp.ndarray): The input MW map to be transformed.
            L (int): Maximum multipole moment for the wavelet transform; lmax+1.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float, optional): Wavelet parameter, default is 2.0.   

        Returns:
            tuple: A tuple containing the wavelet coefficients and scaling coefficients.
        """

        # default JAX path
        #j_filter = filters.filters_directional_vectorised(L, N_directions, lam = lam)
        j_filter = build_axisym_filter_bank(L, lam=lam)
        #j_filter = build_axisym_filter_bank(L, J0=0)

        
        wavelet_coeffs, scaling_coeffs = s2wav.analysis(
            mw_map,
            N       = N_directions,
            L       = L,
            lam     = lam, 
            filters = j_filter,
            reality = True,
        )
        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)   
        print("scaling_coeffs shape:", scaling_coeffs.shape, "dtype:", scaling_coeffs.dtype)
        print("scaling_coeffs min/max:", float(jnp.min(scaling_coeffs)), float(jnp.max(scaling_coeffs)))
        print("scaling_coeffs L2 norm:", float(jnp.linalg.norm(scaling_coeffs)))

        wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs  
    '''

    @staticmethod
    def wavelet_transform_from_map(
        mw_map: jnp.ndarray,
        L: int,
        N_directions: int,
        lam: float,   # kept for API compatibility, but we’ll override internally
    ):
        # --- define the filter bank here ---
        ell_peaks = [64, 128, 256, 512, 705, 917,
                     1192, 1550, 2015, 2539, 3047, 3600]
        lam_list  = [2.0,  2.0,  2.0, 1.377,
                     1.3,  1.3,  1.3, 1.3,
                     1.26005, 1.2001, 1.2, 1.1815]

        # 1) build filters (wav_jln, scal_l)
        wav_jln, scal_l = SimpleHarmonicWindows.build_s2wav_filters(
            L, ell_peaks, lam_list
        )
        filters = (wav_jln, scal_l)

        # 2) use a safe lambda (even though samples.py is patched)
        lam_safe = float(np.max(lam_list))

        print("\n[DEBUG analysis] L =", L, "N =", N_directions, "lam_safe =", lam_safe)
        print("[DEBUG analysis] wav_jln shape:", np.asarray(wav_jln).shape, "dtype:", np.asarray(wav_jln).dtype)
        print("[DEBUG analysis] scal_l shape:", np.asarray(scal_l).shape, "dtype:", np.asarray(scal_l).dtype)

        wav = np.asarray(wav_jln)   # expect (J+1, L) or (J+1, L, ...)
        phi = np.asarray(scal_l)    # expect (L,)

        # if wav has extra dims, reduce to power in ell
        if wav.ndim > 2:
            wav_pow = np.sum(np.abs(wav)**2, axis=tuple(range(2, wav.ndim)))
        else:
            wav_pow = np.abs(wav)**2

        S = np.abs(phi)**2 + np.sum(wav_pow, axis=0)

        imin = int(np.argmin(S))
        imax = int(np.argmax(S))

        print("[DEBUG analysis] S(ell)=|phi|^2+sum|psi|^2  min/max:",
              float(S[imin]), float(S[imax]))
        print(f"[DEBUG analysis]   min at ell = {imin}")
        print(f"[DEBUG analysis]   max at ell = {imax}")

        print("[DEBUG analysis] S tail (last 20 ells):", S[-20:])

        # peak ell per wavelet (fingerprint)
        print("[DEBUG analysis] wavelet peak ells:")
        for j in range(wav_pow.shape[0]):
            print(f"  j={j:02d} peak ell ~ {int(np.argmax(wav_pow[j]))}")

        # 3) run analysis
        wavelet_coeffs, scaling_coeffs = s2wav.analysis(
            mw_map,
            L=L,
            N=N_directions,
            J_min=0,
            lam=lam_safe,
            spin=0,
            sampling="mw",
            nside=None,
            reality=True,
            filters=filters,
        )
        print("[DEBUG analysis] returned #wavelet blocks (no scaling):", len(wavelet_coeffs))
        for i, blk in enumerate(wavelet_coeffs):
            arr = np.asarray(blk)
            print(f"  wav_block[{i:02d}] shape={arr.shape} dtype={arr.dtype}")

        print("[DEBUG analysis] scaling_coeffs shape:", np.asarray(scaling_coeffs).shape)
        print("[DEBUG analysis] filters wav_jln shape:", np.asarray(wav_jln).shape)
        print("[DEBUG analysis] filters scal_l shape:", np.asarray(scal_l).shape)
        print("[DEBUG analysis] len(wavelet_coeffs) returned by s2wav.analysis:", len(wavelet_coeffs))

        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)   
        wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs
    
    
    @staticmethod
    def wavelet_to_mw_alm(
        wavelet_coeffs,                # single band array (or list, see below)
        L: int,
        N_directions: int,
        lam: float,                    # kept for API symmetry; not used (lam_safe used instead)
        reality: bool = True,          # inverse_wavelet_transform uses reality=True
        band_index: int | None = None,
        template_wavelet_coeffs: list | None = None,  # OPTIONAL but recommended for exact shapes
        debug: bool = False,
    ):
        """
        Consistent with inverse_wavelet_transform:
          - uses SimpleHarmonicWindows.build_s2wav_filters with ell_peaks/lam_list
          - uses lam_safe = max(lam_list)
          - calls s2wav.synthesis(... sampling="mw", spin=0, reality=True, filters=(wav_jln, scal_l))

        Input:
          - wavelet_coeffs: a SINGLE wavelet band array (directional: (N, Lj, 2Lj-1) or axisym: (Lj, 2Lj-1))
          - band_index: which j (0-based) this band corresponds to in wav_jln
          - template_wavelet_coeffs (optional): full coefficient list from a real forward transform
            [f_scal_tiled, wav0, wav1, ...]. If provided, we create zero arrays with *exact* expected shapes.
        """

        if band_index is None:
            raise ValueError("Must specify band_index (0-based) when passing a single wavelet band.")

        # ---- same ell_peaks / lam_list as inverse_wavelet_transform ----
        ell_peaks = [64, 128, 256, 512, 705, 917,
                     1192, 1550, 2015, 2539, 3047, 3600]
        lam_list  = [2.0,  2.0,  2.0, 1.377,
                     1.3,  1.3,  1.3, 1.3,
                     1.26005, 1.2001, 1.2, 1.1815]

        wav_jln, scal_l = SimpleHarmonicWindows.build_s2wav_filters(L, ell_peaks, lam_list)
        filters = (wav_jln, scal_l)
        lam_safe = float(np.max(lam_list))

        # Jplus1 = number of wavelet bands implied by the filterbank
        Jplus1 = int(np.asarray(wav_jln).shape[0])

        if not (0 <= int(band_index) < Jplus1):
            raise ValueError(f"band_index={band_index} out of range; filters imply J+1={Jplus1}")

        # ---- prepare the passed band ----
        w_band = jnp.array(wavelet_coeffs)

        # axisymmetric: (Lj, 2Lj-1) -> (1, Lj, 2Lj-1)
        if w_band.ndim == 2:
            if N_directions != 1:
                raise ValueError(f"Got 2D band (axisymmetric), but N_directions={N_directions} != 1")
            w_band = w_band[jnp.newaxis, ...]
        elif w_band.ndim == 3:
            if w_band.shape[0] != N_directions:
                raise ValueError(f"Directional band first axis must be N={N_directions}, got {w_band.shape[0]}")
        else:
            raise ValueError(f"wavelet_coeffs must be 2D or 3D, got ndim={w_band.ndim}")

        # ---- build full wav_coeffs list with correct shapes ----
        if template_wavelet_coeffs is not None:
            # template format should match inverse: [f_scal_tiled, wav0, wav1, ...]
            if len(template_wavelet_coeffs) != (1 + Jplus1):
                raise ValueError(
                    f"template_wavelet_coeffs length must be 1+Jplus1={1+Jplus1}, got {len(template_wavelet_coeffs)}"
                )

            f_scal_tiled = jnp.zeros_like(jnp.array(template_wavelet_coeffs[0]))

            wav_coeffs = []
            for j in range(Jplus1):
                tmpl = jnp.array(template_wavelet_coeffs[1 + j])
                if j == int(band_index):
                    if w_band.shape != tmpl.shape:
                        raise ValueError(
                            f"Your band shape {tuple(w_band.shape)} != template shape {tuple(tmpl.shape)} for j={j}"
                        )
                    wav_coeffs.append(w_band)
                else:
                    wav_coeffs.append(jnp.zeros_like(tmpl))

        else:
            # Fallback: assume all wavelet bands share the same shape as the provided band.
            # This is OK for "minimal" use but not guaranteed if pipeline uses true multires shapes.
            wav_coeffs = [jnp.zeros_like(w_band) for _ in range(Jplus1)]
            wav_coeffs[int(band_index)] = w_band

            # scaling tile: make something compatible-looking for synthesis
            # (inverse passes f_scal=f_scal_tiled and expects tiled scaling)
            f_scal_tiled = jnp.zeros((N_directions, L, 2 * L - 1), dtype=w_band.dtype)

        if debug:

            print("\n[DEBUG wavelet_to_mw_alm] L =", L, "N =", N_directions, "lam_safe =", lam_safe)
            print("[DEBUG wavelet_to_mw_alm] wav_jln shape:", np.asarray(wav_jln).shape)
            print("[DEBUG wavelet_to_mw_alm] scal_l shape:", np.asarray(scal_l).shape)
            print("[DEBUG wavelet_to_mw_alm] Jplus1 =", Jplus1, "band_index =", int(band_index))
            print("[DEBUG wavelet_to_mw_alm] f_scal_tiled shape:", np.asarray(f_scal_tiled).shape)
            print("[DEBUG wavelet_to_mw_alm] wav_coeffs len:", len(wav_coeffs))
            print("[DEBUG wavelet_to_mw_alm] selected band shape:", tuple(w_band.shape))

        # ---- synthesis: consistent with inverse_wavelet_transform ----
        mw_map = s2wav.synthesis(
            wav_coeffs,
            f_scal=f_scal_tiled,
            L=L,
            N=N_directions,
            lam=lam_safe,
            spin=0,
            sampling="mw",
            nside=None,
            reality=True,        
            filters=filters,
        )

        # MW map -> MW alm
        mw_alm = s2fft.forward(mw_map, L=L, reality=True)

        # 7) (Optional) per-band spectra plot (kept as-is but using the normalized 'bands')
        ell_all, Dl_all = [], []
        for j in range(Jplus1):
            sc0 = jnp.zeros_like(f_scal_tiled)
            bands_j = [wav_coeffs[k] if k == j else jnp.zeros_like(wav_coeffs[k]) for k in range(Jplus1)]
            f_band = s2wav.synthesis(
                bands_j, f_scal=sc0, L=L, lam=lam, filters=filters, reality=reality, N=N_directions
            )
            alm_band = s2fft.forward(f_band, L=L, reality=reality)
            ell, Cl = PowerSpectrumTT.from_mw_alm(alm_band)
            Dl = PowerSpectrumTT.cl_to_Dl(ell, Cl, input_unit="K")
            ell_all.append(ell); Dl_all.append(Dl)

        plt.figure(figsize=(6, 4))
        for j, (ell, Dl) in enumerate(zip(ell_all, Dl_all)):
            plt.plot(ell, Dl, label=f"Band {j}")
        plt.xlabel(r"$\ell$"); plt.ylabel(r"$D_\ell\ [\mu\mathrm{K}^2]$")
        plt.grid(alpha=0.4); plt.legend(); plt.tight_layout(); plt.show()

        return mw_alm


    @staticmethod
    def save_wavelet_scaling_coeffs(wavelet_coeffs: list, scaling_coeffs: np.ndarray, comp: str, frequency: str, realisation: int, lmax: int, lam: float, wav_template: str, scal_template: str):
        """ Saves the wavelet and scaling coefficients to files.    

        Parameters:
            comp (str): The component for which the coefficients are saved (e.g., 'sync', 'noise').
            wavelet_coeffs (list): List of wavelet coefficients for each scale.
            scaling_coeffs (np.ndarray): Scaling coefficients.  
            frequency (str): Frequency of the map.
            realisation (int): realisation number for the map.
            lmax (int): Maximum multipole for the wavelet transform.
            lam (float): lambda factor (scaling) for the wavelet transform.
            wav_template (str): Template for the wavelet coefficient file path.
            scal_template (str): Template for the scaling coefficient file path.

        Returns:
            None
        """
        # Save scaling coefficients (kept separate; do NOT treat as "scale 0")
        np_scaling = np.array(scaling_coeffs)
        # If N_directions==1 the scaling may be (1, L, 2L-1) — squeeze to (L, 2L-1)
        if np_scaling.ndim == 3 and np_scaling.shape[0] == 1:
            np_scaling = np_scaling[0]
        np.save(
            scal_template.format(comp=comp, frequency=frequency, realisation=realisation, lmax=lmax, lam=lam),
            np_scaling,
        )
    
        # Save each **wavelet band** at each scale. (Disk scale 0 == first wavelet band.)
        for scale, wav in enumerate(wavelet_coeffs):
            np_wav = np.array(wav)
            # If N_directions==1 and wav is (1, L, 2L-1), squeeze to (L, 2L-1)
            if np_wav.ndim == 3 and np_wav.shape[0] == 1:
                np_wav = np_wav[0]
            np.save(
                wav_template.format(comp=comp, frequency=frequency, scale=scale, realisation=realisation, lmax=lmax, lam=lam),
                np_wav,
            )
            

    @staticmethod
    def inverse_wavelet_transform(wavelet_coeffs: list, L: int, lam: float, N_directions: int = 1):

        # same ell_peaks / lam_list as forward
        ell_peaks = [64, 128, 256, 512, 705, 917,
                     1192, 1550, 2015, 2539, 3047, 3600]
        lam_list  = [2.0,  2.0,  2.0, 1.377,
                     1.3,  1.3,  1.3, 1.3,
                     1.26005, 1.2001, 1.2, 1.1815]

        wav_jln, scal_l = SimpleHarmonicWindows.build_s2wav_filters(L, ell_peaks, lam_list)
        filters = (wav_jln, scal_l)
        lam_safe = float(np.max(lam_list))
        print("\n[DEBUG synthesis] L =", L, "N =", N_directions, "lam_safe =", lam_safe)
        print("[DEBUG synthesis] wav_jln shape:", np.asarray(wav_jln).shape, "dtype:", np.asarray(wav_jln).dtype)
        print("[DEBUG synthesis] scal_l shape:", np.asarray(scal_l).shape, "dtype:", np.asarray(scal_l).dtype)
        print("[DEBUG synthesis] total blocks (incl scaling):", len(wavelet_coeffs))

        f_scal_tiled = wavelet_coeffs[0]
        f_scal = f_scal_tiled[0]  # (Lj, 2Lj-1)

        print("[DEBUG synthesis] wavelet_coeffs total blocks (incl scaling):", len(wavelet_coeffs))
        print("[DEBUG synthesis] f_scal_tiled shape:", np.asarray(wavelet_coeffs[0]).shape)

        wav_coeffs = wavelet_coeffs[1:]
        Jplus1 = int(np.asarray(wav_jln).shape[0])
        print("[DEBUG synthesis] filters imply J+1 =", Jplus1, "but len(wav_coeffs) =", len(wav_coeffs))

        # (3) the S(ell) check (partition-of-unity)
        wav = np.asarray(wav_jln)
        phi = np.asarray(scal_l)
    
        if wav.ndim > 2:
            wav_pow = np.sum(np.abs(wav)**2, axis=tuple(range(2, wav.ndim)))
        else:
            wav_pow = np.abs(wav)**2
    
        S = np.abs(phi)**2 + np.sum(wav_pow, axis=0)
        print("[DEBUG synthesis] S min/max:", float(S.min()), float(S.max()))
        print("[DEBUG synthesis] S tail (last 20):", S[-20:])
    
        # OPTIONAL: print peak ell of each psi_j
        print("[DEBUG synthesis] wavelet peak ells:")
        for j in range(wav_pow.shape[0]):
            print(f"  j={j:02d} peak ell ~ {int(np.argmax(wav_pow[j]))}")
            
        print("[DEBUG synthesis] filters wav_jln shape:", np.asarray(wav_jln).shape)
        print("[DEBUG synthesis] len(wav_coeffs) passed into s2wav.synthesis:", len(wav_coeffs))

        mw_map = s2wav.synthesis(
            wav_coeffs,
            f_scal=f_scal_tiled,
            L=L,
            N=N_directions,
            lam=lam_safe,
            spin=0,
            sampling="mw",
            nside=None,
            reality=True,     
            filters=filters,
        )
        return mw_map
    
    '''
    @staticmethod
    def inverse_wavelet_transform(wavelet_coeffs: list, L: int, lam: float, N_directions: int = 1):
        """
        Performs an inverse wavelet transform on the given wavelet coefficients (assuming scaling coefficients are included at the first index).

        Parameters:
            wavelet_coeffs (list): List of wavelet coefficients.
            L (int): Maximum multipole moment for the wavelet transform; lmax+1.
            N_directions (int, options): Number of directions for the wavelet transform, default is 1.
            lam (float, optional): Wavelet parameter, default is 2.0.

        Returns:
            jnp.ndarray: The reconstructed MW map from the wavelet coefficients.
        """
        #j_filter = filters.filters_directional_vectorised(L, N_directions, lam=lam)
        j_filter = build_axisym_filter_bank(L, lam=lam)
        #print(wavelet_coeffs[0].shape)
        f_scal = wavelet_coeffs[0]  # Scaling coefficients are at the first index
        wavelet_coeffs = wavelet_coeffs[1:]  # Remove scaling coefficients from
        #print(wavelet_coeffs[0].shape)
    
        mw_map = s2wav.synthesis(
            wavelet_coeffs,
            f_scal  = f_scal,
            L       = L,
            lam     = lam,
            filters = j_filter,
            reality = False,
            N = N_directions
        )
        return mw_map
    '''

    @staticmethod
    def load_wavelet_scaling_coeffs(frequency: str, num_wavelets: int, realisation: int, wav_template: str, scal_template: str):
        """
        Loads the wavelet and scaling coefficients from files.

        Parameters:
            frequency (str): Frequency of the map.
            num_wavelets (int): Number of wavelet coefficients to load.
            realisation (int): realisation number for the map.  
            wav_template (str): Template for the wavelet coefficient file path.
            scal_template (str): Template for the scaling coefficient file path.
        
        Returns:
            tuple: A tuple containing the wavelet coefficients and scaling coefficients.
        """
        wavelet_coeffs = [np.real(np.load(wav_template.format(frequency=frequency, scale=scale, realisation=realisation))) for scale in range(num_wavelets)]
        scaling_coeffs = np.real(np.load(scal_template.format(frequency=frequency, realisation=realisation)))
        return wavelet_coeffs, scaling_coeffs
    
    
    @staticmethod
    def visualise_mw_map(mw_map: np.ndarray, title: str = None, coord: list = ["G"], unit: str = r"K", directional: bool = False, method = "jax_cuda",):
        """
        Visualizes a MW pixel wavelet coefficient map using HEALPix mollview.

        Parameters:
            mw_map (numpy array): Array representing the wavelet coefficient map.
            title (str): Title for the visualization plot.  
            coord (list): List of coordinate systems to use for the visualization.
            directional (bool): If plotting wavelet transform maps, set to True (even if N_directions = 1). If plotting normal MW maps, set to False. 
            unit (str): Unit of the map data, default is Kelvin (K).  
        """
        if directional:
            nrows = 1
            ncols = mw_map.shape[0] # number of directions
            fig = plt.figure(figsize=(5*ncols, 5*nrows))
            
            lmax = mw_map.shape[1] - 1
            for i in range(ncols):
                hp_map = SamplingConverters.mw_map_2_hp_map(mw_map[i], lmax, method=method)
                panel = i + 1
                hp.mollview(
                    hp_map,
                    coord=coord,
                    title=title+f", dir {i+1}",
                    unit=unit,
                    fig = fig.number,
                    sub = (nrows, ncols, panel)
                    # min=min, max=max,  # Uncomment and adjust these as necessary for better visualization contrast
                )
                # plt.figure(dpi=1200)
        else:
            lmax = mw_map.shape[0] - 1
            hp_map = SamplingConverters.mw_map_2_hp_map(mw_map, lmax, method=method)
            hp.mollview(
                hp_map,
                coord=coord,
                title=title,
                unit=unit,
                # min=min, max=max,  # Uncomment and adjust these as necessary for better visualization contrast
            )                                           
        plt.savefig(f'{title}.png')
        plt.show()

    '''
    @staticmethod
    def visualise_axisym_wavelets(L: int, lam: float = 2.0):
        """
        Plots the wavelet filters for the given parameters.
        Only works for axisymmetric wavelets (N_directions = 1).

        Parameters:
            L (int): Maximum multipole moment for the wavelet transform; lmax+1
            lam (float): Wavelet parameter, default is 2.0. 
        """
        #j_filter = filters.filters_directional_vectorised(L=L, N=1, lam=lam)[0]
        j_filter = build_axisym_filter_bank(L=L, lam=lam)[0]
        shape = j_filter.shape
        #print ('shape:', j_filter[0].shape) 
        #print ('shape:', j_filter[8].shape) 
        l_list = np.arange(L)
        middle_m = L-1 # m = 0 along which the axisymmetric wavelet is defined
        
        for i in range(shape[0]):
            plt.plot(l_list, np.real(j_filter[i][:,middle_m]), label = f"scale {i}")
            plt.xlabel("l", fontsize=16)
            plt.ylabel("Real part of wavelet filter", fontsize=16)
            plt.title(f"lambda = {lam}, axisym")
            #plt.xscale('log')
            plt.legend()
            #plt.xlim(0,2)            #check the removal of monopole and dipole
            plt.grid(ls=':')
        plt.show()
    '''

    @staticmethod
    def visualise_axisym_wavelets(
        L: int,
        ell_peaks,
        lam_list,
        scal_ell_cut: float = 64.0,
        scal_lam: float | None = None,
        truncate: bool = True,
    ):

        j_filter = SimpleHarmonicWindows.build_s2wav_filters(
            L=int(L),
            ell_peaks=np.asarray(ell_peaks),
            lam_list=np.asarray(lam_list),
            scal_ell_cut=float(scal_ell_cut),
            scal_lam=scal_lam,
            truncate=bool(truncate),
        )


        l_list = np.arange(int(L))
        middle_m = int(L) - 1  # m=0 index in MW layout

        plt.figure()
        if np.asarray(j_filter).ndim == 3:
            # (n_scales, L, 2L-1): take m=0 slice
            for i in range(j_filter.shape[0]):
                plt.plot(l_list, np.real(j_filter[i][:, middle_m]), label=f"scale {i}")
        elif np.asarray(j_filter).ndim == 2:
            # (n_scales, L): already axisymmetric
            for i in range(j_filter.shape[0]):
                plt.plot(l_list, np.real(j_filter[i]), label=f"scale {i}")
        else:
            raise ValueError(f"Unexpected filter shape: {np.asarray(j_filter).shape}")

        plt.xlabel("l", fontsize=16)
        plt.ylabel("Real part of wavelet filter (m=0)", fontsize=16)
        plt.title("axisym (SimpleHarmonicWindows)", fontsize=16)
        plt.legend()
        plt.grid(ls=":")
        plt.show()



class SamplingConverters():
    """Converters between Healpy and MW sampling"""
    
    @staticmethod
    def hp_alm_2_mw_alm(hp_alm: np.ndarray, lmax: int):
        """
        Converts spherical harmonics (alm) from healpy to a matrix representation for use in MW sampling.

        This function takes 1D Healpix spherical harmonics coefficients (alm) and converts them into a matrix form 
        that is in (MW sampling, McEwen & Wiaux) sampling. The matrix form is complex-valued and indexed by multipole 
        moment and azimuthal index.

        Parameters:
            hp_alm (numpy.ndarray): The input healpix spherical harmonics coefficients (alm).
            lmax (int): The maximum multipole moment to be represented in the output matrix.
        
        Note: # lmax = 4 | l = 0,1,2,3 , m = -3...0...(lmax-1 = 3)| number of m = 2(lmax-1)+1 = 2lmax-1
        MW sampling fills in positive and negative m, while healpy only stores m >= 0.

        Returns:
            MW_alm (numpy.ndarray): 2D array of shape (Lmax, 2*Lmax-1) MW spherical harmonics coefficients 
        """
        L = lmax + 1 # L as defined in MW sampling
        MW_alm = np.zeros((L, 2 * L - 1), dtype=np.complex128) # MW does not invoke reality theorem
        for l in range(L):
            for m in range(l + 1):
                index = hp.Alm.getidx(lmax, l, m)
                col = m + L - 1
                hp_point = hp_alm[index]
                MW_alm[l, col] = hp_point
                if m > 0: 
                    MW_alm[l, L-m-1] = (-1)**m * hp_point.conj() # fill m < 0 by symmetry
        return MW_alm
        
    
    @staticmethod
    def mw_alm_2_hp_alm(mw_alm: np.ndarray, lmax:int):
        """
        Converts spherical harmonics (alm) from MW sampling to healpy representation.

        This function takes a 2D alm array in MW form (MW Sampling, McEwen & Wiaux) and converts them 
        into a 1D array used in healpy sampling. The matrix form is complex-valued and indexed by multipole 
        moment and azimuthal index.

        Notea: MW sampling runs from 1,...,L while healpy runs from 0,...,L-1. Hence the lmax param from Healpy
        is L-1 in MW sampling.
        Healpy only stores m >= 0, while MW sampling fills in both positive and negative m.
        
        Parameters:
            MW_alm (numpy.ndarray): The input MW spherical harmonics coefficients in matrix form.
        
        Returns:
            hp_alm (numpy.ndarray): 1D array of healpy spherical harmonics coefficients
        """
        L = mw_alm.shape[0] 
        lmax = L-1 
        hp_alm = np.zeros(hp.Alm.getsize(lmax), np.complex128)
        for l in range(L):
            for m in range(l+1):
                col = lmax + m
                idx = hp.Alm.getidx(lmax, l, m)
                hp_alm[idx] = mw_alm[l, col]
        return hp_alm
    

    @staticmethod
    def hp_map_2_mw_map(hp_map: np.ndarray, lmax: int, method = "jax_cuda"):
        """
        Converts a Healpix map to a MW map by transforming the map to spherical harmonics and then converting the alm to MW sampling.

        Parameters:
            hp_map (numpy.ndarray): The input Healpix map.
            lmax (int): The maximum multipole moment for the spherical harmonics.
        
        Returns:
            mw_map (numpy.ndarray): The converted MW map in spherical harmonics representation.
        """
        L = lmax + 1
        hp_alm = hp.map2alm(hp_map, lmax=lmax )
        mw_alm = SamplingConverters.hp_alm_2_mw_alm(hp_alm, lmax)
        mw_map = s2fft.inverse(mw_alm, L=L, method=method, reality = True)
        return mw_map
    
    @staticmethod
    def mw_map_2_hp_map(mw_map: np.ndarray, lmax: int, method = "jax_cuda"):
        """
        Converts a MW map to a Healpix map by transforming the map to spherical harmonics 
        and then converting the alm to Healpix sampling.

        Parameters:
            mw_map (numpy.ndarray): The input MW map in spherical harmonics representation.
            lmax (int): The maximum multipole moment for the spherical harmonics.
        
        Returns:
            hp_map (numpy.ndarray): The converted Healpix map.
        """
        L = lmax + 1 
        mw_alm = s2fft.forward(mw_map, L=L, method=method, reality = True)
        hp_alm = SamplingConverters.mw_alm_2_hp_alm(mw_alm, lmax)
        hp_map = hp.alm2map(hp_alm, nside=HPTools.get_nside_from_lmax(lmax))
        return hp_map
    
    
    def mw_map_2_mwss_map(mw_map, L: int, ): 
        """
        Soft wrapper to convert a MW map to a MWSS map.

        Parameters:
            mw_map (numpy.ndarray): The input MW map in spherical harmonics representation.
            L (int): The maximum multipole moment for the spherical harmonics.
        
        Returns:
            mwss_map (numpy.ndarray): The converted MWSS map.
        """
        return s2fft.utils.resampling_jax.mw_to_mwss(mw_map, L=L)
    

    def mwss_map_2_mw_map(mwss_map, L: int):
            """
            Convert a MWSS map to a MW map. No s2fft function exists for this,
            but the process is simply performing a harmonic transform.
    
            Parameters:
                mwss_map (numpy.ndarray): The input MWSS map in spherical harmonics representation.
                L (int): The maximum multipole moment for the spherical harmonics.
    
            Returns:
                mw_map (numpy.ndarray): The converted MW map.
            """
            mw_alm = s2fft.forward(mwss_map, L=L, sampling = "mwss", reality = True)
            return s2fft.inverse(mw_alm, L=L, sampling = "mw", reality = True)
    