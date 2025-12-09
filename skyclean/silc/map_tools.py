import os 
import healpy as hp
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import s2fft
import s2wav
import s2wav.filters as filters
from .utils import *
from .harmonic_response import build_axisym_filter_bank
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


    def beam_convolve(hp_map: np.ndarray, lmax: int, standard_fwhm_rad: float, ell_taper_frac: float = 0.94):
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
        # bl *= taper

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
            unit_conversion = 58.0356
            hp_map /= unit_conversion
        if frequency == "857":
            unit_conversion = 2.2681
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
        # hp_map_noP = HPTools.pixwin_deconvolve(hp_map_beamed, lmax=lmax)
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
        # hp_map_noP = HPTools.pixwin_deconvolve(hp_map_beamed, lmax=lmax)

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
        wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs  



    ''' 
    @staticmethod
    def wavelet_transform_from_map(mw_map: jnp.ndarray, L: int, N_directions: int, lam: float = 2.0):
        """
        Performs a wavelet transform on MW map using scale-discretised filters
        for axisym (N_directions=1), otherwise the library's directional filters
        """
        # --- minimal fix: choose filters based on N_directions ---
        if N_directions == 1:
            j_filter = build_axisym_filter_bank(L, lam) 
            print ('shape:', j_filter[0].shape) 
            print ('shape:', j_filter[1].shape) 
        else:
            j_filter = filters.filters_directional_vectorised(L, N_directions, lam=lam)

        wavelet_coeffs, scaling_coeffs = s2wav.analysis(
            mw_map,
            N       = N_directions,
            L       = L,
            lam     = lam, 
            filters = j_filter,
            reality = True,
        )

        # output format
        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)
        wavelet_coeffs.insert(0, scaling_coeffs)
        return wavelet_coeffs, scaling_coeffs 
    ''' 


    @staticmethod
    def wavelet_transform_from_alm(mw_alm: jnp.ndarray, L: int, N_directions: int, lam: float):
        """
        Performs a wavelet transform on MW alm using the S2WAV library.

        Parameters:
            mw_alm (jnp.ndarray): The input MW alm to be transformed.
            L (int): Maximum multipole moment for the wavelet transform; lmax+1.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float, optional): Wavelet parameter, default is 2.0.

        Returns:
            tuple: A tuple containing the wavelet coefficients and scaling coefficients.
        """
        #j_filter = filters.filters_directional_vectorised(L, N_directions, lam = lam)
        j_filter = build_axisym_filter_bank(L, lam=lam)

        wavelet_coeffs, scaling_coeffs = s2wav.flm_to_analysis(
            mw_alm,
            N       = N_directions,
            L       = L,
            lam     = lam,
            filters = j_filter,
            reality = False,
        )
        scaling_coeffs = np.expand_dims(scaling_coeffs, axis=0)  # Ensure scaling coefficients are in the same format as wavelet coefficients
        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)   
        wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs


    @staticmethod
    def wavelet_to_mw_alm(
        wavelet_coeffs,
        L: int,
        N_directions: int,
        lam: float,
        reality: bool = False,
        band_index: int | None = None,
    ):
        """
        Minimal version — accepts only a single wavelet band (not a list).
        Reconstructs MW alm for that band by zeroing all others.
        """
        
        # 1) Build filter bank
        #j_filter = filters.filters_directional_vectorised(L, N_directions, J_min=0, lam=float(lam))
        j_filter = build_axisym_filter_bank(L, lam=lam)
        J = len(j_filter[0])  # number of expected wavelet bands
        #print ("number of wavelet bands J:", J)

        # --- Must have band_index if passing one band ---
        if band_index is None:
            raise ValueError("Must specify band_index (0-based) when passing a single wavelet band.")

        # 2) Prepare the single wavelet band
        w_band = jnp.array(wavelet_coeffs)

        # Axisymmetric (L, 2L-1) → add dir axis for N=1
        if w_band.ndim == 2 and N_directions == 1:
            assert w_band.shape == (L, 2 * L - 1), f"band shape {w_band.shape} != (L, 2L-1)"
            w_band = w_band[jnp.newaxis, ...]
        else:
            # Directional case sanity check
            assert w_band.shape[1:] == (L, 2 * L - 1), f"band shape {w_band.shape} incompatible with L={L}"

        # 3) Create 2D scaling (low-pass)
        scaling = jnp.zeros((L, 2 * L - 1), dtype=w_band.dtype)
        #print("scaling shape:", scaling.shape)

        # 4) Build exactly J bands (others zero)
        bands = [jnp.zeros_like(w_band) for _ in range(J)]
        bands[band_index] = w_band                   
        #print("band shape:", w_band.shape)      
        #print("length of bands:", len(bands))

        # 5) Reconstruct MW map
        f_mw = s2wav.synthesis(
            bands,
            f_scal=scaling,
            L=L,
            lam=lam,
            filters=j_filter,
            reality=reality,
            N=N_directions,
        )

        # 6) Map → alm (MW)
        mw_alm = s2fft.forward(f_mw, L=L, reality=reality)

        # 7) (Optional) per-band spectra plot (kept as-is but using the normalized 'bands')
        ell_all, Dl_all = [], []
        for j in range(J):
            sc0 = jnp.zeros_like(scaling)
            bands_j = [bands[k] if k == j else jnp.zeros_like(bands[k]) for k in range(J)]
            f_band = s2wav.synthesis(
                bands_j, f_scal=sc0, L=L, lam=lam, filters=j_filter, reality=reality, N=N_directions
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
    def visualise_axisym_wavelets(L: int, rel_thresholds=(0.0, 1e-3, 1e-4)):
        """
        Plots the axisymmetric wavelet filters and the scaling function,
        then prints band tables for multiple relative thresholds.

        rel_thresholds : iterable of floats
            Each value is a fraction of the maximum amplitude used to
            decide support:
                support = {ℓ : |w(ℓ)| > thr * max_ℓ |w(ℓ)|}
        """
        psi, phi_l = build_axisym_filter_bank(L=L)

        j_filter = psi
        J = j_filter.shape[0]
        shape = j_filter.shape
        ells = np.arange(L)
        middle_m = L - 1  # m = 0 column

        # --------- plot (independent of threshold) ----------
        plt.figure(figsize=(7, 4))
        base = plt.get_cmap("tab20")

        scal_color = base(0)
        wavelet_colors = [base(i + 1) for i in range(J)]

        # scaling function
        w_scal = np.real(phi_l)
        if w_scal.max() != 0:
            w_scal_plot = w_scal / w_scal.max()
        else:
            w_scal_plot = w_scal

        plt.plot(ells, w_scal_plot, "--", color=scal_color, label="Scal.")

        # wavelets
        for j in range(shape[0]):
            wj = np.real(j_filter[j][:, middle_m])
            if wj.max() != 0:
                wj_plot = wj / wj.max()
            else:
                wj_plot = wj
            plt.plot(ells, wj_plot, color=wavelet_colors[j], label=fr"$j = {j}$")

        plt.xlabel(r"$\ell$", fontsize=16)
        plt.ylabel("Normalised harmonic response", fontsize=16)
        plt.title("Axisym wavelets")
        plt.grid(ls=':')

        plt.legend(bbox_to_anchor=(1.02, 1.0),
                   loc="upper left",
                   borderaxespad=0.)
        plt.tight_layout()
        plt.show()

        # --------- band tables for each threshold ----------
        for thr in rel_thresholds:
            bands = []

            # ----- scaling -----
            w_abs = np.abs(w_scal)
            if w_abs.max() != 0:
                cutoff = thr * w_abs.max()
                support = np.where(w_abs > cutoff)[0]
            else:
                support = np.array([], dtype=int)

            if support.size > 0:
                ell_min  = int(support[0])
                ell_max  = int(support[-1])
                ell_peak = int(ells[np.argmax(w_abs)])
                bands.append(("Scal.", ell_min, ell_peak, ell_max))

            # ----- wavelets -----
            for j in range(shape[0]):
                wj = np.real(j_filter[j][:, middle_m])
                w_abs = np.abs(wj)

                if w_abs.max() == 0:
                    continue

                cutoff = thr * w_abs.max()
                support = np.where(w_abs > cutoff)[0]
                if support.size == 0:
                    continue

                ell_min  = int(support[0])
                ell_max  = int(support[-1])
                ell_peak = int(ells[np.argmax(w_abs)])
                bands.append((j, ell_min, ell_peak, ell_max))

            # ----- print table for this threshold -----
            print(f"\nBand table (relative threshold = {thr:g})")
            print("Wavelet scale j   ell_min^j   ell_peak^j   ell_max^j")
            for label, ell_min, ell_peak, ell_max in bands:
                if isinstance(label, str):   # scaling row
                    lab_str = f"{label:>13s}"
                else:                        # integer j
                    lab_str = f"{label:14d}"
                print(f"{lab_str}   {ell_min:9d}   {ell_peak:11d}   {ell_max:9d}")


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
    

    
