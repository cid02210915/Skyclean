import os 
import healpy as hp
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import s2fft
import s2wav
import s2wav.filters as filters
from .utils import *

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
        hp_alm = hp.map2alm(hp_map, lmax=lmax)
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
        alm = hp.map2alm(hp_map, lmax=lmax)
        # Pixel window function
        pixwin = hp.sphtfunc.pixwin(nside, lmax=lmax, pol=False)
        # Divide out pixel window function
        alm_deconv = hp.almxfl(alm, 1/pixwin)
        # Convert back to map
        hp_map_deconv = hp.alm2map(alm_deconv, nside=nside)
        return hp_map_deconv
    
    
    def beam_convolve(hp_map: np.ndarray, lmax: int, standard_fwhm_rad: float):
        """
        Converts healpix map to alm space, applies a standard beam,
        and converts back to map space.
    
        Parameters:
            hp_map (numpy.ndarray): Input healpix map.
            lmax (int): Maximum multipole moment for spherical harmonics.
            standard_fwhm_rad (float): Standard beam FWHM in radians.   
    
        Returns:
            numpy.ndarray: Reconstructed healpix map after beam convolution.
        """
        nside = hp.get_nside(hp_map)
        alm = hp.map2alm(hp_map, lmax=lmax)
        # Standard beam for the desired FWHM
        Standard_bl = hp.sphtfunc.gauss_beam(standard_fwhm_rad, lmax=lmax, pol=False)
        # Apply standard beam
        alm_conv = hp.almxfl(alm, Standard_bl)
        # Convert back to map
        hp_map_conv = hp.alm2map(alm_conv, nside=nside)
        return hp_map_conv
    
        
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
    
    # For synch/dust/tSZ/noise (have pixel window): beam -> deconv Pℓ -> reduce
    @staticmethod
    def convolve_and_reduce(hp_map: np.ndarray, lmax: int, nside: int, standard_fwhm_rad: float) -> np.ndarray:
        """
        Deconvolve pixel window, convolve with standard beam, then reduce resolution.
        Use for components with pixel window (skyinbands foregrounds, noise).
        """
        
        hp_map_beamed = HPTools.beam_convolve(hp_map, lmax=lmax, standard_fwhm_rad=standard_fwhm_rad)
        hp_map_noP = HPTools.pixwin_deconvolve(hp_map_beamed, lmax=lmax)
        hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(hp_map_noP, lmax=lmax, nside=nside)
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
    def wavelet_transform_from_map(mw_map: jnp.ndarray, L: int, N_directions: int, lam: float = 2.0,):
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
        j_filter = filters.filters_directional_vectorised(L, N_directions, lam = lam)
        
        wavelet_coeffs, scaling_coeffs = s2wav.analysis(
            mw_map,
            N       = N_directions,
            L       = L,
            lam     = lam,
            filters = j_filter,
            reality = True,
        )
        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)   
        #wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs

    
    @staticmethod
    def wavelet_transform_from_alm(mw_alm: jnp.ndarray, L: int, N_directions: int, lam: float = 2.0):
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
        j_filter = filters.filters_directional_vectorised(L, N_directions, lam = lam)
        # remove the last filter (temporary)
        wavelet_coeffs, scaling_coeffs = s2wav.flm_to_analysis(
            mw_alm,
            N       = N_directions,
            L       = L,
            lam     = lam,
            filters = j_filter,
            reality = True,
        )
        scaling_coeffs = np.expand_dims(scaling_coeffs, axis=0)  # Ensure scaling coefficients are in the same format as wavelet coefficients
        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)   
        #wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs
    
    @staticmethod
    def inverse_wavelet_transform(wavelet_coeffs: list, scaling_coeffs, L: int, N_directions: int = 1, lam: float = 2.0,):
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
        j_filter = filters.filters_directional_vectorised(L, N_directions, lam = lam)
        f_scal = scaling_coeffs[0] if getattr(scaling_coeffs, "ndim", 0) == 3 else scaling_coeffs

        mw_map = s2wav.synthesis(
         wavelet_coeffs,
         L       = L,
         f_scal  = f_scal,
         lam     = lam,
         filters = j_filter,
         reality = True,
         N = N_directions
        )
        return mw_map

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
        plt.savefig(f'{title}')
        plt.show()

    
    @staticmethod
    def visualise_axisym_wavelets(L: int, lam: float = 2.0):
        """
        Plots the wavelet filters for the given parameters.
        Only works for axisymmetric wavelets (N_directions = 1).

        Parameters:
            L (int): Maximum multipole moment for the wavelet transform; lmax+1
            lam (float): Wavelet parameter, default is 2.0. 
        """
        j_filter = filters.filters_directional_vectorised(L=L, N=1, lam=lam)[0]
        shape = j_filter.shape
        l_list = np.arange(L)
        middle_m = shape[2]//2 # m = 0 along which the axisymmetric wavelet is defined
        for i in range(shape[0]):
            plt.plot(l_list, np.real(j_filter[i][:,middle_m]), label = f"scale {i}")
            plt.xlabel("l", fontsize=16)
            plt.ylabel("Real part of wavelet filter", fontsize=16)
            plt.title(f"lambda = {lam}, axisym")
            plt.xscale('log')
            plt.legend()
            plt.grid(ls=':')
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
        hp_alm = hp.map2alm(hp_map, lmax=lmax)
        mw_alm = SamplingConverters.hp_alm_2_mw_alm(hp_alm, lmax)
        mw_map = s2fft.inverse(mw_alm, L=L, method=method, reality = True)
        return mw_map
    
    @staticmethod
    def mw_map_2_hp_map(mw_map: np.ndarray, lmax: int, method = "jax_cuda"):
        """
        Converts a MW map to a Healpix map by transforming the map to spherical harmonics and then converting the alm to Healpix sampling.

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

