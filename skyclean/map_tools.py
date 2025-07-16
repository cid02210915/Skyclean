from utils import *
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

class HPMapTools():
    """Tools to process healpy/MW maps"""

    @staticmethod
    def reduce_hp_map_resolution(hp_map: np.ndarray, lmax: int, nside: int) -> tuple:
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
    
    @staticmethod
    def beam_convolve(hp_map, lmax: int, standard_fwhm_rad: float) -> np.ndarray: 
        """
        Converts healpix map to alm space, deconvolves the pixel window function,
        applies a standard beam, and converts back to map space.

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
        # Pixel window function
        pixwin = hp.sphtfunc.pixwin(nside, lmax=lmax, pol=False)
        # Divide out pixel window function
        alm_deconv = hp.almxfl(alm_deconv, 1/pixwin)
        # Apply standard beam
        alm_reconv = hp.almxfl(alm_deconv, Standard_bl)
        # Convert back to map
        hp_map_reconv = hp.alm2map(alm_reconv, nside=nside)
        return hp_map_reconv
    
    @staticmethod
    def unit_convert(hp_map: np.ndarray, frequency: str) -> np.ndarray:
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
        return hp_map

class MWMapTools():
    """Tools to process maps in MW (McEwen & Wiaux) sampling"""

    @staticmethod
    def wavelet_transform(mw_map: jnp.ndarray, L_max: int, N_directions: int, lam: float = 2.0,) -> tuple:
        """
        Performs a wavelet transform on a MW map using the S2WAV library.   

        Parameters: 
            mw_map (jnp.ndarray): The input MW map to be transformed.
            L_max (int): Maximum multipole moment for the wavelet transform.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float, optional): Wavelet parameter, default is 2.0.   

        Returns:
            tuple: A tuple containing the wavelet coefficients and scaling coefficients.
        """
        # default JAX path
        j_filter = filters.filters_directional_vectorised(L_max, N_directions, lam = lam)
        wavelet_coeffs, scaling_coeffs = s2wav.analysis(
            mw_map,
            N       = N_directions,
            L       = L_max,
            lam     = lam,
            filters = j_filter,
            reality = False,
        )
        #scaling_coeffs = np.expand_dims(scaling_coeffs, axis=0)  # Ensure scaling coefficients are in the same format as wavelet coefficients
        scaling_coeffs = np.repeat(scaling_coeffs[np.newaxis, ...], 2*N_directions-1, axis=0)    # -> shape (N,1,1)
        wavelet_coeffs.insert(0, scaling_coeffs) #include scaling coefficients at the first index
        return wavelet_coeffs, scaling_coeffs

    @staticmethod
    def save_wavelet_scaling_coeffs(wavelet_coeffs: list, scaling_coeffs: np.ndarray, frequency: str, realization: int, wav_template: str, scal_template: str) -> None:
        """ Saves the wavelet and scaling coefficients to files.    

        Parameters:
            wavelet_coeffs (list): List of wavelet coefficients for each scale.
            scaling_coeffs (np.ndarray): Scaling coefficients.  
            frequency (str): Frequency of the map.
            realization (int): Realization number for the map.
            wav_template (str): Template for the wavelet coefficient file path.
            scal_template (str): Template for the scaling coefficient file path.

        Returns:
            None
        """
        # Save wavelet coefficients
        for scale, wav in enumerate(wavelet_coeffs):
            np_wav = np.array(wav)  # Convert JAX array to numpy array
            np.save(wav_template.format(frequency=frequency, scale=scale, realization=realization), np_wav)
        
        # Scaling coefficient is the same for all scales'
        np_scaling = np.array(scaling_coeffs)  # Convert JAX array to numpy array
        np.save(scal_template.format(frequency=frequency, realization=realization), np_scaling)

    @staticmethod
    def load_wavelet_scaling_coeffs(frequency, num_wavelets, realization, wav_template, scal_template):
        """
        Loads the wavelet and scaling coefficients from files.

        Parameters:
            frequency (str): Frequency of the map.
            num_wavelets (int): Number of wavelet coefficients to load.
            realization (int): Realization number for the map.  
            wav_template (str): Template for the wavelet coefficient file path.
            scal_template (str): Template for the scaling coefficient file path.
        
        Returns:
            tuple: A tuple containing the wavelet coefficients and scaling coefficients.
        """
        wavelet_coeffs = [np.real(np.load(wav_template.format(frequency=frequency, scale=scale, realization=realization))) for scale in range(num_wavelets)]
        scaling_coeffs = np.real(np.load(scal_template.format(frequency=frequency, realization=realization)))
        return wavelet_coeffs, scaling_coeffs
    
    @staticmethod
    def visualise_mw_map(mw_map, title, coord=["G"], unit = r"K"):
        """
        Visualizes a MW pixel wavelet coefficient map using HEALPix mollview.

        Parameters:
            mw_map (numpy array): Array representing the wavelet coefficient map.
            title (str): Title for the visualization plot.  
            coord (list): List of coordinate systems to use for the visualization.
            unit (str): Unit of the map data, default is Kelvin (K).  
        """  
        nrows = 1
        ncols = mw_map.shape[0]
        fig = plt.figure(figsize=(5*ncols, 5*nrows))
        
        L_max = mw_map.shape[1]
        for i in range(ncols):
            original_map_alm = s2fft.forward(mw_map[i], L=L_max, method = "jax_cuda")
            #print("ME alm shape:", original_map_alm.shape)
            original_map_hp_alm = mw_alm_2_hp_alm(original_map_alm)
            original_hp_map = hp.alm2map(original_map_hp_alm, nside=L_max//2)
            panel = i + 1
            hp.mollview(
                original_hp_map,
                coord=coord,
                title=title+f", dir {i+1}",
                unit=unit,
                fig = fig.number,
                sub = (nrows, ncols, panel)
                # min=min, max=max,  # Uncomment and adjust these as necessary for better visualization contrast
            )
            # plt.figure(dpi=1200)
        plt.show()


class ProcessMaps():
    """Process downloaded maps."""
    def __init__(self, components: list, frequencies: list, realisations: int, directory: str = "data/CMB_realisations", noise: bool = True): 
        """
        Parameters: 
            components (list): List of foreground components to download. Includes: 'sync' (synchrotron)
            directory (str): Directory to save the downloaded data.
            frequencies (list): Frequencies of the data to be downloaded.
            realisations (int): Number of realisations to download.
            noise (bool, optional): Whether to download noise realisations.
        """
        self.components = components
        self.frequencies = frequencies
        self.realisations = realisations
        self.directory = directory
        self.noise = noise

    