import jax
jax.config.update("jax_enable_x64", True)
import s2fft
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
# import os
from astropy.io import fits #For beam deconvolution

def hp_alm_2_mw_alm(hp_alm, L_max):
    """
    Converts spherical harmonics (alm) to a matrix representation for use in MW sampling.

    This function takes 1D Healpix spherical harmonics coefficients (alm) and converts them into a matrix form 
    that is in (MW sampling, McEwen & Wiaux) sampling. The matrix form is complex-valued 
    and indexed by multipole moment and azimuthal index.

    Parameters:
        hp_alm (numpy.ndarray): The input healpix spherical harmonics coefficients (alm).
        L_max (int): The maximum multipole moment to be represented in the output matrix.
    
    Note: # L_max = 4 | l = 0,1,2,3 , true lmax is L_max-1 = 3 | m = -3...0...(L_max-1 = 3)| number of m = 2(L_max-1)+1 = 2L_max-1

    Returns:
        MW_alm (numpy.ndarray): 2D array of shape (Lmax, 2*Lmax-1) MW spherical harmonics coefficients 
    """

    MW_alm = np.zeros((L_max, 2 * L_max - 1), dtype=np.complex128)

    for l in range(L_max):
        for m in range(-l, l + 1):
            index = hp.Alm.getidx(L_max - 1, l, abs(m))
            if m < 0:
                MW_alm[l, L_max + m - 1] = (-1) ** m * np.conj(hp_alm[index])
            else:
                MW_alm[l, L_max + m - 1] = hp_alm[index]

    return MW_alm


def mw_alm_2_hp_alm(MW_alm, lmax):
    '''MW_alm: 2D array of shape (Lmax, 2*Lmax-1) (MW sampling, McEwen & Wiaux)
    '''
    # Initialize the 1D hp_alm array with the appropriate size
    hp_alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
        
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            index = hp.Alm.getidx(lmax, l, abs(m))
            if m < 0:
                hp_alm[index] = (-1)**m * np.conj(MW_alm[l, lmax + m])
            else:
                hp_alm[index] = MW_alm[l, lmax + m]

    return hp_alm

def hp_fits_map_to_MW_map(file_path):
    """
    Process a FITS file containing a HEALPix map and convert it into a map 
    using spherical harmonics and molecular wave (MW) coefficients.

    Parameters:
    -----------
    file_path : str
        Path to the FITS file containing the HEALPix map.

    Returns:
    --------
    np.ndarray
        A numpy array representing the real part of the map obtained from the 
        inverse transformation of MW alm coefficients.

    Notes:
    ------
    - This function reads a HEALPix map from a FITS file.
    - It calculates the maximum multipole order (L_max) based on the nside of the map.
    - The map is converted to spherical harmonics (alm coefficients).
    - The HEALPix alm coefficients are then converted to MW (Molecular Wave) alm coefficients.
    - Finally, the MW alm coefficients are transformed back to a map, and the real part 
      of the result is returned.

    Example:
    --------
    >>> file_path = "CMB_total/CSN_HP_Map_F100_L128_R0000.fits"
    >>> resulting_map = process_fits_to_map(file_path)
    """
    # Read the input map from a FITS file
    hp_map = hp.read_map(file_path)
    
    # Determine the maximum multipole order
    L_max = hp.get_nside(hp_map) * 2
    
    # Convert the map to spherical harmonics (alm coefficients)
    hp_alm = hp.map2alm(hp_map, lmax=L_max - 1)
    
    # Convert the HEALPix alm coefficients to MW (Molecular Wave) alm coefficients
    MW_alm = hp_alm_2_mw_alm(hp_alm, L_max)
    
    # Convert the MW alm coefficients back to a map (real part only)
    MW_map = np.real(s2fft.inverse(MW_alm, L=L_max))
    
    return MW_map



def reduce_hp_map_resolution(hp_map, lmax, nside):
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


def beam_deconvolution(hp_map, frequency, lmax, standard_fwhm_rad, beam_path, LFI_beam_fwhm = {"030": 32.33, "044": 27.01, "070": 13.25}):
    """
    Performs beam deconvolution on the given CMB map data and returns the deconvolved map.

    Parameters:
        cmb_map (fits): CMB map data.
        frequency (str): Frequency identifier (e.g., "030", "044").
        lmax (int): Maximum multipole moment.
        standard_fwhm_rad (float): Standard beam full-width half-maximum in radians.
        beam_path (str): Path to the beam data file specific to the frequency.
        LFI_beam_fwhm (dict): Dictionary of beam full-width half-maximum (FWHM) in arcminutes for LFI frequencies.
    Returns:
      deconvolved_map (fits): The deconvolved CMB map.
    """

    nside = hp.get_nside(hp_map)
    cmb_alm = hp.map2alm(hp_map, lmax=lmax)

    
    # Standard beam for the desired FWHM
    Standard_bl = hp.sphtfunc.gauss_beam(standard_fwhm_rad, lmax=lmax-1, pol=False)
    
    # Pixel window function
    pixwin = hp.sphtfunc.pixwin(nside, lmax=lmax, pol=False)
    
    # LFI beam deconvolution
    if frequency in {"030", "044", "070"}:
        # Deconvolution for lower frequencies
        fwhm_rad = np.radians(LFI_beam_fwhm[frequency] / 60)
        bl = hp.sphtfunc.gauss_beam(fwhm_rad, lmax=lmax-1, pol=False)
        new_cmb_alm = hp.almxfl(cmb_alm, 1/bl)
    # HFI beam deconvolution
    else:
        # Deconvolution using FITS file for higher frequencies
        hfi = fits.open(beam_path)
        beam = hfi[1].data["TEMPERATURE"]
        new_cmb_alm = hp.almxfl(cmb_alm, 1/beam)
    
    # Apply pixel window function and standard beam
    new_cmb_alm = hp.almxfl(new_cmb_alm, 1/pixwin)
    new_cmb_alm = hp.almxfl(new_cmb_alm, Standard_bl)
    
    # Convert back to map
    deconvolved_map = hp.alm2map(new_cmb_alm, nside=nside)
    
    return deconvolved_map


def visualize_MW_Pix_map(MW_Pix_Map, title, coord=["G"], unit = r"K"):
    """
    Processes a MW pixel wavelet coefficient map and visualizes it using HEALPix mollview.

    Parameters:
        MW_Pix_Map (numpy array): Array representing the wavelet coefficient map.
        title (str): Title for the visualization plot.

    Returns:
        Only Displays a mollview map.
    """
    # The newly generated wavelet coefficient map is in three dimensions
    if len(MW_Pix_Map.shape) == 3:
        L_max = MW_Pix_Map.shape[1]
    else:
        L_max = MW_Pix_Map.shape[0]
    original_map_alm = s2fft.forward(MW_Pix_Map, L=L_max)
    print(" Alm shape (Lmax, 2 * Lmax -1): ", original_map_alm.shape)
    
    original_map_hp_alm = mw_alm_2_hp_alm(original_map_alm, L_max - 1)
    original_hp_map = hp.alm2map(original_map_hp_alm, nside=(L_max - 1)//2)

    hp.mollview(
        original_hp_map,
        coord=coord,
        title=title,
        unit=unit,
        # min=min, max=max,  # Uncomment and adjust these as necessary for better visualization contrast
    )
    # plt.figure(dpi=1200)
    plt.show()