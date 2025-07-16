import numpy as np
import healpy as hp


def hp_alm_2_mw_alm(hp_alm, L_max):
    """
    Converts spherical harmonics (alm) from healpy to a matrix representation for use in MW sampling.

    This function takes 1D Healpix spherical harmonics coefficients (alm) and converts them into a matrix form 
    that is in (MW sampling, McEwen & Wiaux) sampling. The matrix form is complex-valued and indexed by multipole 
    moment and azimuthal index.

    Parameters:
        hp_alm (numpy.ndarray): The input healpix spherical harmonics coefficients (alm).
        L_max (int): The maximum multipole moment to be represented in the output matrix.
    
    Note: # L_max = 4 | l = 0,1,2,3 , m = -3...0...(L_max-1 = 3)| number of m = 2(L_max-1)+1 = 2L_max-1
    MW sampling fills in positive and negative m, while healpy only stores m >= 0.

    Returns:
        MW_alm (numpy.ndarray): 2D array of shape (Lmax, 2*Lmax-1) MW spherical harmonics coefficients 
    """
    L = L_max + 1 
    MW_alm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    for l in range(L):
        for m in range(l + 1):
            index = hp.Alm.getidx(L_max, l, m)
            col = m + L - 1
            hp_point = hp_alm[index]
            MW_alm[l, col] = hp_point
            if m > 0: 
                MW_alm[l, L-m-1] = (-1)**m * hp_point.conj() # fill m < 0 by symmetry
    return MW_alm

def mw_alm_2_hp_alm(MW_alm):
    """
    Converts spherical harmonics (alm) from MW sampling to healpy representation.

    This function takes a 2D alm array in MW form (MW Sampling, McEwen & Wiaux) and converts them 
    into a 1D array used in healpy sampling. The matrix form is complex-valued and indexed by multipole 
    moment and azimuthal index.

    Notea: MW sampling runs from 1,...,L while healpy runs from 0,...,L-1. Hence the L_max param from Healpy
    is L-1 in MW sampling.
    Healpy only stores m >= 0, while MW sampling fills in both positive and negative m.
    
    Parameters:
        MW_alm (numpy.ndarray): The input MW spherical harmonics coefficients in matrix form.
    
    Returns:
        hp_alm (numpy.ndarray): 1D array of healpy spherical harmonics coefficients
    """
    L = MW_alm.shape[0]
    L_max = L-1
    hp_alm = np.zeros(hp.Alm.getsize(L_max), np.complex128)
    for l in range(L):
        for m in range(l+1):
            col = L_max + m
            idx = hp.Alm.getidx(L_max, l, m)
            hp_alm[idx] = MW_alm[l, col]
    return hp_alm

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




