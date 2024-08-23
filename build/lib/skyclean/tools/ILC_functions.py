import jax
jax.config.update("jax_enable_x64", True)
import s2fft
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
# import os
from astropy.io import fits #For beam deconvolution


def MW_Map_doubleworker(MW_Pix_Map):
    '''
    Input: MW_Pix_Map: list of mw maps at different scales 
    Each pixel map is a wavelet pixel map of shape (1, Lmax, 2*Lmax-1) (MW sampling, McEwen & Wiaux)
    It is the output of s2wav.analysis
    (Scale: 0, size (1, 4, 7))

    Process:
    1. Covert MW Pixel Map to MW alm space using s2fft.forward

    2. Add zero to the mw alms  
    
    3. Convert mw alm to mw map 
    
    '''
    if MW_Pix_Map.ndim == 3:
        forward_L = MW_Pix_Map.shape[1]
    else:
        forward_L = MW_Pix_Map.shape[0]
    MW_alm = s2fft.forward(MW_Pix_Map, L = forward_L)
    L = MW_alm.shape[0]
    padded_alm = np.zeros((2*L-1,2*(2*L-1)-1),dtype=np.complex128)
    # L = 4 | l = 0,1,2,3 , true lmax is L-1 = 3 | m = -3...0...(L-1)| m = 2(L-1)+1 = 2L-1      
    # double true lmax: 2*3 = 6, and add 1, new L = 7 = 2(L-1)+1 = 2L-1
    # new m = -6...0...(new L-1) | new m = 2*(2L-1)-1
    inner_matrix_middle = MW_alm.shape[1] // 2
    outer_matrix_middle = padded_alm.shape[1] // 2
    start_col = (outer_matrix_middle - inner_matrix_middle)
    end_col = start_col + MW_alm.shape[1] # not included
      
    padded_alm[:MW_alm.shape[0], start_col:end_col] = MW_alm
    # print(padded_alm[:MW_alm.shape[0], start_col:start_col + end_col].shape)
    # print("padded alm size", padded_alm)
    # print(padded_alm.shape)
    
    MW_Pix_Map_doubled = np.real(s2fft.inverse(padded_alm, L = padded_alm.shape[0]))
    # print("Scale:","doubled map size", MW_Pix_Map_doubled.shape)
    # Note
    # assert imaginery part is around zero
    # print(np.imag(MW_Pix_Map_doubled))
    # MW_Pix_Map_doubled = s2fft.inverse(MW_alm_doubled, L = MW_alm_doubled.shape[0])
    
    return MW_Pix_Map_doubled