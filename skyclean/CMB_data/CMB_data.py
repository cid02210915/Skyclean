import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import numpy as np
import s2wav


class CMB_Data():
    ''' help(CMB_Data())
    Class to hold the CMB map 
    '''
    existing_CMB_Data = []
    def __init__(self, path):

        # local path of the Healpix map 
        self.path = path
        
        # Map data before the wavelet transform (analysis)
        self.original_hp_map = hp.read_map(self.path)
        self.original_hp_alm = None
        self.original_mw_map = None
        self.original_mw_alm = None

        # Parameters
        self.nside = hp.get_nside(self.hp_map)
        self.lmax = self.nside * 2


        # Wavelet transform
        self.wavelet_coeff = None   # Wavelet coefficients
        self.Scaling_coeff = None # Scaling coefficients

        # After the wavelet transform 
        self.reconstructed_hp_map = None
        self.reconstructed_hp_alm = None

        self.reconstructed_mw_map = None
        self.reconstructed_mw_alm = None


        print("hello from CMB_Data")
        pass
    @classmethod
    def show_existing_CMB_Data(cls):
        '''
        class -->())
        '''
        print(cls.existing_CMB_Data)
        pass

    def mw_alm_to_hp_alm(MW_alm, lmax):
        '''
        It takes the MW_alm (Spherical harmonic Coefficient) and lmax (level of details) and returns the healpix-style alm
        mapping the coefficients from 2D array to 1D array.
        
        Note:
        Usually used after the wavelet transform.
        Intermediary function to convert the MW map to healpix-style map.
        Typical previous step: s2fft.forward(MW_map, lmax) to get the MW_alm.
        Next Step: hp.alm2map(healpix_alm, nside) to get the healpix map
       
        '''

        n_coeff = hp.Alm.getsize(lmax-1)
        healpix_alm = np.zeros(n_coeff, dtype=np.complex128)

        for l in range(lmax):
            for m in range(-l, l + 1):
                alm_index = hp.Alm.getidx(lmax - 1, l, abs(m))
                if alm_index >= n_coeff:
                    print(f"Index {alm_index} is out of bounds for alm with size {n_coeff}")
                    continue
                if m < 0:
                    healpix_alm[alm_index] = (-1) ** m * np.conj(MW_alm[l, lmax + m - 1])
                else:
                    healpix_alm[alm_index] = MW_alm[l, lmax + m - 1]
        
        return healpix_alm
        