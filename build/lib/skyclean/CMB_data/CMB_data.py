import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import numpy as np
import s2wav


class CMB_Data():
    ''' help(CMB_Data())
    Class to hold the CMB map 
    '''
    #existing_CMB_Data = []
    def __init__(self, path):

        # local path of the Healpix map 
        self.path = path
        
        # Map data before the wavelet transform (analysis)
        self.original_hp_map = hp.read_map(self.path)
        self.original_hp_alm = np.array([])
        self.original_mw_map = np.array([])
        self.original_mw_alm = np.array([])

        # Parameters
        self.nside = hp.get_nside(self.original_hp_map)
        self.lmax = self.nside * 2


        # Wavelet transform
        self.wavelet_coeff = np.array([])   # Wavelet coefficients
        self.Scaling_coeff = np.array([]) # Scaling coefficients

        # After the wavelet transform 
        self.reconstructed_hp_map = np.array([])
        self.reconstructed_hp_alm = np.array([])

        self.reconstructed_mw_map = np.array([])
        self.reconstructed_mw_alm = np.array([])


        print("CMB_Data object created, (use show_attributes() to check the attributes)")

    
    def show_attributes(self):
        '''
        Display all the attributes of the CMB_Data object
        '''

        attributes = [["path", self.path],
                      ["nside", self.nside],
                      ["lmax usually 2 * nisde", self.lmax]]
        
        maps = [["original_hp_map", "Available" if self.original_hp_map.any() else None],
                ["original_mw_map", "Available" if self.original_mw_map.any() else None],
                ["reconstructed_hp_map", "Available" if self.reconstructed_hp_map.any() else None],
                ["reconstructed_mw_map", "Available" if self.reconstructed_mw_map.any() else None]]    
        
        alms = [["original_hp_alm", "Available" if self.original_hp_alm.any() else None],
                ["reconstructed_hp_alm", "Available" if self.reconstructed_hp_alm.any() else None]]
        
        wavelet = [["wavelet_coeff", "Available" if self.wavelet_coeff.any() else None],
                   ["Scaling_coeff", "Available" if self.Scaling_coeff.any() else None]]
        

        print("Attributes:\n", attributes)
        print("Maps:\n", maps)
        print("Alms:\n", alms)
        print("Wavelet_related:\n", wavelet)



        
    


    def hp_alm_to_mw_alm(self, hp_alm,lmax):
        '''
        It takes the healpix-style alm (Spherical harmonic Coefficient) and lmax (level of details) and returns the MW_alm
        mapping the coefficients from 1D array to 2D array.

        '''
        # Rearrange coefficients for s2wav: from 1 dimensional to 2 dimensional
        # s2fft only works with alm in 2d
        MW_alm = np.zeros((lmax, 2 * lmax - 1), dtype=np.complex128)

        for l in range(lmax):
            for m in range(-l, l + 1):
                index = hp.Alm.getidx(lmax - 1, l, abs(m))
                if m < 0:
                    MW_alm[l, lmax + m - 1] = (-1)**m * np.conj(hp_alm[index])
                else:
                    MW_alm[l, lmax + m - 1] = hp_alm[index]

        return MW_alm

    def mw_alm_to_hp_alm(self, MW_alm, lmax):
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
        
    def plot_mollview (self, map, title = "Map in Mollweide view", coord = ["G"], unit=r"$\mu$K",min=-300, max=300, enhence = 1e6):

        '''
        Plot the Mollweide view of the map
        '''
        
        hp.mollview(
            map*enhence,
            coord=coord,
            title=title,
            unit=unit,
            min=min, 
            max=max,
        )

