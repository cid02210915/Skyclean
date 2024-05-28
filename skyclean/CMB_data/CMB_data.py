import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import numpy as np
import s2wav


class CMB_Data():
    existing_CMB_Data = []
    def __init__(self, path):

        # local path of the Healpix map 
        self.path = path
        self.raw_map = hp.read_map(self.path)
        self.nside = hp.get_nside(self.raw_map)
        self.lmax = None
        self.alm = None
        print("hello from CMB_Data")
        pass
    @classmethod
    def show_existing_CMB_Data(cls):
        print(cls.existing_CMB_Data)
        pass

    def test():
        print("hello from test")
        pass