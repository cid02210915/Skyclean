import numpy as np

# Convert arcminutes to radians
def arcmin_to_radians(arcmin):
    '''
    Takes arcminutes and converts them to radians, 
    used for beam_fwhm = {30: 32.33, 44: 27.01, 70: 13.25},
    hp.sphtfunc.gauss_beam(fwhm_rad, lmax=lmax-1, pol=False),
    '''
    return np.radians(arcmin / 60)
