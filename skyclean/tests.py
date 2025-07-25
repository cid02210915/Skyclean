import pytest
import healpy as hp
import numpy as np
from map_tools import *
from utils import *
import s2fft

def test_hp_2_mw_2_hp(): 
    lmax = 64
    cls = np.ones(lmax+1)
    hp_alm = hp.synalm(cls, lmax=lmax)
    mw_alm = SamplingConverters.hp_alm_2_mw_alm(hp_alm, lmax=lmax)
    hp_alm_back = SamplingConverters.mw_alm_2_hp_alm(mw_alm)
    assert np.allclose(hp_alm, hp_alm_back), "The HP alm and the back-converted HP alm do not match."

def test_wavelet_transform(method = "jax_cuda"):
    lmax = 64
    L = lmax + 1 
    cls = np.ones(lmax+1)
    hp_alm = hp.synalm(cls, lmax=lmax)
    mw_alm = SamplingConverters.hp_alm_2_mw_alm(hp_alm, lmax=lmax)
    mw_map = s2fft.inverse(mw_alm, L = L, method = method)
    wavelet_coeffs, scaling_coeffs = MWTools.wavelet_transform(mw_map, L=L, N_directions=1, lam=2.0)
    mw_map_back = MWTools.inverse_wavelet_transform(wavelet_coeffs, scaling_coeffs, L=L, N_directions=1, lam=2.0)
    assert np.allclose(mw_map, mw_map_back), "The MW map and the back-converted MW map do not match."


