import pytest
import healpy as hp
import numpy as np
from map_tools import *
from utils import *
import s2fft
import re
import glob

class Tests(): 
    def __init__(self, directory): 
        self.directory = directory
        self.saved_directories = {
            "cfn": os.path.join(self.directory, "CFN_realisations"),
            "processed_maps": os.path.join(self.directory, "processed_maps"),
            "wavelet_coeffs": os.path.join(self.directory, "wavelet_transforms/wavelet_coeffs"),
            "scaling_coeffs": os.path.join(self.directory, "wavelet_transforms/scaling_coeffs"),
            "ilc_synthesised": os.path.join(self.directory, "SILC/ilc_synthesised_maps"),
        }

        self.file_templates = {
            "cfn": os.path.join(self.saved_directories["cfn"], "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
            "cmb": os.path.join(self.saved_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
            "sync": os.path.join(self.saved_directories["processed_maps"], "processed_sync_f{frequency}_lmax{lmax}.npy"),
            "dust": os.path.join(self.saved_directories["processed_maps"], "processed_dust_f{frequency}_lmax{lmax}.npy"),
            "noise": os.path.join(self.saved_directories["processed_maps"], "processed_noise_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
            "wavelet_coeffs": os.path.join(self.saved_directories["wavelet_coeffs"], "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}.npy"),
            "scaling_coeffs": os.path.join(self.saved_directories["scaling_coeffs"], "{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
            "ilc_synthesised": os.path.join(self.saved_directories["ilc_synthesised"], "ILC_synthesised_Map_R{realisation:04d}_lmax{lmax}.npy"),
        }

    @staticmethod
    def test_hp_2_mw_2_hp(): 
        lmax = 64
        cls = np.ones(lmax+1)
        hp_alm = hp.synalm(cls, lmax=lmax)
        mw_alm = SamplingConverters.hp_alm_2_mw_alm(hp_alm, lmax=lmax)
        hp_alm_back = SamplingConverters.mw_alm_2_hp_alm(mw_alm)
        assert np.allclose(hp_alm, hp_alm_back), "The HP alm and the back-converted HP alm do not match."

    @staticmethod
    def test_wavelet_transform(method = "jax_cuda"):
        lmax = 64
        L = lmax + 1 
        cls = np.ones(lmax+1)
        hp_alm = hp.synalm(cls, lmax=lmax)
        mw_alm = SamplingConverters.hp_alm_2_mw_alm(hp_alm, lmax=lmax)
        mw_map = s2fft.inverse(mw_alm, L = L, method = method)
        wavelet_coeffs, _ = MWTools.wavelet_transform(mw_map, L=L, N_directions=1, lam=2.0)
        mw_map_back = MWTools.inverse_wavelet_transform(wavelet_coeffs, L=L, N_directions=1, lam=2.0)
        mw_alm_back = s2fft.forward(mw_map_back, L=L, method=method)
        hp_alm_back = SamplingConverters.mw_alm_2_hp_alm(mw_alm_back)
        assert np.allclose(mw_map, mw_map_back), "The MW map and the back-converted MW map do not match."
        assert np.allclose(mw_alm, mw_alm_back), "The MW alm and the back-converted MW alm do not match."
        assert np.allclose(hp_alm, hp_alm_back), "The HP alm and the back-converted HP alm do not match."

    def test_ilc_shape(self): 
        # Find available ILC files using regex
        ilc_pattern = os.path.join(self.saved_directories["ilc_synthesised"], "ILC_synthesised_Map_R*_lmax*.npy")
        ilc_files = glob.glob(ilc_pattern)
        
        if not ilc_files:
            pytest.skip("No ILC synthesised files found")
        
        # Extract lmax from the first available ILC file
        ilc_regex = r"ILC_synthesised_Map_R(\d+)_lmax(\d+)\.npy"
        ilc_match = re.search(ilc_regex, ilc_files[0])
        if not ilc_match:
            pytest.skip("Could not parse lmax from ILC filename")
        
        ilc_realisation, ilc_lmax = int(ilc_match.group(1)), int(ilc_match.group(2))
        
        # Find corresponding CFN files using regex
        cfn_pattern = os.path.join(self.saved_directories["cfn"], f"cfn_f*_r{ilc_realisation:04d}_lmax{ilc_lmax}.npy")
        cfn_files = glob.glob(cfn_pattern)
        
        if not cfn_files:
            pytest.skip(f"No CFN files found for realisation {ilc_realisation} and lmax {ilc_lmax}")
        
        # Extract frequency from the first available CFN file
        cfn_regex = r"cfn_f(\d+)_r(\d+)_lmax(\d+)\.npy"
        cfn_match = re.search(cfn_regex, cfn_files[0])
        if not cfn_match:
            pytest.skip("Could not parse frequency from CFN filename")
    
        # Load the files
        ilc_mw = np.load(ilc_files[0])  # MW map form
        ilc_hp = SamplingConverters.mw_map_2_hp_map(ilc_mw, lmax=ilc_lmax)
        cfn = hp.read_map(cfn_files[0])
        print(cfn.shape, ilc_hp.shape)
        assert ilc_hp.shape == cfn.shape, f"ILC MW map shape ({ilc_mw.shape}) does not match CFN map shape ({cfn.shape})"

directory = "/Scratch/matthew/data"
tests = Tests(directory)
tests.test_ilc_shape()


