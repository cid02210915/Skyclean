# from .CMB_data import cmb_data
from skyclean.CMB_data import CMB_Data
# from .tools.useful_functions import arcmin_to_radians
from skyclean.tools.useful_functions import arcmin_to_radians, download_cmb_realizations
from skyclean.tools.map_functions import beam_deconvolution, reduce_hp_map_resolution, hp_alm_2_mw_alm
from skyclean.tools.wavelets_functions import wavelet_transform, save_wavelet_scaling_coeffs, load_wavelet_scaling_coeffs
# print(type(CMB_Data()))


