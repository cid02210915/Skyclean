import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import s2wav
import s2wav.filters as filters

def wavelet_transform(mw_pix_map, L_max, N_directions):
    filter = filters.filters_directional_vectorised(L_max, N_directions)
    MW_Pix_wavelet_coeffs, MW_Pix_scaling_coeffs = s2wav.analysis(mw_pix_map, N=N_directions, L=L_max, filters=filter, reality=False)
    return MW_Pix_wavelet_coeffs, MW_Pix_scaling_coeffs

def save_wavelet_scaling_coeffs(wavelet_coeffs, scaling_coeffs, frequency, realization, wav_template = "wavelet_transform/wavelets/wav_MW_maps/Wav_MW_Pix_F{frequency}_S{scale}_R{realization:04d}.npy"
                                , scal_template = "wavelet_transform/wavelets/scal_coeffs/Scal_MW_Pix_F{frequency}_R{realization:04d}.npy"):
    
    # Save wavelet coefficients
    for scale, wav in enumerate(wavelet_coeffs):
        np_wav = np.array(wav)  # Convert JAX array to numpy array
        np.save(wav_template.format(frequency=frequency, scale=scale, realization=realization), np_wav)
    
    # Scaling coefficient is the same for all scales
    np_scaling = np.array(scaling_coeffs)  # Convert JAX array to numpy array
    np.save(scal_template.format(frequency=frequency, realization=realization), np_scaling)



def load_wavelet_scaling_coeffs(frequency, num_wavelets, realization, wav_template = "wavelet_transform/wavelets/wav_MW_maps/Wav_MW_Pix_F{frequency}_S{scale}_R{realization:04d}.npy",
                                scal_template = "wavelet_transform/wavelets/scal_coeffs/Scal_MW_Pix_F{frequency}_R{realization:04d}.npy"):

    wavelet_coeffs = [np.real(np.load(wav_template.format(frequency=frequency, scale=scale, realization=realization))) for scale in range(num_wavelets)]
    scaling_coeffs = np.real(np.load(scal_template.format(frequency=frequency, realization=realization)))
    return wavelet_coeffs, scaling_coeffs
