import os
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2wav
import math 
import healpy as hp
from s2wav import filters
import concurrent.futures
from .map_tools import *
from .utils import *
from .file_templates import FileTemplates
import concurrent.futures
import time

from .utils import normalize_targets   
from .utils import save_array 

class SILCTools():
    '''Tools for Scale-discretised, directional wavelet ILC (SILC).'''
    @staticmethod
    def Single_Map_doubleworker(mw_map: np.ndarray, method: str):
        """
        Doubles the resolution of a single MW pixel map using s2fft.

        Params:
            mw_map (np.ndarray): MW pixel map at original resolution.
            method (str): s2fft method to use for doubling the resolution.
        
        Returns:
            mw_map_doubled: The MW pixel map with increased resolution.
        """
        
        # use jax/numpy
        #print(mw_map.shape)
        alm = s2fft.forward(mw_map, L=mw_map.shape[1], method = method, spmd = False, reality = True)

        L = alm.shape[0]
        H = 2*L - 1
        W = 2*H - 1
        padded = np.zeros((H, W), dtype=np.complex128)
        mid_in = alm.shape[1]//2
        mid_out = W//2
        start = mid_out - mid_in
        padded[:L, start:start+alm.shape[1]] = alm

        x2 = np.real(s2fft.inverse(padded, L=H, method = method, spmd = False, reality = True))

        return x2
    
    @staticmethod
    def smoothed_covariance(MW_Map1: np.ndarray, MW_Map2: np.ndarray, method: str):
        """
        Parameters:
            MW_Map1, MW_Map2 (np.ndarray): same‐shape complex np.ndarray wavelet maps

        Returns:
            R_covariance_map: real‐valued covariance map as np.ndarray
        """
        smoothing_L = MW_Map1.shape[0]
        # 1) pixel covariance
        map1 = np.real(MW_Map1)
        map2 = np.real(MW_Map2)
        Rpix = np.multiply(map1, map2) + 0.j

        # 2) forward (smooth in harmonic space for efficieny)
        Ralm = s2fft.forward(Rpix, L=smoothing_L, method = method, spmd = False, reality = True)

        # 3) Gaussian beam
        nsamp = 1200.0
        lmax = Ralm.shape[0]
        npix = hp.nside2npix(1 << (int(0.5*lmax)-1).bit_length())
        scale_fwhm = 4.0 * math.sqrt(nsamp / npix)
        gauss_beam = hp.gauss_beam(scale_fwhm, lmax=lmax-1)

        # 4) convolve
        convolved = np.zeros_like(Ralm, dtype=np.complex128)
        convolved = Ralm * gauss_beam[:,None]

        # 5) inverse
        Rmap = np.real(s2fft.inverse(convolved, L=smoothing_L, method = method, spmd = False, reality = True))

        return Rmap
    
    @staticmethod
    def compute_covariance(task):
        """
        Computes the covariance between two frequency maps at a given scale.

        Args:
        task (tuple): A tuple containing (i, fq, frequencies, scale, doubled_MW_wav_c_j).

        Returns:
        tuple: A tuple containing indices i, fq and the computed covariance matrix.
        """
        i, fq, freqs, scale_i, data_dict, method = task  

        key_i  = (freqs[i],  scale_i)
        key_fq = (freqs[fq], scale_i)
    
        cov = SILCTools.smoothed_covariance(
            data_dict[key_i], data_dict[key_fq], method  
        )
        return i, fq, cov
    

    @staticmethod
    def calculate_covariance_matrix(frequencies: list, doubled_MW_wav_c_j: dict, scale: int, 
                                    realisation: int, method: str, path_template: str):
        """
        Calculates the covariance matrices for given frequencies and saves them to disk,
        accommodating any size of the input data arrays.
        
        Parameters:
            frequencies (list): List of frequency indices.
            doubled_MW_wav_c_j (dict): Dictionary containing data arrays for covariance calculations.
            scale (int): The scale.
            realisation (int): The realisation.

        Returns:
            full_array: np.ndarray: A 4D array containing the covariance matrices for the given frequencies.
        """
        # Check dimensions of the first item to set the size of the covariance matrices
        if not frequencies:
            raise ValueError("Frequency list is empty.")

        # --- minimal normalization to fit current pipeline ---
        norm_freqs = [str(f).zfill(3) for f in frequencies]
        scale_i = int(scale)

        # Check dimensions from a sample
        sample_data = doubled_MW_wav_c_j[(norm_freqs[0], scale_i)]  # CHANGED
        n_rows, n_cols = sample_data.shape

        total_frequency = len(norm_freqs)
        full_array = np.zeros((total_frequency, total_frequency, n_rows, n_cols))

        # Upper-triangle tasks; compute_covariance should accept (i, fq, freqs, scale, data_dict)
        tasks = [(i, fq, norm_freqs, scale_i, doubled_MW_wav_c_j, "jax_cuda")
                 for i in range(total_frequency) for fq in range(i, total_frequency)]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(SILCTools.compute_covariance, tasks)
            for result in results:
                i, fq, covariance_matrix = result
                full_array[i, fq] = covariance_matrix

        # Fill symmetric part
        for l1 in range(1, total_frequency):
            for l2 in range(l1):
                full_array[l1, l2] = full_array[l2, l1]

        # Save using normalized frequency tags
        f_str = "_".join(norm_freqs)
        save_path = path_template.format(frequencies=f_str, scale=scale_i, realisation=realisation)
        np.save(save_path, full_array)

        return full_array


    @staticmethod
    def double_wavelet_maps(original_wavelet_c_j: dict, frequencies: list, scales: list, 
                            realisation: int, method: 'jax', *,
                            # --- optional, for saving; safe no-ops if not provided ---
                            path_template: str | None = None,
                            component: str | None = None,
                            lmax: int | None = None,
                            lam: float | None = None):
        """
        Doubles the resolution of wavelet maps and returns them as a dictionary.
    
        Parameters:
            original_wavelet_c_j (dict): Dictionary containing the original wavelet maps.
            frequencies (list): List of frequency strings.
            scales (list): List of scale indices.
            realisation (int): The realisation number for file naming.
            method (str): s2fft method to use for forward/inverse.
    
        Optional (for saving):
            path_template (str): e.g. ".../doubled_{component}_f{frequency}_s{scale}_r{realisation}_lmax{lmax}_lam{lam}.npy"
            component (str): value for {component}/{comp} in filenames (e.g. "cmb", "cfn").
            lmax (int): value for {lmax} in filenames.
            lam (float): value for {lam} in filenames.
    
        Returns:
            dict: A dictionary with keys as (frequency, scale) and values as doubled wavelet maps.
        """
    
        # minimal compatibility shim:
        # if 'method' looks like a path/template, treat it as the save template
        KNOWN = {"jax_cuda", "jax"}
        if path_template is None and isinstance(method, str) and method not in KNOWN:
            if (os.sep in method) or ("{" in method) or str(method).endswith(".npy"):
                path_template = method
                method = "jax"
            else:
                raise ValueError(f"Method {method} not recognised.")    
        # ---- GPU compute (single process, single JAX context) ----
        doubled_MW_wav_c_j = {}
        for i in frequencies:
            for j in scales:
                arr = SILCTools.Single_Map_doubleworker(original_wavelet_c_j[(i, j)], method)
                doubled_MW_wav_c_j[(i, j)] = arr  # likely a jnp.DeviceArray    
        # ---- Optional: threaded I/O (no pickling, overlaps disk) ----
        if path_template is not None:
            realisation_int = int(realisation)
            def freq_label(f): return f if isinstance(f, str) else f"{int(f):03d}"
            comp_val = component if component is not None else "cmb"
            lmax_val = 512 if lmax is None else lmax
            lam_val  = 2.0 if lam  is None else lam
            F_str = "_".join(freq_label(ff) for ff in frequencies)    
            tasks = []
        for f in frequencies:
            for j in scales:
                out_path = path_template.format(
                    component=comp_val, comp=comp_val, extract_comp=comp_val,
                    frequency=freq_label(f),
                    scale=int(j),
                    realisation=realisation_int,
                    lmax=lmax_val,
                    lam=lam_val,
                    frequencies=F_str,
                )
                arr = np.asarray(doubled_MW_wav_c_j[(f, j)])  # ensure NumPy
                tasks.append((out_path, arr))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(executor.map(save_array, tasks))
    

    
    @staticmethod
    def find_f_from_extract_comp(F, extract_comp_or_comps, reference_vectors, allow_sign_flip=False, atol=1e-8):

        if isinstance(extract_comp_or_comps, (str, bytes)):
            names = [extract_comp_or_comps]
        else:
            names = list(extract_comp_or_comps)

        N_comp = F.shape[1]
        f = np.zeros(N_comp, dtype=float)

        for name in names:
            key = name.lower()
            if key not in reference_vectors:
                raise ValueError(f"Reference vector for '{name}' not set in reference_vectors.")
            target_vec = reference_vectors[key]
            t = target_vec / np.linalg.norm(target_vec)

            matched = False
            for j in range(N_comp):
                col = F[:, j] / np.linalg.norm(F[:, j])
                if np.allclose(col, t, atol=atol) or (allow_sign_flip and np.allclose(col, -t, atol=atol)):
                    f[j] = 1.0          # ← always +1 (no -1 branch)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Component '{name}' not found in F.")
        return f


    @staticmethod
    def compute_weight_vector(R: np.ndarray, scale: int, realisation: int):
        """
        Processes the given 4D matrix R by computing and saving the weight vectors for each matrix in the first two dimensions.
        Also stores results in memory as arrays and saves them to disk. Adjusts the size of the identity vector based on sub-matrix size.

        Parameters:
            R (np.ndarray): A 4D matrix with dimensions suitable for swapping and inverting.
            scale (int): The scale.
            realisation (int): The realisation.
        Returns:
            inverses: (np.ndarray): An Array containing the inverse matrices
            weight_vectors (np.ndarray): A 3D Array containing the weight vector.
            The size of the first two dimensions of the weight vector is the size of the wavelet coefficient map at the given scale.
            The third dimension is the weight vector (The contribution from each frequency).
            Each element of the weight vector is a 1D array.
            singular_matrices_location (list): The locations of singular matrices.
        """
        # print(R.shape)
        # Swap the axes to get R_Pix
        R_Pix = np.swapaxes(np.swapaxes(R, 0, 2), 1, 3) #(pix,pix,freq,freq)
        # Get dimensions for looping and size of sub-matrices
        dim1, dim2, subdim1, subdim2 = R_Pix.shape
        # print(dim1, dim2, subdim1, subdim2)
        # Create arrays to store inverses and weight vectors
        inverses = np.zeros((dim1, dim2, subdim1, subdim2))
        weight_vectors = np.zeros((dim1, dim2, subdim1)) # weight vector at each pixel (dim1,dim2) and channel
        # Realistion 6 has a singular matrix
        # Adjust identity vector size based on sub-matrix dimensions
        identity_vector = np.ones(subdim2, dtype=float)
        singular_matrices_location = []
        singular_matrices = []
        for i in range(dim1):
            for j in range(dim2):
                det = np.linalg.det(R_Pix[i, j])
                if det == 0:
                    zeros = np.zeros((subdim1))
                    singular_matrices_location.append((i,j))
                    singular_matrices.append(R_Pix[i, j])
                    weight_vectors[i, j] = zeros
                else:
                    inverses[i, j] = np.linalg.inv(R_Pix[i, j])
                    numerator = np.dot(inverses[i, j], identity_vector)
                    denominator = np.dot(np.dot(inverses[i, j], identity_vector),identity_vector)
                    weight_vectors[i, j] = numerator / denominator
        if len(singular_matrices_location) > 0:
            print("Discovered ", len(singular_matrices_location), "singular matrices at scale", scale, "realisation", realisation)
        return weight_vectors
    
    

    @staticmethod
    def compute_weights_generalised(
        R, scale, realisation,
        weight_vector_matrix_template,
        comp, extract_comp,
        constraint=False, F=None, f=None,
        reference_vectors=None
    ):
        """
        Computes weight vectors from a covariance matrix R using either standard or generalized ILC.

        Args:
            R (np.ndarray): (Nf,Nf) or (H,W,Nf,Nf)
            scale (int): Wavelet scale index.
            realisation (str): realisation id string.
            weight_vector_matrix_template (str): Save path template.
            comp (str): Component tag for filenames.
            extract_comp (str or None): Target component name (for constrained ILC).
            constraint (bool): Use constrained ILC if True.
            F (np.ndarray): Spectral response, shape (Nf, Nc) when constraint=True.
            f (np.ndarray): Constraint vector, shape (Nc,).
            reference_vectors (dict): Dict of named reference spectra (for auto-f).

        Returns:
            inverses, weight_vectors, singular_matrices_location, extract_comp
        """
        # --- shape handling ---
        if R.ndim == 4:
            R_Pix = np.swapaxes(np.swapaxes(R, 0, 2), 1, 3)  # -> (H,W,Nf,Nf)
            dim1, dim2 = R_Pix.shape[:2]
            subdim1, subdim2 = R_Pix.shape[2:]               # Nf, Nf
        elif R.ndim == 2:
            R_Pix = R
            dim1, dim2 = 1, 1
            subdim1, subdim2 = R_Pix.shape                   # Nf, Nf
        else:
            raise ValueError(f"Unexpected array dimension: {R.ndim}")

        N_freq = subdim2  # convenience; equals number of channels

        # --- allocate outputs (before branch) ---
        inverses = np.zeros((dim1, dim2, subdim1, subdim2)) if R.ndim == 4 else np.zeros((subdim1, subdim2))
        weight_vectors = np.zeros((dim1, dim2, subdim1)) if R.ndim == 4 else np.zeros(subdim1)
        singular_matrices_location = []

        # --- branch config ---
        if constraint:
            if F is None:
                raise ValueError("F must be provided when constraint=True")
            Nf_F, N_comp = F.shape
            if Nf_F != N_freq:
                raise ValueError(f"F has {Nf_F} rows but R has {N_freq} channels")

            # Automatically set f from extract_comp if given
            if f is None and extract_comp is not None:
                f = SILCTools.find_f_from_extract_comp(F, extract_comp, reference_vectors)
            if f is None:
                raise ValueError("Constraint vector f must be provided (or inferable) when constraint=True")
            if f.shape != (N_comp,):
                raise ValueError(f"Constraint vector f must have shape ({N_comp},)")
        else:
            # Unconstrained ILC uses the all-ones vector; no F/extract_comp needed
            identity_vector = np.ones(N_freq, dtype=float)   # <<< fix: ones, not a picked index

            # ----------------------------------------------------------

        # --- per-pixel (or single) solve ---
        for i in range(dim1):
            for j in range(dim2):
                try:
                    R_ij = R_Pix[i, j] if R.ndim == 4 else R_Pix
                    R_inv = np.linalg.inv(R_ij)

                    if constraint:
                        # Step 1: Fᵗ R⁻¹
                        FT_Rinv = np.dot(F.T, R_inv)                 # (Nc, Nf)
                        # Step 2: Fᵗ R⁻¹ F
                        constraint_matrix = np.dot(FT_Rinv, F)       # (Nc, Nc)
                        # Step 3: (Fᵗ R⁻¹ F)⁻¹
                        constraint_matrix_inv = np.linalg.inv(constraint_matrix)
                        # Step 4: build temp = (Fᵗ R⁻¹ F)⁻¹ f
                        temp = np.dot(constraint_matrix_inv, f)      # (Nc,)
                        # Step 5: F temp
                        F_temp = np.dot(F, temp)                     # (Nf,)
                        # Step 6: w = R⁻¹ F (Fᵗ R⁻¹ F)⁻¹ f
                        w = np.dot(R_inv, F_temp)                    # (Nf,)
                        w = np.asarray(w).ravel()
                    else:
                        num = np.dot(R_inv, identity_vector)         # (Nf,)
                        den = float(np.dot(num, identity_vector))    # scalar
                        w = (num / den).ravel()                      # (Nf,)

                    if R.ndim == 4:
                        inverses[i, j] = R_inv
                        weight_vectors[i, j] = w
                    else:
                        inverses = R_inv
                        weight_vectors = w

                except np.linalg.LinAlgError:
                    singular_matrices_location.append((i, j))
                    singular_matrix_path = weight_vector_matrix_template.format(
                        comp=comp,
                        extract_comp=extract_comp,
                        scale=scale,
                        realisation=realisation
                    ).replace(".npy", f"_singular_{i}_{j}.npy")
                    np.save(singular_matrix_path, R_Pix[i, j] if R.ndim == 4 else R_Pix)
                    if R.ndim == 4:
                        weight_vectors[i, j] = np.zeros(N_freq)
                    else:
                        weight_vectors = np.zeros(N_freq)

        # save final weight vector matrix
        np.save(
            weight_vector_matrix_template.format(
                comp=comp,
                extract_comp=extract_comp,
                scale=scale,
                realisation=realisation
            ),
            weight_vectors
        )

        return inverses, weight_vectors, singular_matrices_location, extract_comp


    @staticmethod
    def create_doubled_ILC_map(frequencies, scale, weight_vector, doubled_MW_wav_c_j):
        """
        Builds a single (H,W,F) array of wavelet coefficients,
        then does one broadcasted multiply+sum per scale.

        Parameters:
            frequencies (list): List of frequency bands.
            scale (int): Wavelet scale.
            weight_vector (np.ndarray): Weight vector for the ILC at given scale.
            doubled_MW_wav_c_j (dict): Dictionary of doubled MW wavelet coefficients.

        Returns:
            np.ndarray: The doubled ILC map for the given scale.
        """
        wav_coeffs = np.stack(
            [doubled_MW_wav_c_j[(f, scale)] for f in frequencies],
            axis=-1
        )  # shape = (H, W, F)

        # 2) do the ILC multiply & sum over frequency axis.
        doubled_map = np.sum(weight_vector[scale] * wav_coeffs, axis=-1)

        return doubled_map


    @staticmethod
    def trim_to_original(MW_Doubled_Map: np.ndarray, scale: int, realisation: int, method: str):
        """
        Trim a doubled‐resolution MW Pixel map back to its original resolution.

        Parameters:
            MW_Doubled_Map (ndarray): MW pixel map at doubled resolution.
            scale (int): wavelet scale (for filename only).
            realisation (int): realisation index (for filename only).

        Returns:
            mw_map_original (ndarray): trimmed pixel map at original resolution.
        """
        # compute trim indices once
        L2 = MW_Doubled_Map.shape[0]
        inner_v = (L2 + 1) // 2
        inner_h = 2 * inner_v - 1
        outer_mid = MW_Doubled_Map.shape[1] // 2
        start_col = outer_mid - (inner_h // 2)
        end_col = start_col + inner_h
        # numpy pathway
        # forward
        alm_doubled = s2fft.forward(MW_Doubled_Map, L=L2, method = method, spmd = False, reality = True)

        # trim
        trimmed_alm = alm_doubled[:inner_v, start_col:end_col]
        # inverse
        pix = s2fft.inverse(trimmed_alm, L=inner_v, method = method, spmd = False, reality = True)
        mw_map_original = pix[np.newaxis, ...]

        return mw_map_original
    

    @staticmethod
    def load_frequency_data(file_template: str, frequencies: list, scales: list, comp: str, 
                        realisation: int, lmax: int = 512, lam: float = 2.0):
        """
        Load NumPy arrays from dynamically generated file paths for each frequency and scale.
        
        Parameters:
            file_template (str): The template for the file names, with placeholders for frequency and scale.
            frequencies (list): A list of frequency names.
            scales (list): A list of scales.
            comp (str): The component name (e.g., "cfn").
            realisation (int): The realisation index.
            lmax (int): The maximum multipole.
            lam (float): The lambda parameter for the wavelet transform.

        Returns:
            dict: A dictionary where keys are tuples of (frequency, scale) and values are loaded NumPy arrays.
        """
        realisation = int(realisation)

        frequency_data = {}
        for frequency in frequencies:
            for scale in scales:
                filename = file_template.format(
                    comp=comp,
                    component=comp,          # ← minimal fix: provide {component}
                    frequency=frequency,
                    scale=scale,
                    realisation=realisation,
                    lmax=lmax,
                    lam=lam,
                )
                try:
                    frequency_data[(frequency, scale)] = np.load(filename)
                except Exception as e:
                    print(f"Error loading {filename} for frequency {frequency} and scale {scale}: {e}.")
        return frequency_data
    
    @staticmethod
    def visualize_MW_Pix_map(MW_Pix_Map, title, coord=["G"], unit=r"K", is_MW_alm=False):
        from map_tools import SamplingConverters
        """
        Visualize an MW pixel map by converting to HEALPix and drawing a mollview.

        Args:
            MW_Pix_Map (np.ndarray): MW pixel map (spatial) or MW alm if is_MW_alm=True.
            title (str): Plot title.
            coord (list): Coordinate transform for mollview, e.g., ["G"].
            unit (str): Unit string for the colorbar.
            is_MW_alm (bool): If True, MW_Pix_Map is already MW alm.
        """
        # NOTE: relies on s2fft, healpy as hp, matplotlib.pyplot as plt,
        # and mw_alm_2_hp_alm being importable (you already import utils*).
        if not is_MW_alm:
            # Detect L from map shape
            if MW_Pix_Map.ndim == 3:
                L_max = MW_Pix_Map.shape[1]
            else:
                L_max = MW_Pix_Map.shape[0]
            original_map_alm = s2fft.forward(MW_Pix_Map, L=L_max)
            print("MW alm shape:", original_map_alm.shape)
        else:
            original_map_alm = MW_Pix_Map
            L_max = original_map_alm.shape[0]

        original_map_hp_alm = SamplingConverters.mw_alm_2_hp_alm(original_map_alm, L_max - 1)
        nside = (L_max - 1) // 2
        original_hp_map = hp.alm2map(original_map_hp_alm, nside=nside)

        hp.mollview(
            original_hp_map,
            coord=coord,
            title=title,
            unit=unit,
            # min=..., max=...  # enable if you want fixed color range
        )
        plt.show()


    @staticmethod
    def synthesize_ILC_maps_generalised(trimmed_maps, realisation, file_templates, lmax, N_directions, 
                                        lam=2.0, comp=None, extract_comp=None,
                                        visualise = False, constraint=None):
        """
    Synthesizes full-sky ILC or cILC map from trimmed wavelet coefficient maps.

    Args:
        trimmed_maps (list): Trimmed wavelet maps across scales.
        realisation (str): realisation string (e.g., '0000').
        output_templates (dict): Output file templates.
        L_max (int): Maximum spherical harmonic degree.
        N_directions (int): Number of wavelet directions.
        component_name (str or None): Component name for constrained ILC (e.g., 'cmb').

    Returns:
        np.ndarray: Final synthesized ILC map.
    """ 
    
        # 1) normalise realisation formatting
        if isinstance(realisation, int):
            realisation_str = f"{realisation:04d}"
        else:
            realisation_str = str(realisation).zfill(4)
        
        # 2) use the passed parameters 
        file_tmpl = file_templates
        # lam is already a parameter to this function; ensure caller passes it
        
        # 3) load f_scal (template expects {realisation,lmax,lam})
        f_scal = np.load(
            file_tmpl["f_scal"].format(
                comp=comp, realisation=realisation_str, lmax=lmax, lam=lam
            )
        )
        
        # 4) build filters and synthesise
        L = lmax + 1
        filter_bank = filters.filters_directional_vectorised(L, N_directions, lam=lam)
        MW_Pix = s2wav.synthesis(
            trimmed_maps, L=L, f_scal=f_scal, filters=filter_bank, N=N_directions
        )
        
        # 5) (optional) visualise
        if visualise:
            try:
                prefix = "cILC" if extract_comp else "ILC"
                name = extract_comp.upper() if extract_comp else ""
                title = f"{prefix} {name} | r={realisation_str}, lmax={lmax}, N={N_directions}, λ={lam}".strip()
                SILCTools.visualize_MW_Pix_map(MW_Pix, title)
            except NameError:
                pass
                
        # 6) save to ilc_synth (template expects {realisation,lmax,lam})
        out_path = file_tmpl["ilc_synth"].format(
            realisation=realisation_str, lmax=lmax, lam=lam
        )
        np.save(out_path, MW_Pix)
        return MW_Pix


class ProduceSILC():
    """Perform  Scale-discretised, directional wavelet ILC (SILC)."""
    def __init__(self, 
                 ilc_components: list, 
                 frequencies: list, 
                 realisations: int, 
                 start_realisation: int,
                 lmax: int, 
                 N_directions: int = 1, 
                 lam: float = 2.0, 
                 synthesise = True, 
                 directory: str = "data/", 
                 method: str = "jax_cuda", 
                 overwrite: bool = False):
        """
        Parameters:
            ilc_components (list): List of components to produce an ILC for.
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            start_realisation (int): Starting realisation number for processing.
            lmax (int): Maximum multipole for the wavelet transform.
            synthesise (bool): Whether to synthesise the ILC map from the wavelet transforms.
            directory (str): Directory where data is stored / saved to.
            method (str): s2fft method to use for forward/inverse.
            overwrite (bool): Whether to overwrite existing files.
        """
        self.ilc_components = ilc_components
        self.frequencies = frequencies
        self.realisations = realisations
        self.start_realisation = start_realisation
        self.lmax = lmax
        self.N_directions = N_directions  
        self.lam = lam
        self.synthesise = synthesise
        self.directory = directory
        self.method = method
        self.overwrite = overwrite

        files = FileTemplates(self.directory)
        self.file_templates = files.file_templates

        # ilc_wavelet_paths, wavelet_map_directories, silc_output_dir, file_templates

        filter_sample = filters.filters_directional_vectorised(lmax+1, N_directions, lam = lam) # use length of filter to obtain n_scales
        self.scales = range(len(filter_sample[0]) + 1) 
    
    def process_wavelet_maps(self, save_intermediates: bool = False, visualise: bool = False):
        """
        Process wavelet maps for the specified components and frequencies.

        Args:
            save_intermediates (bool): Whether to save intermediate results.
            visualise (bool): Whether to visualise the results.

        Returns:
            None
        """
        ### UNFINISHED TASK: CHECK PROCESS WORKS FOR N_DIRECTIONS>1
        frequencies = self.frequencies
        scales = self.scales # assuming all components have the same scales
        method = self.method
        lmax = self.lmax
        L = self.lmax + 1 
        N_directions = self.N_directions
        lam = self.lam
        for comp in self.ilc_components:
            for realisation in range(self.realisations):
                realisation += self.start_realisation  # Adjust for starting realisation
                print(f"Processing realisation {realisation} for component {comp}...")
                path_test = self.file_templates['ilc_synth'].format(realisation=realisation, lmax=lmax, lam=lam)
                if os.path.exists(path_test) and self.overwrite == False:
                    print(f"File {path_test} already exists. Skipping to the next realisation.")
                    continue

                # Load original wavelet maps
                wavelet_template = self.file_templates['wavelet_coeffs']
                original_wavelet_c_j = SILCTools.load_frequency_data(wavelet_template, frequencies, scales, comp, realisation, lmax, lam)
                # Double the resolution of the wavelet maps
                doubled_MW_wav_c_j = {}
                for i in frequencies:
                    for j in scales:
                        doubled_map_path = self.file_templates['doubled_maps'].format(frequency=i, scale=j, realisation=realisation, lmax=lmax, lam=lam)
                        if os.path.exists(doubled_map_path) and self.overwrite == False:
                            doubled_MW_wav_c_j[(i, j)] = np.load(doubled_map_path)
                            print(f"Doubled map for frequency {i}, scale {j} already exists. Skipping doubling.")
                        else:
                            doubled_MW_wav_c_j[(i, j)] = SILCTools.Single_Map_doubleworker(original_wavelet_c_j[(i, j)], method)
                            if save_intermediates:
                                np.save(doubled_map_path, doubled_MW_wav_c_j[(i, j)])
                                print(f"Saved doubled map for frequency {i}, scale {j}.")
                # Calculate the covariance matrices for each scale
                R_covariance = {}
                for scale in scales:
                    covariance_path = self.file_templates['covariance_matrices'].format(frequencies='_'.join(frequencies), scale=scale, realisation=realisation, lmax=lmax, lam=lam)
                    if os.path.exists(covariance_path) and self.overwrite == False:
                        print(f"Covariance matrix for scale {scale}, realisation {realisation} already exists. Skipping covariance calculation.")
                        R_covariance[scale] = np.load(covariance_path)
                    else:
                        R_covariance[scale] = SILCTools.calculate_covariance_matrix(frequencies, doubled_MW_wav_c_j, scale, realisation, method)
                        if save_intermediates:
                            np.save(covariance_path, R_covariance[scale])
                            print(f"Saved covariance matrix for scale {scale}, realisation {realisation}.")
                # Calculate the weight vectors for each scale
                weight_vectors = []
                for scale in scales:
                    weight_vector_path = self.file_templates['weight_vector_matrices'].format(scale=scale, realisation=realisation, lmax=lmax, lam=lam)
                    if os.path.exists(weight_vector_path) and self.overwrite == False:
                        print(f"Weight vector for scale {scale}, realisation {realisation} already exists. Skipping weight vector calculation.")
                        weight_vectors.append(np.load(weight_vector_path))
                    else:
                        weight_vector_scale = SILCTools.compute_weight_vector(R_covariance[scale], scale, realisation)
                        weight_vectors.append(weight_vector_scale)
                        np.save(weight_vector_path, weight_vector_scale) # we will save weight vectors even if save_intermediates is False
                        print(f"Saved weight vector for scale {scale}, realisation {realisation}.")
                # Create the doubled resolution ILC map for each scale
                doubled_maps = []
                for scale in scales:
                    ilc_doubled_map_path = self.file_templates['ilc_doubled_maps'].format(scale=scale, realisation=realisation, lmax=lmax, lam=lam)
                    if os.path.exists(ilc_doubled_map_path) and self.overwrite == False:
                        print(f"ILC doubled map for scale {scale}, realisation {realisation} already exists. Skipping ILC map creation.")
                        doubled_maps.append(np.load(ilc_doubled_map_path))
                    else:
                        doubled_map_scale = SILCTools.create_doubled_ILC_map(frequencies, scale, weight_vectors, doubled_MW_wav_c_j)
                        doubled_maps.append(doubled_map_scale)
                        if save_intermediates:
                            np.save(ilc_doubled_map_path, doubled_map_scale)
                            print(f"Saved ILC doubled map for scale {scale}, realisation {realisation}.")

                # Trim the doubled resolution ILC map back to its original resolution
                trimmed_maps = []
                for i, scale in enumerate(scales):
                    ilc_trimmed_map_path = self.file_templates['ilc_trimmed_maps'].format(scale=scale, realisation=realisation, lmax=lmax, lam=lam)
                    if os.path.exists(ilc_trimmed_map_path) and self.overwrite == False:
                        print(f"ILC trimmed map for scale {scale}, realisation {realisation} already exists. Skipping trimming.")
                        trimmed_maps.append(np.load(ilc_trimmed_map_path))
                    else:
                        trimmed_map = SILCTools.trim_to_original(doubled_maps[i], scale, realisation, method)
                        trimmed_maps.append(trimmed_map)
                        np.save(ilc_trimmed_map_path, trimmed_map)
                        print(f"Saved trimmed map for scale {scale}, realisation {realisation}.")

                # Synthesise the ILC map from the trimmed wavelet maps
                if self.synthesise:
                    ilc_synth_map_path = self.file_templates['ilc_synth'].format(realisation=realisation, lmax=lmax, lam=lam)
                    if os.path.exists(ilc_synth_map_path) and self.overwrite == False:
                        print(f"ILC synthesised map for realisation {realisation} already exists. Skipping synthesis.")
                    else:
                        mw_pix = MWTools.inverse_wavelet_transform(trimmed_maps, L, N_directions, lam)
                        np.save(ilc_synth_map_path, mw_pix)
                        print(f"Saved synthesised ILC map for realisation {realisation}.")
                    
        return None


    def ILC_wav_coeff_maps_MP(file_template, frequencies, scales, realisations,
        output_templates, L_max, N_directions,
        comp,                                     # input component maps
        constraint=False, F=None, 
        extract_comp=None,  # component to extract
        reference_vectors=None):

        # --- Prepare constraint vector / tags (unchanged) ---
        if constraint:

            if F is None or extract_comp is None:
                raise ValueError("Must provide F and extract_comp if constraint=True")
            target_names, extract_comp = normalize_targets(extract_comp)
            if len(target_names) == 0:
                raise ValueError("Provide at least one target component name when constraint=True")
            f = SILCTools.find_f_from_extract_comp(F, target_names, reference_vectors, allow_sign_flip=False)
        else:
            if isinstance(extract_comp, (list, tuple, np.ndarray)):
                raise ValueError("For unconstrained ILC, pass a single extract_comp (e.g., 'cmb').")
            _, extract_comp = normalize_targets(extract_comp)
            f = None  # not used in unconstrained mode
        for realisation in realisations:
            realisation_str = str(realisation).zfill(4)

            print(f"Processing realisation {realisation_str} for component {comp}")
            # 1) Load original wavelet maps
            original_wavelet_c_j = SILCTools.load_frequency_data(

                file_template=file_template,
                frequencies=frequencies,
                scales=scales,
                comp=comp,
                realisation=realisation,   # int
                lmax=L_max,
                lam=2.0,
            )
            # 2) Double resolution and save (single call)
            t0 = time.perf_counter()

            SILCTools.double_wavelet_maps(
                original_wavelet_c_j,
                frequencies,
                scales,
                realisation,
                method="jax_cuda",  # or "jax"
                path_template=output_templates['doubled_maps'],
                component=comp,
                lmax=L_max,
                lam=2.0,
            )
            print(f'Doubled and saved wavelet maps in {time.perf_counter() - t0:.2f} seconds')

            # 3) Load doubled resolution wavelet maps from disk
            doubled_MW_wav_c_j = SILCTools.load_frequency_data(
                file_template=output_templates['doubled_maps'],
                frequencies=frequencies,
                scales=scales,
                comp=comp,
                realisation=realisation,   # int
                lmax=L_max,
                lam=2.0,
            )

            # 4) Compute covariance matrices (MP: one job per scale)
            t0 = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        SILCTools.calculate_covariance_matrix,
                        frequencies,
                        doubled_MW_wav_c_j,
                        scale,
                        realisation_str,
                        comp,
                        output_templates['covariance_matrices']
                    )
                    for scale in scales
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
            print(f'Calculated covariance matrices in {time.perf_counter() - t0:.2f} seconds')

            # 5) Load covariance matrices (per scale)
            F_str = '_'.join(frequencies)
            R_covariance = [
                np.load(
                    output_templates['covariance_matrices'].format(
                        component=comp,
                        frequencies=F_str,
                        scale=scale,
                        realisation=realisation_str
                    )
                )
                for scale in scales
            ]

            # 6) Compute weight vectors (MP: one job per scale)
            t0 = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        SILCTools.compute_weights_generalised,
                        R_covariance[idx],
                        scale,
                        realisation_str,
                        output_templates['weight_vector_matrices'],
                        comp,
                        extract_comp,
                        constraint,
                        F,
                        f,
                        reference_vectors
                    )
                    for idx, scale in enumerate(scales)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
            print(f'Calculated weight vector matrices in {time.perf_counter() - t0:.2f} seconds')

            # Load weights (in order)
            weight_vector_load = []
            W_for_final_check = None
            name = f"cilc_{extract_comp}" if constraint else "weight_vector"
            for scale in scales:
                weight_vector_path = output_templates['weight_vector_matrices'].format(
                    component=comp,
                    extract_comp=extract_comp,
                    type=name,
                    scale=scale,
                    realisation=realisation_str
                )
                W = np.load(weight_vector_path)
                if W.ndim == 2 and 1 in W.shape:
                    W = W.reshape(-1)
                weight_vector_load.append(W)
                W_for_final_check = W

            # 7) Create doubled ILC maps (serial; you noted MP here is slower)
            t0 = time.perf_counter()
            doubled_maps = []
            for i, scale in enumerate(scales):
                map_ = SILCTools.create_doubled_ILC_map(
                    frequencies,
                    scale,
                    weight_vector_load[i],
                    doubled_MW_wav_c_j,
                    realisation_str,
                    component=comp,
                    constraint=constraint,
                    extract_comp=extract_comp
                )
                doubled_maps.append(map_)
                np.save(
                    output_templates['ilc_maps'].format(
                        component=comp,
                        extract_comp=extract_comp,
                        scale=scale,
                        realisation=realisation_str
                    ),
                    map_
                )
            print(f'Created ILC maps in {time.perf_counter() - t0:.2f} seconds')

            # 8) Trim to original resolution (MP: one job per scale)
            t0 = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        SILCTools.trim_to_original,
                        doubled_maps[i],
                        scales[i],
                        realisation_str,
                        comp,
                        extract_comp,
                        output_templates['trimmed_maps']
                    )
                    for i in range(len(scales))
                ]
                # keep order aligned with `scales`
                trimmed_maps = [None] * len(scales)
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    # expect either (scale, trimmed_array) or just array; handle both
                    if isinstance(res, tuple) and len(res) == 2:
                        sc, trimmed = res
                        idx = scales.index(sc)
                        trimmed_maps[idx] = trimmed
                    else:
                        # if only array returned, append later by position
                        pass
                # fill any Nones by loading from disk (if your trim saves to disk)
                for idx, tm in enumerate(trimmed_maps):
                    if tm is None:
                        trimmed_maps[idx] = np.load(
                            output_templates['trimmed_maps'].format(
                                component=comp,
                                extract_comp=extract_comp,
                                scale=scales[idx],
                                realisation=realisation_str
                            )
                        )
            print(f'Trimmed maps to original resolution in {time.perf_counter() - t0:.2f} seconds')

            # 9) Synthesize final map (serial)
            synthesized_map = SILCTools.synthesize_ILC_maps_generalised(
                trimmed_maps,
                realisation_str,
                output_templates,
                L_max,
                N_directions,
                extract_comp=extract_comp,
                component=comp,
                constraint=constraint
            )
            synthesized_maps.append(synthesized_map)

            # 10) One-time verification per realisation
            if constraint and (W_for_final_check is not None):
                _check_against_F(W_for_final_check, F, f)

        return synthesized_maps



# ilc_producer = ProduceSILC(ilc_components = ["cfn"], frequencies = ["030", "070"], realisations=1, lmax=1024, directory="/Scratch/matthew/data/", synthesise=True, method="jax_cuda")
# ilc_producer.process_wavelet_maps(save_intermediates=True, visualise=False)