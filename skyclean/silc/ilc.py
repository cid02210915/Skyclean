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

from .map_tools import SamplingConverters
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
        # If there is a leading direction axis (N_directions==1), squeeze it
        if mw_map.ndim == 3 and mw_map.shape[0] == 1:
            mw_map = mw_map[0]

        # L is the number of rows; width must be 2L-1
        L = mw_map.shape[0]
        W = mw_map.shape[1]
        assert W == 2*L - 1, f"MW map has wrong shape {mw_map.shape}; expected (L, 2L-1)"

        # forward with the correct L
        alm = s2fft.forward(mw_map, L=L, method=method, spmd=False, reality=True)

        # double: L2 = 2L-1, width = 2*L2-1
        L2 = 2*L - 1
        W2 = 2*L2 - 1

        padded = np.zeros((L2, W2), dtype=alm.dtype)
        mid_out = W2 // 2
        mid_in  = W  // 2
        start   = mid_out - mid_in
        padded[:L, start:start+W] = alm

        x2 = np.real(s2fft.inverse(padded, L=L2, method=method, spmd=False, reality=True))
        return x2

    @staticmethod
    def smoothed_covariance(MW_Map1: np.ndarray, MW_Map2: np.ndarray, method: str):
        #print("smoothed_covariance", flush = True)
        """
        Parameters:
            MW_Map1, MW_Map2 (np.ndarray): same‐shape complex np.ndarray wavelet maps

        Returns:
            R_covariance_map: real‐valued covariance map as np.ndarray
        """

        # --- local helper: force MW sampling (nphi == 2L-1) by tile/truncate ---
        def _ensure_mw_sampling(a: np.ndarray) -> np.ndarray:
            L = a.shape[0]
            exp = 2 * L - 1
            nphi = a.shape[1] if a.ndim >= 2 else 1
            if nphi == exp:
                return a
            if nphi > exp:            # too many columns -> truncate
                return a[:, :exp]
            # too few columns -> periodic tile then crop
            reps = (exp + nphi - 1) // nphi
            return np.tile(a, reps)[:, :exp]

        smoothing_L = int(MW_Map1.shape[0])

        # 1) Coveriance of pixel 
        map1 = np.real(MW_Map1)
        map2 = np.real(MW_Map2)

        # coerce both maps to MW sampling for the given L
        map1 = _ensure_mw_sampling(map1)
        map2 = _ensure_mw_sampling(map2)

        Rpix = np.multiply(map1, map2) + 0.j
 
        #print(f"([smoothed_covariance] Rpix.shape={Rpix.shape}, L={smoothing_L}, method={method})", flush=True)

        # --- guard: s2fft.forward is not happy for L < 2; just return pixel product ---
        if smoothing_L < 2 or Rpix.shape[1] != (2 * smoothing_L - 1):
            # minimal, safe fallback (no harmonic smoothing possible)
            return np.real(Rpix)

        # 2) forward (smooth in harmonic space for efficiency)
        Ralm = s2fft.forward(Rpix, L=smoothing_L, method=method, spmd=False, reality=True)

        # 3) Gaussian beam
        nsamp = 1200.0
        lmax = Ralm.shape[0]
        npix = hp.nside2npix(1 << (int(0.5 * lmax) - 1).bit_length())
        scale_fwhm = 4.0 * math.sqrt(nsamp / npix)
        gauss_beam = hp.gauss_beam(scale_fwhm, lmax=lmax - 1)

        # 4) convolve
        convolved = Ralm * gauss_beam[:, None]

        # 5) inverse
        Rmap = np.real(s2fft.inverse(convolved, L=smoothing_L, method=method, spmd=False, reality=True))

        return Rmap


    @staticmethod 
    def compute_covariance(task):
        #print("compute_covariance", flush = True)
        """
        Computes the covariance between two frequency maps at a given scale.

        Args:
        task (tuple): A tuple containing (i, fq, frequencies, scale, doubled_MW_wav_c_j).

        Returns:
        tuple: A tuple containing indices i, fq and the computed covariance matrix.
        """
        i, fq, frequencies, scale, doubled_MW_wav_c_j, method  = task
        key_i = (frequencies[i], scale)
        key_fq = (frequencies[fq], scale)
        if key_i not in doubled_MW_wav_c_j or key_fq not in doubled_MW_wav_c_j:
            raise KeyError(f"Missing data for keys {key_i} or {key_fq}.")
        return i, fq, SILCTools.smoothed_covariance(doubled_MW_wav_c_j[key_i], doubled_MW_wav_c_j[key_fq], method)


    @staticmethod
    def calculate_covariance_matrix(frequencies: list, doubled_MW_wav_c_j: dict, scale: int,
                                    realisation: int, method: str, path_template: str, *,
                                    component: str = "cfn", lmax: int = 64, lam: float | str = 2.0,):
        
        #print("calculate_covariance_matrix", flush = True)

        """
        Calculates the covariance matrices for given frequencies and saves them to disk,
        accommodating any size of the input data arrays.
        """
        #print('calculate_covariance_matrix', flush=True)

        if not frequencies:
            raise ValueError("Frequency list is empty.")

        # --- minimal normalization to fit current pipeline ---
        norm_freqs = [str(f).zfill(3) for f in frequencies]
        scale_i = int(scale)

        # Size from a sample
        sample_data = doubled_MW_wav_c_j[(norm_freqs[0], scale_i)]
        n_rows, n_cols = sample_data.shape

        total_frequency = len(norm_freqs)
        full_array = np.zeros((total_frequency, total_frequency, n_rows, n_cols))

        # Build work items (upper triangle)
        tasks = [(i, fq, norm_freqs, scale_i, doubled_MW_wav_c_j, method)
                 for i in range(total_frequency) for fq in range(i, total_frequency)]

        # ---------- MINIMAL FIX HERE ----------
        # JAX/JAX+CUDA is NOT fork-safe -> avoid ProcessPool in that case.
        if method != "numpy":
            # serial fallback for jax/jax_cuda to avoid BrokenProcessPool
            for t in tasks:
                i, fq, cov = SILCTools.compute_covariance(t)
                full_array[i, fq] = cov
        else:
            # CPU/NumPy path can safely use processes
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx = mp.get_context("spawn")  # more robust than fork for native libs
            with ProcessPoolExecutor(mp_context=ctx) as executor:
                futures = [executor.submit(SILCTools.compute_covariance, t) for t in tasks]
                for fut in as_completed(futures):
                    i, fq, covariance_matrix = fut.result()
                    full_array[i, fq] = covariance_matrix

        # --------------------------------------
        # Fill symmetric part
        for l1 in range(1, total_frequency):
            for l2 in range(l1):
                full_array[l1, l2] = full_array[l2, l1]

        # Save using normalized frequency tags
        f_str = "_".join(norm_freqs)
        lam_str = str(lam)
        save_path = path_template.format(
            component=component,
            frequencies=f_str,
            scale=scale_i,
            realisation=int(realisation),
            lmax=int(lmax),
            lam=lam_str,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, full_array)

        return full_array

    '''
    @staticmethod
    def double_wavelet_maps(
        original_wavelet_c_j: dict,
        frequencies: list,
        scales: list,
        realisation: int,
        method: str = "jax",
        *,
        path_template: str | None = None,
        component: str | None = None,
        lmax: int | None = None,
        lam: float | str | None = None,
    ):
        """
        Doubles the resolution of wavelet maps and (optionally) saves them.

        If `path_template` is provided, files are saved with fields:
        {component}, {comp}, {extract_comp}, {frequency}, {scale}, {realisation:04d}, {lmax}, {lam}, {frequencies}
        """
        # Allow passing a template in the 'method' parameter by mistake (compat shim)
        KNOWN = {"jax_cuda", "jax", "numpy"}
        if path_template is None and isinstance(method, str) and method not in KNOWN:
            if (os.sep in method) or ("{" in method) or str(method).endswith(".npy"):
                path_template = method
                method = "jax"
            else:
                raise ValueError(f"Method {method} not recognised.")

        # ---- compute doubled maps ----
        doubled_MW_wav_c_j = {}
        for f in frequencies:
            for j in scales:
                arr = SILCTools.Single_Map_doubleworker(original_wavelet_c_j[(f, j)], method)
                doubled_MW_wav_c_j[(f, j)] = arr  # may be JAX array

        # ---- optional save to disk ----
        if path_template is not None:
            realisation_int = int(realisation)
            def freq_label(ff): return ff if isinstance(ff, str) else f"{int(ff):03d}"
            comp_val = component if component is not None else "cmb"
            lmax_val = int(lmax) if lmax is not None else 512
            lam_val  = str(lam) if lam is not None else "2.0"
            F_str = "_".join(freq_label(ff) for ff in frequencies)

            for f in frequencies:
                for j in scales:
                    out_path = path_template.format(
                        component=comp_val, comp=comp_val, extract_comp=comp_val,
                        frequency=freq_label(f),
                        scale=int(j),
                        realisation=realisation_int,      # <-- int so template {:04d} works
                        lmax=lmax_val,
                        lam=lam_val,
                        frequencies=F_str,
                    )
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, np.asarray(doubled_MW_wav_c_j[(f, j)]))

        return doubled_MW_wav_c_j
    '''

    # inside the same module

    @staticmethod
    def save_doubled_wavelet_map(args):
        # match the tuple structure you build in tasks:
        # (arr, freq, scale, realisation, comp, path_template, lmax, lam, method)
        arr, freq, scale, realisation, comp, path_template, lmax, lam, method = args

        # normalize
        freq_tag = freq if isinstance(freq, str) else f"{int(freq):03d}"
        save_path = path_template.format(
            comp=comp,                          # use {comp}, not {component}
            frequency=freq_tag,
            scale=int(scale),
            realisation=int(realisation),       # {:04d} handled by template
            lmax=int(lmax),
            lam=str(lam),
        )

        doubled_map = SILCTools.Single_Map_doubleworker(arr, method)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, np.asarray(doubled_map))
        return save_path


    @staticmethod
    def double_and_save_wavelet_maps(original_wavelet_c_j, frequencies, scales, realisation, 
                                     component, path_template, *, lmax=64, lam="2.0", method="jax"):
        
        """Minimal fix: compute + save doubled maps serially (no MP)."""
        for f in frequencies:
            for s in scales:
                arr = original_wavelet_c_j[(f, s)]
                doubled = SILCTools.Single_Map_doubleworker(arr, method)

                freq_tag = f if isinstance(f, str) else f"{int(f):03d}"
                out_path = path_template.format(
                    component=component,
                    frequency=freq_tag,
                    scale=int(s),
                    realisation=int(realisation),   # template pads via {:04d}
                    lmax=int(lmax),
                    lam=str(lam),
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, np.asarray(doubled))
                #print("saved:", out_path)


    @staticmethod
    def compute_weight_vector(R: np.ndarray, scale: int, realisation: int):
        #print("compute_weight_vector", flush = True)
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
    
    
    def compute_weights_generalised(R, scale, realisation, weight_vector_matrix_template, comp, L_max, 
                                    extract_comp, *, constraint=False, F=None, f=None, 
                                    reference_vectors=None, lam="2.0"):
        #print("compute_weights_generalised", flush=True)
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
    
        # Common filename fields (provide both 'component' and 'comp'; pass ints so {:04d} works)
        name = f"cilc_{extract_comp}" if constraint else "weight_vector"
        fmt = dict(
            component=comp,
            comp=comp,
            type=name,
            extract_comp=extract_comp,
            scale=int(scale),
            realisation=int(realisation),
            lmax=int(L_max-1),
            lam=str(lam),
        )
    
        # --- per-pixel (or single) solve ---
        for i in range(dim1):
            for j in range(dim2):
                try:
                    R_ij = R_Pix[i, j] if R.ndim == 4 else R_Pix
    
                    # robust inverse: try inv, then ridge, else pinv
                    try:
                        R_inv = np.linalg.inv(R_ij)
                    except np.linalg.LinAlgError:
                        eps = 1e-8
                        try:
                            R_inv = np.linalg.inv(R_ij + eps * np.eye(R_ij.shape[0]))
                        except np.linalg.LinAlgError:
                            R_inv = np.linalg.pinv(R_ij)
    
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
                    singular_matrix_path = weight_vector_matrix_template.format(**fmt).replace(
                        ".npy", f"_singular_{i}_{j}.npy"
                    )
                    np.save(singular_matrix_path, R_Pix[i, j] if R.ndim == 4 else R_Pix)
                    if R.ndim == 4:
                        weight_vectors[i, j] = np.zeros(N_freq)
                    else:
                        weight_vectors = np.zeros(N_freq)
                    continue
                
        # save final weight vector matrix
        np.save(weight_vector_matrix_template.format(**fmt), weight_vectors)
        return inverses, weight_vectors, singular_matrices_location, extract_comp
    
    @staticmethod
    def compute_ILC_for_pixel(i, j, frequencies, scale, weight_vector_load, doubled_MW_wav_c_j):
        """
        w_scale: 1D weight vector for THIS scale, shape (F,)
        """
        pix_vector = np.array([
        doubled_MW_wav_c_j[(frequencies[k], scale)][i, j] for k in range(len(frequencies))])
        return np.dot(weight_vector_load[i, j], pix_vector)


    @staticmethod
    def create_doubled_ILC_map(frequencies, scale, weight_vector_load, doubled_MW_wav_c_j, *_, **__):
        """
        weight_vector_load is expected to be the 1D weight vector for THIS scale (shape (F,))
        """
        # tiny sanity (optional)
        size = doubled_MW_wav_c_j[(frequencies[0], scale)].shape
        doubled_map = np.zeros((size[0], size[1]))

        for i in range(doubled_map.shape[0]):
            for j in range(doubled_map.shape[1]):
                doubled_map[i, j] = SILCTools.compute_ILC_for_pixel(i, j, frequencies, scale, weight_vector_load, doubled_MW_wav_c_j)

        return doubled_map
        
    '''
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
    '''


    @staticmethod
    def trim_to_original(MW_Doubled_Map: np.ndarray, scale: int, realisation: int, method: str, *, 
                         path_template:str, component: str, extract_comp: str, lmax:int, lam: str = 2.0):

        #print("trim_to_original", flush=True)
        """
        Trim a doubled‐resolution MW Pixel map back to its original resolution.

        Parameters:
            MW_Doubled_Map (ndarray): MW pixel map at doubled resolution (L2, 2L2-1).
            scale (int): wavelet scale (for filename only).
            realisation (int): realisation index (for filename only).

        Returns:
            mw_map_original (ndarray or None): trimmed pixel map at original resolution. 
        """
        
        # --- minimal debug & guards ---
        #print(f"[DEBUG] scale={scale}, realisation={realisation}, shape={MW_Doubled_Map.shape}", flush=True)
        if MW_Doubled_Map.ndim != 2:
            raise ValueError(f"[DEBUG] Not 2D: got {MW_Doubled_Map.shape}")

        L2, W2 = MW_Doubled_Map.shape
        if W2 != 2 * L2 - 1:
            raise ValueError(f"[DEBUG] Not MW grid: shape={L2}x{W2}, expected {L2}x{2*L2-1}")
        #print('L2:',L2)
     
        L2, W2 = MW_Doubled_Map.shape
        #print(f"[trim pid={os.getpid()}] scale={scale} r={realisation} L2={L2} shape={MW_Doubled_Map.shape}",flush=True)
        # compute trim indices once
        inner_v = (L2 + 1) // 2           # original bandlimit L (rows) after trimming
        inner_h = 2 * inner_v - 1         # original width 2L-1
        outer_mid = W2 // 2
        start_col = outer_mid - (inner_h // 2)
        end_col = start_col + inner_h

        # forward (numpy path)
        alm_doubled = s2fft.forward(MW_Doubled_Map, L=L2, method='numpy', spmd=False, reality=True)
        # trim in harmonic space
        trimmed_alm = alm_doubled[:inner_v, start_col:end_col]

        # inverse back to pixels
        pix = s2fft.inverse(trimmed_alm, L=inner_v, method='numpy', spmd=False, reality=True)
        mw_map_original = pix[np.newaxis, ...]

        # ---- MIN SAVE: only if a template and tags are provided ----
        if path_template and component and extract_comp:
            lmax = inner_v - 1
            lam_str = str(lam)
            save_path = path_template.format(
                component=component,
                extract_comp=extract_comp,
                scale=scale,
                realisation=int(realisation),
                lmax=int(lmax),
                lam=lam_str,
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, mw_map_original)
            #print(f"[SAVE] Trimmed map -> {save_path}")

        return int(scale), mw_map_original


    @staticmethod
    def load_frequency_data(file_template: str, frequencies: list, scales: list, comp: str, lmax: int, *,
                        realisation: int, lam: float = 2.0):
        #print("load_frequency_data", flush = True)
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
                    component=comp,        
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
    def synthesize_ILC_maps_generalised(
        trimmed_maps, realisation, file_templates, lmax, N_directions,
        lam=2.0, component=None, extract_comp=None, 
        visualise=False, constraint=None, frequencies=None,
    ):
        #print("synthesize_ILC_maps_generalised", flush=True)
        """
        Synthesizes full-sky ILC or cILC map from trimmed wavelet coefficient maps.
        ...
        """

        # 1) normalise realisation formatting
        if isinstance(realisation, int):
            realisation_str = f"{realisation:04d}"
        else:
            realisation_str = str(realisation).zfill(4)

        # 2) use the passed parameters
        file_tmpl = file_templates  # lam already passed

        # --- build frequencies tag for filename ---
        if isinstance(frequencies, (list, tuple)):
            freq_tag = "_".join(map(str, frequencies))
            freq0 = str(frequencies[0])
        elif frequencies is None:
            freq_tag, freq0 = "unknown", None
        else:
            freq_tag = str(frequencies)
            freq0 = freq_tag.split("_")[0]

        # 3) load f_scal (template expects {realisation,lmax,lam})
        #    accept either 'f_scal' or 'scaling_coeffs' in templates
        f_key = "f_scal" if "f_scal" in file_tmpl else "scaling_coeffs"
        f_t = file_tmpl[f_key]
        if "{frequency}" in f_t:
            if not freq0:
                raise ValueError("Scaling-coeffs template requires {frequency} but none was provided.")
            f_scal = np.load(
                f_t.format(
                    comp=component,               
                    frequency=freq0,
                    realisation=int(realisation_str),
                    lmax=int(lmax),
                    lam=str(lam),
                )
            )
        else:
            f_scal = np.load(
                f_t.format(
                    comp=component,
                    realisation=int(realisation_str),
                    lmax=int(lmax),
                    lam=str(lam),
                )
            )

        # 4) build filters and synthesise
        L = int(lmax) + 1
        filter_bank = filters.filters_directional_vectorised(L, N_directions, lam=float(lam))
        MW_Pix = s2wav.synthesis(trimmed_maps, L=L, f_scal=f_scal, filters=filter_bank, N=N_directions)

        # 5) (optional) visualise
        if visualise:
            try:
                prefix = "cILC" if extract_comp else "ILC"
                name = extract_comp.upper() if extract_comp else ""
                title = f"{prefix} {name} | r={realisation_str}, lmax={int(lmax)}, N={N_directions}, λ={lam}".strip()
                SILCTools.visualize_MW_Pix_map(MW_Pix, title)
            except NameError:
                pass

        # 6) save to ilc_synth (template expects {extract_comp},{component},{frequencies},{realisation},{lmax},{lam})
        out_path = file_tmpl["ilc_synth"].format(
            extract_comp=extract_comp,
            component=component,
            frequencies=freq_tag,              # <- FIX: use defined tag, not 'freq_'
            realisation=int(realisation_str),
            lmax=int(lmax),
            lam=str(lam),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
                                
                # Calculate the covariance matrices for each scale  (MIN-FIX)
                R_covariance = {}
                
                norm_freqs = [str(f).zfill(3) for f in frequencies]
                f_str = "_".join(norm_freqs)
                lam_str = str(lam)
                
                for scale in scales:
                    covariance_path = self.file_templates['covariance_matrices'].format(
                        component=comp,
                        frequencies=f_str,
                        scale=int(scale),
                        realisation=int(realisation),   # template pads -> r0000
                        lmax=int(lmax),
                        lam=lam_str,
                    )
                    if os.path.exists(covariance_path) and not self.overwrite:
                        R_covariance[scale] = np.load(covariance_path)
                    else:
                        R_covariance[scale] = SILCTools.calculate_covariance_matrix(
                            frequencies=norm_freqs,
                            doubled_MW_wav_c_j=doubled_MW_wav_c_j,
                            scale=int(scale),
                            realisation=int(realisation),
                            method=method,
                        )
                        if save_intermediates:
                            os.makedirs(os.path.dirname(covariance_path), exist_ok=True)
                            np.save(covariance_path, R_covariance[scale])
                
                
                # Calculate the weight vectors for each scale
                weight_vectors = []
                for scale in scales:
                    weight_vector_path = self.file_templates['weight_vector_matrices'].format(scale=scale, realisation=int(realisation), lmax=lmax, lam=lam)
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
                        doubled_map_scale = SILCTools.create_doubled_ILC_map(frequencies, scale, weight_vectors, doubled_MW_wav_c_j, realisation)
                        doubled_maps.append(doubled_map_scale)
                        if save_intermediates:
                            np.save(ilc_doubled_map_path, doubled_map_scale)
                            print(f"Saved ILC doubled map for scale {scale}, realisation {realisation}.")

                # Trim the doubled resolution ILC map back to its original resolution
                trimmed_maps = []
                for i, scale in enumerate(scales):
                    ilc_trimmed_map_path = self.file_templates['trimmed_maps'].format(scale=scale, realisation=int(realisation), lmax=lmax, lam=lam)
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
                    ilc_synth_map_path = self.file_templates['ilc_synth'].format(realisation=int(realisation), lmax=lmax, lam=lam)
                    if os.path.exists(ilc_synth_map_path) and self.overwrite == False:
                        print(f"ILC synthesised map for realisation {realisation} already exists. Skipping synthesis.")
                    else:
                        mw_pix = MWTools.inverse_wavelet_transform(trimmed_maps, L, N_directions, lam)
                        np.save(ilc_synth_map_path, mw_pix)
                        print(f"Saved synthesised ILC map for realisation {realisation}.")
        return None


    def ILC_wav_coeff_maps_MP(file_template, frequencies, scales, realisations,
                             output_templates, L_max, N_directions,
                             comp,                            
                             constraint=False, F=None, 
                             extract_comp=None,
                             reference_vectors=None):

        lmax = L_max - 1  
        realisations = [int(r) for r in realisations]
        def _check_against_F(W, F, f, tol=1e-6):
            W = np.asarray(W)
            if W.ndim == 2 and 1 in W.shape:   # (1,Nf) or (Nf,1) -> (Nf,)
                W = W.reshape(-1)
            resp = np.tensordot(W, F, axes=([-1], [0]))  # (..., N_comp)
            ok = np.allclose(resp, f, atol=tol, rtol=0.0)
            print("FINAL CHECK  F^T w == f  ->", ok)
            if not ok:
                print("max |F^T w - f| =", float(np.max(np.abs(resp - f))))
            return ok
        
        timings = {   # store timings per step
            "double_and_save": [],
            "covariance": [],
            "weights": [],
            "create_ilc_maps": [],
            "trim": [],
        }
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
            
        synthesized_map = []
        
        for realisation in realisations:
            realisation_str = str(realisation).zfill(4)
            print(f"Processing realisation {realisation_str} for component {comp}")

            # 1) Load original wavelet maps
            original_wavelet_c_j = SILCTools.load_frequency_data(
                file_template=file_template,
                frequencies=frequencies,
                scales=scales,
                comp=comp,
                realisation=int(realisation),   # int
                lmax=L_max-1,
                lam=2.0,
            )

            # 2) Double resolution and save (single call)
            t0 = time.perf_counter()
            SILCTools.double_and_save_wavelet_maps(
                original_wavelet_c_j,
                frequencies,
                scales,
                realisation,
                method="jax_cuda",  # or "jax"
                path_template=output_templates['doubled_maps'],
                component=comp,
                lmax=L_max-1,
                lam=2.0,
            )
            dt = time.perf_counter() - t0
            print(f'Doubled and saved wavelet maps in {dt:.2f} seconds')
            timings["double_and_save"].append(dt)

            # 3) Load doubled resolution wavelet maps from disk
            doubled_MW_wav_c_j = SILCTools.load_frequency_data(
                file_template=output_templates['doubled_maps'],
                frequencies=frequencies,
                scales=scales,
                comp=comp,
                realisation=int(realisation),   # int
                lmax=L_max-1,
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
                        realisation=int(realisation),
                        method="numpy",                             # now it’s controlled outside
                        path_template=output_templates["covariance_matrices"],
                        component=comp,
                        lmax=L_max-1,
                        lam=2.0,
                    )
                    for scale in scales
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
            dt = time.perf_counter() - t0
            print(f'Calculated covariance matrices in {dt:.2f} seconds')
            timings["covariance"].append(dt)
            # 5) Load covariance matrices (per scale)
            F_str = '_'.join(frequencies)
            R_covariance = [
                np.load(
                    output_templates['covariance_matrices'].format(
                        component=comp,
                        frequencies=F_str,
                        scale=scale,
                        realisation=int(realisation),
                        lmax=L_max-1,
                        lam = '2.0', 
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
                        int(realisation),
                        output_templates['weight_vector_matrices'],
                        comp,
                        L_max, 
                        extract_comp,
                        constraint=constraint,
                        F=F,
                        f=f,
                        reference_vectors=reference_vectors,
                        lam='2.0', 
                    )
                    for idx, scale in enumerate(scales)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
            dt = time.perf_counter() - t0
            print(f'Calculated weight vector matrices in {dt:.2f} seconds')
            timings["weights"].append(dt)
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
                    realisation=int(realisation), 
                    lmax=L_max-1,
                    lam='2.0',
                )
                W = np.load(weight_vector_path)
                if W.ndim == 2 and 1 in W.shape:
                    W = W.reshape(-1)
                weight_vector_load.append(W)
                W_for_final_check = W

            # 7) Create doubled ILC maps (serial; when noted MP here is slower)
            t0 = time.perf_counter()
            doubled_maps = []
            for i, scale in enumerate(scales):
                map_ = SILCTools.create_doubled_ILC_map(
                    frequencies,
                    scale,
                    weight_vector_load[i],
                    doubled_MW_wav_c_j,
                    realisation=int(realisation),
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
                        realisation=int(realisation),
                        lmax=lmax,
                        lam='2.0',
                    ),
                    map_
                )
            dt = time.perf_counter() - t0
            print(f'Created ILC maps in {dt:.2f} seconds')
            timings["create_ilc_maps"].append(dt)

            # 8) Trim to original resolution (MP: one job per scale)
            t0 = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        SILCTools.trim_to_original,
                        doubled_maps[i],
                        scales[i],
                        int(realisation),
                        'jax_cuda',
                        path_template=output_templates["trimmed_maps"],
                        component=comp,
                        extract_comp=extract_comp,
                        lmax=lmax,
                        lam='2.0',
                    )
                    for i in range(len(scales))
                ]
                # keep order aligned with `scales`
                trimmed_maps = [None] * len(scales)
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    if isinstance(res, tuple) and len(res) == 2:
                        sc, trimmed = res
                        idx = scales.index(sc)
                        trimmed_maps[idx] = trimmed
                for idx, tm in enumerate(trimmed_maps):
                    if tm is None:
                        trimmed_maps[idx] = np.load(
                            output_templates['trimmed_maps'].format(
                                component=comp,
                                extract_comp=extract_comp,
                                scale=scales[idx],
                                realisation=int(realisation),
                                lmax=L_max-1,
                                lam='2.0',
                            )
                        )
            dt = time.perf_counter() - t0
            print(f'Trimmed maps to original resolution in {dt:.2f} seconds')
            timings["trim"].append(dt)

            # 9) Synthesize final map (serial)        
            synth_map = SILCTools.synthesize_ILC_maps_generalised(
             trimmed_maps=trimmed_maps,
             realisation=int(realisation_str),
             file_templates=output_templates,
             lmax=lmax,             
             N_directions=N_directions,
             lam=2.0,
             component=comp,
             extract_comp=extract_comp,
             frequencies=frequencies,
             visualise=True,
             constraint=constraint,
            )         
            synthesized_map.append(np.asarray(synth_map))

            # 10) One-time verification per realisation
            if constraint and (W_for_final_check is not None):
                _check_against_F(W_for_final_check, F, f)
            return synthesized_map, timings
        
# ilc_producer = ProduceSILC(ilc_components = ["cfn"], frequencies = ["030", "070"], realisations=1, lmax=1024, directory="/Scratch/matthew/data/", synthesise=True, method="jax_cuda")
# ilc_producer.process_wavelet_maps(save_intermediates=True, visualise=False)