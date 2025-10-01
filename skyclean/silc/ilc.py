import os
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2wav
import math 
import healpy as hp
from s2wav import filters
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from jax import config
config.update("jax_enable_x64", True)
from functools import lru_cache

from .map_tools import *
from .utils import *
from .file_templates import FileTemplates
from .utils import normalize_targets   
from .utils import save_array
from skyclean.silc.mixing_matrix_constraint import ILCConstraints 
import concurrent.futures
import time

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
        #print('Single_Map_doubleworker', flush = True)

        alm = s2fft.forward(mw_map, L=mw_map.shape[0], method = method, spmd = False, reality = True)

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

    # ---- cached Gaussian beam per L ----
    @lru_cache(maxsize=32)
    def _cached_beam(L: int):
        nsamp = 1200.0
        lmax  = L
        npix  = hp.nside2npix(1 << (int(0.5 * lmax) - 1).bit_length())
        scale_fwhm = 4.0 * math.sqrt(nsamp / npix)
        return hp.gauss_beam(scale_fwhm, lmax=lmax - 1)   # (L,)


    @staticmethod
    def smoothed_covariance(MW_Map1: np.ndarray, MW_Map2: np.ndarray, method='jax_cuda'):
        """
        Return real-valued covariance map, smoothed by a Gaussian beam in harmonic space.
        """
        L = int(MW_Map1.shape[0])

        # --- ensure MW sampling width=2L-1 only if needed ---
        exp = 2 * L - 1
        def _ensure_mw(a: np.ndarray) -> np.ndarray:
            nphi = a.shape[1]
            if nphi == exp:      return a
            if nphi >  exp:      return a[:, :exp]
            reps = (exp + nphi - 1) // nphi
            return np.tile(a, reps)[:, :exp]

        map1 = _ensure_mw(np.real(MW_Map1))
        map2 = _ensure_mw(np.real(MW_Map2))

        # trivial guard
        if L < 2 or map1.shape[1] != exp:
            return np.real(map1 * map2)

        # pixelwise product (keep real; reality=True below)
        Rpix = map1 * map2

        # forward -> beam -> inverse
        Ralm = s2fft.forward(Rpix, L=L, method=method, spmd=False, reality=True)
        beam = SILCTools._cached_beam(Ralm.shape[0])  # CPU array

        if method == "numpy":
            # match dtype to avoid implicit casts
            convolved = Ralm * beam[:, None].astype(Ralm.dtype, copy=False)
            out = s2fft.inverse(convolved, L=L, method="numpy", spmd=False, reality=True)
            return np.real(out)

        # JAX path: keep multiply on device and dtype-aligned
        jbeam = jnp.asarray(beam, dtype=Ralm.dtype)
        convolved = Ralm * jbeam[:, None]
        out = s2fft.inverse(convolved, L=L, method=method, spmd=False, reality=True)
        return np.asarray(jnp.real(out))


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

        if not frequencies:
            raise ValueError("Frequency list is empty.")

        # --- normalization to fit current pipeline ---
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

        if method != "numpy":
            # serial fallback for jax/jax_cuda to avoid BrokenProcessPool
            for t in tasks:
                i, fq, cov = SILCTools.compute_covariance(t)
                full_array[i, fq] = cov
        else:
            # CPU/NumPy path can safely use processes
            ctx = mp.get_context("spawn")  # more robust than fork for native libs
            with ProcessPoolExecutor(mp_context=ctx) as executor:
                futures = [executor.submit(SILCTools.compute_covariance, t) for t in tasks]
                for fut in as_completed(futures):
                    i, fq, covariance_matrix = fut.result()
                    full_array[i, fq] = covariance_matrix
                    

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


    @staticmethod
    def double_and_save_wavelet_maps(
        original_wavelet_c_j, frequencies, scales, realisation, 
        component, path_template, *, lmax=64, lam="2.0", method="jax_cuda"
    ):
        #print ('double_and_save_wavelet_maps', flush = True)
        """Minimal fix: compute + save doubled maps serially (no MP)."""
        doubled_MW_wav_c_j = {}
        for f in frequencies:
            for s in scales:
                doubled = SILCTools.Single_Map_doubleworker(original_wavelet_c_j[(f, s)], method)
                doubled_MW_wav_c_j[(f, s)] = doubled

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
        return doubled_MW_wav_c_j


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
        #print('R:', R.shape)
        
        # --- shape handling --- what?
        # Swap the axes to get R_Pix
        R_Pix = np.swapaxes(np.swapaxes(R, 0, 2), 1, 3) #(pix,pix,freq,freq)
        # Get dimensions for looping and size of sub-matrices
        dim1, dim2, subdim1, subdim2 = R_Pix.shape
        # print(dim1, dim2, subdim1, subdim2)
        # Create arrays to store inverses and weight vectors
        inverses = np.zeros((dim1, dim2, subdim1, subdim2))
        weight_vectors = np.zeros((dim1, dim2, subdim1)) # weight vector at each pixel (dim1,dim2) and channel
        # Realiztion 6 has a singular matrix
        # Adjust identity vector size based on sub-matrix dimensions
        identity_vector = np.ones(subdim2, dtype=float)
        singular_matrices_location = []
        singular_matrices = []
        
        N_freq = subdim2 

        # --- branch config ---
        if constraint:
            if F is None:
                raise ValueError("F must be provided when constraint=True")
            #print ('F:',F.shape)
            Nf_F, N_comp = F.shape
            if Nf_F != N_freq:
                raise ValueError(f"F has {Nf_F} rows but R has {N_freq} channels")
    
            # Automatically set f from extract_comp if given
            if f is None and extract_comp is not None:
                f = ILCConstraints.find_f_from_extract_comp(F, extract_comp, reference_vectors)
            #print ('f:', f.shape)
            if f is None:
                raise ValueError("Constraint vector f must be provided (or inferable) when constraint=True")
            if f.shape != (N_comp,):
                raise ValueError(f"Constraint vector f must have shape ({N_comp},)")
        else:
            # Unconstrained ILC uses the all-ones vector; no F/extract_comp needed
            identity_vector = np.ones(N_freq, dtype=float)   
    
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
    
        for i in range(dim1):
            for j in range(dim2):
    
                det = np.linalg.det(R_Pix[i, j])
                if det == 0:
                    zeros = np.zeros((subdim1))
                    singular_matrices_location.append((i,j))
                    singular_matrices.append(R_Pix[i, j])
                    weight_vectors[i, j] = zeros
                else:
                    # compute inverse
                    R_inv = np.linalg.inv(R_Pix[i, j])
                    #print ('R_inv:', R_inv.shape)

                    if constraint:
                        # Step 1: Fᵗ R⁻¹
                        FT_Rinv = np.dot(F.T, R_inv)                 # (Nc, Nf)
                        #print ('FT_Rinv:', FT_Rinv.shape)
                        # Step 2: Fᵗ R⁻¹ F
                        constraint_matrix = np.dot(FT_Rinv, F)       # (Nc, Nc)
                        #print ('constraint_matrix:', constraint_matrix.shape)
                        # Step 3: (Fᵗ R⁻¹ F)⁻¹
                        constraint_matrix_inv = np.linalg.inv(constraint_matrix)
                        #print ('constraint_matrix_inv:', constraint_matrix_inv.shape)
                        # Step 4: build temp = (Fᵗ R⁻¹ F)⁻¹ f
                        temp = np.dot(constraint_matrix_inv, f)      # (Nc,)
                        #print ('temp:', temp.shape)
                        # Step 5: F temp
                        F_temp = np.dot(F, temp)                     # (Nf,)
                        #print ('F_temp:', F_temp.shape)
                        # Step 6: w = R⁻¹ F (Fᵗ R⁻¹ F)⁻¹ f
                        w = np.dot(R_inv, F_temp)                    # (Nf,)
                        w = np.asarray(w).ravel()
                        #print('w:', w.shape)
                    else:
                        num = np.dot(R_inv, identity_vector)         # (Nf,)
                        den = float(np.dot(num, identity_vector))    # scalar
                        w = (num / den).ravel()                      # (Nf,)

                    if R.ndim == 4:
                        inverses[i, j] = R_inv
                        weight_vectors[i, j] = w
                        #print('weight_vectors[i,j]:', weight_vectors[i,j].shape)
                    else:
                        inverses[i, j] = np.linalg.inv(R_Pix[i, j])
                        numerator = np.dot(inverses[i, j], identity_vector)
                        denominator = np.dot(np.dot(inverses[i, j], identity_vector),identity_vector)
                        weight_vectors[i, j] = numerator / denominator

        if len(singular_matrices_location) > 0:
            print("Discovered ", len(singular_matrices_location), "singular matrices at scale", scale, "realisation", realisation)

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
        cube = np.stack([doubled_MW_wav_c_j[(f, scale)] for f in frequencies], axis=-1)

        W = np.asarray(weight_vector_load)

        if W.ndim == 1:
            H, Wd, F = cube.shape
            return np.dot(cube.reshape(-1, F), W).reshape(H, Wd)
        
        if W.ndim == 3:
            # Elementwise multiply then sum over freq
            return np.einsum('ijc,ijc->ij', cube, W)

        raise ValueError(f"Unexpected weight_vector shape {W.shape}; expected (F,) or (H,W,F).")
    

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
        '''
        for d in jax.devices():
            ms = d.memory_stats()
            print(f"Device {d.id}:",
                f"bytes_in_use={ms['bytes_in_use']}",
                f"peak_bytes_in_use={ms['peak_bytes_in_use']}",
                f"bytes_limit={ms['bytes_limit']}",
                f"largest_free_chunk={ms.get('largest_free_chunk', 'n/a')}",
                f"num_allocs={ms.get('num_allocs', 'n/a')}")
        '''

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
        alm_doubled = s2fft.forward(MW_Doubled_Map, L=L2, method=method, spmd=False, reality=True)
   
        # trim in harmonic space
        trimmed_alm = alm_doubled[:inner_v, start_col:end_col]

        # inverse back to pixels
        pix = s2fft.inverse(trimmed_alm, L=inner_v, method=method, spmd=False, reality=True)

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
        # and mw_alm_2_hp_alm being importable
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
        )
        plt.show()

        
    @staticmethod
    def build_ilc_scaling_coeff(
        f_scal_template: str, *,
        frequencies: list[str],
        realisation: int,
        lmax: int,
        lam: float | str,
        component: str,
        weight_vector_matrix_template: str,
        L_max: int | None = None,
        extract_comp: str | None = None,
        constraint: bool = False,
        F: np.ndarray | None = None,            # shape (N_freq, N_constraints)
        f: np.ndarray | None = None,            # shape (N_constraints,)
        reference_vectors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an ILC'ed scaling-coefficient map (L, 2L-1)."""
    
        realisation = int(realisation)
        lam_str = str(lam)
        L = int(lmax) + 1
        if L_max is None:
            L_max = L
    
        # (1) load per-frequency scaling maps -> each (L, 2L-1)
        S_list = []
        for ff in frequencies:
            path = f_scal_template.format(
                comp=component, component=component,
                frequency=str(ff), realisation=realisation,
                lmax=int(lmax), lam=lam_str,
            )
            S_list.append(np.load(path))
        X = np.stack(S_list, axis=-1)             # (L, 2L-1, F)
        _, _, Fch = X.shape
    
        # (optional) quick sanity for cILC shapes
        if constraint:
            if F is None:
                raise ValueError("constraint=True requires F")
            if F.shape[0] != Fch:
                raise ValueError(f"F has {F.shape[0]} rows but #freq is {Fch}")
            if f is not None and f.shape != (F.shape[1],):
                raise ValueError(f"f must have shape ({F.shape[1]},), got {f.shape}")
    
        # (2) global covariance across pixels in channel space (F,F)
        Xp = X.reshape(-1, Fch)                   # (Npix, F)
        R  = (Xp.T @ Xp) / Xp.shape[0]            # (F, F)
    
        # (3) one global weight vector
        _, w_scal_global, _, _ = SILCTools.compute_weights_generalised(
            R=R,
            scale=-1,                              # sentinel: “scaling band”
            realisation=realisation,
            weight_vector_matrix_template=weight_vector_matrix_template,
            comp=component,
            L_max=L_max,
            extract_comp=extract_comp,
            constraint=constraint,
            F=F, f=f,
            reference_vectors=reference_vectors,
            lam=lam_str,
        )  
        # w_scal_global shape == (F,)
    
        # (4) apply weights per pixel -> (L, 2L-1)
        f_scal_ilc = (X @ w_scal_global).astype(X.dtype)
        return f_scal_ilc, w_scal_global
    

    @staticmethod
    def synthesize_ILC_maps_generalised(
        trimmed_maps, realisation, file_templates, lmax, N_directions,lam=2.0, component=None, 
        extract_comp=None, visualise=False, constraint=None, frequencies=None, F=None, f=None, 
        reference_vectors=None,
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
        file_tmpl = file_templates 

        # build frequencies tag for filename
        if isinstance(frequencies, (list, tuple)):
            freq_tag = "_".join(map(str, frequencies))
            freq0 = str(frequencies[0])
        elif frequencies is None:
            freq_tag, freq0 = "unknown", None
        else:
            freq_tag = str(frequencies)
            freq0 = freq_tag.split("_")[0]

        # 3) build filters and synthesise
        L = int(lmax) + 1

        MW_Pix = MWTools.inverse_wavelet_transform(trimmed_maps, L, N_directions=1, lam=float(lam))

        # 4) Save
        out_path = file_tmpl["ilc_synth"].format(
            extract_comp=extract_comp,
            component=component,
            frequencies=freq_tag,
            realisation=int(realisation_str),
            lmax=int(lmax),
            lam=str(lam),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, MW_Pix)
    
        # 5) Visualise
        if visualise:
            try:
                prefix = "cILC" if extract_comp else "ILC"
                name = extract_comp.upper() if extract_comp else ""
                title = f"{prefix} {name} | r={realisation_str}, lmax={int(lmax)}, N={N_directions}, λ={lam}".strip()
                SILCTools.visualize_MW_Pix_map(MW_Pix, title)
            except Exception:
                pass
            
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


    def ILC_wav_coeff_maps_MP(file_template, frequencies, scales, realisations, output_templates, L_max, 
                              N_directions, comp, constraint=False, F=None, extract_comp=None,
                             reference_vectors=None):

        lmax = L_max - 1  
        realisations = [int(r) for r in realisations]

        '''
        def _check_against_F(W, F, f, tol=1e-6, scale=None):
            W = np.asarray(W)
            if W.ndim == 2 and 1 in W.shape:
                W = W.reshape(-1)
            resp = np.tensordot(W, F, axes=([-1], [0]))
            ok = np.allclose(resp, f, atol=tol, rtol=0.0)
            tag = f" [scale={scale}]" if scale is not None else ""
            print(f"FINAL CHECK F^T w == f{tag} -> {ok}")
            if not ok:
                print(f"max |F^T w - f|{tag} = {float(np.max(np.abs(resp - f)))}")
            return ok
        '''
        
        def _check_against_F(W, F, f, tol=1e-6):
            W = np.asarray(W)
            if W.ndim == 2 and 1 in W.shape:   # (1,Nf) or (Nf,1) -> (Nf,)
                W = W.reshape(-1)
            resp = np.tensordot(W, F, axes=([-1], [0]))  # (..., N_comp)
            ok = np.allclose(resp, f, atol=tol, rtol=0.0)
            print("FINAL CHECK  F^T w == f  ->", ok)
            if not ok:
                print("|F^T w - f| =", float(np.max(np.abs(resp - f))))
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
            f = ILCConstraints.find_f_from_extract_comp(F, target_names, reference_vectors)
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

            # 4) Compute covariance matrices (serial, JAX backend)
            t0 = time.perf_counter()

            for scale in scales:
                SILCTools.calculate_covariance_matrix(
                    frequencies=frequencies,
                    doubled_MW_wav_c_j=doubled_MW_wav_c_j,
                    scale=int(scale),
                    realisation=int(realisation),
                    method= "jax_cuda"  ,                  
                    path_template=output_templates["covariance_matrices"],
                    component=comp,
                    lmax=L_max - 1,
                    lam=2.0,
                )

            dt = time.perf_counter() - t0
            print(f"Calculated covariance matrices in {dt:.2f} seconds")
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

            # 6) Compute weight vectors (serial; no MP to avoid pickling big arrays)
            t0 = time.perf_counter()
            for idx, scale in enumerate(scales):
                SILCTools.compute_weights_generalised(
                    R=R_covariance[idx],
                    scale=scale,
                    realisation=int(realisation),
                    weight_vector_matrix_template=output_templates['weight_vector_matrices'],
                    comp=comp,
                    L_max=L_max,
                    extract_comp=extract_comp,
                    constraint=constraint,
                    F=F,
                    f=f,
                    reference_vectors=reference_vectors,
                    lam='2.0',
                )

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
                W = np.load(weight_vector_path)  # mmap_mode='r' optional
                if W.ndim == 2 and 1 in W.shape:
                    W = W.reshape(-1)            # handle saved (1,F) or (F,1)
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

            # 8) Trim to original resolution (serial, in-order)
            t0 = time.perf_counter()

            trimmed_maps = []
            for i, sc in enumerate(scales):
                save_path = output_templates['trimmed_maps'].format(
                    component=comp,
                    extract_comp=extract_comp,
                    scale=int(sc),
                    realisation=int(realisation),
                    lmax=int(lmax),
                    lam='2.0',
                )

                if os.path.exists(save_path):
                    #print(f"ILC trimmed map for scale {sc}, realisation {realisation} already exists. Loading.")
                    tm = np.load(save_path)
                    trimmed_maps.append(tm)
                    continue
                
                # Optional tiny guard: skip trimming if not truly doubled (e.g., j=0)
                arr = np.asarray(doubled_maps[i])
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr2d = arr[0]
                else:
                    arr2d = arr
                L2, W2 = arr2d.shape
                Lorig = (L2 + 1) // 2
                if W2 != 2 * L2 - 1 or L2 != 2 * Lorig - 1 or L2 <= 1:
                    tm = arr  # pass-through
                else:
                    # Use same backend choice you’ve been using; here 'jax_cuda' to match your submit
                    _, tm = SILCTools.trim_to_original(
                        MW_Doubled_Map=arr2d,
                        scale=int(sc),
                        realisation=int(realisation),
                        method='jax_cuda',
                        path_template=output_templates["trimmed_maps"],
                        component=comp,
                        extract_comp=extract_comp,
                        lmax=int(lmax),
                        lam='2.0',
                    )

                # Ensure saved on disk (trim_to_original already saved when path_template provided,
                # but save again if we took the pass-through branch)
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, tm)

                trimmed_maps.append(tm)

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
             F=F, 
             f=f, 
             reference_vectors=reference_vectors,
            )         
            synthesized_map.append(np.asarray(synth_map))

            '''
            # 10) Verify constraints
            if constraint:
                # per wavelet scale
                for j, W in zip(scales, weight_vector_load):
                    _check_against_F(W, F, f, scale=j)
            
                # scaling band (if available)
                try:
                    _check_against_F(w_scal_global, F, f, scale="j=0 (scaling)")
                except NameError:
                    pass
                '''

            # 10) One-time verification per realisation
            if constraint and (W_for_final_check is not None):
                _check_against_F(W_for_final_check, F, f)
            return synthesized_map, timings

    
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
                        weight_vector_scale = SILCTools.compute_weights_generalised(R_covariance[scale], scale, realisation)
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