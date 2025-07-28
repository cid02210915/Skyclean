from utils import *
from map_tools import *
import os
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2wav
import math 
import healpy as hp

class SILCTools():
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
        alm = s2fft.forward(mw_map, L=mw_map.shape[1], method = method, spmd = False, reality = False)

        L = alm.shape[0]
        H = 2*L - 1
        W = 2*H - 1
        padded = np.zeros((H, W), dtype=np.complex128)
        mid_in = alm.shape[1]//2
        mid_out = W//2
        start = mid_out - mid_in
        padded[:L, start:start+alm.shape[1]] = alm

        x2 = np.real(s2fft.inverse(padded, L=H, method = method, spmd = False, reality = False))

        return x2

    @staticmethod
    def calculate_covariance_matrix(frequencies: list, doubled_MW_wav_c_j: dict, scale: int, realisation: int, method: str):
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
        if frequencies:
            sample_data = doubled_MW_wav_c_j[(frequencies[0], scale)]
            n_rows, n_cols = sample_data.shape
        else:
            raise ValueError("Frequency list is empty.")
        
        total_frequency = len(frequencies)
        # Initialize a 4D array to store the covariance matrices
        full_array = np.zeros((total_frequency, total_frequency, n_rows, n_cols))

        # Calculate the covariance matrix and save each one
        # Calculate the upper triangle only since the matrix is symmetric
        for i in range(total_frequency):
            for fq in range(i, total_frequency):
                full_array[i, fq] = SILCTools.smoothed_covariance(doubled_MW_wav_c_j[(frequencies[i], scale)],
                                                        doubled_MW_wav_c_j[(frequencies[fq], scale)],
                                                        method)
                # Save the computed covariance matrix
                # np.save(f"ILC/covariance_matrix/cov_MW_Pix2_F{frequencies[i]}_F{frequencies[fq]}_S{scale}", full_array[i, fq])
        # Testing if single process output is the same as multiprocessing output
        # np.save(f"ILC/covariance_matrix/half_original_{scale}_R{realisation}", full_array)
        # Fill the symmetric part of the matrix
        for l1 in range(1, total_frequency):
            for l2 in range(l1):
                full_array[l1, l2] = full_array[l2, l1]
        # print(full_array.shape)
        return full_array

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
        Ralm = s2fft.forward(Rpix, L=smoothing_L, method = method, spmd = False, reality = False)

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
        Rmap = np.real(s2fft.inverse(convolved, L=smoothing_L, method = method, spmd = False, reality = False))

        return Rmap

    @staticmethod
    def double_wavelet_maps(original_wavelet_c_j: dict, frequencies: list, scales: list, realisation: int, method: str):
        """
        Doubles the resolution of wavelet maps and returns them as a dictionary.

        Parameters:
            original_wavelet_c_j (dict): Dictionary containing the original wavelet maps.
            frequencies (list): List of frequency strings.
            scales (list): List of scale indices.
            realisation (int): The realisation number for file naming.
            method (str): s2fft method to use for forward/inverse.

        Returns:
            dict: A dictionary with keys as (frequency, scale) and values as doubled wavelet maps.
        """
        doubled_MW_wav_c_j = {}
        for i in frequencies:
            for j in scales:
                # Perform the doubling of the wavelet map for the given frequency and scale
                wavelet_mw_map_doubled = SILCTools.Single_Map_doubleworker(original_wavelet_c_j[(i, j)], method)
                doubled_MW_wav_c_j[(i, j)] = wavelet_mw_map_doubled
        return doubled_MW_wav_c_j

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
        # Realiztion 6 has a singular matrix
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
        Trim a doubled‐resolution MW Pixel map back to its original resolution,
        using either NumPy or PyTorch spherical transforms.

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
        alm_doubled = s2fft.forward(MW_Doubled_Map, L=L2, method = method, spmd = False, reality = False)

        # trim
        trimmed_alm = alm_doubled[:inner_v, start_col:end_col]
        # inverse
        pix = s2fft.inverse(trimmed_alm, L=inner_v, method = method, spmd = False, reality = False)
        mw_map_original = pix[np.newaxis, ...]

        return mw_map_original
    
    @staticmethod
    def load_frequency_data(file_template: str, comp: str, frequencies: list, scales=None, realisation=None, lmax=None):
        """
        Load NumPy arrays from dynamically generated file paths for each frequency and scale.
        
        Parameters:
            file_template (str): The template for the file names, with placeholders for frequency and scale.
            frequencies (list): A list of frequency names.
            scales (list): A list of scales.
            realisation (int): The realisation index.
            lmax (int): The maximum multipole.

        Returns:
            dict: A dictionary where keys are tuples of (frequency, scale) and values are loaded NumPy arrays.
        """
        frequency_data = {}
        for frequency in frequencies:
            for scale in scales:
                # Generate the file path using the template and the current frequency and scale
                path = file_template.format(comp=comp, frequency=frequency, scale=scale, realisation=realisation, lmax=lmax)
                try:
                    frequency_data[(frequency, scale)] = np.load(path)
                except Exception as e:
                    print(f"Error loading {path} for frequency {frequency} and scale {scale}: {e}, realisation {realisation}")
        return frequency_data


class ProduceSILC():
    """Perform  Scale-discretised, directional wavelet ILC (SILC)."""
    def __init__(self, ilc_components: list, frequencies: list, realisations: int, lmax: int, 
    N_directions: int = 1, lam: float = 2.0, synthesise = True, directory: str = "data/", method: str = "jax_cuda"):
        """
        Parameters:
            ilc_components (list): List of components to produce an ILC for.
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            lmax (int): Maximum multipole for the wavelet transform.
            synthesise (bool): Whether to synthesise the ILC map from the wavelet transforms.
            directory (str): Directory where data is stored / saved to.
            method (str): s2fft method to use for forward/inverse.
        """
        self.ilc_components = ilc_components
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions  
        self.lam = lam
        self.synthesise = synthesise
        self.directory = directory
        self.method = method

        output_dir = os.path.join(directory, "SILC/")
        create_dir(output_dir)

        wavelet_map_directories = os.path.join(self.directory, "wavelet_transforms/wavelet_coeffs")

        self.ilc_wavelet_paths = {} 
        for comp in self.ilc_components:
            self.ilc_wavelet_paths[comp] = os.path.join(wavelet_map_directories, f"{{comp}}_wavelet_f{{frequency}}_s{{scale}}_r{{realisation:05d}}_lmax{{lmax}}.npy")

        silc_output_dir = {
            "doubled_maps": os.path.join(output_dir, "doubled_maps"),
            "covariance_matrix": os.path.join(output_dir, "covariance_matrix"),
            "weight_vector_data": os.path.join(output_dir, "weight_vector_data"),
            "ilc_doubled_wavelet_maps": os.path.join(output_dir, "ilc_doubled_wavelet_maps"),
            "ilc_trimmed_maps": os.path.join(output_dir, "ilc_trimmed_maps"),
            "ilc_synthesised_maps": os.path.join(output_dir, "ilc_synthesised_maps"),
        }
        for _, value in silc_output_dir.items():
            create_dir(value)
        # Example usage

        self.output_paths = {
            'doubled_maps': os.path.join(silc_output_dir["doubled_maps"], "doubled_maps_F{frequency}_S{scale}_R{realisation:04d}_lmax{lmax}.npy"),
            'covariance_matrices': os.path.join(silc_output_dir["covariance_matrix"],"cov_MW_F{frequencies}_S{scale}_R{realisation:04d}_lmax{lmax}.npy"),
            'weight_vector_matrices': os.path.join(silc_output_dir["weight_vector_data"], "weight_vector_S{scale}_R{realisation:04d}_lmax{lmax}.npy"),
            'ilc_doubled_maps': os.path.join(silc_output_dir["ilc_doubled_wavelet_maps"], "ILC_doubled_Map_S{scale}_R{realisation:04d}_lmax{lmax}.npy"),
            'ilc_trimmed_maps': os.path.join(silc_output_dir["ilc_trimmed_maps"], "ILC_trimmed_wav_Map_S{scale}_R{realisation:04d}_lmax{lmax}.npy"),
            'ilc_synthesised_maps': os.path.join(silc_output_dir["ilc_synthesised_maps"], "ILC_synthesised_Map_R{realisation:04d}_lmax{lmax}.npy"),
        }

        self.scales = detect_scales(
            directory=wavelet_map_directories,
            comp=self.ilc_components[0],  # Assuming all components have the same scales
            realisation=0,  # Use a dummy realisation to detect scales
            pad=5
        )
    
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
                print(f"Processing realisation {realisation} for component {comp}...")
                path_test = self.output_paths['ilc_synthesised_maps'].format(realisation=realisation, lmax=lmax)
                if os.path.exists(path_test):
                    print(f"File {path_test} already exists. Skipping to the next realisation.")
                    continue

                # Load original wavelet maps
                original_wavelet_c_j = SILCTools.load_frequency_data(self.ilc_wavelet_paths[comp], comp, frequencies, scales, realisation, lmax)
                # Double the resolution of the wavelet maps
                doubled_MW_wav_c_j = {}
                for i in frequencies:
                    for j in scales:
                        doubled_map_path = self.output_paths['doubled_maps'].format(frequency=i, scale=j, realisation=realisation, lmax=lmax)
                        if os.path.exists(doubled_map_path):
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
                    covariance_path = self.output_paths['covariance_matrices'].format(frequencies='_'.join(frequencies), scale=scale, realisation=realisation, lmax=lmax)
                    if os.path.exists(covariance_path):
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
                    weight_vector_path = self.output_paths['weight_vector_matrices'].format(scale=scale, realisation=realisation, lmax=lmax)
                    if os.path.exists(weight_vector_path):
                        print(f"Weight vector for scale {scale}, realisation {realisation} already exists. Skipping weight vector calculation.")
                        weight_vectors.append(np.load(weight_vector_path))
                    else:
                        weight_vector_scale = SILCTools.compute_weight_vector(R_covariance[scale], scale, realisation)
                        weight_vectors.append(weight_vector_scale)
                        if save_intermediates:
                            np.save(weight_vector_path, weight_vector_scale)
                            print(f"Saved weight vector for scale {scale}, realisation {realisation}.")
                # Create the doubled resolution ILC map for each scale
                doubled_maps = []
                for scale in scales:
                    ilc_doubled_map_path = self.output_paths['ilc_doubled_maps'].format(scale=scale, realisation=realisation, lmax=lmax)
                    if os.path.exists(ilc_doubled_map_path):
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
                    ilc_trimmed_map_path = self.output_paths['ilc_trimmed_maps'].format(scale=scale, realisation=realisation, lmax=lmax)
                    if os.path.exists(ilc_trimmed_map_path):
                        print(f"ILC trimmed map for scale {scale}, realisation {realisation} already exists. Skipping trimming.")
                        trimmed_maps.append(np.load(ilc_trimmed_map_path))
                    else:
                        trimmed_map = SILCTools.trim_to_original(doubled_maps[i], scale, realisation, method)
                        trimmed_maps.append(trimmed_map)
                        np.save(ilc_trimmed_map_path, trimmed_map)
                        print(f"Saved trimmed map for scale {scale}, realisation {realisation}.")

                # Synthesise the ILC map from the trimmed wavelet maps
                if self.synthesise:
                    ilc_synthesised_map_path = self.output_paths['ilc_synthesised_maps'].format(realisation=realisation, lmax=lmax)
                    if os.path.exists(ilc_synthesised_map_path):
                        print(f"ILC synthesised map for realisation {realisation} already exists. Skipping synthesis.")
                    else:
                        mw_pix = MWTools.inverse_wavelet_transform(trimmed_maps, L, N_directions, lam)
                        np.save(ilc_synthesised_map_path, mw_pix)
                        print(f"Saved synthesised ILC map for realisation {realisation}.")
                    
        return None


# ilc_producer = ProduceSILC(ilc_components = ["cfn"], frequencies = ["030", "070"], realisations=1, lmax=1024, directory="/Scratch/matthew/data/", synthesise=True, method="jax_cuda")
# ilc_producer.process_wavelet_maps(save_intermediates=True, visualise=False)
