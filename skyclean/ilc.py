from utils import *
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
    def Single_Map_doubleworker(mw_map, method):
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
        alm = s2fft.forward(mw_map, L=mw_map.shape[1], method = method, spmd = False)

        L = alm.shape[0]
        H = 2*L - 1
        W = 2*H - 1
        padded = np.zeros((H, W), dtype=np.complex128)
        mid_in = alm.shape[1]//2
        mid_out = W//2
        start = mid_out - mid_in
        padded[:L, start:start+alm.shape[1]] = alm

        x2 = np.real(s2fft.inverse(padded, L=H, method = method, spmd = False))

        return x2

    @staticmethod
    def calculate_covariance_matrix(frequencies, doubled_MW_wav_c_j, scale, realisation, method):
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
        f = '_'.join(frequencies)
        
        # Testing if single process output is the same as multiprocessing output
        # np.save(f"ILC/covariance_matrix/half_original_{scale}_R{realisation}", full_array)
        # Fill the symmetric part of the matrix
        for l1 in range(1, total_frequency):
            for l2 in range(l1):
                full_array[l1, l2] = full_array[l2, l1]
        # print(full_array.shape)
        return full_array

    @staticmethod
    def smoothed_covariance(MW_Map1, MW_Map2, method):
        """
        Parameters:
            MW_Map1, MW_Map2: same‐shape complex np.ndarray wavelet maps

        Returns:
            R_covariance_map: real‐valued covariance map as np.ndarray
        """
        smoothing_L = MW_Map1.shape[0]
            # 1) pixel covariance
        map1 = np.real(MW_Map1)
        map2 = np.real(MW_Map2)
        Rpix = np.multiply(map1, map2) + 0.j

        # 2) forward (smooth in harmonic space for efficieny)
        Ralm = s2fft.forward(Rpix, L=smoothing_L, method = method, spmd = False)

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
        Rmap = np.real(s2fft.inverse(convolved, L=smoothing_L, method = method, spmd = False))

        return Rmap

    @staticmethod
    def double_and_save_wavelet_maps(original_wavelet_c_j, frequencies, scales, realisation, method):
        """
        Doubles the resolution of wavelet maps and saves them with the realisation number in the file name.

        Parameters:
            original_wavelet_c_j (dict): Dictionary containing the original wavelet maps.
            frequencies (list): List of frequency strings.
            scales (list): List of scale indices.
            realisation (int): The realisation number for file naming.
        """
        for i in frequencies:
            for j in scales:
                # Perform the doubling of the wavelet map for the given frequency and scale
                wavelet_mw_map_doubled = SILCTools.Single_Map_doubleworker(original_wavelet_c_j[(i, j)], method)

    @staticmethod
    def compute_weight_vector(R,scale,realisation):
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
        singular_matrices_lcoation = []
        singular_matrices = []
        for i in range(dim1):
            for j in range(dim2):
                
                det = np.linalg.det(R_Pix[i, j])
                if det == 0:
                    print(i,j)
                    print(R_Pix[i, j].shape)
                    print(det)
                    print(R_Pix[i, j])
                    print("Pixel", i,j)
                    print("The matrix is singular.")
                    # np.linalg.inv(R_Pix[i, j])
                    zeros = np.zeros((subdim1))

                    singular_matrices_lcoation.append((i,j))
                    singular_matrices.append(R_Pix[i, j])
                    weight_vectors[i, j] = zeros
                    np.save(f"data/ILC/weight_vector_data/inverse_singular_matrix_{i}_{j}_S{scale}_R{realisation}.npy", R_Pix[i,j])
                    print("saved at ", f"ILC/weight_vector_data/inverse_singular_matrix_{i}_{j}_S{scale}_R{realisation}.npy")
                    
                else:
                    # print("The matrix is not singular.")
                    # Invert the matrix at position (i, j)
                    inverses[i, j] = np.linalg.inv(R_Pix[i, j])
                
                    # Compute the weight vector
                    numerator = np.dot(inverses[i, j], identity_vector)
                    denominator = np.dot(np.dot(inverses[i, j], identity_vector),identity_vector)
                    weight_vectors[i, j] = numerator / denominator

        return inverses, weight_vectors,singular_matrices_lcoation,singular_matrices
    
    @staticmethod
    def compute_ILC_for_pixel(i, j, frequencies, scale, weight_vector_load, doubled_MW_wav_c_j):
        """
        Computes the Internal Linear Combination (ILC) value for a specific pixel using the provided wavelet coefficients and weight vectors.

        Parameters:
            i (int): The row index of the pixel in the map.
            j (int): The column index of the pixel in the map.
            frequencies (list): A list of frequency identifiers corresponding to different channels.
            scale (int): The scale of the wavelet coefficient map.
            weight_vector_load (list): A list where each element corresponds to the weight vector map at a scale.
            doubled_MW_wav_c_j (dict): A dictionary with keys as tuples of (frequency, scale) and values as 2D arrays of wavelet coefficients for each pixel.

        Returns:
            float: The ILC value computed for the pixel at position (i, j).
        """
        # Create a vector of pixel values of all frequencies at the given pixel position
        pix_vector = np.array([
            doubled_MW_wav_c_j[(frequencies[k], scale)][i, j] for k in range(len(frequencies))
        ])
        return np.dot(weight_vector_load[scale][i, j], pix_vector)

    @staticmethod
    def create_doubled_ILC_map(frequencies, scale, weight_vector_load, doubled_MW_wav_c_j, realisation):
        
        """
        Creates a doubled Internal Linear Combination (ILC) map for a given scale and realisation.
        Doubled because the resolution of the wavelet coefficient map is doubled.
        
        Parameters:
            frequencies (list): A list of frequency identifiers corresponding to different channels.
            scale (int): The wavelet coefficient scale.
            weight_vector_load (list): A list where each element corresponds to the weight vector map at a scale.
            doubled_MW_wav_c_j (dict): A dictionary with keys as tuples of (frequency, scale) and values as 2D arrays of wavelet coefficients for each pixel.
            realisation (int): The realisation index used for saving the resulting ILC map.

        Returns:
            doubled_map (np.ndarray): The generated ILC map as a 2D numpy array.
        """
        # Get the size of the wavelet map
        size = doubled_MW_wav_c_j[(frequencies[0],scale)].shape
        
        # Initialize the doubled map
        doubled_map = np.zeros((size[0], size[1]))
        
        # Compute the ILC value for each pixel in the map
        for i in range(doubled_map.shape[0]):
            for j in range(doubled_map.shape[1]):
                doubled_map[i, j] = SILCTools.compute_ILC_for_pixel(i, j, frequencies, scale,weight_vector_load, doubled_MW_wav_c_j)
        return doubled_map

    @staticmethod
    def trim_to_original(MW_Doubled_Map, scale, realisation, method):
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
        alm_doubled = s2fft.forward(MW_Doubled_Map, L=L2, method = method, spmd = False)

        # trim
        trimmed_alm = alm_doubled[:inner_v, start_col:end_col]
        # inverse
        pix = s2fft.inverse(trimmed_alm, L=inner_v, method = method, spmd = False)
        mw_map_original = pix[np.newaxis, ...]

        return mw_map_original
    
    @staticmethod
    def load_frequency_data(base_path, file_template, frequencies, scales=None, realisation = None):
        """
        Load NumPy arrays from dynamically generated file paths for each frequency and scale.
        
        Parameters:
            base_path (str): The base path where the files are located.
            file_template (str): The template for the file names, with placeholders for frequency and scale.
            frequencies (list): A list of frequency names.
            scales_: A lists of scales.
            
        Returns:
            dict: A dictionary where keys are tuples of (frequency, scale) and values are loaded NumPy arrays.
        """
        frequency_data = {}
        realisation = str(realisation).zfill(4)
        for frequency in frequencies:
            for scale in scales:
                # Generate the file path using the template and the current frequency and scale
                path = f"{base_path}/{file_template.format(frequency, scale, realisation)}"
                try:
                    frequency_data[(frequency, scale)] = np.load(path)
                except Exception as e:
                    print(f"Error loading {path} for frequency {frequency} and scale {scale}: {e}, realisation {realisation}")
        return frequency_data


class ProduceSILC():
    """Perform  Scale-discretised, directional wavelet ILC (SILC)."""
    def __init__(self, components: list, frequencies: list, realisations: int, directory: str = "data/", method: str = "jax_cuda"):
        """
        Parameters:
            components (list): List of foreground components to process.
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            directory (str): Directory where data is stored / saved to.
            method (str): s2fft method to use for forward/inverse.
        """
        self.components = components
        self.frequencies = frequencies
        self.realisations = realisations
        self.directory = directory
        self.method = method

        output_dir = os.path.join(directory, "SILC/")
        create_dir(output_dir)

        wavelet_map_directories = os.path.join(self.directory, "wavelet_transforms/wavelet_coeffs")

        ilc_wavelet_paths = {} 
        for comp in self.components:
            ilc_wavelet_paths[comp] = os.path.join(wavelet_map_directories, f"{comp}_wavelet_f{{frequency}}_s{{scale}}_r{{realisation:05d}}_lmax{{lmax}}.npy")

        silc_output_dir = {
            "covariance_matrix": os.path.join(output_dir, "covariance_matrix"),
            "ILC_doubled_maps": os.path.join(output_dir, "ILC_doubled_maps"),
            "ILC_processed_wavelet_maps": os.path.join(output_dir, "ILC_processed_wavelet_maps"),
            "synthesised_ILC_MW_maps": os.path.join(output_dir, "synthesised_ILC_MW_maps"),
            "wavelet_doubled": os.path.join(output_dir, "wavelet_doubled"),
            "weight_vector_data": os.path.join(output_dir, "weight_vector_data")
        }
        for key, value in silc_output_dir.items():
            create_dir(value)
        # Example usage

        self.output_paths = {
            'doubled_maps': "data/ILC/wavelet_doubled/Wav_Pix2_F{frequency}_S{scale}_R{realisation}_MP.npy",
            'covariance_matrices': "data/ILC/covariance_matrix/cov_MW_Pix2_F{frequencies}_S{scale}_R{realisation}_MP.npy",
            'weight_vector_matrices': "data/ILC/weight_vector_data/{type}_S{scale}_R{realisation}_MP.npy",
            'ilc_maps': "data/ILC/ILC_doubled_maps/ILC_Map_S{scale}_R{realisation}_MP.npy",
            'trimmed_maps': "data/ILC/ILC_processed_wavelet_maps/ILC_processed_wav_Map_S{scale}_R{realisation}_MP.npy",
            'synthesized_maps': "data/ILC/synthesized_ILC_MW_maps/ILC_MW_Map_R{realisation}_MP.npy",
            'f_scal': "data/ILC/scal_coeffs/Scal_MW_Pix_F100_R{realisation}.npy"
        }

        self.scales = detect_scales(
            directory=wavelet_map_directories,
            comp=self.components[0],  # Assuming all components have the same scales
            realisation=0,  # Use a dummy realisation to detect scales
            pad=5
        )
    
    def process_wavelet_maps(self, save_intermediates: bool = False, visualize: bool = False):
        ### UNFINISHED.

        """
        Process wavelet maps for the specified components and frequencies.

        Args:
            save_intermediates (bool): Whether to save intermediate results.
            visualize (bool): Whether to visualize the results.

        Returns:
            None
        """
        frequencies = self.frequencies
        scales = self.scales # assuming all components have the same scales
        method = self.method

        for comp in self.components:
            for realisation in range(self.realisations):
                realisation_str = str(realisation).zfill(4)
                print(f"Processing realisation {realisation_str}")
                path = f"data/ILC/ILC_processed_wavelet_maps/ILC_processed_wav_Map_S5_R{realisation_str}.npy"
                if os.path.exists(path):
                        print(f"File {path} already exists.")
                        continue
                original_wavelet_c_j = SILCTools.load_frequency_data(base_path, file_template, frequencies, scales, realisation_str)
                
                # Double the resolution of the wavelet maps
                SILCTools.double_and_save_wavelet_maps(original_wavelet_c_j, frequencies, scales, realisation_str, method = method)
                if save_intermediates:
                    doubled_MW_wav_c_j = SILCTools.load_frequency_data("data/ILC/wavelet_doubled", "Wav_Pix2_F{}_S{}_R{}.npy", frequencies, scales, realisation_str)
            
                # Calculate the covariance matrices for each scale
                for i in range(len(scales)):      
                    scale = i
                    SILCTools.calculate_covariance_matrix(frequencies, doubled_MW_wav_c_j, scale, realisation_str, method = method)

                F_str = '_'.join(frequencies)
                R_covariance = [np.load(f"data/ILC/covariance_matrix/cov_MW_Pix2_F{F_str}_S{i}_R{realisation_str}_Full.npy") for i in range(len(scales))] 

                # Calculate the weight vectors for each frequency wavelet coefficient map using covariance matrix and the euqation.
                for scale in range(len(R_covariance)):
                    # print(scale)
                    SILCTools.compute_weight_vector(R_covariance[scale], scale, realisation_str)
                weight_vector_load = [np.load(f"data/ILC/weight_vector_data/weight_vector_S{i}_R{realisation_str}.npy") for i in range(len(scales))]

                doubled_maps = []
                # Create the doubled resolution ILC map for each scale
                for i, scale in enumerate(scales):
                    doubled_maps.append(SILCTools.create_doubled_ILC_map(frequencies, scale, weight_vector_load, doubled_MW_wav_c_j, realisation=realisation_str))
                doubled_maps = [np.load(f"data/ILC/ILC_doubled_maps/ILC_Map_S{i}_R{realisation_str}.npy") for i in range(len(scales))]
                # Trim the doubled resolution ILC map back to the original resolution
                trimmed_maps = [SILCTools.trim_to_original(doubled_maps[i], i, realisation_str, method) for i in range(len(scales))]
                
                # if visualize: 
                #     start = time.perf_counter()
                #     for i in range(len(scales)):
                #         tilte = "ILC wavelet coefficient map at scale: "
                #         visualize_wavelet_coefficient_map(trimmed_maps[i], tilte, str(i), method = method)
                #     visualize_time = time.perf_counter() - start

                # return trimmed_maps



