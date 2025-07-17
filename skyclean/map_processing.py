from map_tools import *
import os
import healpy as hp


class ProcessMaps():
    """Process downloaded maps."""
    def __init__(self, components: list, frequencies: list, realisations: int, desired_lmax:int, directory: str = "data/", method = "jax_cuda"): 
        """
        Parameters: 
            components (list): List of foreground components to process. Includes: 'sync' (synchrotron)
            directory (str): Directory where data is stored / saved to.
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            desired_lmax (int): Desired maximum multipole for the processed maps.
            method (str): s2fft method
        """
        self.components = components
        self.frequencies = frequencies
        self.realisations = realisations
        self.desired_lmax = desired_lmax
        self.directory = directory
        
        downloaded_map_directories = os.path.join(self.directory, "CMB_realisations")
        self.downloaded_data_filepaths = {
            "cmb": os.path.join(downloaded_map_directories, "cmb_r{realisation:04d}.fits"),
            "sync": os.path.join(downloaded_map_directories, "sync_f{frequency}.fits"),
            "noise": os.path.join(downloaded_map_directories, "noise_f{frequency}_r{realisation:05d}.fits")
        }

        self.output_directories = {
            "cfn": os.path.join(self.directory, "CFN_realisations"),
            "processed_maps": os.path.join(self.directory, "processed_maps"),
            "wavelet_coeffs": os.path.join(self.directory, "wavelet_transforms/wavelet_coeffs"),
            "scaling_coeffs": os.path.join(self.directory, "wavelet_transforms/scaling_coeffs"),
        }

        self.output_paths = {
            "cfn": os.path.join(self.output_directories["cfn"], "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
            "cmb": os.path.join(self.output_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
            "sync": os.path.join(self.output_directories["processed_maps"], "processed_sync_f{frequency}_lmax{lmax}.npy"),
            "noise": os.path.join(self.output_directories["processed_maps"], "processed_noise_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
            "wavelet_coeffs": os.path.join(self.output_directories["wavelet_coeffs"], "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}.npy"),
            "scaling_coeffs": os.path.join(self.output_directories["scaling_coeffs"], "{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}.npy")
        }

    def create_cfn(self, frequency: str, realisation: int, save = True): 
        """
        Create a CFN (Cmb + Foreground + Noise) for a given frequency and realisation, by convolving
        the CMB and foregrounds with the standard beam and adding noise.

        Parameters:
            frequency (str): The frequency for which to create the CFN.
            realisation (int): The realisation number for which to create the CFN.
            lmax (int): The maximum multipole desired for the CFN map.
            save (bool): Whether to save each processed component.
        
        Returns:
            np.ndarray: The CFN map in HP format.
        """
        def save_map(hp_map, filepath):
            """Save the processed map to the specified filepath."""
            if save:
                if os.path.exists(filepath):
                    print(f"File {filepath} already exists. Skipping saving.")
                else:
                    hp.write_map(filepath, hp_map)
        desired_lmax = self.desired_lmax
        standard_fwhm_rad = np.radians(5/60)
        nside = desired_lmax // 2 # desired n
        cfn = np.zeros(hp.nside2npix(nside), dtype=np.float64)
        for comp in self.components:
            output_path = self.output_paths[comp].format(frequency=frequency, realisation=realisation, lmax = desired_lmax)
            if os.path.exists(output_path):
                # Certain foreground components are frequency-independent, so we can skip processing
                hp_map_reduced = hp.read_map(output_path)
            else:
                filepath = self.downloaded_data_filepaths[comp].format(frequency=frequency, realisation=realisation)
                hp_map = hp.read_map(filepath)
                hp_map = HPTools.unit_convert(hp_map, frequency)
                if comp == "noise" and self.noise:
                    hp_map_reduced = HPTools.reduce_hp_map_resolution(hp_map, lmax=desired_lmax, nside=nside)
                else:
                    hp_map_reduced = HPTools.convolve_and_reduce(hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad)
                if save:
                    save_map(hp_map_reduced, self.output_paths[comp].format(frequency=frequency, realisation=realisation, lmax=desired_lmax))
            cfn+= hp_map_reduced
        return cfn

    def produce_and_save_cfns(self):
        """
        Produce CFN maps across realisations and frequencies.

        Returns:
            None
        """
        desired_lmax = self.desired_lmax
        output_dir = self.output_directories["cfn"]
        create_dir(output_dir)
        create_dir(self.output_directories["processed_maps"])
        for realisation in range(self.realisations):
            for frequency in self.frequencies:
                cfn_output_path = self.output_paths["cfn"].format(frequency=frequency, realisation=realisation, lmax=desired_lmax)
                if os.path.exists(cfn_output_path):
                    print(f"CFN map at {frequency} GHz for realisation {realisation + 1} already exists. Skipping processing.")
                    continue
                cfn_map = self.create_cfn(frequency, realisation, desired_lmax, save=True)
                hp.write_map(cfn_output_path, cfn_map)
                print(f"CFN map at {frequency} GHz for realisation {realisation + 1} saved to {cfn_output_path}")


    def create_wavelet_transform(self, comp: str, frequency: str, realisation: int, N_directions: int = 1, lam: float = 2.0, method = "jax_cuda", visualise = False):
        """
        Create a wavelet transform of the specified component.

        Parameters:
            comp (str): The component to process (e.g., 'sync', 'noise').
            frequency (str): The frequency of the map.
            realisation (int): The realisation number.
            lmax (int): The maximum multipole for the wavelet transform.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float): lambda factor (scaling) for the wavelet transform.
            visualise (bool): Whether to visualise the wavelet transform.

        Returns:
            np.ndarray: The wavelet transformed map.
        """
        lmax = self.desired_lmax
        L = lmax + 1
        filepath = self.output_paths[comp].format(frequency=frequency, realisation=realisation, lmax=lmax) # input path
        wavelet_coeffs_output_dir = self.output_directories["wavelet_coeffs"]
        scaling_coeffs_output_dir = self.output_directories["scaling_coeffs"]
        create_dir(wavelet_coeffs_output_dir)
        create_dir(scaling_coeffs_output_dir)
        wavelet_coeffs_path = self.output_paths["wavelet_coeffs"]
        scaling_coeffs_path = self.output_paths["scaling_coeffs"]
        if os.path.exists(wavelet_coeffs_path.format(comp=comp, frequency=frequency, scale=0, realisation=realisation, lmax=lmax)):
            # test if scale 0 exists; this means the transform has already been created
            print(f"Wavelet coefficients for {comp} at {frequency} GHz for realisation {realisation + 1} already exist. Skipping generation.")
            return None
        hp_map = hp.read_map(filepath)
        mw_map = SamplingConverters.hp_map_2_mw_map(hp_map, lmax=lmax)
        wavelet_coeffs, scaling_coeffs = MWTools.wavelet_transform(mw_map, L=L, N_directions=N_directions, lam=lam)
        MWTools.save_wavelet_scaling_coeffs(wavelet_coeffs, scaling_coeffs, comp, frequency, realisation, lmax, wavelet_coeffs_path, scaling_coeffs_path)
        if visualise:
            MWTools.visualise_mw_map(mw_map, title=f"{comp} map at {frequency} GHz", coord=["G"], unit="K", method=method)
        return wavelet_coeffs, scaling_coeffs
    

    def produce_and_save_wavelet_transforms(self, comp: str, N_directions: int = 1, lam: float = 2.0, method = "jax_cuda", visualise = False):
        """
        Produce and save wavelet transforms for all components across realisations and frequencies.

        Parameters:
            lmax (int): The maximum multipole for the wavelet transform.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float): lambda factor (scaling) for the wavelet transform.
            method (str): Method to use for visualisation.
            visualise (bool): Whether to visualise the wavelet transform.

        Returns:
            None
        """
        lmax = self.desired_lmax
        for realisation in range(self.realisations):
            for frequency in self.frequencies:
                self.create_wavelet_transform(comp, frequency, realisation, lmax, N_directions=N_directions, lam=lam, method=method, visualise=visualise)
                print(f"Wavelet transform for {comp} at {frequency} GHz for realisation {realisation + 1} saved.")

processor = ProcessMaps(components=["cmb", "sync"], frequencies=["030", "044"], realisations=2, directory="data/")
processor.produce_and_save_cfns(desired_lmax=1024)
processor.produce_and_save_wavelet_transforms(comp="cfn", lmax=1024, N_directions=1, lam=4.0, method="jax_cuda", visualise=False)

