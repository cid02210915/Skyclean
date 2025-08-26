import os
import healpy as hp
from .file_templates import FileTemplates
from .map_tools import *


class ProcessMaps():
    """Process downloaded maps."""
    def __init__(self, 
                 components: list, 
                 wavelet_components: list, 
                 frequencies: list, 
                 realisations: int,
                 start_realisation: int,
                 desired_lmax:int, 
                 directory: str = "data/", 
                 method = "jax_cuda", 
                 overwrite: bool = False): 
        """
        Parameters: 
            components (list): List of foreground components to process. Includes: 'sync' (synchrotron)
            wavelet_components (list): List of components to produce wavelet transforms for.
            directory (str): Directory where data is stored / saved to.
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            start_realisation (int): Starting realisation number for processing.
            desired_lmax (int): Desired maximum multipole for the processed maps.
            lam (int): lambda factor (scaling) for the wavelet transform.
            method (str): s2fft method
            overwrite (bool): Whether to overwrite existing files.
        """
        self.components = components
        self.wavelet_components = wavelet_components
        self.frequencies = frequencies
        self.realisations = realisations
        self.start_realisation = start_realisation
        self.method = method
        self.desired_lmax = desired_lmax
        self.directory = directory
        self.overwrite = overwrite

        files = FileTemplates(directory)
        self.file_templates = files.file_templates
        

        # file_template, file_templates

    def _find_max_noise_realisation(self, frequency: str):
        """
        Find the maximum noise realisation number available for a given frequency.
        
        Parameters:
            frequency (str): The frequency to check for noise files.
            
        Returns:
            int: Maximum noise realisation number found, or 299 as fallback.
        """
        import glob
        # Get the CMB realisations directory
        cmb_dir = os.path.join(self.directory, "CMB_realisations")
        
        # Pattern to match noise files for this frequency
        pattern = os.path.join(cmb_dir, f"noise_f{frequency}_r*.fits")
        noise_files = glob.glob(pattern)
        
        if not noise_files:
            print(f"Warning: No noise files found for frequency {frequency}. Using fallback max of 299.")
            return 299
            
        # Extract realisation numbers from filenames
        realisations = []
        for filepath in noise_files:
            filename = os.path.basename(filepath)
            # Extract number between 'r' and '.fits'
            # Format: noise_f{frequency}_r{realisation:05d}.fits
            try:
                realisation_str = filename.split('_r')[1].split('.fits')[0]
                realisations.append(int(realisation_str))
            except (IndexError, ValueError):
                continue
                
        if realisations:
            max_realisation = max(realisations)
            print(f"Found {len(realisations)} noise files for frequency {frequency}, max realisation: {max_realisation}")
            return max_realisation
        else:
            print(f"Warning: Could not parse realisation numbers for frequency {frequency}. Using fallback max of 299.")
            return 299

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
        desired_lmax = self.desired_lmax
        standard_fwhm_rad = np.radians(5/60)
        nside = HPTools.get_nside_from_lmax(desired_lmax)
        cfn = np.zeros(hp.nside2npix(nside), dtype=np.float64)
        for comp in self.components:
            # make 'comp' label 'processed_comp' for dictionary parsing
            processed_comp = "processed_" + comp
            output_path = self.file_templates[processed_comp].format(frequency=frequency, realisation=realisation, lmax = desired_lmax)
            if os.path.exists(output_path) and self.overwrite == False:
                # Certain foreground components are frequency-independent, so we can skip processing
                hp_map_reduced = hp.read_map(output_path)
            else:
                if comp == "noise": 
                    # Find the maximum noise realisation number from downloaded files
                    max_noise_realisation = self._find_max_noise_realisation(frequency)
                    print(f'max noise: {max_noise_realisation}')
                    noise_realisation = np.random.randint(0, max_noise_realisation + 1)
                    filepath = self.file_templates[comp].format(frequency=frequency, realisation=noise_realisation)
                else:
                    filepath = self.file_templates[comp].format(frequency=frequency, realisation=realisation)
                hp_map = hp.read_map(filepath)
                hp_map = HPTools.unit_convert(hp_map, frequency)
                if comp == "noise":
                    hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(hp_map, lmax=desired_lmax, nside=nside)
                else:
                    hp_map_reduced = HPTools.convolve_and_reduce(hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad)
                if save:
                    save_map(output_path, hp_map_reduced, self.overwrite)
            cfn+= hp_map_reduced
        return cfn

    def produce_and_save_cfns(self):
        """
        Produce CFN maps across realisations and frequencies.

        Returns:
            None
        """
        desired_lmax = self.desired_lmax
        for realisation in range(self.realisations):
            realisation += self.start_realisation  # Adjust for starting realisation
            for frequency in self.frequencies:
                cfn_output_path = self.file_templates["cfn"].format(frequency=frequency, realisation=realisation, lmax=desired_lmax)
                if os.path.exists(cfn_output_path) and self.overwrite == False:
                    print(f"CFN map at {frequency} GHz for realisation {realisation} already exists. Skipping processing.")
                    continue
                cfn_map = self.create_cfn(frequency, realisation, save=True)
                hp.write_map(cfn_output_path, cfn_map, overwrite=True)
                print(f"CFN map at {frequency} GHz for realisation {realisation} saved to {cfn_output_path}")


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
        # load in processed map
        filepath = self.file_templates[comp].format(frequency=frequency, realisation=realisation, lmax=lmax, lam = lam) # input path
        wavelet_coeffs_path = self.file_templates["wavelet_coeffs"]
        scaling_coeffs_path = self.file_templates["scaling_coeffs"]
        if os.path.exists(wavelet_coeffs_path.format(comp=comp, frequency=frequency, scale=0, realisation=realisation, lmax=lmax, lam = lam)) and self.overwrite == False:
            # test if scale 0 exists; this means the transform has already been created
            print(f"Wavelet coefficients for {comp} at {frequency} GHz for realisation {realisation} already exist. Skipping generation.")
            return None
        hp_map = hp.read_map(filepath)
        L = lmax + 1
        mw_map = SamplingConverters.hp_map_2_mw_map(hp_map, lmax=lmax, method = method)
        MWTools.visualise_mw_map(mw_map, title=f"{comp}", directional = False)
        wavelet_coeffs, scaling_coeffs = MWTools.wavelet_transform_from_map(mw_map, L=L, N_directions=N_directions, lam=lam)
        MWTools.save_wavelet_scaling_coeffs(wavelet_coeffs, scaling_coeffs, comp, frequency, realisation, lmax, lam, wavelet_coeffs_path, scaling_coeffs_path)
        return wavelet_coeffs, scaling_coeffs
    

    def produce_and_save_wavelet_transforms(self, N_directions: int = 1, lam: float = 2.0, method = "jax_cuda", visualise = False):
        """
        Produce and save wavelet transforms for all components across realisations and frequencies.

        Parameters:
            N_directions (int): Number of directions for the wavelet transform.
            lam (float): lambda factor (scaling) for the wavelet transform.
            method (str): Method to use for visualisation.
            visualise (bool): Whether to visualise the wavelet transform.

        Returns:
            None
        """
        lmax = self.desired_lmax
        for comp in self.wavelet_components:
            for realisation in range(self.realisations):
                realisation += self.start_realisation
                for frequency in self.frequencies:
                    self.create_wavelet_transform(comp, frequency, realisation, N_directions=N_directions, lam=lam, method=method, visualise=visualise)
                    print(f"Wavelet transform for {comp} at {frequency} GHz for realisation {realisation} saved.")

# processor = ProcessMaps(components=["cmb", "sync"], wavelet_components=["cfn"], frequencies = ["030", "044"], realisations=0, desired_lmax=256, directory="/Scratch/matthew/data/", overwrite=False)
# processor.produce_and_save_cfns()
# processor.produce_and_save_wavelet_transforms(N_directions=1, lam=4.0, method="jax_cuda", visualise=False)

