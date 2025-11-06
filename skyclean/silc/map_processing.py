import os, glob
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
                 file_templates: FileTemplates | None = None,
                 directory: str = "data/", 
                 method = "jax_cuda", 
                 overwrite: bool = False, 
                ): 
        """
        Parameters: 
            components (list): List of foreground components to process. Includes: 'sync', 'tsz', 'dust' 
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
        self.templates = FileTemplates(directory="/Scratch/agnes/data")

        files = FileTemplates(directory)
        self.file_templates = files.file_templates
        # file_template, file_templates
        

    def _find_max_noise_realisation(self, frequency: str):
        """
        Find the maximum noise realisation number available for a given frequency.

        Returns:
            int: Maximum noise realisation number found.

        Raises:
            FileNotFoundError: If no noise files are found.
            ValueError: If files exist but no realisation numbers can be parsed.
        """
        import glob
        cmb_dir = os.path.join(self.directory, "CMB_realisations")
        pattern = os.path.join(cmb_dir, f"noise_f{frequency}_r*.fits")
        noise_files = glob.glob(pattern)

        if not noise_files:
            print(f"Warning: No noise files found for frequency {frequency}.")
            raise FileNotFoundError(f"No noise files found for frequency {frequency}.")

        realisations = []
        for filepath in noise_files:
            filename = os.path.basename(filepath)
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
            print(f"Warning: Could not parse realisation numbers for frequency {frequency}.")
            raise ValueError(f"Could not parse realisation numbers for frequency {frequency}.")


    def create_cfn(self, frequency: str, realisation: int, save=True):
        """
        Create a CFN (Cmb + Foreground + Noise) for a given frequency and realisation, by convolving
        the CMB with the standard beam and 
            foregrounds with the standard beam + pixel window and 
            adding noise 

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
            processed_comp = "processed_" + comp
            output_path = self.file_templates[processed_comp].format(
                frequency=frequency, realisation=realisation, lmax=desired_lmax
            )
    
            if os.path.exists(output_path) and self.overwrite is False:
                hp_map_reduced = hp.read_map(output_path)
            else:
                if comp == "noise":
                    # Choose an existing noise realisation ID from disk
                    noise_dir = os.path.join(self.directory, "CMB_realisations")
                    pattern = os.path.join(noise_dir, f"noise_f{frequency}_r*.fits")
                    files = glob.glob(pattern)
                    available = []
                    for p in files:
                        try:
                            rid = int(os.path.basename(p).rsplit(".fits", 1)[0].rsplit("_r", 1)[1])
                            available.append(rid)
                        except Exception:
                            continue
                    if not available:
                        raise FileNotFoundError(
                            f"No noise files found for frequency '{frequency}' matching '{pattern}'."
                        )
                    noise_realisation = int(np.random.choice(sorted(set(available))))
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=noise_realisation
                    )
                else:
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
    
                hp_map = hp.read_map(filepath)
                if comp != "cmb":
                    hp_map = HPTools.unit_convert(hp_map, frequency)
                    print('1')

                if comp == "noise":
                    beam_path = None
                    if frequency not in {"030","044","070"}:
                        beam_path = self.templates.hfi_beam_path(frequency)
                        
                    hp_map_reduced = HPTools.deconvolve_and_convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, frequency=frequency,
                        standard_fwhm_rad=standard_fwhm_rad, beam_path=beam_path
                    )
                    print('2')
                else:
                    hp_map_reduced = HPTools.convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
                    )
                    print('3')

                if save:
                    save_map(output_path, hp_map_reduced, self.overwrite)
            cfn += hp_map_reduced
        return cfn

    
    def process_single_component(self, comp: str, frequency: str, realisation: int,
                             save: bool = True, noise_realisation: int | None = None):
        """
        Process ONE component (e.g. 'cmb', 'sync', 'dust', 'tsz', or 'noise') at a given
        frequency and realisation, applying unit conversion, beam smoothing (except noise),
        and band-limit reduction. Returns the processed map (HEALPix 1D array).

        Parameters:
            comp (str): One of {'cmb','sync','dust','tsz','noise'}.
            frequency (str): Frequency channel, e.g. '143'.
            realisation (int): Processing realisation index (used for filenames & CMB).
            save (bool): If True, writes to file_templates['processed_' + comp].
            noise_realisation (int | None): If comp=='noise', which MC (0.299) to use.
                                            If None, picks a random one (mirrors create_cfn).

        Returns:
            np.ndarray: Processed HEALPix map at target nside (from desired_lmax).
        """
        desired_lmax = self.desired_lmax
        nside = HPTools.get_nside_from_lmax(desired_lmax)
        standard_fwhm_rad = np.radians(5/60)

        processed_key = "processed_" + comp
        if processed_key not in self.file_templates:
            raise KeyError(f"Missing file_templates['{processed_key}'] for component '{comp}'.")

        out_path = self.file_templates[processed_key].format(
            frequency=frequency, realisation=realisation, lmax=desired_lmax
        )

        # Fast path: reuse if exists and not overwriting
        if (not self.overwrite) and os.path.exists(out_path):
            return hp.read_map(out_path)

        # Build input path
        if comp == "noise":
            if noise_realisation is None:
                noise_realisation = np.random.randint(self.start_realisation, self.start_realisation + self.realisations)
            in_path = self.file_templates["noise"].format(
                frequency=frequency, realisation=noise_realisation
            )
        else:
            in_path = self.file_templates[comp].format(
                frequency=frequency, realisation=realisation
            )

        # Load, convert units, and process
        hp_map = hp.read_map(in_path)
        if comp != "cmb":  
            hp_map = HPTools.unit_convert(hp_map, frequency)
        
        if comp == "noise":
            hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(
                hp_map, lmax=desired_lmax, nside=nside
            )
        else:
            hp_map_reduced = HPTools.convolve_and_reduce(
                hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
            )
            
        if save:
            save_map(out_path, hp_map_reduced, self.overwrite)
    
        return hp_map_reduced


    def produce_and_save_maps(self):
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


    def create_wavelet_transform(self, comp: str, frequency: str, realisation: int, N_directions: int = 1, 
                                 lam: float = 2.0, method = "jax_cuda", visualise = False):
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