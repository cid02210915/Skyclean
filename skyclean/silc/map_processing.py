import os, glob
import healpy as hp
from .file_templates import FileTemplates
from .map_tools import *
from .add_point_source import PointSource


def read_map_with_known_order(path: str, comp: str | None = None, frequency: str | int | None = None):
    # Only low-frequency noise maps are stored in NESTED ordering.
    freq = str(frequency).zfill(3) if frequency is not None else None
    nest = (comp == "noise") and (freq in {"030", "044", "070"})
    return hp.read_map(path, verbose=False, nest=nest)


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
                 ps_component: str = 'faintradiops',
                 n_points: int = 10,
                 lon_range: tuple = None,
                 lat_range: tuple = (20.0, 90.0),
                 brightness_percentile: tuple = (75.0, 100.0),
                 mode: str = "random",
                 random_seed: int = 1,
                 factor: int | float = 50.0,
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
        self.templates = FileTemplates(directory=directory)
        files = FileTemplates(directory)

        self.file_templates = files.file_templates
        self.download_templates = files.download_templates

        if 'extra_feature' in components:
            self.ps_component = ps_component
            self.n_points = n_points
            self.lon_range = lon_range
            self.lat_range = lat_range
            self.brightness_percentile = brightness_percentile
            self.mode = mode
            self.random_seed = random_seed
            self.factor = factor

            #self.pointsource = PointSource(directory=directory)
            self.ps_lon = None
            self.ps_lat = None
            self.ps_rad = None
            self.ps_val = None
            self.ps_sed_lists = None
            
    
    def build_point_sources(self):
        """
        Build a point-source catalogue (positions, sizes, amplitudes, and SEDs)

        Returns:
        lon_deg (array): 1D array of longitudes in degrees, shapeshape = (N,). N = number of features per frequency map.
        lat_deg (array): 1D array of latitudes in degrees, shape shape = (N,).
        rad_deg (array): 1D array of (possibly enlarged) equivalent radii in degrees, shape = (N,).
                                 If a size-enlargement factor is used, this corresponds to rad_deg_original * factor.
        val_ref (array): 1D array of summed component values in the reference frequency (usually the first frequency), shape = (N,).
        sed_lists (list): List of length N; each entry is a list of length N_freq containing per-frequency summed values in the order of the frequencies list.
        """
        ps = PointSource(
            ps_component=self.ps_component,
            frequencies=self.frequencies,
            n_points=self.n_points,
            lon_range=self.lon_range,
            lat_range=self.lat_range,
            brightness_percentile=self.brightness_percentile,
            mode=self.mode,
            random_seed=self.random_seed,
            factor=self.factor,
            directory=self.directory,
        )
        lon, lat, rad, val, sed_lists = ps.create_and_output_catalogue()
        self.ps_lon = lon
        self.ps_lat = lat
        self.ps_rad = rad
        self.ps_val = val
        self.ps_sed_lists = sed_lists
        return lon, lat, rad, val, sed_lists
    

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


    def add_filled_circle_hp(self, hp_map, n_side, center_lon_deg, center_lat_deg, radius_deg, value):
        """
        Add a filled circle to a HEALPix map (lon/lat in degrees).
        Returns a copy with the circle applied.
        """
        # healpy uses colatitude theta = 90 - lat
        print(f'center = ({center_lon_deg}, {center_lat_deg}), radius = {radius_deg}˚, value = {value} K')
        theta = np.deg2rad(90.0 - center_lat_deg)
        phi   = np.deg2rad(center_lon_deg)
        vec = hp.ang2vec(theta, phi)
        pix = hp.query_disc(n_side, vec, np.deg2rad(radius_deg))
        hp_map[pix] = value
        if pix.size == 0:
            print(f"Warning: No pixels found for circle at ({center_lon_deg}, {center_lat_deg}) with radius {radius_deg}˚.")
    

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

        if 'all' in self.components:
            # load all components in download_templates except for 'extra_feature', 'cib', and 'mask' 
            components = ['cmb']
            for comp in self.download_templates.keys():
                if comp == "extra_feature" or comp == "cib" or comp == "mask":
                    continue
                else:
                    components.append(comp)
            print(f"Loading all components for CFN creation: {components}")
        else: 
            components = self.components
        
        for comp in components:
         
            if comp == "cib" and frequency not in {"353", "545", "857"}:
                continue

            if comp != "extra_feature":
                processed_comp = "processed_" + comp

            output_path = self.file_templates[processed_comp].format(
                frequency=frequency, realisation=realisation, lmax=desired_lmax
            )
    
            if os.path.exists(output_path) and self.overwrite is False:
                hp_map_reduced = read_map_with_known_order(output_path)
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
                        frequency=frequency, realisation=realisation
                    )
                elif comp != "extra_feature":
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
    
                #hp_map = hp.read_map(filepath)
                hp_map = read_map_with_known_order(filepath, comp=comp, frequency=frequency)

                if comp == "cib":
                    hp_map = HPTools.unit_convert_cib(hp_map, frequency)
                elif comp != "cmb":
                    hp_map = HPTools.unit_convert(hp_map, frequency)
                
                if comp == "noise":
                    beam_path = None
                    if frequency not in {"030","044","070"}:
                        beam_path = self.templates.hfi_beam_path(frequency)
                        
                    hp_map_reduced = HPTools.deconvolve_and_convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, frequency=frequency,
                        standard_fwhm_rad=standard_fwhm_rad, beam_path=beam_path
                    )
                elif comp == "cib":
                    # CIB GNILC already has 5' arcmin beam:
                    # (https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/previews/COM_CompMap_CIB-GNILC-F353_2048_R2.00/header.txt)
                    # just remove pixel window and reduce NSIDE
                    hp_map_noP = HPTools.pixwin_deconvolve(hp_map, lmax=desired_lmax)
                    hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(
                        hp_map_noP, lmax=desired_lmax, nside=nside
                    )
                elif comp != "extra_feature":
                    hp_map_reduced = HPTools.convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
                    )
                    
                if save:
                    save_map(output_path, hp_map_reduced, self.overwrite)
            cfn += hp_map_reduced
        return cfn

    def create_cfn_with_extra_features(self, frequency: str, realisation: int, 
                                      center_lon_deg: list, center_lat_deg: list, radius_deg: list, value: list, sed_factors: np.array,
                                      save=True):
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
            center_lon_deg (float): Longitude of the center of the extra feature in degrees.
            center_lat_deg (float): Latitude of the center of the extra feature in degrees
            radius_deg (float): Radius of the extra feature in degrees.
            value (float): Temperature value to fill the extra feature with.
            sed_factors (array): Normalised SED factor at a specific frequency for n features. Shape = (n features,)
        
        Returns:
            np.ndarray: The CFN map in HP format.
        """
        desired_lmax = self.desired_lmax
        standard_fwhm_rad = np.radians(5/60)
        nside = HPTools.get_nside_from_lmax(desired_lmax)
        cfn = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    
        for comp in self.components:
         
            if comp == "cib" and frequency not in {"353", "545", "857"}:
                continue
            if comp != "extra_feature":
                processed_comp = "processed_" + comp
            output_path = self.file_templates[processed_comp].format(
                frequency=frequency, realisation=realisation, lmax=desired_lmax
            )
    
            if os.path.exists(output_path) and self.overwrite is False:
                hp_map_reduced = read_map_with_known_order(output_path)
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
                        frequency=frequency, realisation=realisation
                    )
                elif comp != "extra_feature":
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
    
                hp_map = read_map_with_known_order(filepath, comp=comp, frequency=frequency)

                if comp == "cib":
                    hp_map = HPTools.unit_convert_cib(hp_map, frequency)
                elif comp != "cmb":
                    hp_map = HPTools.unit_convert(hp_map, frequency)
                
                if comp == "noise":
                    beam_path = None
                    if frequency not in {"030","044","070"}:
                        beam_path = self.templates.hfi_beam_path(frequency)
                    hp_map_reduced = HPTools.deconvolve_and_convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, frequency=frequency,
                        standard_fwhm_rad=standard_fwhm_rad, beam_path=beam_path
                    )
                elif comp == "cib":
                    # CIB GNILC already has 5' arcmin beam:
                    # (https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/previews/COM_CompMap_CIB-GNILC-F353_2048_R2.00/header.txt)
                    # just remove pixel window and reduce NSIDE
                    hp_map_noP = HPTools.pixwin_deconvolve(hp_map, lmax=desired_lmax)
                    hp_map_reduced, _ = HPTools.reduce_hp_map_resolution(
                        hp_map_noP, lmax=desired_lmax, nside=nside
                    )
                elif comp == "extra_feature":
                    hp_map = np.zeros(hp.nside2npix(nside), dtype=np.float64)
                    for i in range(len(center_lon_deg)):
                        self.add_filled_circle_hp(hp_map, nside, center_lon_deg[i], center_lat_deg[i], radius_deg[i], value[i]*sed_factors[i])
                        hp_map_reduced = HPTools.convolve_and_reduce(
                            hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
                        )
                else: 
                    hp_map_reduced = HPTools.convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
                    )
                
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
            return read_map_with_known_order(out_path)

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
        hp_map = read_map_with_known_order(in_path, comp=comp, frequency=frequency)
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
        center_lon_deg (list): list for longitudgnal positions (in degree) of n features on a single frequency map. With shape = (n features,)

        sed (list): list for spectral dependence factor for each features on a map across different channels. Shape = (n features, m frequenceis)

        Returns:
            None
        """
        desired_lmax = self.desired_lmax
        frequencies = self.frequencies
        for realisation in range(self.realisations):
            i=0
            realisation += self.start_realisation  # Adjust for starting realisation
            for (i, frequency) in enumerate(frequencies):
                if 'extra_feature' in self.components:
                    cfn_output_path = self.file_templates["cfne"].format(frequency=frequency, realisation=realisation, lmax=desired_lmax)
                    if os.path.exists(cfn_output_path) and self.overwrite == False:
                        print(f"CFN map at {frequency} GHz for realisation {realisation} already exists. Skipping processing.")
                        continue
                    center_lon_deg, center_lat_deg, radius_deg, value, sed = self.build_point_sources()
                    print(f'Creating CFN with injected feature at {frequency} GHz...')
                    sed_array = np.array(sed)
                    cfn_map = self.create_cfn_with_extra_features(frequency, realisation, save=True, 
                                                                    center_lon_deg = center_lon_deg,
                                                                    center_lat_deg = center_lat_deg,
                                                                    radius_deg = radius_deg,
                                                                    value = value, 
                                                                    sed_factors = sed_array[:, i]/sed_array[:, 0])
                    # visualise the map to check
                    # hp.mollview(cfn_map, title=f"CFN @ {frequency} GHz with injected feature, realisation {realisation}")
                else:
                    cfn_output_path = self.file_templates["cfn"].format(frequency=frequency, realisation=realisation, lmax=desired_lmax)
                    if os.path.exists(cfn_output_path) and self.overwrite == False:
                        print(f"CFN map at {frequency} GHz for realisation {realisation} already exists. Skipping processing.")
                        continue
                    cfn_map = self.create_cfn(frequency, realisation, save=True)
                    # hp.mollview(cfn_map, title=f"CFN @ {frequency} GHz, realisation {realisation}")
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
        if os.path.exists(wavelet_coeffs_path.format(comp=comp, frequency=frequency, scale=0, 
                                                     realisation=realisation, lmax=lmax, lam = lam)) and self.overwrite == False:
            # test if scale 0 exists; this means the transform has already been created
            print(f"Wavelet coefficients for {comp} at {frequency} GHz for realisation {realisation} already exist. Skipping generation.")
            return None
        hp_map = read_map_with_known_order(filepath)
        L = lmax + 1
        mw_map = SamplingConverters.hp_map_2_mw_map(hp_map, lmax=lmax, method = method)
        MWTools.visualise_mw_map(mw_map, title=f"{comp}", directional = False)
        wavelet_coeffs, scaling_coeffs = MWTools.wavelet_transform_from_map(mw_map, L=L, N_directions=N_directions, lam=lam)
        MWTools.save_wavelet_scaling_coeffs(wavelet_coeffs, scaling_coeffs, comp, frequency, 
                                            realisation, lmax, lam, wavelet_coeffs_path, scaling_coeffs_path)
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
