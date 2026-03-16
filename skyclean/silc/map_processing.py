import os, glob
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from .file_templates import FileTemplates
from .map_tools import *
from .add_point_source import (CircularPSInjector, PixelPSInjector, PointSource, measure_source_values)


def read_map_with_known_order(path: str, comp: str | None = None, frequency: str | int | None = None):
    # Only low-frequency noise maps are stored in NESTED ordering.
    freq = str(frequency).zfill(3) if frequency is not None else None
    nest = (comp == "noise") and (freq in {"030", "044", "070"})
    # Explicitly check for file existence and non-emptiness before attempting to read
    if not os.path.exists(path):
        raise FileNotFoundError(f"Map file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size == 0:
        detail = f" (component={comp}, frequency={freq})" if (comp is not None or freq is not None) else ""
        raise OSError(
            f"Empty map file detected at '{path}'{detail}. "
            "Delete/redownload this file and rerun."
        )
    try:
        return hp.read_map(path, verbose=False, nest=nest)
    except Exception as exc:
        detail = f"component={comp}, frequency={freq}, nest={nest}, size={file_size} bytes"
        raise OSError(f"Failed to read map '{path}' ({detail}): {exc}") from exc


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
                 ps_component: str = 'strongirps',
                 n_points: int = 10,
                 lon_range: tuple = None,
                 lat_range: tuple = (-90.0, 90.0), # fixed for pipeline
                 brightness_percentile: tuple = (75.0, 100.0),
                 mode: str = "random",
                 random_seed: int = 1,
                 ps_radius_range: tuple = (1.0, 1.0),
                 ps_brightness_scale: float = 1.0,
                 ps_injection_mode: str = "circular_ps",
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
        self.ps = PointSource(
                    ps_component=ps_component,
                    frequencies=frequencies,
                    n_points=n_points,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    brightness_percentile=brightness_percentile,
                    mode=mode,
                    random_seed=random_seed,
                    ps_radius_range=ps_radius_range,
                    ps_brightness_scale=ps_brightness_scale,
                    directory=directory,
                )

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
            self.ps_radius_range = tuple(ps_radius_range)
            self.ps_brightness_scale = float(ps_brightness_scale)
            self.ps_injection_mode = str(ps_injection_mode)


    def _repair_and_read_map(self, path: str, comp: str, frequency: str, realisation: int):
        """
        Attempt to read a map; if invalid/missing, regenerate or redownload it once, then retry.
        """
        try:
            return read_map_with_known_order(path, comp=comp, frequency=frequency)
        except (FileNotFoundError, OSError) as exc:
            print(
                f"Map read failed for component='{comp}', frequency='{frequency}', "
                f"realisation={realisation}: {exc}. Attempting one repair."
            )
            try:
                from .download import DownloadData
                downloader = DownloadData(
                    components=[comp],
                    frequencies=[frequency],
                    realisations=1,
                    start_realisation=realisation,
                    directory=self.directory,
                )
                if comp == "cmb":
                    downloader.generate_and_save_cmb_realisation(realisation)
                elif comp == "noise":
                    downloader.download_foreground_component("noise", frequency, realisation)
                elif comp != "extra_feature":
                    downloader.download_foreground_component(comp, frequency)
                else:
                    raise RuntimeError("Component 'extra_feature' has no source file to repair.")
                return read_map_with_known_order(path, comp=comp, frequency=frequency)
            except Exception as repair_exc:
                raise OSError(
                    f"Failed to repair map '{path}' for component='{comp}', "
                    f"frequency='{frequency}', realisation={realisation}: {repair_exc}"
                ) from repair_exc
            
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

    @staticmethod
    def _co_is_missing_channel(comp: str, frequency: str | int) -> bool:
        """CO is absent in FFP10 at 30/44/70 GHz; treat as exact zero."""
        return (comp == "co") and (str(frequency).zfill(3) in {"030", "044", "070", "217"})
    

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
    
            reuse_processed = os.path.exists(output_path) and self.overwrite is False
            if reuse_processed:
                try:
                    hp_map_reduced = read_map_with_known_order(output_path)
                except (FileNotFoundError, OSError) as exc:
                    print(f"Processed map '{output_path}' is invalid ({exc}). Regenerating from source map.")
                    reuse_processed = False

            if not reuse_processed:
                if self._co_is_missing_channel(comp, frequency):
                    hp_map_reduced = np.zeros(hp.nside2npix(nside), dtype=np.float64)
                    if save:
                        save_map(output_path, hp_map_reduced, self.overwrite)
                    print(f"Using zero CO map at {frequency} GHz (no CO signal for this channel).")
                    cfn += hp_map_reduced
                    continue

                if comp == "noise":
                    # Choose an existing noise realisation ID from disk
                    noise_dir = os.path.join(self.directory, "CMB_realisations")
                    pattern = os.path.join(noise_dir, f"noise_f{frequency}_r*.fits")
                    files = glob.glob(pattern)
                    available = []
                    for p in files:
                        try:
                            if os.path.getsize(p) == 0:
                                continue
                            rid = int(os.path.basename(p).rsplit(".fits", 1)[0].rsplit("_r", 1)[1])
                            available.append(rid)
                        except Exception:
                            continue
                    if not available:
                        raise FileNotFoundError(
                            f"No valid non-empty noise files found for frequency '{frequency}' matching '{pattern}'."
                        )
                    # noise_realisation = int(np.random.choice(sorted(set(available))))
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
                elif comp != "extra_feature":
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
    
                if comp != "extra_feature":
                    hp_map = self._repair_and_read_map(filepath, comp=comp, frequency=frequency, realisation=realisation)

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
            print(f"Added component '{comp}' to CFN at {frequency} GHz (realisation {realisation}).")
            print(f"Shape of component '{comp}': {hp_map_reduced.shape}")
            hp.mollview(
                hp_map_reduced,
                title=f"Component '{comp}' @ {frequency} GHz, r{realisation} (before CFN sum)",
            )
            plt.show()
            cfn += hp_map_reduced
        return cfn

    def create_cfn_with_extra_features(self, frequency: str, realisation: int, 
                                      center_lon_deg: list | None = None, center_lat_deg: list | None = None, radius_deg: list | None = None,
                                      value: list | None = None, sed_factors: np.ndarray | None = None,
                                      extra_feature_map: np.ndarray | None = None,
                                      save=True,
                                      return_components: bool = True):
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
        processed_component_maps = {} if return_components else None
    
        for comp in self.components:
         
            if comp == "cib" and frequency not in {"353", "545", "857"}:
                continue
            if comp != "extra_feature":
                processed_comp = "processed_" + comp
                output_path = self.file_templates[processed_comp].format(
                    frequency=frequency, realisation=realisation, lmax=desired_lmax
                )
            else:
                output_path = None
    
            reuse_processed = (output_path is not None) and os.path.exists(output_path) and self.overwrite is False
            if reuse_processed:
                try:
                    hp_map_reduced = read_map_with_known_order(output_path)
                except (FileNotFoundError, OSError) as exc:
                    print(f"Processed map '{output_path}' is invalid ({exc}). Regenerating from source map.")
                    reuse_processed = False

            if not reuse_processed:
                if self._co_is_missing_channel(comp, frequency):
                    hp_map_reduced = np.zeros(hp.nside2npix(nside), dtype=np.float64)
                    if save:
                        save_map(output_path, hp_map_reduced, self.overwrite)
                    print(f"Using zero CO map at {frequency} GHz (no CO signal for this channel).")
                    cfn += hp_map_reduced
                    continue

                if comp == "noise":
                    # Choose an existing noise realisation ID from disk
                    noise_dir = os.path.join(self.directory, "CMB_realisations")
                    pattern = os.path.join(noise_dir, f"noise_f{frequency}_r*.fits")
                    files = glob.glob(pattern)
                    available = []
                    for p in files:
                        try:
                            if os.path.getsize(p) == 0:
                                continue
                            rid = int(os.path.basename(p).rsplit(".fits", 1)[0].rsplit("_r", 1)[1])
                            available.append(rid)
                        except Exception:
                            continue
                    if not available:
                        raise FileNotFoundError(
                            f"No valid non-empty noise files found for frequency '{frequency}' matching '{pattern}'."
                        )
                    #noise_realisation = int(np.random.choice(sorted(set(available))))
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
                elif comp != "extra_feature":
                    filepath = self.file_templates[comp].format(
                        frequency=frequency, realisation=realisation
                    )
    
                if comp != "extra_feature":
                    hp_map = self._repair_and_read_map(filepath, comp=comp, frequency=frequency, realisation=realisation)

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
                    if extra_feature_map is not None:
                        hp_map = np.asarray(extra_feature_map, dtype=np.float64) * self.ps_brightness_scale
                        if self.ps_injection_mode == "pixel_ps":
                            hp_map_reduced = HPTools.beam_convolve(hp_map, lmax=desired_lmax, standard_fwhm_rad=standard_fwhm_rad)
                        else: # "circular_ps"
                            hp_map_reduced = HPTools.convolve_and_reduce(hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad)
                    else:
                        # Fallback to the original uniform-disc injection if no real source map is supplied.
                        hp_map = np.zeros(hp.nside2npix(nside), dtype=np.float64)
                        if center_lon_deg is None or center_lat_deg is None or radius_deg is None or value is None or sed_factors is None:
                            raise ValueError(
                                "extra_feature_map is None and circle-parameter fallback is incomplete."
                            )
                        for i in range(len(center_lon_deg)):
                            self.add_filled_circle_hp(
                                hp_map,
                                nside,
                                center_lon_deg[i],
                                center_lat_deg[i],
                                radius_deg[i],
                                self.ps_brightness_scale * value[i] * sed_factors[i],
                            )
                        hp_map_reduced = HPTools.convolve_and_reduce(
                            hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
                        )
                else: 
                    hp_map_reduced = HPTools.convolve_and_reduce(
                        hp_map, lmax=desired_lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
                    )
                
                if save and output_path is not None:
                    save_map(output_path, hp_map_reduced, self.overwrite)
            if return_components and comp == "extra_feature":
                # Only expose the processed extra-feature map (not other components).
                processed_component_maps["extra_feature"] = np.asarray(hp_map_reduced, dtype=np.float64)
            hp.mollview(
                hp_map_reduced,
                title=f"Component '{comp}' @ {frequency} GHz, r{realisation} (before CFN sum)",
            )
            plt.show()
            print(f"Added component '{comp}' to CFN at {frequency} GHz (realisation {realisation}).")
            cfn += hp_map_reduced
        if return_components:
            return cfn, processed_component_maps
        return cfn

    def _load_component_map_in_K(self, comp: str, frequency: str, realisation: int) -> np.ndarray:
        """
        Load a raw component map in K units.
        """
        path = self.file_templates[comp].format(frequency=frequency, realisation=realisation)
        m = self._repair_and_read_map(path, comp=comp, frequency=frequency, realisation=realisation)
        if comp == "cib":
            return HPTools.unit_convert_cib(m, frequency)
        if comp != "cmb":
            return HPTools.unit_convert(m, frequency)
        return m
    
    
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
            try:
                return read_map_with_known_order(out_path)
            except (FileNotFoundError, OSError) as exc:
                print(f"Processed map '{out_path}' is invalid ({exc}). Regenerating from source map.")

        if self._co_is_missing_channel(comp, frequency):
            hp_map_reduced = np.zeros(hp.nside2npix(nside), dtype=np.float64)
            if save:
                save_map(out_path, hp_map_reduced, self.overwrite)
            return hp_map_reduced

        # Build input path
        if comp == "noise":
            if noise_realisation is None:
                noise_dir = os.path.join(self.directory, "CMB_realisations")
                pattern = os.path.join(noise_dir, f"noise_f{frequency}_r*.fits")
                files = glob.glob(pattern)
                available = []
                for p in files:
                    try:
                        if os.path.getsize(p) == 0:
                            continue
                        rid = int(os.path.basename(p).rsplit(".fits", 1)[0].rsplit("_r", 1)[1])
                        available.append(rid)
                    except Exception:
                        continue
                if not available:
                    raise FileNotFoundError(
                        f"No valid non-empty noise files found for frequency '{frequency}' matching '{pattern}'."
                    )
                noise_realisation = int(np.random.choice(sorted(set(available))))
            in_path = self.file_templates["noise"].format(
                frequency=frequency, realisation=noise_realisation
            )
        else:
            in_path = self.file_templates[comp].format(
                frequency=frequency, realisation=realisation
            )

        # Load, convert units, and process
        hp_map = self._repair_and_read_map(in_path, comp=comp, frequency=frequency, realisation=realisation)
        # extra_feature is already prepared in converted units by add_point_source;
        # do not re-apply unit conversion here.
        if comp not in {"cmb", "extra_feature"}:
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
        target_nside = HPTools.get_nside_from_lmax(desired_lmax)
        output_key = "cfne_circ" if (
            'extra_feature' in self.components and self.ps_injection_mode == "circular_ps"
        ) else ("cfne" if 'extra_feature' in self.components else "cfn")
        for realisation in range(self.realisations):
            i=0
            realisation += self.start_realisation  # Adjust for starting realisation
            if 'extra_feature' in self.components:
                center_theta, center_phi, radius_deg, sed_lists = self.ps.create_and_output_catalogue()
                center_pix = hp.ang2pix(target_nside, center_theta, center_phi).astype(np.int64)
                center_pixel_sed = np.full((len(center_theta), len(frequencies)), np.nan, dtype=np.float64)
                if self.ps_injection_mode == "pixel_ps":
                    pixel_injector = PixelPSInjector(
                        components=self.components,
                        frequencies=self.frequencies,
                        ps_component=self.ps_component,
                        desired_lmax=self.desired_lmax,
                        map_loader=self._load_component_map_in_K,
                    )
                    extra_feature_maps_by_freq, sed_matrix = pixel_injector.build_maps(
                        realisation=realisation,
                        center_theta=center_theta,
                        center_phi=center_phi,
                    )
                else:
                    circular_injector = CircularPSInjector(
                        ps_component=self.ps_component,
                        map_loader=self._load_component_map_in_K,
                    )
                    extra_feature_maps_by_freq = None
                    sed_matrix = np.zeros((len(center_theta), len(frequencies)), dtype=np.float64)
            for (i, frequency) in enumerate(frequencies):
                if 'extra_feature' in self.components:
                    cfn_output_path = self.file_templates[output_key].format(frequency=frequency, realisation=realisation, lmax=desired_lmax)
                    if os.path.exists(cfn_output_path) and self.overwrite == False:
                        print(f"{output_key.upper()} map at {frequency} GHz for realisation {realisation} already exists. Skipping processing.")
                        continue
                    print(f"Creating {output_key.upper()} with injected feature at {frequency} GHz...")
                    if self.ps_injection_mode == "pixel_ps":
                        extra_feature_map = extra_feature_maps_by_freq[frequency]
                    else:
                        source_values = np.asarray([sed[i] for sed in sed_lists], dtype=np.float64)
                        extra_feature_map = circular_injector.build_map(
                            frequency=frequency,
                            realisation=realisation,
                            center_theta=center_theta,
                            center_phi=center_phi,
                            radius_deg=radius_deg,
                            values=source_values,
                        )
                    cfn_map, processed_maps = self.create_cfn_with_extra_features(
                        frequency,
                        realisation,
                        save=True,
                        extra_feature_map=extra_feature_map,
                        return_components=True,
                    )
                    # Extract SED after beam convolution and resolution reduction
                    if "extra_feature" in processed_maps:
                        center_pixel_sed[:, i] = processed_maps["extra_feature"][center_pix]
                    if self.ps_injection_mode == "pixel_ps":
                        sed_matrix[:, i] = measure_source_values(
                            extra_feature_map,
                            center_theta=center_theta,
                            center_phi=center_phi,
                        )
                    if self.ps_injection_mode != "pixel_ps" and "extra_feature" in processed_maps:
                        sed_matrix[:, i] = measure_source_values(
                            processed_maps["extra_feature"],
                            center_theta=center_theta,
                            center_phi=center_phi,
                        )
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
            if 'extra_feature' in self.components:
                sed_path = os.path.join(
                    self.directory,
                    "processed_maps",
                    f"processed_extra_feature_sed_{output_key}_r{realisation:04d}_lmax{desired_lmax}.npz",
                )
                np.savez(
                    sed_path,
                    frequencies=np.array(frequencies),
                    theta=np.asarray(center_theta, dtype=np.float64),
                    phi=np.asarray(center_phi, dtype=np.float64),
                    sed=sed_matrix,
                    center_pixel_sed=center_pixel_sed,
                )
                print(f"Saved processed extra-feature per-source SED to {sed_path}")
                if center_pixel_sed.size > 0 and np.isfinite(center_pixel_sed).any():
                    ref_vals = center_pixel_sed[:, [0]]
                    center_pixel_sed_norm = np.divide(
                        center_pixel_sed,
                        ref_vals,
                        out=np.full_like(center_pixel_sed, np.nan, dtype=np.float64),
                        where=np.isfinite(ref_vals) & (ref_vals != 0),
                    )
                    n_plot = min(5, center_pixel_sed_norm.shape[0])
                    freq_axis = np.asarray([int(f) for f in frequencies], dtype=np.int64)
                    plt.figure(figsize=(8, 5))
                    for src_idx in range(n_plot):
                        plt.plot(freq_axis, center_pixel_sed_norm[src_idx], marker="o", label=(f"PS {src_idx} "
                                f"(theta={center_theta[src_idx]:.2f}, phi={center_phi[src_idx]:.2f})"))
                    plt.axhline(1.0, color="k", linewidth=1.0, linestyle="--")
                    plt.xlabel("Frequency [GHz]")
                    plt.ylabel(f"Center-pixel SED / {frequencies[0]} GHz")
                    plt.title(f"Processed center-pixel SEDs ({output_key}, first {n_plot} PS, r{realisation:04d})")
                    plt.legend(fontsize=8)
                    plt.tight_layout()
                    sed_plot_path = os.path.join(
                        self.directory,
                        "processed_maps",
                        f"processed_extra_feature_center_pixel_sed_{output_key}_r{realisation:04d}_lmax{desired_lmax}.png",
                    )
                    plt.savefig(sed_plot_path, dpi=150, bbox_inches="tight")
                    plt.show()
                    plt.close()
                    print(f"Saved processed center-pixel SED plot to {sed_plot_path}")


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
