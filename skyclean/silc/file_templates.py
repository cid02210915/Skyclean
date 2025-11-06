import os 

class FileTemplates():

    # --- HFI beam filenames live at the data root, HFI_RIMO_BEAMS_R3.01.TAR.GZ from https://pla.esac.esa.int/#docsw ---
    HFI_BEAM_FILE = {
        "100": "Bl_T_R3.01_fullsky_100x100.fits",
        "143": "Bl_T_R3.01_fullsky_143x143.fits",
        "217": "Bl_T_R3.01_fullsky_217x217.fits",
        "353": "Bl_T_R3.01_fullsky_353x353.fits",
        "545": "Bl_T_R3.01_fullsky_545x545.fits",
        "857": "Bl_T_R3.01_fullsky_857x857.fits",
    }

    def __init__(self, directory = "data/", start_realisation: int = 0):
        self.directory = directory
        self.output_directories = {
            # downloaded maps
            "cmb_realisations": os.path.join(directory, "CMB_realisations/"),
            # processed maps
            "cfn": os.path.join(directory, "CFN_realisations"),
            "processed_maps": os.path.join(directory, "processed_maps"),
            # wavelet transforms
            "wavelet_coeffs": os.path.join(directory, "wavelet_transforms/wavelet_coeffs"),
            "scaling_coeffs": os.path.join(directory, "wavelet_transforms/scaling_coeffs"),
            "f_scal": os.path.join(directory, "SILC/f_scal"),
            # ilc maps
            "doubled_maps": os.path.join(directory, "SILC/doubled_maps"),
            "covariance_matrix": os.path.join(directory, "SILC/covariance_matrix"),
            "weight_vector_data": os.path.join(directory, "SILC/weight_vector_data"),
            "ilc_doubled_wavelet_maps": os.path.join(directory, "SILC/ilc_doubled_wavelet_maps"),
            "ilc_trimmed_maps": os.path.join(directory, "SILC/ilc_trimmed_maps"),
            "ilc_synthesised_maps": os.path.join(directory, "SILC/ilc_synthesised_maps"),
            # ML 
            "ml_maps": os.path.join(directory, "ML/maps"),
            "ml_models": os.path.join(directory, "ML/models"),
            "ilc_improved_maps": os.path.join(directory, "SILC/ilc_improved_maps")
        }

        for key, value in self.output_directories.items():
            if not os.path.exists(value):
                print(f"Creating directory: {value}")
                os.makedirs(value)
        
        self.download_templates = {
            "sync": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_synchrotron-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits",
            "dust": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_thermaldust-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits",
            "noise": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_{frequency}_full_map_mc_{realisation:05d}.fits",
            'tsz': "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_thermalsz-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits"
        }

        self.file_templates = {
        # ---------------- downloaded maps ----------------
        "cmb":   os.path.join(self.output_directories["cmb_realisations"], "cmb_r{realisation:04d}.fits"),
        "sync":  os.path.join(self.output_directories["cmb_realisations"], "sync_f{frequency}.fits"),
        "dust":  os.path.join(self.output_directories["cmb_realisations"], "dust_f{frequency}.fits"),
        "noise": os.path.join(self.output_directories["cmb_realisations"], "noise_f{frequency}_r{realisation:05d}.fits"),
        "tsz":   os.path.join(self.output_directories["cmb_realisations"], "tsz_f{frequency}.fits"),

        # ---------------- processed maps ----------------
        "processed_cmb":   os.path.join(self.output_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
        "processed_sync":  os.path.join(self.output_directories["processed_maps"], "processed_sync_f{frequency}_lmax{lmax}.npy"),
        "processed_dust":  os.path.join(self.output_directories["processed_maps"], "processed_dust_f{frequency}_lmax{lmax}.npy"),
        "processed_noise": os.path.join(self.output_directories["processed_maps"], "processed_noise_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
        "processed_tsz":   os.path.join(self.output_directories["processed_maps"], "processed_tsz_f{frequency}_lmax{lmax}.npy"),
        "cfn":             os.path.join(self.output_directories["cfn"],           "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),

        # ---------------- wavelet transforms ----------------
        "wavelet_coeffs": os.path.join(
            self.output_directories["wavelet_coeffs"],
            "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_lam{lam}.npy"
        ),
        # Alias for older code that expects 'wavelet_c_j'
        "wavelet_c_j": os.path.join(
            self.output_directories["wavelet_coeffs"],
            "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_lam{lam}.npy"
        ),
        "scaling_coeffs": os.path.join(
            self.output_directories["scaling_coeffs"],
            "{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}_lam{lam}.npy"
        ),
        "f_scal": os.path.join(
            self.output_directories["f_scal"],
            "f_scal_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # ---------------- ILC intermediates & outputs (matches ILC_wav_coeff_maps_MP) ----------------
        # IMPORTANT: these use {component}, {extract_comp}, and US spelling {realisation}
        # Per-frequency, per-scale doubled wavelet maps (still per input component)
        "doubled_maps": os.path.join(
            self.output_directories["doubled_maps"],
            "doubled_{component}_f{frequency}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # Covariance matrices per scale over the whole band-set (frequencies join tag, e.g. 30_44_70_...)
        "covariance_matrices": os.path.join(
            self.output_directories["covariance_matrix"],
            "cov_MW_{component}_f{frequencies}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # Weights per scale; {type} is "weight_vector" (or "cilc_cmb" for constrained case)
        "weight_vector_matrices": os.path.join(
            self.output_directories["weight_vector_data"],
            "{component}_{type}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # Per-scale ILC maps at doubled resolution (function expects key 'ilc_maps')
        "ilc_maps": os.path.join(
            self.output_directories["ilc_doubled_wavelet_maps"],
            "ilc_doubled_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # Legacy alias (same path)
        "ilc_doubled_maps": os.path.join(
            self.output_directories["ilc_doubled_wavelet_maps"],
            "ilc_doubled_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),
        
        # Per-scale maps trimmed back to original resolution (function expects key 'trimmed_maps')
        "trimmed_maps": os.path.join(
            self.output_directories["ilc_trimmed_maps"],
            "ilc_trimmed_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),
        

        # Final synthesized map — records target (extract_comp), source (component), and band-set
        "ilc_synth": os.path.join(
            self.output_directories["ilc_synthesised_maps"],
            "{extract_comp}_from-{component}_f{frequencies}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # Optional: power spectrum
        "ilc_spectrum": os.path.join(
            self.output_directories["ilc_synthesised_maps"],
            "{extract_comp}_from-{component}_spectrum_f{frequencies}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"
        ),

        # ---------------- ML (left unchanged) ----------------
        "foreground_estimate": os.path.join(self.output_directories["ml_maps"], "foreground_estimate_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
        "ilc_residual":       os.path.join(self.output_directories["ml_maps"], "ilc_residual_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
        "ilc_mwss":           os.path.join(self.output_directories["ml_maps"], "ilc_mwss_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
        }

    def hfi_beam_path(self, frequency: str) -> str:
        return os.path.join(self.directory, self.HFI_BEAM_FILE[str(frequency)])
     
    @staticmethod
    def create_dir(directory: str):
        """
        Create a directory if it does not exist.

        Parameters:
            dir (str): The path of the directory to create.

        Returns:
            None
        """
        if not os.path.exists(directory):
            print("Creating directory:", directory)
            os.makedirs(directory)
        else:
            pass

    def print_one_example_per_type(self):
        """
        Print a single example file (if any) from each managed output directory.
        """
        base = os.path.abspath(self.directory)
        for key, root in self.output_directories.items():
            example = None
            for r, _, files in os.walk(root):
                if files:
                    # pick the first file we see; simple and fast
                    example = os.path.join(r, files[0])
                    break
            if example:
                rel = os.path.relpath(example, base)
                print(f"{key}: {rel}")
            else:
                print(f"{key}: (no files)")
    