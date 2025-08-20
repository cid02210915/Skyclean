import os 

class FileTemplates():
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
            # downloaded maps
            # downloaded maps
            "cmb": os.path.join(self.output_directories["cmb_realisations"], "cmb_r{realisation:04d}.fits"),
            "sync": os.path.join(self.output_directories["cmb_realisations"], "sync_f{frequency}.fits"),
            "dust": os.path.join(self.output_directories["cmb_realisations"], "dust_f{frequency}.fits"),
            "noise": os.path.join(self.output_directories["cmb_realisations"], "noise_f{frequency}_r{realisation:05d}.fits"),
            "tsz": os.path.join(self.output_directories["cmb_realisations"], "tsz_f{frequency}.fits"), 
            # processed maps
            "processed_cmb": os.path.join(self.output_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
            "processed_sync": os.path.join(self.output_directories["processed_maps"], "processed_sync_f{frequency}_lmax{lmax}.npy"),
            "processed_dust": os.path.join(self.output_directories["processed_maps"], "processed_dust_f{frequency}_lmax{lmax}.npy"),
            "processed_noise": os.path.join(self.output_directories["processed_maps"], "processed_noise_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
            "processed_tsz": os.path.join(self.output_directories["processed_maps"], "processed_tsz_f{frequency}_lmax{lmax}.npy"),
            "cfn": os.path.join(self.output_directories["cfn"], "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
        
            # wavelet transforms
            "wavelet_coeffs": os.path.join(self.output_directories["wavelet_coeffs"], "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_lam{lam}.npy"),
            "scaling_coeffs": os.path.join(self.output_directories["scaling_coeffs"], "{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}_lam{lam}.npy"),
            # ilc intermediates and maps
            'doubled_maps': os.path.join(self.output_directories["doubled_maps"], "doubled_maps_f{frequency}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            'covariance_matrices': os.path.join(self.output_directories["covariance_matrix"],"cov_MW_f{frequencies}_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            'weight_vector_matrices': os.path.join(self.output_directories["weight_vector_data"], "weight_vector_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            'ilc_doubled_maps': os.path.join(self.output_directories["ilc_doubled_wavelet_maps"], "ilc_doubled_Map_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            'ilc_trimmed_maps': os.path.join(self.output_directories["ilc_trimmed_maps"], "ilc_trimmed_wav_Map_s{scale}_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            'ilc_synth': os.path.join(self.output_directories["ilc_synthesised_maps"], "ilc_synthesised_map_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            'ilc_spectrum': os.path.join(self.output_directories["ilc_synthesised_maps"], "ilc_power_spectrum_R{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            # ML
            "foreground_estimate": os.path.join(self.output_directories["ml_maps"], "foreground_estimate_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            "ilc_residual": os.path.join(self.output_directories["ml_maps"], "ilc_residual_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
            "ilc_mwss": os.path.join(self.output_directories["ml_maps"], "ilc_mwss_r{realisation:04d}_lmax{lmax}_lam{lam}.npy"),
        }

    
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








