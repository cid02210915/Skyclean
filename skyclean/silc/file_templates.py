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

    def __init__(self, directory = "data/", start_realisation: int = 0, topology: str | None = None,):

        valid_topologies = {
            None,
            "Toy",
            "E1",
            "E2",
            "E3",
        }

        if topology not in valid_topologies:
            raise ValueError(
                "topology must be None, 'Toy', 'E1', 'E2', or 'E3'."
            )
        
        self.directory = directory
        self.topology = topology

        if topology is None:
            output_root = directory
        else:
            output_root = os.path.join(
                directory,
                topology,
            )

        self.output_root = output_root
        self.output_directories = {
            # downloaded maps
            "cmb_realisations": os.path.join(directory, "CMB_realisations/"),
            # SO maps
            "so_realisations": os.path.join(directory, "SO_realisations/"),
            # processed maps
            "cfn": os.path.join(output_root, "CFN_realisations"),
            "fn": os.path.join(output_root, "FN_realisations"),
            "processed_maps": os.path.join(output_root, "processed_maps"),
            # wavelet transforms
            "wavelet_coeffs": os.path.join(output_root, "wavelet_transforms/wavelet_coeffs"),
            "scaling_coeffs": os.path.join(output_root, "wavelet_transforms/scaling_coeffs"),
            "f_scal": os.path.join(output_root, "SILC/f_scal"),
            # ilc maps
            "doubled_maps": os.path.join(output_root, "SILC/doubled_maps"),
            "covariance_matrix": os.path.join(output_root, "SILC/covariance_matrix"),
            "weight_vector_data": os.path.join(output_root, "SILC/weight_vector_data"),
            "ilc_doubled_wavelet_maps": os.path.join(output_root, "SILC/ilc_doubled_wavelet_maps"),
            "ilc_trimmed_maps": os.path.join(output_root, "SILC/ilc_trimmed_maps"),
            "ilc_synthesised_maps": os.path.join(output_root, "SILC/ilc_synthesised_maps"),
            # ilc_synth spectra
            "ilc_synthesised_spec": os.path.join(output_root, "SILC/ilc_synthesised_spec"),
            "ilc_synthesised_cross_spec": os.path.join(output_root, "SILC/ilc_synthesised_cross_spec"),
            # ML 
            "ml_maps": os.path.join(output_root, "ML/maps"),
            "ml_models": os.path.join(output_root, "ML/models"),
            "ilc_improved_maps": os.path.join(output_root, "SILC/ilc_improved_maps")
        }
        for key, value in self.output_directories.items():
            if not os.path.exists(value):
                print(f"Creating directory: {value}")
            os.makedirs(value, exist_ok=True)

        
        self.download_templates = {
            "sync": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_synchrotron-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits",
            "dust": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_thermaldust-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits",
            "noise":"http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_{frequency}_full_map_mc_{realisation:05d}.fits",
            'tsz':  "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_thermalsz-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits",
            "cib":  "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_CIB-GNILC-F{frequency}_2048_R2.00.fits",
            # point sources 
            "strong_irps": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_strongirps-ffp10-skyinbands-{frequency}_4096_R3.00_full.fits",
            "faint_radiops": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_faintradiops-ffp10-skyinbands-{frequency}_4096_R3.00_full.fits",
            # diffuse
            "freefree": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_freefree-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits",
            "faint_irps":    "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_faintirps-ffp10-skyinbands-{frequency}_4096_R3.00_full.fits",
            "strong_uchii":  "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_stronguchii-ffp10-skyinbands-{frequency}_4096_R3.00_full.fits",

            # total FFP10 simulated frequency maps: CMB + foregrounds + noise/instrument simulation
            "total": "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=ffp10_newdust_total_{frequency}_full_map.fits",
            "real": {
                "030": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_full.fits",
                "044": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_full.fits",
                "070": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_full.fits",
                "100": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_full.fits",
                "143": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits",
                "217": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_217_2048_R3.01_full.fits",
                "353": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_353_2048_R3.01_full.fits",
                "545": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_545_2048_R3.01_full.fits",
                "857": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_full.fits",
            },
        }
        
        # ==================================================
        # Simons Observatory / LAMBDA 2019 simulations
        # ==================================================
        self.so_download_templates = {
            # Frequency-independent in K_CMB / thermodynamic units
            "ksz": "https://lambda.gsfc.nasa.gov/data/tools/so_v3/148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits",
            # Frequency-independent Compton-y map
            "tsz": "https://lambda.gsfc.nasa.gov/data/tools/so_v3/tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits",
            # Frequency-dependent foreground maps
            "cib": "https://lambda.gsfc.nasa.gov/data/tools/so_v3/{frequency}_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75.fits",
            "radio": "https://lambda.gsfc.nasa.gov/data/tools/so_v3/{frequency}_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp.fits",
            "dust": "https://lambda.gsfc.nasa.gov/data/tools/so_v3/{frequency}_dust_healpix_nopell_Nside4096_DeltaT_uK_lininterp.fits",
            "lowfreq_galactic": "https://lambda.gsfc.nasa.gov/data/tools/so_v3/commander_synch_freefree_AME_{frequency}_Equ_uK_smooth17arcmin.fits",
        }

        if topology is None:
            cmb_input_file = os.path.join(
                self.output_directories["cmb_realisations"],
                "cmb_r{realisation:04d}.fits",
            )
        elif topology == "Toy":
            cmb_input_file = (
                "/home/agnes/Skyclean/docs/circle/"
                "step1_injected_pair_nside512_lmax800/"
                "injected_map.fits"
            )
        elif topology == "E1":
            cmb_input_file = (
                "/home/agnes/Skyclean/Skyclean/docs/circle/"
                "E1_r{realisation:03d}_nside256_lmax600.fits" # has to be changed to a way that accept a more general naming
            )
        elif topology == "E2":
            cmb_input_file = (
                "/home/agnes/Skyclean/Skyclean/docs/circle/"
                "E2_r{realisation:03d}.fits"
            )
        elif topology == "E3":
            cmb_input_file = (
                "/home/agnes/Skyclean/Skyclean/docs/circle/"
                "E3_r{realisation:03d}.fits"
            )
        else:
            raise ValueError(
                "topology must be None, 'Toy', 'E1', 'E2', or 'E3'."
            )
        
        # ==================================================
        # SO downloaded maps
        # ==================================================
        self.so_file_templates = {
            # Use the same unlensed CMB realisation as the existing pipeline
            "cmb": cmb_input_file,

            # Frequency-independent SO components
            "ksz": os.path.join(self.output_directories["so_realisations"], "ksz.fits"),
            "tsz": os.path.join(self.output_directories["so_realisations"], "tsz_y.fits"),

            # Frequency-dependent SO foregrounds
            "cib": os.path.join(self.output_directories["so_realisations"], "cib_f{frequency}.fits"),
            "radio": os.path.join(self.output_directories["so_realisations"],"radio_f{frequency}.fits"),
            "dust": os.path.join(self.output_directories["so_realisations"], "dust_f{frequency}.fits"),
            "lowfreq_galactic": os.path.join(self.output_directories["so_realisations"],"lowfreq_galactic_f{frequency}.fits"),
        }

        self.file_templates = {
        # ---------------- downloaded maps ----------------
        "cmb":   cmb_input_file,
        "sync":  os.path.join(self.output_directories["cmb_realisations"], "sync_f{frequency}.fits"),
        "dust":  os.path.join(self.output_directories["cmb_realisations"], "dust_f{frequency}.fits"),
        "noise": os.path.join(self.output_directories["cmb_realisations"], "noise_f{frequency}_r{realisation:05d}.fits"),
        "tsz":   os.path.join(self.output_directories["cmb_realisations"], "tsz_f{frequency}.fits"),
        "cib":   os.path.join(self.output_directories["cmb_realisations"], "cib_f{frequency}.fits"),
        "strong_irps":  os.path.join(self.output_directories["cmb_realisations"], "strong_irps_f{frequency}.fits"),
        "freefree":      os.path.join(self.output_directories["cmb_realisations"], "freefree_f{frequency}.fits"),
        "faint_irps":    os.path.join(self.output_directories["cmb_realisations"], "faint_irps_f{frequency}.fits"),
        "faint_radiops":  os.path.join(self.output_directories["cmb_realisations"], "faint_radiops_f{frequency}.fits"),
        "strong_uchii":   os.path.join(self.output_directories["cmb_realisations"], "strong_uchii_f{frequency}.fits"),
        "total": os.path.join(self.output_directories["cmb_realisations"], "total_f{frequency}.fits"),
        "real": os.path.join(self.output_directories["cmb_realisations"], "real_f{frequency}.fits"),

        # ---------------- processed maps ----------------
        "processed_cmb":   os.path.join(self.output_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
        "processed_sync":  os.path.join(self.output_directories["processed_maps"], "processed_sync_f{frequency}_lmax{lmax}.npy"),
        "processed_dust":  os.path.join(self.output_directories["processed_maps"], "processed_dust_f{frequency}_lmax{lmax}.npy"),
        "processed_noise": os.path.join(self.output_directories["processed_maps"], "processed_noise_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
        "processed_tsz":   os.path.join(self.output_directories["processed_maps"], "processed_tsz_f{frequency}_lmax{lmax}.npy"),
        "processed_cib":   os.path.join(self.output_directories["processed_maps"], "processed_cib_f{frequency}_lmax{lmax}.npy"),
        "processed_strong_irps":   os.path.join(self.output_directories["processed_maps"], "processed_strong_irps_f{frequency}_lmax{lmax}.npy"),
        "processed_freefree":      os.path.join(self.output_directories["processed_maps"], "processed_freefree_f{frequency}_lmax{lmax}.npy"),
        "processed_faint_irps":    os.path.join(self.output_directories["processed_maps"], "processed_faint_irps_f{frequency}_lmax{lmax}.npy"),
        "processed_faint_radiops": os.path.join(self.output_directories["processed_maps"], "processed_faint_radiops_f{frequency}_lmax{lmax}.npy"),
        "processed_strong_uchii":  os.path.join(self.output_directories["processed_maps"], "processed_strong_uchii_f{frequency}_lmax{lmax}.npy"),
        "cfn":             os.path.join(self.output_directories["cfn"],            "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
        "fn":              os.path.join(self.output_directories["fn"], "fn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
        "processed_total": os.path.join(self.output_directories["processed_maps"], "processed_total_f{frequency}_lmax{lmax}.npy"),
        "processed_real": os.path.join(self.output_directories["processed_maps"], "processed_real_f{frequency}_lmax{lmax}.npy"),
        "processed_real": os.path.join(self.output_directories["processed_maps"],"processed_real_f{frequency}_lmax{lmax}.fits"),

        # ---------------- SO processed maps ----------------
        "so_processed_cmb": os.path.join(self.output_directories["processed_maps"], "so_processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
        "so_processed_ksz": os.path.join(self.output_directories["processed_maps"], "so_processed_ksz_lmax{lmax}.npy"),
        "so_processed_tsz": os.path.join(self.output_directories["processed_maps"], "so_processed_tsz_f{frequency}_lmax{lmax}.npy"),
        "so_processed_cib": os.path.join(self.output_directories["processed_maps"], "so_processed_cib_f{frequency}_lmax{lmax}.npy"),
        "so_processed_radio": os.path.join(self.output_directories["processed_maps"], "so_processed_radio_f{frequency}_lmax{lmax}.npy"),
        "so_processed_dust": os.path.join(self.output_directories["processed_maps"], "so_processed_dust_f{frequency}_lmax{lmax}.npy"),
        "so_processed_lowfreq_galactic": os.path.join(self.output_directories["processed_maps"], "so_processed_lowfreq_galactic_f{frequency}_lmax{lmax}.npy"),
        "so_processed_noise": os.path.join(self.output_directories["processed_maps"], "so_processed_noise_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
        "so_cfn": os.path.join(self.output_directories["cfn"], "so_cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
        "so_fn": os.path.join(self.output_directories["fn"], "so_fn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),

        # ---------------- wavelet transforms ----------------
        "wavelet_coeffs": os.path.join(
            self.output_directories["wavelet_coeffs"],
            "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
        ),
        # Alias for older code that expects 'wavelet_c_j'
        "wavelet_c_j": os.path.join(
            self.output_directories["wavelet_coeffs"],
            "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
        ),
        "scaling_coeffs": os.path.join(
            self.output_directories["scaling_coeffs"],
            "{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
        ),
        "f_scal": os.path.join(
            self.output_directories["f_scal"],
            "f_scal_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
        ),

        # ---------------- SO wavelet transforms ----------------
        "so_wavelet_coeffs": os.path.join(
            self.output_directories["wavelet_coeffs"], 
            "so_{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
            ),
        "so_wavelet_c_j": os.path.join(
            self.output_directories["wavelet_coeffs"], 
            "so_{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
            ),
        "so_scaling_coeffs": os.path.join(
            self.output_directories["scaling_coeffs"], 
            "so_{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
        ),
        "so_f_scal": os.path.join(
            self.output_directories["f_scal"], 
            "so_f_scal_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}.npy"
            ),

        # ---------------- ILC intermediates & outputs (matches ILC_wav_coeff_maps_MP) ----------------
        # IMPORTANT: these use {component}, {extract_comp}, and US spelling {realisation}
        # Per-frequency, per-scale doubled wavelet maps (still per input component)
        "doubled_maps": os.path.join(
            self.output_directories["doubled_maps"],
            "doubled_{component}_f{frequency}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Covariance matrices per scale over the whole band-set (frequencies join tag, e.g. 30_44_70_...)
        "covariance_matrices": os.path.join(
            self.output_directories["covariance_matrix"],
            "cov_MW_{component}_f{frequencies}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Weights per scale; {type} is "weight_vector" (or "cilc_cmb" for constrained case)
        "weight_vector_matrices": os.path.join(
            self.output_directories["weight_vector_data"],
            "{component}_{type}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Per-scale ILC maps at doubled resolution (function expects key 'ilc_maps')
        "ilc_maps": os.path.join(
            self.output_directories["ilc_doubled_wavelet_maps"],
            "{mode}_ilc_doubled_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Legacy alias (same path)
        "ilc_doubled_maps": os.path.join(
            self.output_directories["ilc_doubled_wavelet_maps"],
            "{mode}_ilc_doubled_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Per-scale maps trimmed back to original resolution (function expects key 'trimmed_maps')
        "trimmed_maps": os.path.join(
            self.output_directories["ilc_trimmed_maps"],
            "{mode}_ilc_trimmed_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Final synthesized map — records target (extract_comp), source (component), and band-set
        "ilc_synth": os.path.join(
            self.output_directories["ilc_synthesised_maps"],
            "{mode}_{extract_comp}_from-{component}_f{frequencies}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Optional: power spectrum
        "ilc_spectrum": os.path.join(
            self.output_directories["ilc_synthesised_spec"],
            "{mode}_{extract_comp}_from-{component}_spectrum_f{frequencies}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # Optional: cross power spectrum (e.g. ILC vs processed)
        "ilc_cross_spectrum": os.path.join(
            self.output_directories["ilc_synthesised_cross_spec"],
            "{mode}_{extract_comp}_from-{component}_x-{other}_spectrum_f{frequencies}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
        ),
        # ---------------- SO ILC intermediates & outputs ----------------
        "so_doubled_maps": os.path.join(
            self.output_directories["doubled_maps"], 
            "so_doubled_{component}_f{frequency}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_covariance_matrices": os.path.join(
            self.output_directories["covariance_matrix"],
            "so_cov_MW_{component}_f{frequencies}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_weight_vector_matrices": os.path.join(
            self.output_directories["weight_vector_data"], 
            "so_{component}_{type}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_ilc_maps": os.path.join(
            self.output_directories["ilc_doubled_wavelet_maps"], 
            "so_{mode}_ilc_doubled_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_ilc_doubled_maps": os.path.join(
            self.output_directories["ilc_doubled_wavelet_maps"], 
            "so_{mode}_ilc_doubled_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_trimmed_maps": os.path.join(
            self.output_directories["ilc_trimmed_maps"], 
            "so_{mode}_ilc_trimmed_{component}_{extract_comp}_s{scale}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_ilc_synth": os.path.join(
            self.output_directories["ilc_synthesised_maps"], 
            "so_{mode}_{extract_comp}_from-{component}_f{frequencies}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_ilc_spectrum": os.path.join(
            self.output_directories["ilc_synthesised_spec"], 
            "so_{mode}_{extract_comp}_from-{component}_spectrum_f{frequencies}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),
        "so_ilc_cross_spectrum": os.path.join(
            self.output_directories["ilc_synthesised_cross_spec"], 
            "so_{mode}_{extract_comp}_from-{component}_x-{other}_spectrum_f{frequencies}_r{realisation:04d}_lmax{lmax}_N{N_directions}_lam{lam}_nsamp{nsamp}.npy"
            ),

        # ------------- ML -------------
        "foreground_estimate": os.path.join(self.output_directories["ml_maps"], "foreground_estimate_r{realisation:04d}_lmax{lmax}_lam{lam}_nsamp{nsamp}.npy"),
        "ilc_residual": os.path.join(self.output_directories["ml_maps"], "ilc_residual_r{realisation:04d}_lmax{lmax}_lam{lam}_nsamp{nsamp}.npy"),
        "ilc_mwss": os.path.join(self.output_directories["ml_maps"], "ilc_mwss_r{realisation:04d}_lmax{lmax}_lam{lam}_nsamp{nsamp}.npy"),

        # ------------- SO ML -------------
        "so_foreground_estimate": os.path.join(self.output_directories["ml_maps"], "so_foreground_estimate_r{realisation:04d}_lmax{lmax}_lam{lam}_nsamp{nsamp}.npy"),
        "so_ilc_residual": os.path.join(self.output_directories["ml_maps"], "so_ilc_residual_r{realisation:04d}_lmax{lmax}_lam{lam}_nsamp{nsamp}.npy"),
        "so_ilc_mwss": os.path.join(self.output_directories["ml_maps"], "so_ilc_mwss_r{realisation:04d}_lmax{lmax}_lam{lam}_nsamp{nsamp}.npy"),
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
    
    