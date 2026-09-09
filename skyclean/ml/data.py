# Contributed by Qixing Deng

import os
import sys
import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt

from skyclean.silc import utils, HPTools, MWTools, SamplingConverters, FileTemplates
from skyclean.silc.utils import ilc_mode_tag, ilc_mode_candidates
from skyclean.silc.file_templates import register_pixel_ps_component_template
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Reduce TF/XLA log noise.

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False) 

class CMBFreeILC(): 
    def __init__(self, extract_comp: str, component: str, frequencies: list, realisations: int, lmax: int = 1024, N_directions: int = 1, lam: float = 2.0, 
                 nsamp: int = 1200, constraint: bool = False,
                 pcilc: bool = False, pcilc_eps: float | None = None, 
                 batch_size: int = 32, split: list = [0.8, 0.1, 0.1], directory: str = "data/", random: bool = False,
                 prefetch: bool = False, produce_residuals: bool = True, stats_component: str | None = None,
                 run_id: str | None = None):
        """
        Parameters:
            extract_comp (str): Component to be extracted. e.g. "cmb"
            component (str): Maps with components before going to silc. e.g. "cfn", "cfne", "cfne_circ".
            frequencies (list): List of frequency channels for the maps.
            realisations (int): Number of realisations to process.
            lmax (int): The maximum multipole for the wavelet transform.    
            N_directions (int): Number of directions for the wavelet transform.
            lam (float): The lambda parameter for the wavelet transform.
            nsamp (int)
            constraint (bool): Mode for the constrainted ILC method. 
            batch_size (int): Size of the batches for training.
            split (list): List of train/validation/test split ratios.
            directory (str): Directory where data is stored / saved to.
            random (bool): Whether to create random maps for testing purposes. True/False.
            prefetch (bool): Whether to enable tf.data prefetch on the batched datasets.
            produce_residuals (bool): Whether to create the input/target maps for every realisation on construction.
                Pass False to run produce_residuals() explicitly later (the data-preparation step); Train does this.
            stats_component (str): Component whose train-split normalization statistics are applied. Defaults to
                component. A real-map instance (component="real") must pass the simulation product the model was
                trained on (e.g. "cfn") with the same realisations/split: statistics are never fitted on the real sky.
            run_id (str): Model run folder under ML/models where the normalization statistics are saved to / loaded
                from (next to the checkpoints). Required to save or load the statistics.
        """ 
        self.frequencies = frequencies
        self.n_channels_in = len(frequencies)
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.batch_size = batch_size
        self.split = self._normalize_split(split)
        self.directory = directory
        self.component = component
        self.stats_component = stats_component or component
        self.run_id = (run_id or "").strip()
        self.extract_comp = extract_comp
        self.nsamp = nsamp
        self.random = random
        self.prefetch = prefetch
        self.constraint = constraint
        self.pcilc = pcilc
        self.pcilc_eps = pcilc_eps
        # Must match the tag ProduceSILC wrote into the ILC filenames.
        self.mode = ilc_mode_tag(constraint=constraint, pcilc=pcilc, pcilc_eps=pcilc_eps)

        self.a = 1E-5

        files = FileTemplates(directory)
        self.file_templates = files.file_templates
        register_pixel_ps_component_template(self.file_templates, files.output_directories, self.component,)
        self.download_templates = files.download_templates
        # data shapes
        self.H = lmax + 2
        self.W = 2 * (lmax + 1) # for MWSS sampling
        if produce_residuals:
            self.produce_residuals()  # Create residual maps for all realisations
        #self.signed_log_F_mean, self.signed_log_R_mean, self.signed_log_F_std, self.signed_log_R_std = self.find_dataset_mean_std()

    def _split_indices(self):
        """Return deterministic train/validation/test indices for the configured split."""
        idx = np.arange(self.realisations)
        n_train = int(self.split[0] * self.realisations)
        n_val = int(self.split[1] * self.realisations)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        return train_idx, val_idx, test_idx

    def get_split_indices(self):
        """Expose the configured train/validation/test realisation IDs."""
        train_idx, val_idx, test_idx = self._split_indices()
        return {
            "train": train_idx.copy(),
            "val": val_idx.copy(),
            "test": test_idx.copy(),
        }

    def create_random_mwss_maps(self, realisation: int):
        """Generate and save random foreground and residual maps in MWSS sampling format for testing purposes.

        Parameters:
            realisation (int): The realisation number to process.

        Returns:
            foreground_estimate (np.ndarray): Random foreground estimate map of shape (H, W, N_freq).
            ilc_residual (np.ndarray): Random ILC residual map of shape (H, W, 1).
        """
        H, W, lmax, N = self.H, self.W, self.lmax, self.n_channels_in
        
        if os.path.exists(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions)) and os.path.exists(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions)):
            #print(f"Loading existing random test maps for realisation {realisation}...")
            test_foreground_estimate = np.load(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions))
            test_ilc_residual = np.load(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions))
        else:
            print(f"Creating random test maps for realisation {realisation}...")
            np.random.seed(realisation)
            test_foreground_estimate = np.random.randn(H, W, N).astype(np.float32)
            test_ilc_residual = np.random.randn(H, W, 1).astype(np.float32)
            # save the maps up to expected realisations 
            np.save(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions), test_foreground_estimate)
            np.save(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions), test_ilc_residual)
        return test_foreground_estimate, test_ilc_residual
        
    
    def create_residual_mwss_maps(self, realisation: int, component: str | None = None): 
        """For a single realisation, create the ILC residual maps (CMB free) in MWSS sampling format. 

        Parameters:
            realisation (int): The realisation number to process.
            component (str): Input map product to read instead of self.component. Use "real" to build the
                model input from the observed Planck sky (processed_real maps + ilc_synth from-real): there is
                no truth CMB, so ilc_residual is returned as None and the normalisation still comes from
                self.component (the simulations the model was trained on).

        Returns:
            foreground_estimate (np.ndarray): F(i) = CFN(i) - ILC where i is the frequency component (hence have N_freq input channels) of shape (H, W, N_freq)
            ilc_residual (np.ndarray): ILC - CMB of shape (H, W, 1). 
            ilc_map_mwss (np.ndarray): ILC map in MWSS sampling of shape (H, W, 1) (useful for when applying the model).
        """
        H, W, lmax, lam = self.H, self.W, self.lmax, self.lam
        L = lmax + 1
        extract_comp, nsamp, mode = self.extract_comp, self.nsamp, self.mode
        component = component or self.component
        has_truth = component != "real"  # the observed sky has no processed_cmb, so no ILC - CMB residual
        frequencies = '_'.join(self.frequencies)
        if os.path.exists(self.file_templates["foreground_estimate"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode)) and (not has_truth or os.path.exists(self.file_templates["ilc_residual"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode))):
            foreground_estimate = np.load(self.file_templates["foreground_estimate"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode))
            ilc_residual = np.load(self.file_templates["ilc_residual"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode)) if has_truth else None
            ilc_map_mwss = np.load(self.file_templates["ilc_mwss"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode))
        else:
            print(f"Creating residual maps for realisation {realisation}...")
            # load ilc (already in MW sampling)
            ilc_synth_path = self._resolve_ilc_synth_path(realisation, frequencies, lam, component=component)
            ilc_map_mw = np.load(ilc_synth_path)
            ilc_map_mwss = SamplingConverters.mw_map_2_mwss_map(ilc_map_mw, L=L)
            # load cmb and convert to MW sampling (simulations only)
            if has_truth:
                cmb_map_hp = hp.read_map(self.file_templates["processed_cmb"].format(realisation=realisation, lmax=lmax), dtype=np.float32)
                cmb_map_mw = SamplingConverters.hp_map_2_mw_map(cmb_map_hp, lmax) # highly expensive? involves s2fft.forwards.
                cmb_map_mwss = SamplingConverters.mw_map_2_mwss_map(cmb_map_mw, L=L)
            # load cfn maps across frequencies and convert to MW sampling
            # ("real" reads the processed_real maps, which have the same beam/unit treatment as a simulated CFN)
            input_key = "processed_real" if component == "real" else component
            cfn_maps_hp = [hp.read_map(self.file_templates[input_key].format(frequency=frequency, realisation=realisation, lmax=lmax), dtype=np.float32) for frequency in self.frequencies]
            cfn_maps_mw = [SamplingConverters.hp_map_2_mw_map(cfn_map_hp, lmax) for cfn_map_hp in cfn_maps_hp]
            cfn_maps_mwss = [SamplingConverters.mw_map_2_mwss_map(cfn_map_mw, L=L) for cfn_map_mw in cfn_maps_mw]
            # create foreground estimate and ilc residual
            foreground_estimate = np.zeros((H, W, self.n_channels_in), dtype=np.float32)
            ilc_residual = np.zeros((H, W, 1), dtype=np.float32) if has_truth else None
            for i, _ in enumerate(self.frequencies):
                foreground_estimate[:, :, i] = cfn_maps_mwss[i] - ilc_map_mwss
                if has_truth:
                    ilc_residual[:, :, 0] = ilc_map_mwss - cmb_map_mwss
            # save the maps
            np.save(self.file_templates["ilc_mwss"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode), ilc_map_mwss)
            np.save(self.file_templates["foreground_estimate"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode), foreground_estimate)
            if has_truth:
                np.save(self.file_templates["ilc_residual"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode), ilc_residual)
        return foreground_estimate, ilc_residual, ilc_map_mwss

    def _resolve_ilc_synth_path(self, realisation: int, frequencies: str, lam: float, component: str | None = None):
        """Locate the SILC ilc_synth map, tolerating the legacy 'uncon'/'con' mode tag.

        Parameters:
            realisation (int): The realisation number.
            frequencies (str): Underscore-joined frequency tag.
            lam (float): The lambda parameter for the wavelet transform.
            component (str): Input product tag in the ilc_synth filename; defaults to self.component.

        Returns:
            str: Path to an existing ilc_synth file.
        """
        candidates = [
            self.file_templates["ilc_synth"].format(
                extract_comp=self.extract_comp, mode=mode_try, component=component or self.component,
                frequencies=frequencies, realisation=realisation, lmax=self.lmax,
                N_directions=self.N_directions, lam=lam, nsamp=self.nsamp,
            )
            for mode_try in ilc_mode_candidates(
                constraint=self.constraint, pcilc=self.pcilc, pcilc_eps=self.pcilc_eps
            )
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Could not find an ilc_synth map for realisation "
            f"{realisation}. Tried:\n  " + "\n  ".join(candidates) +
            "\nRun the SILC pipeline (step_ilc) for this configuration first."
        )

    def signed_log_transform(self, x: tf.Tensor):
        return jnp.sign(x) * jnp.log1p(jnp.abs(x) / self.a)

    def transform(self, x: tf.Tensor):
        """ Apply a signed-log transform followed by z-score normalization.

        Parameters:
            x (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Transformed and normalized tensor.
        """
        if not hasattr(self, "_cached_stats"):
            stats = self.load_normalization_stats()
            if stats is None:
                train_idx, _, _ = self._split_indices()
                stats = self.find_dataset_mean_std(indices=train_idx)
            self._cached_stats = stats
        (self.signed_log_F_mean,
         self.signed_log_R_mean,
         self.signed_log_F_std,
         self.signed_log_R_std) = self._cached_stats
        # Apply signed-log transform
        signed_log_x = self.signed_log_transform(x)
        
        # determine whether input is F or R by checking number of channels.
        if x.shape[-1] == 1:  
            signed_log_mean = self.signed_log_R_mean
            signed_log_std = self.signed_log_R_std
        else:  # Multiple channels input
            signed_log_mean = self.signed_log_F_mean
            signed_log_std = self.signed_log_F_std
    
        return (signed_log_x - signed_log_mean) / signed_log_std
    
    def inverse_signed_log_transform(self, y: tf.Tensor):
        """Inverse of the signed-log transform.
        
        Parameters:
            y (tf.Tensor): Signed-log transformed tensor.
        Returns:
            tf.Tensor: Original tensor.
        """
        return jnp.sign(y) * self.a * jnp.expm1(jnp.abs(y))

    def inverse_transform(self, z: tf.Tensor):
        """Invert the data transform (reverse z-score normalization then reverse signed-log transform).
        
        Parameters:
            z (tf.Tensor): Normalized transformed tensor.  
        Returns:
            tf.Tensor: Original tensor.
        """
        # Determine whether input is F or R by checking number of channels
        if z.shape[-1] == 1:  
            signed_log_mean = self.signed_log_R_mean
            signed_log_std = self.signed_log_R_std
        else:  # Multiple channels input
            signed_log_mean = self.signed_log_F_mean
            signed_log_std = self.signed_log_F_std

        # Reverse z-score normalization
        y = z * signed_log_std + signed_log_mean
        
        # Reverse signed-log transform
        return self.inverse_signed_log_transform(y)
    
    def _data_generator(self, indices, random):
        """Define a data generator for lazy loading of data.
        
        Parameters:
            indices (list): List of realisation indices to process.
        """
        for realisation in indices:
            if random:
                F, R = self.create_random_mwss_maps(realisation)
                yield F, R
            else:
                F, R, _ = self.create_residual_mwss_maps(realisation)
                # Apply signed-log transform + z-score normalization + cast
                F = self.transform(F).astype(np.float32)
                R = self.transform(R).astype(np.float32)
                yield F, R

    def _make_dataset(self, indices, random, drop_remainder: bool):
        """Build a tf.data.Dataset from a data generator.
        This creates a lazy-loading dataset that processes only the specified indices when requested.
        
        Parameters:
            indices (list): List of realisation indices to process.
        
        Returns:
            tf.data.Dataset: A tf dataset containing the processed data.
        """
        signature = (
            tf.TensorSpec((self.H, self.W, len(self.frequencies)), tf.float32),
            tf.TensorSpec((self.H, self.W, 1), tf.float32),
        ) # tell the generator the type of data it will yield
        ds = tf.data.Dataset.from_generator(
            lambda: self._data_generator(indices, random=random),
            output_signature=signature
        )
        ds = ds.batch(self.batch_size, drop_remainder=drop_remainder)
        if getattr(self, "prefetch", False):
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def produce_residuals(self):
        """Produce and save the residual maps for all realisations."""
        for realisation in range(self.realisations):
            if self.random == False:
                self.create_residual_mwss_maps(realisation)
            else:
                self.create_random_mwss_maps(realisation)

    @staticmethod
    def _normalize_split(split):
        """Normalize split ratios to (train, validation, test)."""
        if len(split) == 2:
            train_ratio, remaining_ratio = split
            validation_ratio = remaining_ratio / 2.0
            test_ratio = remaining_ratio / 2.0
            split = [train_ratio, validation_ratio, test_ratio]
        elif len(split) != 3:
            raise ValueError(f"split must have 2 or 3 values, got {split}.")

        split = np.asarray(split, dtype=float)
        if np.any(split < 0):
            raise ValueError(f"split values must be non-negative, got {split.tolist()}.")
        total = float(np.sum(split))
        if not np.isclose(total, 1.0):
            raise ValueError(f"split must sum to 1.0, got {split.tolist()} (sum={total}).")
        return split.tolist()

    def prepare_data(self):
        """Split indices and return train/validation/test generators.
        
        Returns:
            tuple: Training, validation, and test datasets plus their sizes.
        NOTE: It is recommended to run produce_residuals before running this in the training code.
        """
        random = self.random
        train_idx, val_idx, test_idx = self._split_indices()

        # Freeze normalization on the training split only, then reuse it for all splits.
        if not random:
            stats = self.load_normalization_stats()
            self._cached_stats = stats if stats is not None else self.find_dataset_mean_std(indices=train_idx)

        # TODO: k-fold cross-validation is better.
        train_ds = self._make_dataset(train_idx, random, drop_remainder=True)
        drop_remainder_val = len(val_idx) >= self.batch_size
        if not drop_remainder_val:
            print(f"[WARN] Validation set size ({len(val_idx)}) < batch_size ({self.batch_size}); "
                  "using drop_remainder=False for validation dataset.")
        val_ds = self._make_dataset(val_idx, random, drop_remainder=drop_remainder_val)

        drop_remainder_test = len(test_idx) >= self.batch_size
        if not drop_remainder_test:
            print(f"[WARN] Test set size ({len(test_idx)}) < batch_size ({self.batch_size}); "
                  "using drop_remainder=False for test dataset.")
        test_ds = self._make_dataset(test_idx, random, drop_remainder=drop_remainder_test)
        print("Data generators prepared. Train size:", len(train_idx), "Validation size:", len(val_idx), "Test size:", len(test_idx))
        return train_ds, val_ds, test_ds, len(train_idx), len(val_idx), len(test_idx), drop_remainder_val, drop_remainder_test

    def find_dataset_mean_std(self, indices=None): 
        """Compute normalization statistics for a chosen subset of realizations.
        
        Returns:
            tuple: A tuple containing 4 numpy arrays: (F_mean, R_mean, F_std, R_std).
            F_mean, F_std (np.ndarray) have shape (num_channels,).
            R_mean, R_std (np.ndarray) have shape (1,).
        """
        if self.component == "real":
            raise ValueError(
                "Normalization statistics must be fitted on the simulations the model was trained on, not on the "
                "observed sky. Construct CMBFreeILC(component='real', stats_component='cfn', ...) with the training "
                "realisations/split and prepare the statistics on the 'cfn' instance first."
            )
        if indices is None:
            if hasattr(self, "_cached_stats"):
                return self._cached_stats
            indices, _, _ = self._split_indices()

        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            raise ValueError("Cannot compute normalization statistics from an empty split.")

        F_mean_sum = np.zeros(self.n_channels_in, dtype=np.float64) #per channel mean
        R_mean_sum = 0 # only one output channel
        F_std_sum = np.zeros(self.n_channels_in, dtype=np.float64) #per channel std
        R_std_sum = 0 # only one output channel

        # fit normalization on the training set only, then reuse for val and test.
        for realisation in indices:
            F, R, _ = self.create_residual_mwss_maps(realisation) # load maps
            signed_log_F = self.signed_log_transform(F)
            signed_log_R = self.signed_log_transform(R)
            F_mean_sum += np.mean(signed_log_F, axis=(0, 1))  # Sum over H and W
            R_mean_sum += np.mean(signed_log_R, axis = (0, 1))
            F_std_sum += np.std(signed_log_F, axis=(0, 1))  # Sum over H and W
            R_std_sum += np.std(signed_log_R, axis = (0, 1))

        n_stats = float(indices.size)
        signed_log_F_mean = F_mean_sum / n_stats
        signed_log_R_mean = R_mean_sum / n_stats
        signed_log_F_std = F_std_sum / n_stats
        signed_log_R_std = R_std_sum / n_stats

        stats = (signed_log_F_mean, signed_log_R_mean, signed_log_F_std, signed_log_R_std)
        if np.array_equal(indices, self._split_indices()[0]):
            self._cached_stats = stats
            print(f"Saved normalization statistics to: {self.save_normalization_stats(stats, indices)}")
        return stats

    def normalization_stats_path(self) -> str:
        """Path of the train-split normalization statistics of stats_component for this data configuration,
        under the model run folder ML/models/<run_id>/."""
        if not self.run_id:
            raise ValueError("run_id is required to locate the normalization statistics (ML/models/<run_id>/norm_stats_*.npz).")
        n_train = len(self._split_indices()[0])
        return self.file_templates["ml_norm_stats"].format(run_id=self.run_id, component=self.stats_component, frequencies='_'.join(self.frequencies), lmax=self.lmax, N_directions=self.N_directions, lam=self.lam, nsamp=self.nsamp, mode=self.mode, n_train=n_train)

    def save_normalization_stats(self, stats, train_idx) -> str:
        """Persist (F_mean, R_mean, F_std, R_std) with the training indices they were fitted on. Returns the path."""
        signed_log_F_mean, signed_log_R_mean, signed_log_F_std, signed_log_R_std = stats
        path = self.normalization_stats_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, signed_log_F_mean=np.asarray(signed_log_F_mean, dtype=np.float64), signed_log_R_mean=np.asarray(signed_log_R_mean, dtype=np.float64),
                 signed_log_F_std=np.asarray(signed_log_F_std, dtype=np.float64), signed_log_R_std=np.asarray(signed_log_R_std, dtype=np.float64),
                 train_indices=np.asarray(train_idx, dtype=int), a=np.float64(self.a))
        return path

    def load_normalization_stats(self):
        """Load the persisted train-split statistics of stats_component, or None if they have not been fitted yet."""
        path = self.normalization_stats_path()
        if not os.path.exists(path):
            return None
        with np.load(path) as f:
            return (f["signed_log_F_mean"], f["signed_log_R_mean"], f["signed_log_F_std"], f["signed_log_R_std"])

    def missing_residual_realisations(self) -> list:
        """Realisation IDs whose input/target maps are not on disk yet. Empty for a real-map instance: the single
        observed sky is checked when it is read (create_residual_mwss_maps with component='real')."""
        if self.component == "real":
            return []
        lmax, lam, nsamp, mode, component = self.lmax, self.lam, self.nsamp, self.mode, self.component
        frequencies = '_'.join(self.frequencies)
        missing = []
        for realisation in range(self.realisations):
            if self.random:
                exists = os.path.exists(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions)) and os.path.exists(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax, N_directions=self.N_directions))
            else:
                exists = os.path.exists(self.file_templates["foreground_estimate"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode)) and os.path.exists(self.file_templates["ilc_residual"].format(component=component, frequencies=frequencies, realisation=realisation, lmax=lmax, N_directions=self.N_directions, lam=lam, nsamp=nsamp, mode=mode))
            if not exists:
                missing.append(realisation)
        return missing

    def unprepared_reasons(self) -> list:
        """Why the inputs are not ready (empty once the maps exist and the statistics of stats_component are saved)."""
        reasons = []
        missing = self.missing_residual_realisations()
        if missing:
            reasons.append(f"{len(missing)} of {self.realisations} realisations have no cached input/target maps (first missing: realisation {missing[0]})")
        if not self.random and self.load_normalization_stats() is None:
            reasons.append(f"no normalization statistics for '{self.stats_component}' at {self.normalization_stats_path()}"
                           + (" (fit them on the simulation instance first)" if self.component == "real" else ""))
        return reasons
    

    def load_mask_hp(self,fsky=0.7, apodization=2) -> np.ndarray:
            """
            Load a mask in HEALPix FITS, given the desired f_sky value.
            """
            # Choose a column by index:
            # 0: GAL020, 1: GAL040, 2: GAL060, 3: GAL070,
            # 4: GAL080, 5: GAL090, 6: GAL097, 7: GAL099
            # e.g. GAL070 = 70% sky retained
            get_index = {0.2: 0,
                         0.4: 1,
                         0.6: 2,
                         0.7: 3,
                         0.8: 4,
                         0.9: 5,
                         0.97: 6,
                         0.99: 7}
            if fsky not in get_index:
                raise ValueError(f"Unsupported f_sky={fsky}. Allowed values: {sorted(get_index.keys())}")
            field_index = get_index[fsky]
            mask_path = self.file_templates["mask"].format(apodization=apodization)

            if not os.path.exists(mask_path):
                import urllib.request
                url = self.download_templates["mask"].format(apodization=apodization)
                os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
                urllib.request.urlretrieve(url, mask_path)
                print(f"Download mask for fsky={fsky} apodization={apodization}.")
            else: 
                print(f"Mask already exists: {mask_path} (skipping download)")
            mask = hp.read_map(mask_path, field=field_index)
            print(f"Mask with fsky={fsky} apodization={apodization} loaded from {mask_path}.")
            return mask
    
    def mask_mwss(self,fsky=0.7, apodization=2) -> np.ndarray:
        '''
        Convert a healpix mask to mwss format.
        '''
        lmax = self.lmax
        mask_hp = self.load_mask_hp(fsky=fsky, apodization=apodization)
        mask_mw  = SamplingConverters.hp_map_2_mw_map(mask_hp, lmax)
        L = lmax + 1
        mask_mwss = SamplingConverters.mw_map_2_mwss_map(mask_mw, L=L).astype(np.float32)
        # Ensure (H,W,1)
        mask_mwss = mask_mwss[..., None]
        print(f'MWSS (fsky={fsky}, apodization={apodization}) shape: ', mask_mwss.shape)
        return mask_mwss


    def mask_mwss_beamed(self, fsky=0.7, apodization=2) -> np.ndarray:
        """
        Proceed the mask by convolving and reducing, then converting to MWSS sampling.
        Return a mask in MWSS with shape (H, W, 1).
        """
        lmax = self.lmax
        nside = HPTools.get_nside_from_lmax(lmax)
        standard_fwhm_rad = np.radians(5/60)
        mask_hp = self.load_mask_hp(fsky = fsky, apodization=apodization)
        mask_hp_reduced = HPTools.convolve_and_reduce(
                mask_hp, lmax=lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
            )
        L = lmax + 1
        mask_mw  = SamplingConverters.hp_map_2_mw_map(mask_hp_reduced, lmax)
        mask_mwss = SamplingConverters.mw_map_2_mwss_map(mask_mw, L=L).astype(np.float32)
        # Ensure (H,W,1)
        mask_mwss = mask_mwss[..., None]
        print('Beamed mask MWSS shape: ', mask_mwss.shape)
        return mask_mwss

    
    def mask_mw_beamed(self, fsky=0.7, apodization=2) -> np.ndarray:
        lmax = self.lmax
        nside = HPTools.get_nside_from_lmax(lmax)
        standard_fwhm_rad = np.radians(5/60)
        mask_hp = self.load_mask_hp(fsky, apodization)
        mask_hp_reduced = HPTools.convolve_and_reduce(
            mask_hp, lmax=lmax, nside=nside, standard_fwhm_rad=standard_fwhm_rad
        )
        mask_mw  = SamplingConverters.hp_map_2_mw_map(mask_hp_reduced, lmax)
        return mask_mw