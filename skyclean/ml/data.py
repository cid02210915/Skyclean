import os
import sys
import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt

from skyclean.silc import utils, HPTools, MWTools, SamplingConverters, FileTemplates
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Reduce TF/XLA log noise.

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False) 

class CMBFreeILC(): 
    def __init__(self, extract_comp: str, component: str, frequencies: list, realisations: int, lmax: int = 1024, N_directions: int = 1, lam: float = 2.0, 
                 batch_size: int = 32, shuffle: bool = True,  split: list = [0.8, 0.2], directory: str = "data/", random: bool = False):
        """
        Parameters:
            frequencies (list): List of frequency channels for the maps.
            realisations (int): Number of realisations to process.
            lmax (int): Maximum multipole for the wavelet transform.    
            N_directions (int): Number of directions for the wavelet transform.
            batch_size (int): Size of the batches for training.
            split (list): List of train/validation/test split ratios.
            shuffle (bool): Whether to shuffle the dataset.
            directory (str): Directory where data is stored / saved to.
            component (str): cfn.
            extract_comp (str): cmb.
            random (bool): Whether to create random maps for testing purposes.
        """ 
        self.frequencies = frequencies
        self.n_channels_in = len(frequencies)
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.directory = directory
        self.component = component
        self.extract_comp = extract_comp
        self.random = random

        self.a = 1E-5

        files = FileTemplates(directory)
        self.file_templates = files.file_templates
        self.download_templates = files.download_templates
        # retrieve shapes
        #ilc_map_temp = np.load(self.file_templates["ilc_synth"].format(
        #    mode='uncon',extract_comp=self.extract_comp, component=self.component, frequencies="_".join(str(x) for x in self.frequencies), 
        #    realisation=0, lmax=self.lmax, lam = self.lam, nsamp='1200'))
        #self.H = ilc_map_temp.shape[0]+1
        #self.W = ilc_map_temp.shape[1]+1 # for MWSS sampling
        self.H = lmax + 2
        self.W = 2 * (lmax + 1) # for MWSS sampling
        self.produce_residuals()  # Create residual maps for all realisations
        #self.signed_log_F_mean, self.signed_log_R_mean, self.signed_log_F_std, self.signed_log_R_std = self.find_dataset_mean_std()

    def create_random_mwss_maps(self, realisation: int):
        """Generate and save random foreground and residual maps in MWSS sampling format for testing purposes.

        Parameters:
            realisation (int): The realisation number to process.

        Returns:
            foreground_estimate (np.ndarray): Random foreground estimate map of shape (H, W, N_freq).
            ilc_residual (np.ndarray): Random ILC residual map of shape (H, W, 1).
        """
        H, W, lmax, N = self.H, self.W, self.lmax, self.n_channels_in
        
        if os.path.exists(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax)) and os.path.exists(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax)):
            #print(f"Loading existing random test maps for realisation {realisation}...")
            test_foreground_estimate = np.load(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax))
            test_ilc_residual = np.load(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax))
        else:
            print(f"Creating random test maps for realisation {realisation}...")
            np.random.seed(realisation)
            test_foreground_estimate = np.random.randn(H, W, N).astype(np.float32)
            test_ilc_residual = np.random.randn(H, W, 1).astype(np.float32)
            # save the maps up to expected realisations 
            np.save(self.file_templates["test_foreground_estimate"].format(realisation=realisation, lmax=lmax), test_foreground_estimate)
            np.save(self.file_templates["test_ilc_residual"].format(realisation=realisation, lmax=lmax), test_ilc_residual)
        return test_foreground_estimate, test_ilc_residual
        
    
    def create_residual_mwss_maps(self, realisation: int): 
        """For a single realisation, create the ILC residual maps (CMB free) in MWSS sampling format. 

        Parameters:
            realisation (int): The realisation number to process.

        Returns:
            foreground_estimate (np.ndarray): F(i) = CFN(i) - ILC where i is the frequency component (hence have N_freq input channels) of shape (H, W, N_freq)
            ilc_residual (np.ndarray): ILC - CMB of shape (H, W, 1). 
            ilc_map_mwss (np.ndarray): ILC map in MWSS sampling of shape (H, W, 1) (useful for when applying the model).
        """
        H, W, lmax, lam = self.H, self.W, self.lmax, self.lam
        L = lmax + 1
        frequencies = self.frequencies
        if os.path.exists(self.file_templates["foreground_estimate"].format(realisation=realisation, lmax=lmax, lam=lam)) and os.path.exists(self.file_templates["ilc_residual"].format(realisation=realisation, lmax=lmax, lam=lam)):
            foreground_estimate = np.load(self.file_templates["foreground_estimate"].format(realisation=realisation, lmax=lmax, lam=lam))
            ilc_residual = np.load(self.file_templates["ilc_residual"].format(realisation=realisation, lmax=lmax, lam=lam))
            ilc_map_mwss = np.load(self.file_templates["ilc_mwss"].format(realisation=realisation, lmax=lmax, lam=lam))
        else:
            print(f"Creating residual maps for realisation {realisation}...")
            # load ilc (already in MW sampling)
            ilc_map_mw = np.load(self.file_templates["ilc_synth"].format(
            mode='uncon',extract_comp=self.extract_comp, component=self.component, frequencies="_".join(str(x) for x in self.frequencies), 
            realisation=realisation, lmax=self.lmax, lam = self.lam, nsamp='1200'))
            #ilc_map_mw = np.load(self.file_templates["ilc_synth"].format(realisation=realisation, lmax=lmax, lam=lam))
            ilc_map_mwss = SamplingConverters.mw_map_2_mwss_map(ilc_map_mw, L=L)
            # load cmb and convert to MW sampling
            #cmb_map_hp = hp.read_map(self.file_templates["cmb"].format(realisation=realisation, lmax=lmax), dtype=np.float32)
            cmb_map_hp = hp.read_map(self.file_templates["cmb"].format(realisation=realisation), dtype=np.float32)
            cmb_map_mw = SamplingConverters.hp_map_2_mw_map(cmb_map_hp, lmax) # highly expensive? involves s2fft.forwards.
            cmb_map_mwss = SamplingConverters.mw_map_2_mwss_map(cmb_map_mw, L=L)
            # load cfn maps across frequencies and convert to MW sampling
            cfn_maps_hp = [hp.read_map(self.file_templates["cfn"].format(frequency=frequency, realisation=realisation, lmax=lmax), dtype=np.float32) for frequency in frequencies]
            cfn_maps_mw = [SamplingConverters.hp_map_2_mw_map(cfn_map_hp, lmax) for cfn_map_hp in cfn_maps_hp]
            cfn_maps_mwss = [SamplingConverters.mw_map_2_mwss_map(cfn_map_mw, L=L) for cfn_map_mw in cfn_maps_mw]
            # create foreground estimate and ilc residual
            foreground_estimate = np.zeros((H, W, self.n_channels_in), dtype=np.float32)
            ilc_residual = np.zeros((H, W, 1), dtype=np.float32)
            for i, _ in enumerate(frequencies):
                foreground_estimate[:, :, i] = cfn_maps_mwss[i] - ilc_map_mwss
                ilc_residual[:, :, 0] = ilc_map_mwss - cmb_map_mwss
            # save the maps
            np.save(self.file_templates["ilc_mwss"].format(realisation=realisation, lmax=lmax, lam=lam), ilc_map_mwss)
            np.save(self.file_templates["foreground_estimate"].format(realisation=realisation, lmax=lmax, lam=lam), foreground_estimate)
            np.save(self.file_templates["ilc_residual"].format(realisation=realisation, lmax=lmax, lam=lam), ilc_residual)
        return foreground_estimate, ilc_residual, ilc_map_mwss

    def signed_log_transform(self, x: tf.Tensor):
        return jnp.sign(x) * jnp.log1p(jnp.abs(x) / self.a)

    def transform(self, x: tf.Tensor):
        """ Apply a signed-log transform followed by z-score normalization.

        Parameters:
            x (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Transformed and normalized tensor.
        """
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
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(indices))
        return ds.batch(self.batch_size, drop_remainder=drop_remainder) \
                 .prefetch(tf.data.AUTOTUNE)

    def produce_residuals(self):
        """Produce and save the residual maps for all realisations."""
        for realisation in range(self.realisations):
            if self.random == False:
                self.create_residual_mwss_maps(realisation)
            else:
                self.create_random_mwss_maps(realisation)

    def prepare_data(self):
        """Split indices and return (train_ds, test_ds) generators.
        
        Returns:
            tuple: A tuple containing the training and testing datasets.
        NOTE: It is recommended to run produce_residuals before running this in the training code.
        """
        random = self.random
        idx = np.arange(self.realisations)
        cut = int(self.split[0] * self.realisations)
        train_idx, test_idx = idx[:cut], idx[cut:]
        train_ds = self._make_dataset(train_idx, random, drop_remainder=True)
        drop_remainder_test = len(test_idx) >= self.batch_size
        if not drop_remainder_test:
            print(f"[WARN] Test set size ({len(test_idx)}) < batch_size ({self.batch_size}); "
                  "using drop_remainder=False for test dataset.")
        test_ds = self._make_dataset(test_idx, random, drop_remainder=drop_remainder_test)
        print("Data generators prepared. Train size:", len(train_idx), "Test size:", len(test_idx))
        return train_ds, test_ds, len(train_idx), len(test_idx), drop_remainder_test

    def find_dataset_mean_std(self): 
        """Compute the mean and standard deviation of the inputs (channel-wise) and outputs (single channel). 
        
        Returns:
            tuple: A tuple containing 4 numpy arrays: (F_mean, R_mean, F_std, R_std).
            F_mean, F_std (np.ndarray) have shape (num_channels,).
            R_mean, R_std (np.ndarray) have shape (1,).
        """
        F_mean_sum = np.zeros(self.n_channels_in, dtype=np.float64) #per channel mean
        R_mean_sum = 0 # only one output channel
        F_std_sum = np.zeros(self.n_channels_in, dtype=np.float64) #per channel std
        R_std_sum = 0 # only one output channel

        for realisation in range(self.realisations):
            F, R, _ = self.create_residual_mwss_maps(realisation) # load maps
            signed_log_F = self.signed_log_transform(F)
            signed_log_R = self.signed_log_transform(R)
            F_mean_sum += np.mean(signed_log_F, axis=(0, 1))  # Sum over H and W
            R_mean_sum += np.mean(signed_log_R, axis = (0, 1))
            F_std_sum += np.std(signed_log_F, axis=(0, 1))  # Sum over H and W
            R_std_sum += np.std(signed_log_R, axis = (0, 1))

        signed_log_F_mean = F_mean_sum / self.realisations
        signed_log_R_mean = R_mean_sum / self.realisations
        signed_log_F_std = F_std_sum / self.realisations
        signed_log_R_std = R_std_sum / self.realisations

        return signed_log_F_mean, signed_log_R_mean, signed_log_F_std, signed_log_R_std
    

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
