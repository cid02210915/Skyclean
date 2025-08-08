import os
import sys
import numpy as np
import healpy as hp 
import matplotlib.pyplot as plt

# import from skyclean/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import * 
from map_tools import * 
from file_templates import FileTemplates
import tensorflow as tf

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", False)  # Use 32-bit

class CMBFreeILC(): 
    def __init__(self, frequencies: list, realisations: int, lmax: int = 1024, N_directions: int = 1, lam: float = 2.0, 
                 batch_size: int = 32, shuffle: bool = True,  split: list = [0.8, 0.2], directory: str = "data/", ):
        """
        Parameters:
            frequencies (list): List of frequencies for the maps.
            realisations (int): Number of realisations to process.
            lmax (int): Maximum multipole for the wavelet transform.    
            N_directions (int): Number of directions for the wavelet transform.
            batch_size (int): Size of the batches for training.
            split (list): List of train/validation/test split ratios.
            shuffle (bool): Whether to shuffle the dataset.
            directory (str): Directory where data is stored / saved to.
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

        self.a = 1E-5

        files = FileTemplates(directory)
        self.file_templates = files.file_templates
        # retrieve shapes
        ilc_map_temp = np.load(self.file_templates["ilc_synth"].format(realisation=0, lmax=self.lmax, lam = self.lam))
        self.H = ilc_map_temp.shape[0]+1
        self.W = ilc_map_temp.shape[1]+1 # for MWSS sampling

        self.produce_residuals()  # Create residual maps for all realisations
        self.signed_log_F_mean, self.signed_log_R_mean, self.signed_log_F_std, self.signed_log_R_std = self.find_dataset_mean_std()

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
            ilc_map_mw = np.load(self.file_templates["ilc_synth"].format(realisation=realisation, lmax=lmax, lam=lam))
            ilc_map_mwss = SamplingConverters.mw_map_2_mwss_map(ilc_map_mw, L=L)
            # load cmb and convert to MW sampling
            cmb_map_hp = hp.read_map(self.file_templates["cmb"].format(realisation=realisation, lmax=lmax), dtype=np.float32)
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
    
    def inverse_transform(self, z: tf.Tensor):
        """Invert the data transform. 
        Parameters:
            z (tf.Tensor): Transformed tensor.  
        Returns:
            tf.Tensor: Inverse transformed tensor.
        """
        if z.shape[-1] == 1:  
            mean = self.R_mean
            std = self.R_std
        else:  # Multiple channels input
            mean = self.F_mean
            std = self.F_std
        signed_log_mean = jnp.sign(mean) * jnp.log1p(jnp.abs(mean) / self.a)
        signed_log_std = jnp.sign(std) * jnp.log1p(jnp.abs(std) / self.a)

        y = z * signed_log_std + signed_log_mean
        return jnp.sign(y) * self.a * jnp.expm1(jnp.abs(y))
    
    def _data_generator(self, indices):
        """Define a data generator for lazy loading of data.
        
        Parameters:
            indices (list): List of realisation indices to process.
        """
        for realisation in indices:
            F, R, _ = self.create_residual_mwss_maps(realisation)
            # Apply signed-log transform + z-score normalization + cast
            F = self.transform(F).astype(np.float32)
            R = self.transform(R).astype(np.float32)
            yield F, R

    def _make_dataset(self, indices):
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
            lambda: self._data_generator(indices),
            output_signature=signature
        )
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(indices))
        return ds.batch(self.batch_size, drop_remainder=False) \
                 .prefetch(tf.data.AUTOTUNE)

    def produce_residuals(self):
        """Produce and save the residual maps for all realisations."""
        for realisation in range(self.realisations):
            self.create_residual_mwss_maps(realisation)

    def prepare_data(self):
        """Split indices and return (train_ds, test_ds) generators.
        
        Returns:
            tuple: A tuple containing the training and testing datasets.
        NOTE: It is recommended to run produce_residuals before running this in the training code.
        """
        idx = np.arange(self.realisations)
        cut = int(self.split[0] * self.realisations)
        train_idx, test_idx = idx[:cut], idx[cut:]
        train_ds = self._make_dataset(train_idx)
        test_ds  = self._make_dataset(test_idx)
        print("Data generators prepared. Train size:", len(train_idx), "Test size:", len(test_idx))
        return train_ds, test_ds, len(train_idx), len(test_idx)

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
    
    def apply_model(self, model, realisation: int): 
        """Apply the model to a specific realisation's ILC map, returning the improved prediction (CMB prediction).
        Performed after model training. 

        
        Parameters:
            model: The model to apply.
            realisation (int): The realisation number to process.
        
        Returns:
            np.ndarray: The model's prediction for the given realisation.
        """
        F, _, ilc_mwss = self.create_residual_mwss_maps(realisation)
        F = self.transform(F).astype(np.float32)
        # F currently has shape (H,W,N_freq), we need to add a batch dimension to apply the model
        F = jnp.expand_dims(F, axis=0)  # Add batch dimension
        R_pred_norm = model(F)
        R_pred = self.inverse_transform(R_pred_norm)  # Invert the normalisation
        # R_pred has shape (1,H,W,1), squeeze to (H,W)
        R_pred = jnp.squeeze(R_pred, axis=(0,3)) 
        cmb_pred = ilc_mwss - R_pred # compute the CMB prediction
        # convert this to MW sampling
        return SamplingConverters.mwss_map_2_mw_map(cmb_pred, L=self.lmax + 1)





            


## TESTING
# from model import S2_UNET
# import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
# import jax.numpy as jnp
# lmax = 511
# L = lmax+1
# lam = 4.0
# obj = CMBFreeILC(frequencies=["030", "100", "353"], realisations=10, lmax=lmax, lam=lam, N_directions=1, batch_size=3, shuffle=True, split=[0.8, 0.2], directory="/Scratch/matthew/data/")
# train_ds, test_ds, _, _ = obj.prepare_data()
# model = S2_UNET(L,ch_in = 3, )
# train_iter = iter(tfds.as_numpy(train_ds))
# batch_x, batch_y = next(train_iter)
# image = jnp.asarray(batch_x)
# output = jnp.asarray(batch_y)
# pred = model(image) #shape (3,257,513,2)
# input_ex = image[0, :, :, 0]# First channel of the first image in the batch
# output_ex = output[0, :, :, 0]  # First channel of the first image in the batch
# pred_ex = pred[0, :, :, 0]  # First channel of the first

# fig,ax=plt.subplots(1,3, figsize=(15, 5))
# im0 = ax[0].imshow(input_ex)
# plt.colorbar(im0, ax=ax[0], shrink=0.2)
# ax[0].set_title("Input Foreground Estimate")
# im1 = ax[1].imshow(output_ex)
# plt.colorbar(im1, ax=ax[1], shrink=0.2)
# ax[1].set_title("ILC Residual")
# im2 = ax[2].imshow(pred_ex)
# plt.colorbar(im2, ax=ax[2], shrink=0.2)
# ax[2].set_title("Predicted ILC Residual")
# plt.tight_layout()
# plt.savefig('network.png', bbox_inches='tight', dpi=150)

# # Add histogram plots for data distribution analysis
# fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))

# # Histogram of input data
# ax_hist[0].hist(input_ex.flatten(), bins=50, alpha=0.7, color='blue', density=True)
# ax_hist[0].set_title("Input Foreground Estimate Distribution")
# ax_hist[0].set_xlabel("Pixel Value")
# ax_hist[0].set_ylabel("Density")
# ax_hist[0].grid(True, alpha=0.3)

# # Histogram of output data
# ax_hist[1].hist(output_ex.flatten(), bins=50, alpha=0.7, color='red', density=True)
# ax_hist[1].set_title("ILC Residual Distribution")
# ax_hist[1].set_xlabel("Pixel Value")
# ax_hist[1].set_ylabel("Density")
# ax_hist[1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('data_distributions.png', bbox_inches='tight', dpi=150)
# plt.show()