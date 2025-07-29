import os
import sys
import numpy as np
import healpy as hp 

# import from skyclean/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import * 
from map_tools import * 
import tensorflow as tf
from sklearn.model_selection import train_test_split

class CMBFreeILC(): 
    def __init__(self, frequencies: list, realisations: int, lmax: int = 1024, N_directions: int = 1, 
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
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.directory = directory

        self.saved_directories = {
            "cfn": os.path.join(self.directory, "CFN_realisations"),
            "processed_maps": os.path.join(self.directory, "processed_maps"),
            "ilc_synthesised": os.path.join(self.directory, "SILC/ilc_synthesised_maps"),
        }

        ml_directory = os.path.join(self.directory, "ML/")
        if not os.path.exists(ml_directory):
            create_dir(ml_directory)
        self.saved_directories["ml"] = ml_directory


        self.file_templates = {
            "cfn": os.path.join(self.saved_directories["cfn"], "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
            "cmb": os.path.join(self.saved_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
            "ilc_synthesised": os.path.join(self.saved_directories["ilc_synthesised"], "ilc_synthesised_map_r{realisation:04d}_lmax{lmax}.npy"),
            "foreground_estimate": os.path.join(self.saved_directories["ml"], "foreground_estimate_r{realisation:04d}_lmax{lmax}.npy"),
            "ilc_residual": os.path.join(self.saved_directories["ml"], "ilc_residual_r{realisation:04d}_lmax{lmax}.npy"),
        }
        # retrieve shapes
        ilc_map_temp = np.load(self.file_templates["ilc_synthesised"].format(realisation=0, lmax=self.lmax))
        self.H = ilc_map_temp.shape[0]
        self.W = ilc_map_temp.shape[1]
    
    def create_residual_maps(self, realisation: int): 
        """For a single realisation, create the ILC residual maps (CMB free) in MW sampling format. 

        Parameters:
            realisation (int): The realisation number to process.

        Returns:
            foreground_estimate (np.ndarray): F(i) = CFN(i) - ILC where i is the frequency component (hence have N_freq input channels) of shape (H, W, N_freq)
            ilc_residual (np.ndarray): ILC - CMB of shape (H, W, 1). 
        """
        H, W, lmax = self.H, self.W, self.lmax
        frequencies = self.frequencies
        if os.path.exists(self.file_templates["foreground_estimate"].format(realisation=realisation, lmax=lmax)) and os.path.exists(self.file_templates["ilc_residual"].format(realisation=realisation, lmax=lmax)):
            foreground_estimate = np.load(self.file_templates["foreground_estimate"].format(realisation=realisation, lmax=lmax))
            ilc_residual = np.load(self.file_templates["ilc_residual"].format(realisation=realisation, lmax=lmax))
        else:
            print(f"Creating residual maps for realisation {realisation}...")
            # load ilc (already in MW sampling)
            ilc_map_mw = np.load(self.file_templates["ilc_synthesised"].format(realisation=realisation, lmax=lmax))
            # load cmb and convert to MW sampling
            cmb_map_hp = hp.read_map(self.file_templates["cmb"].format(realisation=realisation, lmax=lmax), dtype=np.float64)
            cmb_map_mw = SamplingConverters.hp_map_2_mw_map(cmb_map_hp, lmax) # highly expensive? involves s2fft.forwards.
            # load cfn maps across frequencies and convert to MW sampling
            cfn_maps_hp = [hp.read_map(self.file_templates["cfn"].format(frequency=frequency, realisation=realisation, lmax=lmax), dtype=np.float64) for frequency in frequencies]
            cfn_maps_mw = [SamplingConverters.hp_map_2_mw_map(cfn_map_hp, lmax) for cfn_map_hp in cfn_maps_hp]
            # create foreground estimate and ilc residual
            foreground_estimate = np.zeros((H, W, len(frequencies)), dtype=np.float64)
            ilc_residual = np.zeros((H, W, 1), dtype=np.float64)
            for i, frequency in enumerate(frequencies):
                foreground_estimate[:, :, i] = cfn_maps_mw[i] - ilc_map_mw
                ilc_residual[:, :, 0] = ilc_map_mw - cmb_map_mw
            # save the maps
            np.save(self.file_templates["foreground_estimate"].format(realisation=realisation, lmax=lmax), foreground_estimate)
            np.save(self.file_templates["ilc_residual"].format(realisation=realisation, lmax=lmax), ilc_residual)
        return foreground_estimate, ilc_residual

    @staticmethod
    def concatenate_maps(self): 
        """Concatenate maps across realisations, appending another axis to the arrays, resulting in a shape of (realisations, H, W, N_freq) 
        for foreground_estimate and (realisations, H, W, 1) for ilc_residual."""
        realisations = self.realisations
        H, W = self.H, self.W
        N_freq = len(self.frequencies)
        X = np.zeros((realisations, H, W, N_freq), dtype=np.float64)
        Y = np.zeros((realisations, H, W, 1), dtype=np.float64)
        for realisation in range(realisations):
            foreground_estimate, ilc_residual = self.create_residual_maps(realisation)
            X[realisation, :, :, :] = foreground_estimate
            Y[realisation, :, :, :] = ilc_residual
        return X, Y

    def split_data(self, X: np.ndarray, Y: np.ndarray):
        """Split data into train and test. 

        Parameters:
            X (np.ndarray): Input data (foreground estimates).
            Y (np.ndarray): Target data (ILC residuals).
            split (list): List of proportions for train, validation, and test splits.

        Returns:
            X_train, Y_train, X_val, Y_val, X_test, Y_test:
        """
        assert sum(self.split) == 1, "Split proportions must sum to 1."

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.split[1], shuffle=self.shuffle)

        return X_train, Y_train, X_test, Y_test

    def prepare_tf_dataset(self, X: np.ndarray, Y: np.ndarray): 
        """Prepare a TensorFlow dataset for training and testing. This includes applying transforms, shuffling, and batching the data.

        Parameters:
            X (np.ndarray): Input data (foreground estimates).
            Y (np.ndarray): Target data (ILC residuals).
        
        Returns:
            tf.data.Dataset: A TensorFlow dataset ready for training.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Optimize for performance
        return dataset

    def prepare_data(self):
        print("Preparing datasets...")
        X, Y = self.concatenate_maps()
        X_train, Y_train, X_test, Y_test = self.split_data(X, Y)
        train_ds = self.prepare_tf_dataset(X_train, Y_train)
        test_ds = self.prepare_tf_dataset(X_test, Y_test)
        return train_ds, test_ds


