"""
CMB-Free ILC Model Inference Class.

This module provides a class-based interface for loading trained models and applying them for inference,
with integrated FileTemplates support for organized data management.
"""

import os
import numpy as np
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from .model import S2_UNET
from .data import CMBFreeILC
from skyclean.silc.file_templates import FileTemplates
from skyclean.silc import SamplingConverters


class Inference:
    """Class for CMB prediction inference using trained models."""
    
    def __init__(self, frequencies, realisations, lmax, N_directions=1, lam=2.0, directory="data/", seed=0):
        """Initialize the CMB inference system.
        
        Parameters:
            frequencies (list): List of frequency strings.
            realisations (int): Number of realisations.
            lmax (int): Maximum multipole.
            N_directions (int): Number of directions for wavelet transform.
            lam (float): Lambda parameter.
            directory (str): Base data directory.
            seed (int): Random seed for model initialization.
        """
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.directory = directory
        self.seed = seed
        
        # Initialize file templates
        self.file_templates = FileTemplates(directory)
        
        # Model and config will be loaded when needed
        self.model = None
        self.config = None
        self.data_handler = CMBFreeILC(
                frequencies=self.frequencies,
                realisations=self.realisations,
                lmax=self.lmax,
                N_directions=self.N_directions,
                lam=self.lam,
                batch_size=1,  # Not used for inference
                directory=self.directory
            )

    
    def load_model(self):
        """Load the latest model weights for inference using fresh start approach.
        
        Parameters:
            model: Unused in fresh start approach (kept for compatibility).
            model_dir (str): Directory containing model checkpoints.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        checkpointer = ocp.StandardCheckpointer()
        model_dir = self.file_templates.output_directories["ml_models"]
        # Find the latest checkpoint
        checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')]
        print(f"Found checkpoint files: {checkpoint_files}")
        if not checkpoint_files:
            print("No checkpoints found.")
            return False
        
        # Get the latest checkpoint by epoch number
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1]))
        checkpoint_path = os.path.abspath(os.path.join(model_dir, latest_checkpoint))
        
        print(f"Loading checkpoint from: {checkpoint_path}")

        # Fresh start approach following Flax NNX documentation
        # First create a concrete model to get the state structure
        L = self.lmax + 1
        ch_in = len(self.frequencies)
        
        # Create a concrete model instance to get the state structure
        temp_model = S2_UNET(L, ch_in, rngs=nnx.Rngs(self.seed))
        # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
        abstract_model = nnx.eval_shape(lambda: temp_model)
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.display(abstract_state)

        state_restored = checkpointer.restore(checkpoint_path, abstract_state)
        nnx.display(state_restored)

        model = nnx.merge(graphdef, state_restored)
        return model
    
    
    def predict_cmb(self, realisation, save_result=True):
        """Predict CMB for a specific realisation.
        
        Parameters:
            realisation (int): The realisation number to process.
            save_result (bool): Whether to save the result using FileTemplates.
            
        Returns:
            np.ndarray: CMB prediction in MW sampling format.
        """
        # Ensure model is loaded
        if self.model is None:
            print("Loading model...")
            self.model = self.load_model()
            print("Loaded model.")
        
        
        print(f"Predicting CMB for realisation {realisation}...")
        
        # Get the data for this realisation
        F, _, ilc_mwss = self.data_handler.create_residual_mwss_maps(realisation)
        
        # Transform and prepare for model input
        F = self.data_handler.transform(F).astype(np.float32)
        F = jnp.expand_dims(F, axis=0)  # Add batch dimension
        
        # Apply model
        R_pred_norm = self.model(F)
        
        # Inverse transform to get residual prediction
        R_pred = self.data_handler.inverse_transform(R_pred_norm)
        R_pred = jnp.squeeze(R_pred, axis=(0, 3))  # Remove batch and channel dims
        
        # Compute CMB prediction
        cmb_pred = ilc_mwss - R_pred
        
        # Convert to MW sampling
        print("Converting prediction to MW sampling...")
        cmb_mw = SamplingConverters.mwss_map_2_mw_map(cmb_pred, L=self.lmax + 1)
        
        # Save result if requested
        if save_result:
            self._save_cmb_prediction(cmb_mw, realisation)
        
        print(f"CMB prediction completed for realisation {realisation}")
        print(f"Prediction shape: {cmb_mw.shape}")
        print(f"Value range: [{cmb_mw.min():.3e}, {cmb_mw.max():.3e}]")
        
        return cmb_mw
        

    
    def _save_cmb_prediction(self, cmb_prediction, realisation):
        """Save CMB prediction using FileTemplates.
        
        Parameters:
            cmb_prediction (np.ndarray): CMB prediction to save.
            realisation (int): Realisation number.
        """
        try:
            # Create a model configuration string for the filename
            model_config = f"lmax{self.lmax}_lam{self.lam}_freq{'_'.join(self.frequencies)}"
            
            # Use FileTemplates to get the save path
            save_path = self.file_templates.file_templates["ilc_improved_map"].format(
                realisation=realisation,
                lmax=self.lmax,
                lam=self.lam,
                model_config=model_config
            )
            
            # Save the prediction
            np.save(save_path, cmb_prediction)
            print(f"Saved CMB prediction to: {save_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save CMB prediction: {str(e)}")
    
    def get_model_info(self):
        """Get information about the loaded model.
        
        Returns:
            dict: Model information including configuration and status.
        """
        info = {
            'model_loaded': self.model is not None,
            'frequencies': self.frequencies,
            'lmax': self.lmax,
            'N_directions': self.N_directions,
            'lam': self.lam,
            'directory': self.directory,
            'model_dir': self.file_templates.output_directories["ml_models"]
        }
        
        if self.config is not None:
            info.update({
                'trained_frequencies': self.config.get('frequencies'),
                'trained_lmax': self.config.get('lmax'),
                'trained_L': self.config.get('L'),
                'trained_channels': self.config.get('ch_in'),
                'training_params': {
                    'batch_size': self.config.get('batch_size'),
                    'learning_rate': self.config.get('learning_rate'),
                    'momentum': self.config.get('momentum')
                }
            })
        
        return info

# Example inference.
if __name__ == "__main__":
    frequencies = ["030", "100", "353"]
    realisations = 1000
    lmax = 511
    N_directions = 1
    lam = 2.0
    directory = "/Scratch/matthew/data/"

    inference = Inference(
        frequencies=frequencies,
        realisations=1000,
        lmax=lmax,
        N_directions=N_directions,
        lam=lam,
        directory=directory
    )
    
    print("\n1. Model Information:")
    info = inference.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("Realisation Prediction:")
    cmb_pred = inference.predict_cmb(realisation=0)
    print("successful.")