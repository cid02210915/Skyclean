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
    
    def __init__(self, frequencies, realisations, lmax, N_directions=1, lam=2.0, directory="data/", seed=0, model_path=None):
        """Initialize the CMB inference system.
        
        Parameters:
            frequencies (list): List of frequency strings.
            realisations (int): Number of realisations.
            lmax (int): Maximum multipole.
            N_directions (int): Number of directions for wavelet transform.
            lam (float): Lambda parameter.
            directory (str): Base data directory.
            seed (int): Random seed for model initialization.
            model_path (str, optional): Specific path to model checkpoint. If None, loads the latest model.
        """
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.directory = directory
        self.seed = seed
        self.model_path = model_path
        
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

    
    def load_model(self, force_load=False):
        """Load model weights for inference using fresh start approach.
        
        Uses the class variable model_path if set, otherwise loads the latest model.
        Includes compatibility checking unless force_load is True.
        
        Parameters:
            force_load (bool): If True, skip compatibility check and force load.
            
        Returns:
            nnx.Module: The loaded model.
        """
        # Check compatibility first unless force_load is True
        if not force_load:
            compatibility = self.check_model_compatibility()
            if not compatibility['compatible']:
                print(f"Model compatibility check failed: {compatibility['message']}")
                print("Use force_load=True to bypass this check if you're sure the model is correct.")
                return False
            else:
                print(f"Model compatibility check passed: {compatibility['message']}")
        
        checkpointer = ocp.StandardCheckpointer()
        
        if self.model_path is not None:
            # Use user-specified model path from class variable
            checkpoint_path = os.path.abspath(self.model_path)
            print(f"Loading user-specified model from: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                print(f"Error: Specified model path does not exist: {checkpoint_path}")
                return False
        else:
            # Find the latest checkpoint automatically
            model_dir = self.file_templates.output_directories["ml_models"]
            checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')]
            print(f"Found checkpoint files: {checkpoint_files}")
            if not checkpoint_files:
                print("No checkpoints found.")
                return False
            
            # Get the latest checkpoint by epoch number
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1]))
            checkpoint_path = os.path.abspath(os.path.join(model_dir, latest_checkpoint))
        
        print(f"Loading checkpoint from: {checkpoint_path}")

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
    
    def check_model_compatibility(self):
        """Check if the model path is compatible with initialized variables.
        
        This method attempts to load the model and verify that its architecture
        matches the current instance parameters (frequencies, lmax, etc.).
        
        Returns:
            dict: Dictionary containing compatibility information with keys:
                - 'compatible': bool indicating if model is compatible
                - 'message': str describing the compatibility status
                - 'model_info': dict with model architecture details (if loadable)
                - 'expected_info': dict with expected architecture details
        """
        expected_L = self.lmax + 1
        expected_ch_in = len(self.frequencies)
        
        result = {
            'compatible': False,
            'message': '',
            'model_info': {},
            'expected_info': {
                'L': expected_L,
                'lmax': self.lmax,
                'channels': expected_ch_in,
                'frequencies': self.frequencies,
                'N_directions': self.N_directions,
                'lam': self.lam
            }
        }
        
        # Check if model path exists
        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                result['message'] = f"Model path does not exist: {self.model_path}"
                return result
        else:
            # Check if default model directory has checkpoints
            model_dir = self.file_templates.output_directories["ml_models"]
            if not os.path.exists(model_dir):
                result['message'] = f"Default model directory does not exist: {model_dir}"
                return result
            
            checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')]
            if not checkpoint_files:
                result['message'] = f"No checkpoint files found in: {model_dir}"
                return result
        
        try:
            # Attempt to create model with current parameters
            temp_model = S2_UNET(expected_L, expected_ch_in, rngs=nnx.Rngs(self.seed))
            
            # Try to load the checkpoint
            checkpointer = ocp.StandardCheckpointer()
            
            if self.model_path is not None:
                checkpoint_path = os.path.abspath(self.model_path)
            else:
                model_dir = self.file_templates.output_directories["ml_models"]
                checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_')]
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1]))
                checkpoint_path = os.path.abspath(os.path.join(model_dir, latest_checkpoint))
            
            # Test restoration
            abstract_model = nnx.eval_shape(lambda: temp_model)
            graphdef, abstract_state = nnx.split(abstract_model)
            
            # Attempt to restore - this will fail if architecture doesn't match
            state_restored = checkpointer.restore(checkpoint_path, abstract_state)
            
            # If we get here, the model is compatible
            result['compatible'] = True
            result['message'] = f"Model is compatible with current parameters"
            result['model_info'] = {
                'checkpoint_path': checkpoint_path,
                'L': expected_L,
                'lmax': self.lmax,
                'channels': expected_ch_in,
                'frequencies': self.frequencies
            }
            
        except Exception as e:
            result['message'] = f"Model incompatibility detected: {str(e)}"
            if "shape" in str(e).lower() or "dimension" in str(e).lower():
                result['message'] += f"\nExpected: L={expected_L}, channels={expected_ch_in}"
                result['message'] += f"\nThis usually indicates the model was trained with different lmax or frequencies."
        
        return result
    
    
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
            'model_path': self.model_path,
            'frequencies': self.frequencies,
            'lmax': self.lmax,
            'N_directions': self.N_directions,
            'lam': self.lam,
            'directory': self.directory,
            'model_dir': self.file_templates.output_directories["ml_models"]
        }
        
        # Add compatibility check information
        compatibility = self.check_model_compatibility()
        info['model_compatibility'] = compatibility
        
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