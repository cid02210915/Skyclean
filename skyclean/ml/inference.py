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
    
    def __init__(self, extract_comp, component, frequencies, realisations, lmax, N_directions=1, lam=2.0, chs=None, directory="data/", seed=0, model_path=None,
                 rn: int = 30, batch_size: int = 32, epochs: int = 120, learning_rate: float = 1e-3, momentum: float = 0.9, nsamp: int = 1200):
        """Initialize the CMB inference system.
        
        Parameters:
            frequencies (list): List of frequency strings.
            realisations (int): Number of realisations.
            lmax (int): Maximum multipole.
            N_directions (int): Number of directions for wavelet transform.
            lam (float): Lambda parameter.
            chs (list): List of channel dimensions for each layer. Default: [1, 16, 32, 32, 64]
            directory (str): Base data directory.
            seed (int): Random seed for model initialization.
            model_path (str, optional): Specific path to model checkpoint. If None, loads the latest model.
        """
        self.extract_comp = extract_comp
        self.component = component
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.chs = chs if chs is not None else [1, 16, 32, 32, 64]
        self.directory = directory
        self.seed = seed
        self.model_path = model_path
        self.rn = rn
        self.batch = batch_size
        self.epochs = epochs
        self.lr = learning_rate 
        self.momentum = momentum 
        self.nsamp = nsamp
        
        # Initialize file templates
        self.file_templates = FileTemplates(directory)
        
        # Model and config will be loaded when needed
        self.model = None
        self.config = None
        self.data_handler = CMBFreeILC(
                extract_comp=self.extract_comp,
                component=self.component,
                frequencies=self.frequencies,
                realisations=self.realisations,
                lmax=self.lmax,
                N_directions=self.N_directions,
                lam=self.lam,
                batch_size=1,  # Not used for inference
                directory=self.directory
            )

    
    '''
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
        temp_model = S2_UNET(L, ch_in, chs=self.chs, rngs=nnx.Rngs(self.seed))
        # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
        abstract_model = nnx.eval_shape(lambda: temp_model)
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.display(abstract_state)

        state_restored = checkpointer.restore(checkpoint_path, abstract_state)
        nnx.display(state_restored)

        model = nnx.merge(graphdef, state_restored)
        return model
    '''

    def load_model(self, force_load: bool = False):
        """Load and return a trained nnx.Module; also sets self.model."""
        # 1) Compatibility gate
        if not force_load:
            compatibility = self.check_model_compatibility()
            if not compatibility.get('compatible', False):
                raise RuntimeError(
                    f"Model compatibility check failed: {compatibility.get('message','')}. "
                    f"Pass force_load=True to bypass."
                )

        # 2) Resolve checkpoint path
        if getattr(self, "model_path", None):
            checkpoint_path = os.path.abspath(self.model_path)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Specified model path does not exist: {checkpoint_path}")
            print(f"Loading user-specified model from: {checkpoint_path}")
        else:
            model_dir = self.file_templates.output_directories["ml_models"]
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            ckpts = [f for f in os.listdir(model_dir) if f.startswith("checkpoint_")]
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {model_dir}")
            latest = max(ckpts, key=lambda x: int(x.split("_")[1]))
            checkpoint_path = os.path.abspath(os.path.join(model_dir, latest))
            print(f"Loading checkpoint from: {checkpoint_path}")

        # 3) Rebuild *exact same* architecture used at train time
        L = self.lmax + 1
        ch_in = len(self.frequencies)
        model = S2_UNET(L, ch_in, chs=self.chs, rngs=nnx.Rngs(getattr(self, "seed", 0)))

        # 4) Split to get graphdef and expected state structure and restore
        graphdef, _ = nnx.split(model)
        ckpt = ocp.StandardCheckpointer().restore(checkpoint_path)
        # tolerate both layouts (either {'params': ...} or just params)
        params = ckpt['params'] if isinstance(ckpt, dict) and 'params' in ckpt else ckpt
        import jax
        params = jax.tree.map(jax.device_put, params) # put the whole tree on device in one go (avoid many tiny transfers)


        # 5) Merge (single object return in your NNX)
        model = nnx.merge(graphdef, {'params': params})

        self.model = model
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
        

        return result
    
    
    def predict_cmb(self, realisation, save_result=True, masked=False):
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
        #print("Converting prediction to MW sampling...")
        cmb_mw = SamplingConverters.mwss_map_2_mw_map(cmb_pred, L=self.lmax + 1)

        
        # Save result if requested
        if save_result:
            if masked:
                mask_mw = self.data_handler.mask_mw_beamed()
                cmb_mw *= mask_mw
                self._save_masked_cmb_prediction(cmb_mw, realisation, masked)
            else:
                self._save_cmb_prediction(cmb_mw, realisation)
        
        print(f"CMB prediction completed for realisation {realisation}")
        print(f"Prediction shape: {cmb_mw.shape}")
        print(f"Value range: [{cmb_mw.min():.3e}, {cmb_mw.max():.3e}]")

        return cmb_mw
        

    def compute_mse(self, comp, realisation, save_result=True, masked=False):
        """
        Compute pixel-space MSE for a single realisation.

        Parameters
        ----------
        comp : str e.g. {"ilc", "nn"}
            - "ilc": MSE of the raw ILC map vs true CMB.
            - "nn" : MSE of the NN-predicted CMB map vs true CMB
                     (using the trained network).
        realisation : int
            Realisation index.

        Returns
        -------
        float
            Mean squared error for this realisation.
        """
        comp = comp.lower()
        if comp not in ("ilc", "nn"):
            raise ValueError("comp must be 'ilc' or 'nn'")
        
        # Get the data for this realisation
        F, R, _ = self.data_handler.create_residual_mwss_maps(realisation)
        # R has shape (H, W, 1); squeeze to (H, W)
        R = np.asarray(R)
        if R.ndim == 3 and R.shape[-1] == 1:
            R = R[..., 0]          # shape (H, W)
        elif R.ndim == 2:
            R = R                  # already (H, W)
        else:
            raise ValueError(f"Unexpected shape for R: {R.shape}")
        
        if masked:
            mask = self.data_handler.mask_mwss_beamed()  # (T, P) or (T, P, 1)
            mask = np.asarray(mask)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")
        else:
            mask = None

        if comp == "ilc":
            print(f"Calculating MSE(ILC) for realisation {realisation}...")
            diff = R
            
        else: # comp == "nn":
            print(f"Calculating MSE(NN) for realisation {realisation}...")
            
            if self.model is None: # Ensure model is loaded
                print("Loading model...")
                self.model = self.load_model()  
                print("Loaded model.")

            # Prepare network input (same pipeline as in predict_cmb)
            F = self.data_handler.transform(F).astype(np.float32)
            F = jnp.expand_dims(F, axis=0)  # Add batch dimension
            
            # Predict normalised residual and inverse-transform
            R_pred_norm = self.model(F)
            R_pred = self.data_handler.inverse_transform(R_pred_norm) # shape: ()
            R_pred = jnp.squeeze(R_pred, axis=(0, 3))  # Remove batch and channel dims
        
            # MSE(NN) = <(R_pred - R_true)^2>
            diff = R - np.asarray(R_pred)
        
        if mask is None:
            mse = float(np.mean(diff ** 2)) # in K
        else:
            w = mask
            num = np.sum(w * diff**2)
            denom = np.sum(w) + 1e-12
            mse = float(num / denom)

        print(mse)
        return mse

    
    def _save_cmb_prediction(self, cmb_prediction, realisation):
        """Save CMB prediction using FileTemplates.
        
        Parameters:
            cmb_prediction (np.ndarray): CMB prediction to save.
            realisation (int): Realisation number.
        """
        try:
            # Create a model configuration string for the filename
            chs = "_".join(str(n) for n in self.chs)
            #model_config = f"r{realisation}_lmax{self.lmax}_lam{self.lam}_nsamp{self.nsamp}_rn{self.rn}_batch{self.batch}_epo{self.epochs}_lr{self.lr}_mom{self.momentum}_chs{chs}.npy"

            # Use FileTemplates to get the save path
            save_path = self.file_templates.file_templates["ilc_improved_map"].format(
                realisation=realisation,
                lmax=self.lmax,
                lam=self.lam,
                nsamp=self.nsamp,
                rn=self.rn,
                batch=self.batch,
                epochs=self.epochs,
                lr=self.lr,
                momentum=self.momentum,
                chs=chs,
            )
            
            # Save the prediction
            np.save(save_path, cmb_prediction)

            print(f"Saved CMB prediction to: {save_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save CMB prediction: {str(e)}")
    
    def _save_masked_cmb_prediction(self, cmb_prediction, realisation, mask):
        try:
            model_config = f"lmax{self.lmax}_lam{self.lam}_freq{'_'.join(self.frequencies)}"
            save_path = self.file_templates.file_templates["ilc_improved_masked_map"].format(
                realisation=realisation,
                lmax=self.lmax,
                lam=self.lam,
                model_config=model_config
            )
            masked_cmb_prediction = cmb_prediction * mask
            np.save(save_path, masked_cmb_prediction)
            print(f"Saved masked CMB prediction to: {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save maked CMB prediction: {str(e)}")
    
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