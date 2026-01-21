# Adapted from: https://github.com/astro-informatics/s2ai
# Original code by: Matthew A. Price, Kevin Mulder, Jason D. McEwen
# License: MIT

import argparse
import os
import sys

# GPU configuration
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # remove this line because it forces the use of physical GPU 0, but Slurm may assign different GPU IDs

import jax
print("available JAX GPU devices:", jax.devices("gpu")) # check available GPUs

jax.config.update("jax_enable_x64", False)  # Use 32-bit
try:
    gpus = jax.devices("gpu")
except Exception:
    gpus = []
if not gpus:
    print("No GPU visible to JAX. devices() =", jax.devices())
else:
    gpu = gpus[0]
    info = gpu.memory_stats()

    if info is None:
        print("JAX memory_stats() not available on this backend.")
        print("Device:", gpu)
    else:
        bytes_limit = info.get("bytes_limit", None)
        bytes_in_use = info.get("bytes_in_use", None)

        if bytes_limit is None:
            print("memory_stats() returned keys:", list(info.keys()))
        else:
            print("GPU 0 bytes_limit:", bytes_limit / 1024**3, "GiB")
            if bytes_in_use is not None:
                print("GPU 0 bytes_in_use:", bytes_in_use / 1024**3, "GiB")
#gpu = jax.devices()[0]
#info = gpu.memory_stats()
#print('Initial GPU usage for device 0:', info["bytes_limit"] / 1024**3, "GiB")

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import s2fft
import functools
import orbax.checkpoint as ocp
import atexit
import re
from pathlib import Path

from .model import S2_UNET
from .data import CMBFreeILC
from skyclean.silc.utils import create_dir
from skyclean.silc.file_templates import FileTemplates

import matplotlib.pyplot as plt

def get_gpu_memory_usage():
    """Get current GPU memory usage as percentage and MB using JAX."""
    devices = jax.devices()
    gpu_devices = [d for d in devices]
    
    # Use the first GPU device
    gpu = gpu_devices[0]
    
    # Get memory info through JAX backend
    memory_info = gpu.memory_stats()
    bytes_in_use = memory_info.get('bytes_in_use', 0)
    bytes_limit = memory_info.get('bytes_limit', 1)
    
    mb_used = bytes_in_use / (1024 ** 2)
    mb_total = bytes_limit / (1024 ** 2) 
    percentage = (bytes_in_use / bytes_limit) * 100 if bytes_limit > 0 else 0
    
    return percentage, mb_used, mb_total

def get_gpu_memory_usage(device_id: int = 0):
    """
    Return (percentage, mb_used, mb_total) for GPU device_id.
    If not available, return (None, None, None).
    """
    # Prefer actual GPU devices; fall back gracefully.
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []

    if not gpus: # No GPU backend visible to JAX
        return None, None, None

    if device_id >= len(gpus):
        device_id = 0

    gpu = gpus[device_id]
    mem = gpu.memory_stats()

    if mem is None: # Some backends return None
        return None, None, None

    # Keys can vary; use .get and guard division.
    bytes_in_use = mem.get("bytes_in_use", None)
    bytes_limit  = mem.get("bytes_limit", None)

    if bytes_in_use is None or bytes_limit is None or bytes_limit == 0:
        return None, None, None

    mb_used = bytes_in_use / (1024 ** 2)
    mb_total = bytes_limit / (1024 ** 2)
    percentage = (bytes_in_use / bytes_limit) * 100.0

    return percentage, mb_used, mb_total

# def print_gpu_usage(stage_name):
def print_gpu_usage(stage_name: str, device_id: int = 0):
    """Print GPU memory usage for a given stage."""
    # percentage, mb_used, mb_total = get_gpu_memory_usage()
    percentage, mb_used, mb_total = get_gpu_memory_usage(device_id=device_id)
    if percentage is not None:
        print(f"[GPU Memory] {stage_name}: {percentage:.1f}% ({mb_used:.0f}/{mb_total:.0f} MB)")
    else:
        print(f"[GPU Memory] {stage_name}: Unable to query GPU memory via JAX memory_stats()")


class Train: 
    def __init__(self,  extract_comp: str, component: str, frequencies: list, realisations: int, 
                 lmax: int = 1024, N_directions: int = 1, lam: float = 2.0,
                 batch_size: int = 32, shuffle: bool = True, split: list = [0.8,0.2], epochs: int = 120, 
                 learning_rate: float = 1e-3, momentum: float = 0.9, chs: list = None, rngs: nnx.Rngs = nnx.Rngs(0), 
                 directory: str = "data/", resume_training: bool = False,  loss_tag: str | None = 'pixel', 
                 random_generator: bool = False):
        """
        Parameters:
            frequencies (list): List of frequencies for the maps.
            realisations (int): Number of realisations to process.
            lmax (int): Maximum multipole for the wavelet transform.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float): lambda factor (scaling) for the wavelet transform.
            batch_size (int): Size of the batches for training.
            shuffle (bool): Whether to shuffle the dataset.
            split (list): List of train/validation/test split ratios.
            epochs (int): Number of epochs to train for.
            learning_rate (float): Learning rate for the optimizer.
            momentum (float): Momentum for the optimizer.
            chs (list): List of channel dimensions for each layer. Default: [1, 16, 32, 32, 64]
            rngs (nnx.Rngs): Random number generators for the model.
            directory (str): Directory where data is stored / saved to.
            resume_training (bool): Whether to resume training from the last checkpoint.
            run_tag (str | None): Optional tag to identify the use of loss function.
            random_generator (bool): Whether to use random generator for test maps.
        """ 
        self.component = component
        self.extract_comp = extract_comp
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.chs = chs if chs is not None else [1, 16, 32, 32, 64]
        self.rngs = rngs
        self.directory = directory
        self.resume_training = resume_training
        self.loss_tag = loss_tag
        self.random_generator = random_generator

        self.dataset = CMBFreeILC(extract_comp, component, frequencies, realisations, lmax, N_directions, lam, batch_size, shuffle, split, directory, random=random_generator)
        
        # Create model directory for checkpoints
        files = FileTemplates(directory)  # Pass directory to FileTemplates
        # Ensure model_dir is absolute path
        self.model_dir = os.path.abspath(files.output_directories["ml_models"])


        #self.save_dir = os.path.join(self.directory, "ML/model")
        #if not os.path.exists(self.save_dir):
        #    create_dir(self.save_dir)
        
        # Initialize Orbax checkpoint manager
        self.checkpointer = ocp.StandardCheckpointer()

        # ensure checkpoints are saved before exit
        atexit.register(lambda: getattr(self.checkpointer, "wait_until_finished", lambda: None)())


    @staticmethod
    def clear_gpu_cache():
        """Clear GPU memory cache and force garbage collection."""
        import gc
        print("[GPU Memory] Clearing cache...")
        jax.clear_caches()
        gc.collect()
        # Synchronize to ensure all operations complete
        jax.block_until_ready(jnp.array([1.0]))
        print("[GPU Memory] Cache cleared")
    
    def save_model(self, model, epoch):
        """Save the model state using Orbax.

        Parameters:
            model (nnx.Module): The model to save.
            epoch (int): Current epoch number.
        """
        try:
            _, state = nnx.split(model)
            checkpoint_path = os.path.abspath(os.path.join(self.model_dir, f"checkpoint_{epoch}"))
            os.makedirs(self.model_dir, exist_ok=True)
            if os.path.exists(checkpoint_path):
                # Remove stale or partial checkpoints before overwriting.
                import shutil
                shutil.rmtree(checkpoint_path)

            self.checkpointer.save(checkpoint_path, state)
            if hasattr(self.checkpointer, "wait_until_finished"):
                self.checkpointer.wait_until_finished()
            # keep only latest checkpoint
            if epoch > 1:
                prev_checkpoint_path = os.path.abspath(os.path.join(self.model_dir, f"checkpoint_{epoch-1}"))
                if os.path.exists(prev_checkpoint_path):
                    import shutil
                    shutil.rmtree(prev_checkpoint_path)

            print(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save model at epoch {epoch}: {str(e)}")
    

    def load_model_for_training(self, model, optimizer):
        """
        Load the latest saved model checkpoint into `model`.

        Returns
        -------
        last_epoch : int
            Last epoch number for which a checkpoint exists.
            Returns 0 if no checkpoint is found.
        """
        model_dir = os.path.abspath(self.model_dir)

        if not os.path.isdir(model_dir):
            print(f"No model directory found at {model_dir}. Starting from scratch.")
            return 0

        # Find all checkpoint_* directories
        entries = [
            d for d in os.listdir(model_dir)
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(model_dir, d))
        ]

        if not entries:
            print("No checkpoints found. Starting from scratch.")
            return 0

        # Extract epoch numbers from "checkpoint_{epoch}"
        epochs = []
        for name in entries:
            m = re.match(r"checkpoint_(\d+)", name)
            if m:
                epochs.append(int(m.group(1)))

        if not epochs:
            print("No valid checkpoint names found. Starting from scratch.")
            return 0

        last_epoch = max(epochs)
        checkpoint_path = os.path.join(model_dir, f"checkpoint_{last_epoch}")

        print(f"Loading model checkpoint from: {checkpoint_path}")

        try:
            # Build abstract state from current model structure
            _, abstract_state = nnx.split(model)

            checkpointer = ocp.StandardCheckpointer()
            restored_state = checkpointer.restore(checkpoint_path, abstract_state)

            # Update the live model in-place
            nnx.update(model, restored_state)

            print(f"Successfully loaded model from epoch {last_epoch}")
            return last_epoch

        except Exception as e:
            print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {e}")
            print("Starting from scratch instead.")
            return 0


    @staticmethod
    def pix_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss):
        """
        Pixel 
        Train to predict the ILC residual ΔT_ILC, as in McCarthy+: L = Σ_p (ΔT̂_ILC(p) - ΔT_ILC(p))^2
        with quadrature weighting over t and averaging over batch.

        Updated: 
            Masked, quadrature-weighted MAE.

        pred_residuals, residuals: (B, T, P, C)
        norm_quad_weights: (T,)
        mask_mwss: (T, P) or (T, P, 1)
        """
        diff_sq = (pred_residuals - residuals)**2                  # (b, t, p, c=1)
        
        mask = jnp.asarray(mask_mwss)
        if mask.ndim == 2:
            mask = mask[None, :, :, None]    # (1, T, P, 1)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[None, :, :, :]       # (1, T, P, 1)
        else:
            raise ValueError(f"Unexpected mask_mwss shape: {mask.shape}")
        
        # --- build weight map w(T,P) = w_quad(T) * mask(T,P) ---
        w_t = jnp.asarray(norm_quad_weights)[None, :, None, None]  # (1, T, 1, 1)
        weights = w_t * mask                 # (1, T, P, 1), broadcasts over batch
        # (Masked) weighted sum over (t, p, c), using norm_quad_weights[t]
        #weighted_sum = jnp.einsum("btpc,t->", diff_sq, weights, optimize=True)
        #return weighted_sum / residuals.shape[0] # Average over batch
        num = jnp.sum(diff_sq * weights)             # Σ_{b,t,p,c} w_t M_{tp} diff^2
        den = jnp.sum(weights) + 1e-12               # Σ_{b,t,p,c} w_t M_{tp}
        loss = num / den
        return loss

    @staticmethod
    def pix_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss):
        """ Pixel
        Accuracy in the McCarthy+ sense:
            acc = 1 - MSE_clean / MSE_ILC

        where:
            MSE_ILC   = ⟨ (ΔT_ILC)^2 ⟩
            MSE_clean = ⟨ (ΔT_ILC - ΔT̂_ILC)^2 ⟩
        we expect a smaller MSE_clean than MSE_ILC. 

        pred_residuals, residuals: (B, T, P, C)
        norm_quad_weights: (T,)
        mask_mwss: (T, P) or (T, P, 1)
        """
        delta_ilc = residuals
        pred_delta_ilc = pred_residuals
        # broadcast mask to (1, T, P, 1)
        mask = jnp.asarray(mask_mwss)
        if mask.ndim == 2:
            mask = mask[None, :, :, None]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[None, :, :, :]
        else:
            raise ValueError(f"Unexpected mask_mwss shape: {mask.shape}")

        # w_t: (1, T, 1, 1)
        w_t = jnp.asarray(norm_quad_weights)[None, :, None, None]

        weights = w_t * mask   # (1, T, P, 1), broadcasts over batch

        # MSE_ILC = < (ΔT_ILC)^2 >
        diff_ilc_sq = delta_ilc**2
        num_ilc = jnp.sum(diff_ilc_sq * weights)
        den = jnp.sum(weights) + 1e-24
        mse_ilc = num_ilc / den

        # MSE_clean = < (ΔT_ILC - ΔT̂_ILC)^2 >
        diff_clean_sq = (delta_ilc - pred_delta_ilc)**2
        num_clean = jnp.sum(diff_clean_sq * weights)
        mse_clean = num_clean / den

        acc = 1.0 - mse_clean / mse_ilc # Fractional improvement
        return acc
    
    def harm_loss_fn_from_pred(
        pred_residuals: jnp.ndarray,
        residuals: jnp.ndarray,
        norm_quad_weights: jnp.ndarray,
        mask_mwss: jnp.ndarray,
        L: int = 1024,
    ):
        """
        Harmonic loss:
        - Compute spherical harmonic coefficients of the residual maps
        - Minimise |a_lm(pred) - a_lm(true)|^2, with quadrature weights over T.
        
        pred_residuals, residuals: (B, T, P, C=1)
        mask_mwss: (T, P) or (T, P, 1)
        """
        # --- handle mask (same logic as pixel loss) ---
        mask = jnp.asarray(mask_mwss)
        if mask.ndim == 2:
            mask = mask[:, :, None]       # (T, P, 1)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            pass                          # already (T, P, 1)
        else:
            raise ValueError(f"Unexpected mask_mwss shape: {mask.shape}")

        # Apply mask in pixel space (same for pred and target)
        pred_residuals = pred_residuals * mask
        residuals      = residuals * mask

        # Remove the channel dimension for the transform: (B, T, P)
        pred_maps   = pred_residuals[..., 0]
        target_maps = residuals[..., 0]

        # ---- spherical harmonic transform, batched over (B, T) ----
        # forward: map (P,) -> alm (ℓ,m) representation
        forward = functools.partial(s2fft.forward, L=L, method="jax_cuda")

        # vmap over time, then over batch: in_axes=(0)-> over T, then (0)-> over 
        forward_b = jax.vmap(forward, in_axes=0, out_axes=0)       # (B, T, P) -> (B, ℓ, m)

        pred_spec   = forward_b(pred_maps)     # (B, ℓ, m)
        target_spec = forward_b(target_maps)   # (B, ℓ, m)

        # Take magnitude so we compare |a_{ℓm}|:
        pred_amps   = jnp.abs(pred_spec)
        target_amps = jnp.abs(target_spec) # why absolute? 
        #pred_amps   = pred_spec
        #target_amps = target_spec

        # pointwise L2 loss in harmonic space: (B, T, ℓ, m)
        losses = optax.l2_loss(target_amps, pred_amps)

        return jnp.mean(losses) # average over batch


    @staticmethod
    def harm_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L):
        """
        Harmonic-space accuracy in the McCarthy+ sense:
            acc = 1 - MSE_clean / MSE_ILC

        computed in spherical-harmonic space, *without* any quadrature
        weighting over T, and only vmapping the forward transform over
        the batch axis.

        pred_residuals, residuals: (B, T, P, C)
        mask_mwss: (T, P) or (T, P, 1)
        L: band-limit for s2fft.forward
        """
        # aliases
        delta_ilc = residuals
        pred_delta_ilc = pred_residuals

        # --- broadcast mask to (1, T, P, 1) like in pix_acc_fn_from_pred ---
        mask = jnp.asarray(mask_mwss)
        if mask.ndim == 2:
            mask = mask[None, :, :, None]    # (1, T, P, 1)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[None, :, :, :]       # (1, T, P, 1)
        else:
            raise ValueError(f"Unexpected mask_mwss shape: {mask.shape}")

        # apply mask in pixel space
        delta_ilc = delta_ilc * mask        # (B, T, P, C)
        pred_delta_ilc = pred_delta_ilc * mask

        # drop channel dimension -> (B, T, P)
        delta_ilc_maps = delta_ilc[..., 0]
        pred_delta_ilc_maps = pred_delta_ilc[..., 0]

        # --- spherical harmonic transform, vmapped only over batch (B) ---
        # forward: (T, P) -> alm(ℓ, m, ...)  (whatever layout your s2fft.forward uses)
        forward = functools.partial(s2fft.forward, L=L, method="jax_cuda")

        # vmap over batch axis 0: in_axes=0, out_axes=0
        forward_batch = jax.vmap(forward, in_axes=0, out_axes=0)

        # alm shapes: (B, ..., ℓ, m) depending on s2fft config
        alm_ilc = forward_batch(delta_ilc_maps)
        alm_pred = forward_batch(pred_delta_ilc_maps)

        # work with amplitudes |a_{ℓm}| to match your earlier pattern
        amp_ilc = jnp.abs(alm_ilc)
        amp_pred = jnp.abs(alm_pred)

        # --- define harmonic MSEs, unweighted over (B, T, ℓ, m, ...) ---
        # MSE_ILC^harm = < |a_ilc|^2 >
        diff_ilc_sq = amp_ilc**2
        mse_ilc = jnp.mean(diff_ilc_sq)

        # MSE_clean^harm = < (|a_ilc| - |a_pred|)^2 >
        diff_clean_sq = (amp_ilc - amp_pred)**2
        mse_clean = jnp.mean(diff_clean_sq)

        acc = 1.0 - mse_clean / (mse_ilc + 1e-24)
        return acc


    def loss_and_acc_fn(self, model: nnx.Module, images: jnp.ndarray, residuals: jnp.ndarray, norm_quad_weights: jnp.ndarray, mask_mwss: jnp.ndarray):
        """
        Forward pass returning:
            - loss: MSE on ΔT_ILC (for gradients)
            - acc : fractional improvement over ILC (for logging)
        """
        # Single forward pass
        pred_residuals = model(images)
        #loss = Train.pix_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss)
        loss = Train.harm_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L=self.lmax+1)
        #accuracy = Train.pix_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss)
        accuracy = Train.harm_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L=self.lmax+1)
        return loss, accuracy # Return loss as main value, accuracy as aux

    #@jax.jit
    @functools.partial(jax.jit, static_argnums=0) # self argument is treated as static
    # 'static' means: treat that argument as compile-time constant, not a JAX value and JAX wont try to stage this.
    def train_step(self, graphdef: nnx.GraphDef, state: nnx.State, images: jnp.ndarray, residuals: jnp.ndarray, 
                   norm_quad_weights: jnp.ndarray, mask_mwss: jnp.ndarray):
        """Perform a single training step on a batch of data.

        Parameters:
            graphdef (nnx.GraphDef): The graph definition containing the model and optimizer.
            state (nnx.State): The current state of the model, optimizer, and metrics.
            images (jnp.ndarray): Input images.
            residuals (jnp.ndarray): Target residuals.
            norm_quad_weights (jnp.ndarray): Normalized quadrature weights.

        Returns:
            nnx.State: Updated state after the training step.
        """
        model, optimizer, metrics = nnx.merge(graphdef, state)
        model.train()

        # value_and_grad will see: (loss, accuracy)
        # loss is used for grads; accuracy is treated as "auxiliary" data
        (loss, accuracy), grads = nnx.value_and_grad(self.loss_and_acc_fn, 
                                                    has_aux=True)(model, images, residuals, norm_quad_weights, mask_mwss)

        optimizer.update(grads)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state

    @functools.partial(jax.jit, static_argnums=0)
    def eval_step(self, graphdef: nnx.GraphDef, state: nnx.State, images: jnp.ndarray, residuals: jnp.ndarray, 
                  norm_quad_weights: jnp.ndarray, mask_mwss: jnp.ndarray):
        """Evaluate the model on a batch of data.

        Parameters:
            graphdef (nnx.GraphDef): The graph definition containing the model and optimizer.
            state (nnx.State): The current state of the model, optimizer, and metrics.
            images (jnp.ndarray): Input images.
            residuals (jnp.ndarray): Target residuals.
            norm_quad_weights (jnp.ndarray): Normalized quadrature weights.
        
        Returns:
            nnx.State: Updated state after the evaluation step.
        """
        model, optimizer, metrics = nnx.merge(graphdef, state)
        model.eval()
        loss, accuracy = self.loss_and_acc_fn(
        model, images, residuals, norm_quad_weights, mask_mwss)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state


    def _training_log_path(self) -> str:
        return os.path.join(self.model_dir, "training_log.npy")

    def _load_training_log(self) -> dict | None:
        p = self._training_log_path()
        if not os.path.exists(p):
            return None
        obj = np.load(p, allow_pickle=True).item()
        if not isinstance(obj, dict):
            return None
        # expected keys
        needed = {"train_loss", "train_accuracy", "eval_loss", "eval_accuracy"}
        if not needed.issubset(set(obj.keys())):
            return None
        return obj

    def _save_training_log(self, metrics_history: dict) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        np.save(self._training_log_path(), metrics_history)

    def _find_latest_checkpoint_epoch(self) -> int:
        """
        Returns the largest epoch number found in save_dir checkpoint folders.
        Expected folder format: checkpoint_<epoch>
        Raises FileNotFoundError if save_dir does not exist or no checkpoint folders are found.
        """
        base = Path(self.model_dir)
        if not base.exists():
            raise FileNotFoundError(f"No existing checkpoints found: save_dir does not exist: {base}")
        best = None
        best_path = None
        pat = re.compile(r"checkpoint_(\d+)$")
        for p in base.iterdir():
            if not p.is_dir():
                continue
            m = pat.search(p.name)
            if m:
                epoch = int(m.group(1))
                if best is None or epoch > best:
                    best = epoch
                    best_path = p
        if best is None:
            raise FileNotFoundError(f"No existing checkpoints found in {base}. "
            f"Expected folders named like 'checkpoint_<epoch>' (e.g. checkpoint_50).")
        print(f"[Checkpoint] Latest checkpoint found: epoch {best} at {best_path}")
        return best

    def _trim_history_to_epoch(self, metrics_history: dict, epoch: int) -> dict:
        """
        Ensure lists are exactly length=epoch (or shorter if not available).
        """
        for k in ["train_loss", "train_accuracy", "eval_loss", "eval_accuracy"]:
            if k in metrics_history and isinstance(metrics_history[k], list):
                metrics_history[k] = metrics_history[k][:epoch]
        return metrics_history


    def execute_training_procedure(self, masked: bool = False, fsky: float = 0.7, apodization: int = 2):
        """Execute the training procedure for the CMB-Free ILC model.

        Parameters:
            masked : bool
                If True, use the apodised MWSS mask to weight the loss/accuracy.
                If False, use an all-ones mask with the same shape (no masking).
        """
        Train.clear_gpu_cache()
        learning_rate = self.learning_rate
        momentum = self.momentum
        epochs = self.epochs
        batch_size = self.batch_size
        N_freq = len(self.frequencies)
        
        L = self.lmax + 1 
        print("Constructing the CMB-Free ILC dataset")
        train_ds, test_ds, n_train, n_test= self.dataset.prepare_data()
        print_gpu_usage("After dataset creation")
        training_batches_per_epoch = (n_train + batch_size - 1) // batch_size
        testing_batches_per_epoch = (n_test + batch_size - 1) // batch_size
        train_iter, test_iter = iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))
        print_gpu_usage("After dataset creation")
        print("Constructing the model")
        model = S2_UNET(L, N_freq, chs=self.chs, rngs = self.rngs)
        print_gpu_usage("After model creation")

        print("Configuring the optimizer")
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
        print_gpu_usage("After optimizer creation")
        # Handle resume training
        if self.resume_training:
            try:
                ckpt_epoch = self._find_latest_checkpoint_epoch()
            except FileNotFoundError as e:
                print(e)
                print("No checkpoint found, starting training from scratch.")
                ckpt_epoch = 0
            # Load training log if exists; otherwise create empty
            metrics_history = self._load_training_log()
            if metrics_history is None:
                metrics_history = {
                    "train_loss": [],
                    "train_accuracy": [],
                    "eval_loss": [],
                    "eval_accuracy": [],
                }
            if ckpt_epoch > 0: # trim if there is a checkpoint epoch > 0
                metrics_history = self._trim_history_to_epoch(metrics_history, ckpt_epoch)
            if ckpt_epoch == epochs and ckpt_epoch > 0: # saved ckpt == requested epochs => skip training
                print(f"Checkpoint already at epoch {ckpt_epoch} (target epochs = {epochs}). Skipping training.")
                self.plot_training_metrics(metrics_history)
                return
            if ckpt_epoch > epochs: # saved ckpt > requested epochs => error
                raise ValueError(
                    f"Requested epochs ({epochs}) is smaller than the latest saved checkpoint epoch ({ckpt_epoch}). "
                    f"Increase self.epochs to >= {ckpt_epoch}, or delete/choose a different checkpoint directory.")
            start_epoch = ckpt_epoch + 1 # resume training from ckpt+1

        else: # not resume, fresh run
            start_epoch = 1
            # create metric history
            metrics_history = {
                "train_loss": [],
                "train_accuracy": [],
                "eval_loss": [],
                "eval_accuracy": [],
            }

        print("Configuring the metrics")
        # nnx metric setup
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            accuracy=nnx.metrics.Average("accuracy"),
        )

        if self.resume_training and start_epoch > 1:
            loaded_epoch = self.load_model_for_training(model, optimizer)
            if loaded_epoch != start_epoch - 1:
                print(f"[WARN] Loaded epoch {loaded_epoch}, but latest checkpoint epoch is {start_epoch-1}.")
            print(f"Resuming training from epoch {start_epoch}")

        # Select a single image for repeated testing
        test_batch = next(iter(test_ds))
        # Pull the quad weights to avoid repeated CPU transfers from the GPU
        norm_quad_weights = model.input_conv.conv.quad_weights.value / (4 * L)

        # Split prior to training loop
        graphdef, state = nnx.split((model, optimizer, metrics))
        if not masked:
            # Use a first sample to infer the mask shape
            _, by0 = next(iter(tfds.as_numpy(train_ds.take(1))))
            by0 = by0[0]
            mask_mwss = jnp.ones_like(jnp.asarray(by0), dtype=jnp.float32)  # same shape as residuals
            print("Training WITHOUT mask (mask_mwss = 1 everywhere), with shape ", np.shape(mask_mwss))
        else:
            mask_mwss = self.dataset.mask_mwss_beamed(fsky=fsky, apodization=apodization)   # (T, P, 1)
            mask_mwss = jnp.asarray(mask_mwss, dtype=jnp.float32)
            print(f"Training WITH mask (mask-weighted loss & accuracy), with shape {np.shape(mask_mwss)}.")
        
        print_gpu_usage("Before training.")
        print("Starting training")
        for epoch in range(start_epoch, epochs + 1):
            # Commence training for the current epoch
            for _ in range(training_batches_per_epoch):
                batch_x, batch_y = next(train_iter)
                images = jnp.asarray(batch_x)
                residuals = jnp.asarray(batch_y)
                state = self.train_step(graphdef, state, images, residuals, 
                                         norm_quad_weights, mask_mwss)
                # Print GPU usage for first batch of first epoch
                if epoch == 1 and _ == 0:
                    print_gpu_usage("After first training step")
            nnx.update((model, optimizer, metrics), state)  # Upd. model/opt/metrics
            train_iter = iter(tfds.as_numpy(train_ds)) # reset iterator after each epoch

            # Compute metrics for the current epoch
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            metrics.reset()

            # Evaluate at the end of the current epoch
            for _ in range(testing_batches_per_epoch):
                batch_x, batch_y = next(test_iter)
                images = jnp.asarray(batch_x)
                residuals = jnp.asarray(batch_y)
                state = self.eval_step(graphdef, state, images, residuals, 
                                        norm_quad_weights, mask_mwss)
            nnx.update((model, optimizer, metrics), state)  # Only updates metrics
            test_iter = iter(tfds.as_numpy(test_ds))
            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(value)
            metrics.reset()

            print(
                "[Train/Test] epoch = {:03d}: train_loss = {:.03f}, eval_loss = {:.3f}, train_acc = {:.3f}, eval_acc = {:.3f}".format(
                    epoch,
                    metrics_history["train_loss"][-1],
                    metrics_history["eval_loss"][-1],
                    metrics_history["train_accuracy"][-1],
                    metrics_history["eval_accuracy"][-1],
                )
            )
            _ = jnp.asarray(metrics_history["eval_loss"][-1]).block_until_ready() # force sync by touching a device value
            print_gpu_usage(f"After epoch {epoch}")

            # Save model checkpoint after each epoch (overwrites previous)
            self.save_model(model, epoch)
            
            #np.save(self.model_dir + "training_log.npy", metrics_history)
            outdir = os.path.join(self.model_dir, f"checkpoint_{epoch}")
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, "training_log.npy"), metrics_history)
            #np.save(os.path.join(self.model_dir, "training_log.npy"), metrics_history)

            
        # Plot training metrics and examples
        # on last epoch, plot metrics and examples
        self.plot_training_metrics(metrics_history)
        self.plot_examples(metrics_history, model, test_batch, n_examples=self.batch_size)

        # make sure all async ops are done before exiting
        if hasattr(self, "checkpointer") and hasattr(self.checkpointer, "wait_until_finished"):
            self.checkpointer.wait_until_finished()

    def plot_training_metrics(self, metrics_history: dict) -> None:
        """Plot and save training metrics.
        
        Parameters:
            metrics_history (dict): Dictionary containing training and evaluation metrics history.
        """
        fig, ((ax1), (ax3)) = plt.subplots(1, 2, figsize=(10, 4))
        total_epochs = len(metrics_history["train_loss"])
        epochs = range(1, total_epochs + 1)
        
        ax1.plot(epochs, metrics_history["train_loss"], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, metrics_history["eval_loss"], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax3.plot(epochs, metrics_history["train_accuracy"], 'b-', label='Training', linewidth=2)
        ax3.plot(epochs, metrics_history["eval_accuracy"], 'r-', label='Validation', linewidth=2)
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        title = f"lmax={self.lmax}, Lam={self.lam}, nsamp={1200}, Realisations={self.realisations}, Batch Size={self.batch_size}, lr={self.learning_rate}, Momentum={self.momentum}, Lam={self.lam}, chs={self.chs}, loss_fc={self.loss_tag}"        
        fig.suptitle(f"Epoch {total_epochs}\n{title}", fontsize=11, y=1.02)
        plt.tight_layout()
        outdir = os.path.join(self.model_dir, f"checkpoint_{total_epochs}")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "training_metrics.png"), bbox_inches='tight', dpi=150)
        plt.close()

    def plot_examples(self, metrics_history, model, test_batch, n_examples: int = 1):
        """Plot input, output, model prediction and residuals for sample examples.
        Parameters:
            metrics_history (dict): Dictionary containing training and evaluation metrics history.
            model (nnx.Module): The trained model.
            test_batch (tuple): A batch of test data (input images and target residuals).
            n_examples (int): Number of examples to plot from the test batch.
        """ 
        # Plot sample input and predictions
        fig, ax = plt.subplots(n_examples, 5, figsize=(25, 5 * n_examples))
        if n_examples == 1:
            ax = ax.reshape(1, -1)  # Ensure 2D array for consistency
        
        foreground, residual = test_batch
        
        for row in range(n_examples):
            input_ex = jnp.asarray(foreground[row, :, :, 0])
            output_ex = jnp.asarray(residual[row, :, :, 0])
            pred_ex = model(input_ex[None, :, :, None])[0, :, :, 0]
            residual_ex = pred_ex - output_ex  # Prediction - Target residual
            
            # Find global min/max for consistent colorbar (excluding residual)
            vmin = min(jnp.min(input_ex), jnp.min(output_ex), jnp.min(pred_ex))
            vmax = max(jnp.max(input_ex), jnp.max(output_ex), jnp.max(pred_ex))
            
            # Residual colorbar limits (symmetric around zero)
            res_max = max(abs(jnp.min(residual_ex)), abs(jnp.max(residual_ex)))
            res_vmin, res_vmax = -res_max, res_max
            
            # Input
            im0 = ax[row, 0].imshow(input_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=ax[row, 0], shrink=0.6)
            ax[row, 0].set_title(f"Input (Ex {row+1})")
            
            # Output
            im1 = ax[row, 1].imshow(output_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=ax[row, 1], shrink=0.6)
            ax[row, 1].set_title(f"Output (Ex {row+1})")
            
            # Prediction
            im2 = ax[row, 2].imshow(pred_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im2, ax=ax[row, 2], shrink=0.6)
            ax[row, 2].set_title(f"Prediction (Ex {row+1})")
            
            # Residual (Prediction - Output)
            im3 = ax[row, 3].imshow(residual_ex, vmin=res_vmin, vmax=res_vmax, cmap='RdBu_r')
            plt.colorbar(im3, ax=ax[row, 3], shrink=0.6)
            ax[row, 3].set_title(f"Residual (Ex {row+1})")
            
            # Combined Histogram of Expected Output and Prediction
            ax[row, 4].hist(output_ex.flatten(), bins=30, alpha=0.6, color='red', density=True, label='Expected Output')
            ax[row, 4].hist(pred_ex.flatten(), bins=30, alpha=0.6, color='blue', density=True, label='Prediction')
            ax[row, 4].set_title(f"Distribution Comparison (Ex {row+1})")
            ax[row, 4].set_xlabel("Pixel Value")
            ax[row, 4].set_ylabel("Density")
            ax[row, 4].grid(True, alpha=0.3)
            ax[row, 4].legend()
        
        plt.tight_layout()
        total_epochs = len(metrics_history["train_loss"])
        fig.suptitle(f"Epoch {total_epochs}", fontsize=16)
        outdir = os.path.join(self.model_dir, f"checkpoint_{total_epochs}")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'prediction.png'), bbox_inches='tight', dpi=150)
        plt.close()


def main():
    jax.config.update("jax_enable_x64", False)
    print(f"JAX 64-bit mode: {jax.config.jax_enable_x64}")
    """Main function to run training with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the CMB-Free ILC neural network with configurable parameters and GPU selection."
    )
    parser.add_argument(
        '--frequencies',
        nargs='+',
        default=["030", "100", "353"],
        help='List of frequencies to process'
    )
    parser.add_argument(
        '--realisations',
        type=int,
        default=1000,
        help='Number of realisations to process'
    )
    parser.add_argument(
        '--lmax',
        type=int,
        default=1023,
        help='Maximum multipole for the wavelet transform'
    )
    parser.add_argument(
        '--N-directions',
        type=int,
        default=1,
        help='Number of directions for the wavelet transform'
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=2.0,
        help='Lambda factor (scaling) for the wavelet transform'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Size of the batches for training'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Disable shuffling of the dataset'
    )
    parser.add_argument(
        '--split',
        nargs=2,
        type=float,
        default=[0.8, 0.2],
        help='Train/validation split ratios (e.g., 0.8 0.2)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train for'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.95,
        help='Momentum for the optimizer'
    )
    parser.add_argument(
        '--chs',
        nargs='+',
        type=int,
        default=[1, 16, 32, 32, 64],
        help='List of channel dimensions for each layer (default: [1, 16, 32, 32, 64])'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='/Scratch/matthew/data/',
        help='Base directory for data'
    )
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Resume training from the last checkpoint'
    )
    
    args = parser.parse_args()

    # Convert arguments to match Train class parameters
    shuffle = not args.no_shuffle
    rngs = nnx.Rngs(args.seed)
    
    # Create trainer instance
    trainer = Train(
        frequencies=args.frequencies,
        realisations=args.realisations,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        batch_size=args.batch_size,
        shuffle=shuffle,
        split=args.split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        chs=args.chs,
        rngs=rngs,
        directory=args.directory,
        resume_training=args.resume_training,
    )
    
    trainer.execute_training_procedure()


if __name__ == '__main__':
    main()




