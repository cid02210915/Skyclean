# Adapted from: https://github.com/astro-informatics/s2ai
# Original code by: Matthew A. Price, Kevin Mulder, Jason D. McEwen
# License: MIT

import argparse
import os
import sys

# GPU configuration
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax
jax.config.update("jax_enable_x64", False)  # Use 32-bit

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import s2fft
import functools
import orbax.checkpoint as ocp

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

def print_gpu_usage(stage_name):
    """Print GPU memory usage for a given stage."""
    percentage, mb_used, mb_total = get_gpu_memory_usage()
    if percentage is not None:
        print(f"[GPU Memory] {stage_name}: {percentage:.1f}% ({mb_used:.0f}/{mb_total:.0f} MB)")
    else:
        print(f"[GPU Memory] {stage_name}: Unable to query GPU memory")


class Train: 
    def __init__(self, frequencies: list, realisations: int, lmax: int = 1024, N_directions: int = 1, lam: float = 2.0,
                 batch_size: int = 32, shuffle: bool = True, split: list = [0.8,0.2], epochs: int = 120, 
                 learning_rate: float = 1e-3, momentum: float = 0.9, chs: list = None, rngs: nnx.Rngs = nnx.Rngs(0), 
                 directory: str = "data/", resume_training: bool = False):
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
        """ 
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

        self.dataset = CMBFreeILC(frequencies, realisations, lmax, N_directions, lam, batch_size, shuffle, split, directory)
        
        # Create model directory for checkpoints
        files = FileTemplates(directory)  # Pass directory to FileTemplates
        # Ensure model_dir is absolute path
        self.model_dir = os.path.abspath(files.output_directories["ml_models"])


        self.save_dir = os.path.join(self.directory, "ML/model")
        if not os.path.exists(self.save_dir):
            create_dir(self.save_dir)
        
        # Initialize Orbax checkpoint manager
        self.checkpointer = ocp.StandardCheckpointer()

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
            checkpointer = ocp.StandardCheckpointer()

            checkpoint_path = os.path.abspath(os.path.join(self.model_dir, f"checkpoint_{epoch}"))
            checkpointer.save(checkpoint_path, state)
            # keep only latest checkpoint
            if epoch > 1:
                prev_checkpoint_path = os.path.abspath(os.path.join(self.model_dir, f"checkpoint_{epoch-1}"))
                if os.path.exists(prev_checkpoint_path):
                    import shutil
                    shutil.rmtree(prev_checkpoint_path)

            print(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save model at epoch {epoch}: {str(e)}")
    
    
    def loss_fn(model: nnx.Module, images: jnp.ndarray, residuals: jnp.ndarray, norm_quad_weights: jnp.ndarray,):
        """Weighted MAE on the sphere loss function.
        
        Parameters:
            model (nnx.Module): model.
            images (jnp.ndarray): Input images.
            residuals (jnp.ndarray): Target residuals.
            norm_quad_weights (jnp.ndarray): Normalized quadrature weights.
        
        Returns:
            jnp.ndarray: Computed loss value.
        """
        pred_residuals = model(images)
        errors = jnp.abs(residuals-pred_residuals) #L2 loss to penalise bigger errors
        # mask = jnp.ones_like(errors)  # Create a mask of ones
        # sin = jnp.sin(jnp.linspace(0, jnp.pi, errors.shape[1]))**5
        # mask = mask * sin[None, :, None, None] 
        # errors = jnp.abs(errors*mask).astype(pred_residuals.dtype)  # Apply the mask to the errors
        #integrate error over theta using quadrature weights.  
        return (jnp.einsum("btpc,t->", errors, norm_quad_weights, optimize=True,)
                / (residuals.shape[0]))  # Average over batch
        
    @staticmethod
    def loss_fn_harmonic(model: nnx.Module, images: jnp.ndarray, residuals: jnp.ndarray, norm_quad_weights: jnp.ndarray):
        L = 512
        pred_residuals = model(images)
        # broadcast out N_channels = 1 dimension
        pred_maps = pred_residuals[..., 0]
        target_maps = residuals[..., 0]

        # edit s2ftt.forward to allow for batch processing
        forward = functools.partial(s2fft.forward,
                                        L=L,
                                        method="jax_cuda")
        forward_batch = jax.vmap(forward, in_axes=0, out_axes=0)

        pred_spec = jnp.abs(forward_batch(pred_maps))     # shape (batch, l, m)
        target_spec = jnp.abs(forward_batch(target_maps)) # same shape

        losses = optax.l2_loss(target_spec, pred_spec)

        return jnp.mean(losses)



    def acc_fn(model: nnx.Module, images: jnp.ndarray, residuals: jnp.ndarray, norm_quad_weights: jnp.ndarray, threshold: float = 1.1,):
        """Compute the accuracy of the model predictions.

        Parameters:
            model (nnx.Module): The model to evaluate.
            images (jnp.ndarray): Input images.
            residuals (jnp.ndarray): Target residuals.
            norm_quad_weights (jnp.ndarray): Normalized quadrature weights.
            threshold (float): Threshold for accuracy calculation.
        
        Returns:
            jnp.ndarray: Accuracy metric.
        """
        pred_residuals = model(images)
        errors = jnp.maximum(residuals / (pred_residuals + 1e-24), pred_residuals / (residuals + 1e-24))
        errors = jnp.where(
            errors < jnp.ones_like(errors) * threshold,
            jnp.ones_like(errors),
            jnp.zeros_like(errors),
        )
        return (jnp.einsum("btpc,t->", errors, norm_quad_weights, optimize=True)/ residuals.shape[0])


    @jax.jit
    def train_step(graphdef: nnx.GraphDef, state: nnx.State, images: jnp.ndarray, residuals: jnp.ndarray, 
                   norm_quad_weights: jnp.ndarray,):
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
        loss, grads = nnx.value_and_grad(Train.loss_fn_harmonic)(model, images, residuals, norm_quad_weights)
        optimizer.update(grads)
        accuracy = Train.acc_fn(model, images, residuals, norm_quad_weights)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state


    def eval_step(graphdef: nnx.GraphDef, state: nnx.State, images: jnp.ndarray, residuals: jnp.ndarray, 
                  norm_quad_weights: jnp.ndarray,):
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
        loss = Train.loss_fn_harmonic(model, images, residuals, norm_quad_weights)
        accuracy = Train.acc_fn(model, images, residuals, norm_quad_weights)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state


    def execute_training_procedure(self):
        """Execute the training procedure for the CMB-Free ILC model.
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
        start_epoch = 1
        if self.resume_training:
            start_epoch = self.load_model_for_training(model, optimizer) + 1
            if start_epoch > 1:
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("No checkpoint found, starting training from scratch")

        print("Configuring the metrics")
        # nnx metric setup
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            accuracy=nnx.metrics.Average("accuracy"),
        )
        # Store metric history
        metrics_history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
        }

        # Select a single image for repeated testing
        test_batch = next(iter(test_ds))
        # Pull the quad weights to avoid repeated CPU transfers from the GPU
        norm_quad_weights = model.input_conv.conv.quad_weights.value / (4 * L)

        # Split prior to training loop
        graphdef, state = nnx.split((model, optimizer, metrics))
        print_gpu_usage("Before training.")
        print("Starting training")
        for epoch in range(start_epoch, epochs + 1):
            # Commence training for the current epoch
            for _ in range(training_batches_per_epoch):
                batch_x, batch_y = next(train_iter)
                images = jnp.asarray(batch_x)
                residuals = jnp.asarray(batch_y)
                state = Train.train_step(graphdef, state, images, residuals, norm_quad_weights)
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
                state = Train.eval_step(graphdef, state, images, residuals, norm_quad_weights)
            nnx.update((model, optimizer, metrics), state)  # Only updates metrics
            test_iter = iter(tfds.as_numpy(test_ds))
            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(value)
            metrics.reset()

            print(
                "[Train/Test] epoch = {:03d}: train_loss = {:.3f}, eval_loss = {:.3f}, train_acc(1.1) = {:.3f}, eval_ac(1.1) = {:.3f}".format(
                    epoch,
                    metrics_history["train_loss"][-1],
                    metrics_history["eval_loss"][-1],
                    metrics_history["train_accuracy"][-1],
                    metrics_history["eval_accuracy"][-1],
                )
            )
            
            # Save model checkpoint after each epoch (overwrites previous)
            self.save_model(model, epoch)
            
            np.save(self.save_dir + "training_log.npy", metrics_history)
            
            # Plot training metrics and examples
            # on last epoch, plot metrics and examples
            self.plot_training_metrics(metrics_history, epoch)
            self.plot_examples(model, test_batch, epoch=epoch, n_examples=5)

    def plot_training_metrics(self, metrics_history, current_epoch):
        """Plot and save training metrics.
        
        Parameters:
            metrics_history (dict): Dictionary containing training and evaluation metrics history.
            current_epoch (int): The current epoch number.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, current_epoch + 1)
        
        # Loss plots
        ax1.plot(epochs, metrics_history["train_loss"], 'b-', label='Training Loss', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, metrics_history["eval_loss"], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Accuracy plots
        ax3.plot(epochs, metrics_history["train_accuracy"], 'b-', label='Training Accuracy', linewidth=2)
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        ax4.plot(epochs, metrics_history["eval_accuracy"], 'r-', label='Validation Accuracy', linewidth=2)
        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_metrics.png', bbox_inches='tight', dpi=150)
        plt.close()  # Close to save memory
    
    def plot_examples(self, model, test_batch, epoch: int, n_examples: int = 1):
        """Plot input, output, model prediction and residuals for sample examples.
        Parameters:
            model (nnx.Module): The trained model.
            test_batch (tuple): A batch of test data (input images and target residuals).
            epoch (int): The current training epoch.
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
        fig.suptitle(f"Epoch {epoch}", fontsize=16)
        plt.savefig(f'prediction.png', bbox_inches='tight', dpi=150)
        plt.show()


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
        default=511,
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
        '--gpu',
        type=int,
        default=0,
        help='Index of the GPU to use (e.g., 0 or 1)'
    )
    parser.add_argument(
        '--mem-fraction',
        type=float,
        default=0.7,
        help='GPU memory fraction to use (default: 0.7)'
    )
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Resume training from the last checkpoint'
    )
    
    args = parser.parse_args()

    # Set GPU configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_fraction)
    print(f"Using GPU {args.gpu} with {args.mem_fraction*100:.0f}% memory fraction for training.")

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


# Example usage:
# python3 -m skyclean.ml.train \
#   --gpu 0 \
#   --frequencies 030 100 353 \
#   --realisations 1000 \
#   --lmax 511 \
#   --lam 2.0 \
#   --batch-size 8 \
#   --epochs 100 \
#   --learning-rate 1e-3 \
#   --momentum 0.95 \
#   --chs 1 16 32 32 64 \
#   --directory /Scratch/matthew/data/ \
#   --mem-fraction 0.7




