# Adapted from: https://github.com/astro-informatics/s2ai
# Original code by: Matthew A. Price, Kevin Mulder, Jason D. McEwen
# License: MIT

import os
import sys
# import from skyclean/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU configuration
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from model import S2_UNET
from data import CMBFreeILC
from utils import *

import matplotlib.pyplot as plt

class Train: 
    def __init__(self, frequencies: list, realisations: int, lmax: int = 1024, N_directions: int = 1, lam: float = 2.0,
                 batch_size: int = 32, shuffle: bool = True, split: list = [0.8,0.2], epochs: int = 120, 
                 learning_rate: float = 1e-3, momentum: float = 0.9, rngs: nnx.Rngs = nnx.Rngs(0), 
                 directory: str = "data/", ):
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
            rngs (nnx.Rngs): Random number generators for the model.
            directory (str): Directory where data is stored / saved to.
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
        self.rngs = rngs
        self.directory = directory

        self.dataset = CMBFreeILC(frequencies, realisations, lmax, N_directions, lam, batch_size, shuffle, split, directory)

        self.save_dir = os.path.join(self.directory, "ML/model")
        if not os.path.exists(self.save_dir):
            create_dir(self.save_dir)
    
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
        return (jnp.einsum("btpc,t->", jnp.abs(residuals - pred_residuals), norm_quad_weights, optimize=True,)
                / residuals.shape[0])


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
        )  # Not differentiating through this as it is just a metric
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
        loss, grads = nnx.value_and_grad(Train.loss_fn)(model, images, residuals, norm_quad_weights)
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
        loss = Train.loss_fn(model, images, residuals, norm_quad_weights)
        accuracy = Train.acc_fn(model, images, residuals, norm_quad_weights)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state


    def execute_training_procedure(self):
        """Execute the training procedure for the CMB-Free ILC model.
        """
        learning_rate = self.learning_rate
        momentum = self.momentum
        epochs = self.epochs
        batch_size = self.batch_size
        N_freq = len(self.frequencies)
        
        L = self.lmax + 1 
        print("Constructing the CMB-Free ILC dataset")
        train_ds, test_ds = self.dataset.prepare_data()
        train_iter, test_iter = iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))
        training_steps_per_epoch, testing_steps_per_epoch = len(train_ds), len(test_ds)
        print("Constructing the model")
        model = S2_UNET(L, N_freq, rngs = self.rngs)

        print("Configuring the optimizer")
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate, momentum))

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

        print("Starting training")
        for epoch in range(1, epochs + 1):
            # Commence training for the current epoch
            for _ in range(training_steps_per_epoch):
                batch_x, batch_y = next(train_iter)
                images = jnp.asarray(batch_x)
                residuals = jnp.asarray(batch_y)
                state = Train.train_step(graphdef, state, images, residuals, norm_quad_weights)
            nnx.update((model, optimizer, metrics), state)  # Upd. model/opt/metrics
            train_iter = iter(tfds.as_numpy(train_ds)) # reset iterator after each epoch

            # Compute metrics for the current epoch
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
            metrics.reset()

            # Evaluate at the end of the current epoch
            for _ in range(testing_steps_per_epoch):
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
            np.save(self.save_dir + "training_log.npy", metrics_history)
            # Plot sample input and predictions
            fig,ax=plt.subplots(1,3)
            foreground, residual = test_batch
            input_ex = jnp.asarray(foreground[0, :, :, 0])
            output_ex = jnp.asarray(residual[0, :, :, 0])
            pred_ex = model(input_ex[None, :, :, None])[0, :, :, 0]
            ax[0].imshow(input_ex)
            ax[0].set_title("Input")
            ax[1].imshow(output_ex)
            ax[1].set_title("Output")
            ax[2].imshow(pred_ex)
            ax[2].set_title(f"Prediction (acc = {metrics_history['eval_accuracy'][-1]:.3f})")
            plt.show()


## Test usage 
frequencies = ["030", "044"]
realisations = 16
lmax = 255
N_directions = 1
lam = 4.0
batch_size = 2
shuffle = True
split = [0.8, 0.2]
epochs = 10
learning_rate = 1e-3
momentum = 0.9
rngs = nnx.Rngs(0)
directory = "/Scratch/matthew/data/"

trainer = Train(frequencies, realisations, lmax, N_directions, lam, batch_size, shuffle, split, epochs, learning_rate, momentum, rngs, directory)
trainer.execute_training_procedure()



