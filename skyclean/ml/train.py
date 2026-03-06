# Adapted from: https://github.com/astro-informatics/s2ai
# Original code by: Matthew A. Price, Kevin Mulder, Jason D. McEwen
# License: MIT

import argparse
import subprocess
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx, serialization
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import s2fft
import functools
import re
from pathlib import Path
from datetime import datetime
import os
import shutil

from .model import S2_UNET
from .data import CMBFreeILC
from skyclean.silc.utils import create_dir
from skyclean.silc.file_templates import FileTemplates


def _get_physical_id(jax_device_id: int = 0) -> str:
    """Map JAX device ID to physical GPU ID using CUDA_VISIBLE_DEVICES."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = [x.strip() for x in vis.split(",") if x.strip()]
        return ids[min(jax_device_id, len(ids) - 1)]
    return str(jax_device_id)


def print_gpu_usage(stage_name: str, jax_device_id: int = 0):
    """Print GPU memory usage for monitoring."""
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        print(f"[GPU Memory] {stage_name}: no GPU visible to JAX")
        return

    jax_device_id = min(jax_device_id, len(gpus) - 1)
    phys_id = _get_physical_id(jax_device_id)

    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={phys_id}", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used_mb, total_mb = [float(x) for x in out.split(",")]
        used_gb = used_mb / 1024.0
        total_gb = total_mb / 1024.0
        pct = (used_mb / total_mb) * 100.0 if total_mb > 0 else float("nan")
        print(f"[GPU Memory] {stage_name}: {pct:.2f}% ({used_gb:.2f}/{total_gb:.2f} GB)")
    except Exception as e:
        print(f"[GPU Memory] {stage_name}: failed to query GPU memory ({e!r})")


class Train:
    def __init__(self, extract_comp: str, component: str, frequencies: list, realisations: int,
                 lmax: int = 1024, N_directions: int = 1, lam: float = 2.0, nsamp: int = 1200, constraint: bool = False,
                 batch_size: int = 32, shuffle: bool = True, split: list = [0.8, 0.2], epochs: int = 120,
                 learning_rate: float = 1e-3, momentum: float = 0.9, chs: list = None, rngs: nnx.Rngs = nnx.Rngs(0),
                 directory: str = "data/", resume_training: bool = False, loss_tag: str | None = 'pixel',
                 random_generator: bool = False, eval_every: int = 1, eval_steps: int = -1,
                 prefetch: bool = False, run_id: str | None = None):

        self.component = component
        self.extract_comp = extract_comp
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.nsamp = nsamp
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
        self.loss_tag = (loss_tag or "pixel").lower()
        self.eval_every = eval_every
        self.eval_steps = eval_steps
        self.prefetch = prefetch
        self.run_id = (run_id or datetime.now().strftime("%Y%m%d_%H%M%S")).strip()

        if self.loss_tag not in {"pixel", "harmonic"}:
            raise ValueError(f"Unsupported loss_tag={self.loss_tag!r}. Use 'pixel' or 'harmonic'.")
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be >= 1, got {self.eval_every}.")
        if not self.run_id:
            raise ValueError("run_id cannot be empty.")
        self.random_generator = random_generator

        self.dataset = CMBFreeILC(extract_comp, component, frequencies, realisations, lmax, N_directions, lam,
                                  nsamp, constraint, batch_size, shuffle, split, directory, random=random_generator)

        files = FileTemplates(directory)
        self.model_dir = os.path.abspath(os.path.join(files.output_directories["ml_models"], self.run_id))
        os.makedirs(self.model_dir, exist_ok=True)

    @staticmethod
    def clear_gpu_cache():
        """Clear JAX and Python GC cache to free GPU memory."""
        import gc
        print("[GPU Memory] Clearing cache...")
        jax.clear_caches()
        gc.collect()
        jax.block_until_ready(jnp.array([1.0]))
        print("[GPU Memory] Cache cleared")

    @staticmethod
    def _to_host_scalar(x):
        """Convert JAX array to host scalar (float)."""
        return float(np.asarray(x))

    # ========== 核心：极简 Checkpoint (无 Orbax) ==========
    def _get_ckpt_path(self, epoch: int) -> Path:
        """Get path for checkpoint file."""
        return Path(self.model_dir) / f"checkpoint_epoch_{epoch}.msgpack"

    def save_model(self, model, optimizer, epoch) -> bool:
        """Save model + optimizer state using pure flax.serialization (no Orbax)."""
        try:
            # Split model and optimizer state
            _, (model_state, opt_state) = nnx.split((model, optimizer))

            # Build checkpoint data dict
            ckpt_data = {
                "model": model_state,
                "opt": opt_state,
                "epoch": epoch
            }

            # Serialize and save
            bytes_data = serialization.to_bytes(ckpt_data)
            ckpt_path = self._get_ckpt_path(epoch)

            with open(ckpt_path, "wb") as f:
                f.write(bytes_data)

            # Clean up old checkpoints to save space
            pat = re.compile(r"checkpoint_epoch_(\d+)\.msgpack$")
            for f in Path(self.model_dir).iterdir():
                m = pat.match(f.name)
                if m and int(m.group(1)) != epoch:
                    os.remove(f)

            print(f"[Checkpoint] Saved successfully at epoch {epoch} to {ckpt_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Checkpoint save failed at epoch {epoch}: {e}")
            return False

    def load_model_for_training(self, model, optimizer) -> int:
        """Load latest checkpoint into model and optimizer, return loaded epoch."""
        model_dir = Path(self.model_dir)
        if not model_dir.exists():
            print("[Checkpoint] No model directory found, starting from scratch.")
            return 0

        # Find latest checkpoint file
        pat = re.compile(r"checkpoint_epoch_(\d+)\.msgpack$")
        latest_epoch = 0
        latest_file = None

        for f in model_dir.iterdir():
            m = pat.match(f.name)
            if m:
                ep = int(m.group(1))
                if ep > latest_epoch:
                    latest_epoch = ep
                    latest_file = f

        if latest_file is None:
            print("[Checkpoint] No checkpoints found, starting from scratch.")
            return 0

        print(f"[Checkpoint] Loading from {latest_file}")
        try:
            # Build empty template for deserialization
            _, (empty_model_state, empty_opt_state) = nnx.split((model, optimizer))
            template = {
                "model": empty_model_state,
                "opt": empty_opt_state,
                "epoch": 0
            }

            # Load and restore
            with open(latest_file, "rb") as f:
                bytes_data = f.read()

            restored = serialization.from_bytes(template, bytes_data)
            nnx.update((model, optimizer), (restored["model"], restored["opt"]))

            print(f"[Checkpoint] Loaded successfully from epoch {restored['epoch']}")
            return restored["epoch"]

        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            return 0

    # ========== 补全缺失的辅助方法 (解决AttributeError) ==========
    def save_run_config(self, config: dict) -> str:
        """Save training configuration to config.json for reproducibility."""
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, "config.json")

        # Add extra metadata to config
        payload = dict(config)
        payload["run_id"] = self.run_id
        payload["model_dir"] = self.model_dir
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        print(f"[Config] Saved to {path}")
        return path

    def _training_log_path(self) -> str:
        """Get path for training log file."""
        return os.path.join(self.model_dir, "training_log.npy")

    def _load_training_log(self) -> dict | None:
        """Load existing training log if available."""
        p = self._training_log_path()
        if not os.path.exists(p):
            return None

        try:
            obj = np.load(p, allow_pickle=True).item()
            if not isinstance(obj, dict):
                return None

            # Validate required keys
            needed = {"train_loss", "train_accuracy", "eval_loss", "eval_accuracy"}
            if not needed.issubset(set(obj.keys())):
                return None

            return obj
        except Exception as e:
            print(f"[WARN] Failed to load training log: {e}")
            return None

    def _save_training_log(self, metrics_history: dict) -> None:
        """Save training metrics history to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        np.save(self._training_log_path(), metrics_history)
        print(f"[Log] Training metrics saved to {self._training_log_path()}")

    # ========== Loss & Accuracy Functions (All Fixed) ==========
    @staticmethod
    def pix_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss):
        """Pixel-wise loss function."""
        diff_sq = (pred_residuals - residuals) ** 2
        mask = jnp.asarray(mask_mwss)

        if mask.ndim == 2:
            mask = mask[None, :, :, None]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[None, :, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        w_t = jnp.asarray(norm_quad_weights)[None, :, None, None]
        weights = w_t * mask
        num = jnp.sum(diff_sq * weights)
        den = jnp.sum(weights) + 1e-12
        return num / den

    @staticmethod
    def pix_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss):
        """Pixel-wise accuracy function."""
        delta_ilc = residuals
        pred_delta_ilc = pred_residuals
        mask = jnp.asarray(mask_mwss)

        if mask.ndim == 2:
            mask = mask[None, :, :, None]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[None, :, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        w_t = jnp.asarray(norm_quad_weights)[None, :, None, None]
        weights = w_t * mask

        diff_ilc_sq = delta_ilc ** 2
        mse_ilc = jnp.sum(diff_ilc_sq * weights) / (jnp.sum(weights) + 1e-24)

        diff_clean_sq = (delta_ilc - pred_delta_ilc) ** 2
        mse_clean = jnp.sum(diff_clean_sq * weights) / (jnp.sum(weights) + 1e-24)

        return 1.0 - mse_clean / mse_ilc

    @staticmethod
    def harm_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L: int = 1024):
        """Harmonic domain loss function (fixed vmap layers)."""
        mask = jnp.asarray(mask_mwss)

        if mask.ndim == 2:
            mask = mask[:, :, None]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        pred_residuals = pred_residuals * mask
        residuals = residuals * mask
        pred_maps = pred_residuals[..., 0]
        target_maps = residuals[..., 0]

        # Double vmap to handle (batch, time, pixels) shape
        forward = functools.partial(s2fft.forward, L=L, method="jax_cuda")
        forward_t = jax.vmap(forward, in_axes=0, out_axes=0)
        forward_b = jax.vmap(forward_t, in_axes=0, out_axes=0)

        pred_spec = forward_b(pred_maps)
        target_spec = forward_b(target_maps)
        losses = optax.l2_loss(jnp.abs(target_spec), jnp.abs(pred_spec))
        return jnp.mean(losses)

    @staticmethod
    def harm_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L):
        """Harmonic domain accuracy function (fixed vmap layers)."""
        delta_ilc = residuals
        pred_delta_ilc = pred_residuals
        mask = jnp.asarray(mask_mwss)

        if mask.ndim == 2:
            mask = mask[None, :, :, None]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[None, :, :, :]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        delta_ilc = delta_ilc * mask
        pred_delta_ilc = pred_delta_ilc * mask
        delta_ilc_maps = delta_ilc[..., 0]
        pred_delta_ilc_maps = pred_delta_ilc[..., 0]

        # Double vmap to handle (batch, time, pixels) shape
        forward = functools.partial(s2fft.forward, L=L, method="jax_cuda")
        forward_t = jax.vmap(forward, in_axes=0, out_axes=0)
        forward_batch = jax.vmap(forward_t, in_axes=0, out_axes=0)

        alm_ilc = forward_batch(delta_ilc_maps)
        alm_pred = forward_batch(pred_delta_ilc_maps)
        mse_ilc = jnp.mean(jnp.abs(alm_ilc) ** 2)
        mse_clean = jnp.mean((jnp.abs(alm_ilc) - jnp.abs(alm_pred)) ** 2)
        return 1.0 - mse_clean / (mse_ilc + 1e-24)

    def loss_and_acc_fn(self, model, images, residuals, norm_quad_weights, mask_mwss):
        """Compute loss and accuracy based on loss_tag (pixel/harmonic)."""
        pred_residuals = model(images)

        if self.loss_tag == "pixel":
            loss = Train.pix_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss)
            accuracy = Train.pix_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss)
        else:
            loss = Train.harm_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss,
                                                L=self.lmax + 1)
            accuracy = Train.harm_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss,
                                                   L=self.lmax + 1)

        return loss, accuracy

    # ========== Training & Evaluation Steps (JIT Compiled) ==========
    @functools.partial(jax.jit, static_argnums=0)
    def train_step(self, graphdef, state, images, residuals, norm_quad_weights, mask_mwss):
        """Single training step (JIT compiled for speed)."""
        model, optimizer, metrics = nnx.merge(graphdef, state)
        model.train()

        (loss, accuracy), grads = nnx.value_and_grad(self.loss_and_acc_fn, has_aux=True)(
            model, images, residuals, norm_quad_weights, mask_mwss
        )

        optimizer.update(grads)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state

    @functools.partial(jax.jit, static_argnums=0)
    def eval_step(self, graphdef, state, images, residuals, norm_quad_weights, mask_mwss):
        """Single evaluation step (JIT compiled for speed)."""
        model, optimizer, metrics = nnx.merge(graphdef, state)
        model.eval()

        loss, accuracy = self.loss_and_acc_fn(model, images, residuals, norm_quad_weights, mask_mwss)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state

    # ========== Main Training Procedure ==========
    def execute_training_procedure(self, masked: bool = False, fsky: float = 0.7, apodization: int = 2):
        """Execute full training pipeline with evaluation and checkpointing."""
        print("[Train] Starting training procedure...")
        print(f"[Train] Run ID: {self.run_id} | Epochs: {self.epochs} | Batch size: {self.batch_size}")
        Train.clear_gpu_cache()

        # Dataset preparation
        L = self.lmax + 1
        print_gpu_usage("Before dataset creation")
        print("[Data] Constructing CMB-Free ILC dataset...")

        train_ds, test_ds, n_train, n_test, drop_remainder_test = self.dataset.prepare_data()

        # Data pipeline optimization
        if self.prefetch:
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
            print("[Data] Prefetch enabled (tf.data.AUTOTUNE)")
        else:
            print("[Data] Prefetch disabled")

        print(f"[Data] Dataset ready - Train: {n_train} samples | Test: {n_test} samples")
        print_gpu_usage("After dataset creation")

        if n_test == 0:
            raise ValueError("Test set is empty! Increase realisations or adjust train/test split.")

        # Calculate batches per epoch
        training_batches_per_epoch = n_train // self.batch_size
        testing_batches_per_epoch = n_test // self.batch_size if drop_remainder_test else (
                                                                                                      n_test + self.batch_size - 1) // self.batch_size

        # Create iterators
        train_iter, test_iter = iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))

        # Model & Optimizer initialization
        print("[Model] Constructing S2_UNET model...")
        model = S2_UNET(L, len(self.frequencies), chs=self.chs, rngs=self.rngs)
        print(f"[Model] Channel configuration: {self.chs}")
        print_gpu_usage("After model creation")

        print("[Optimizer] Configuring Adam optimizer...")
        optimizer = nnx.Optimizer(model, optax.adam(self.learning_rate))
        print_gpu_usage("After optimizer creation")

        # Resume training setup
        start_epoch = 1
        metrics_history = {
            "train_loss": [], "train_accuracy": [],
            "eval_loss": [], "eval_accuracy": []
        }

        if self.resume_training:
            loaded_epoch = self.load_model_for_training(model, optimizer)
            start_epoch = loaded_epoch + 1
            # Load existing metrics if available
            loaded_history = self._load_training_log()
            if loaded_history:
                metrics_history = loaded_history
                print(f"[Train] Resumed metrics history from epoch {loaded_epoch}")

        # Metrics initialization
        print("[Metrics] Configuring training metrics...")
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            accuracy=nnx.metrics.Average("accuracy"),
        )

        # Prepare fixed parameters
        test_batch = next(iter(test_ds))
        norm_quad_weights = model.input_conv.conv.quad_weights.value / (4 * L)
        graphdef, state = nnx.split((model, optimizer, metrics))

        # Mask configuration
        if not masked:
            _, by0 = next(iter(tfds.as_numpy(train_ds.take(1))))
            mask_mwss = jnp.ones_like(jnp.asarray(by0[0]), dtype=jnp.float32)
            print(f"[Mask] Training WITHOUT mask (shape: {mask_mwss.shape})")
        else:
            mask_mwss = self.dataset.mask_mwss_beamed(fsky=fsky, apodization=apodization)
            mask_mwss = jnp.asarray(mask_mwss, dtype=jnp.float32)
            print(f"[Mask] Training WITH mask (shape: {mask_mwss.shape}, fsky={fsky}, apodization={apodization})")

        # Main training loop
        print("[Train] Starting main training loop...")
        print_gpu_usage("Before training loop")

        for epoch in range(start_epoch, self.epochs + 1):
            # Training phase
            for step in range(training_batches_per_epoch):
                batch_x, batch_y = next(train_iter)
                state = self.train_step(graphdef, state, jnp.asarray(batch_x), jnp.asarray(batch_y),
                                        norm_quad_weights, mask_mwss)

                # Log GPU usage after first step of first epoch
                if epoch == 1 and step == 0:
                    print_gpu_usage("After first training step")

            # Update model/optimizer/metrics and reset iterator
            nnx.update((model, optimizer, metrics), state)
            train_iter = iter(tfds.as_numpy(train_ds))

            # Record training metrics
            train_metrics = metrics.compute()
            for metric, value in train_metrics.items():
                metrics_history[f"train_{metric}"].append(self._to_host_scalar(value))
            metrics.reset()

            # Evaluation phase (every eval_every epochs)
            do_eval = (epoch % self.eval_every == 0)
            if do_eval:
                eval_batches = testing_batches_per_epoch if self.eval_steps <= 0 else min(self.eval_steps,
                                                                                          testing_batches_per_epoch)

                for _ in range(eval_batches):
                    batch_x, batch_y = next(test_iter)
                    state = self.eval_step(graphdef, state, jnp.asarray(batch_x), jnp.asarray(batch_y),
                                           norm_quad_weights, mask_mwss)

                # Update model/optimizer/metrics and reset iterator
                nnx.update((model, optimizer, metrics), state)
                test_iter = iter(tfds.as_numpy(test_ds))

                # Record evaluation metrics
                eval_metrics = metrics.compute()
                for metric, value in eval_metrics.items():
                    metrics_history[f"eval_{metric}"].append(self._to_host_scalar(value))
                metrics.reset()

                eval_loss = f"{metrics_history['eval_loss'][-1]:.3f}"
                eval_acc = f"{metrics_history['eval_accuracy'][-1]:.3f}"
            else:
                # Fill with NaN if evaluation is skipped
                metrics_history["eval_loss"].append(float("nan"))
                metrics_history["eval_accuracy"].append(float("nan"))
                eval_loss = "nan"
                eval_acc = "nan"

            # Print progress
            print(
                f"[Train] Epoch {epoch:03d}/{self.epochs}: "
                f"Train Loss = {metrics_history['train_loss'][-1]:.3f} | "
                f"Eval Loss = {eval_loss} | "
                f"Train Acc = {metrics_history['train_accuracy'][-1]:.3f} | "
                f"Eval Acc = {eval_acc}"
                f"{'' if do_eval else f' (eval skipped, eval_every={self.eval_every})'}"
            )
            print_gpu_usage(f"After epoch {epoch}")

            # Save intermediate log
            self._save_training_log(metrics_history)

            # Final save (last epoch only)
            if epoch == self.epochs:
                # Save model + optimizer
                self.save_model(model, optimizer, epoch)

                # Save final metrics and plots
                outdir = os.path.join(self.model_dir, f"checkpoint_{epoch}")
                os.makedirs(outdir, exist_ok=True)

                # Save metrics log
                np.save(os.path.join(outdir, "training_log.npy"), metrics_history)

                # Generate plots
                self.plot_training_metrics(metrics_history)
                self.plot_examples(metrics_history, model, test_batch, n_examples=self.batch_size)

        print("[Train] Training procedure completed successfully!")
        print(f"[Train] All results saved to: {self.model_dir}")

    # ========== Visualization Functions ==========
    def plot_training_metrics(self, metrics_history: dict) -> None:
        """Plot training/evaluation loss and accuracy curves."""
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4))
        total_epochs = len(metrics_history["train_loss"])
        epochs = np.arange(1, total_epochs + 1)

        # Convert to numpy arrays (handle NaNs)
        train_loss = np.asarray(metrics_history["train_loss"], dtype=float)
        train_acc = np.asarray(metrics_history["train_accuracy"], dtype=float)
        eval_loss = np.asarray(metrics_history["eval_loss"], dtype=float)
        eval_acc = np.asarray(metrics_history["eval_accuracy"], dtype=float)

        # Filter valid (non-NaN) evaluation points
        eval_loss_valid = np.isfinite(eval_loss)
        eval_acc_valid = np.isfinite(eval_acc)

        # Plot loss
        ax1.plot(epochs, train_loss, 'b-', label='Training', linewidth=2)
        if np.any(eval_loss_valid):
            ax1.plot(epochs[eval_loss_valid], eval_loss[eval_loss_valid], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot accuracy
        ax3.plot(epochs, train_acc, 'b-', label='Training', linewidth=2)
        if np.any(eval_acc_valid):
            ax3.plot(epochs[eval_acc_valid], eval_acc[eval_acc_valid], 'r-', label='Validation', linewidth=2)
        ax3.set_title('Accuracy Curve')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Add title with key parameters
        title = (
            f"lmax={self.lmax}, λ={self.lam}, nsamp={self.nsamp}, Realisations={self.realisations}, "
            f"Batch={self.batch_size}, lr={self.learning_rate}, Loss={self.loss_tag}"
        )
        fig.suptitle(f"Training Metrics (Epoch {total_epochs})\n{title}", fontsize=11, y=1.02)
        plt.tight_layout()

        # Save plot
        outdir = os.path.join(self.model_dir, f"checkpoint_{total_epochs}")
        os.makedirs(outdir, exist_ok=True)
        plot_path = os.path.join(outdir, "training_metrics.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Plot] Training metrics saved to {plot_path}")

    def plot_examples(self, metrics_history, model, test_batch, n_examples: int = 1):
        """Plot input/output/prediction examples for qualitative analysis."""
        foreground, residual = test_batch
        batch_n = int(foreground.shape[0])
        n_plot = min(n_examples, batch_n)

        if n_plot == 0:
            print("[Plot] No examples to plot (empty test batch)")
            return

        # Create figure
        fig, ax = plt.subplots(n_plot, 5, figsize=(25, 5 * n_plot))
        if n_plot == 1:
            ax = ax.reshape(1, -1)

        # Plot each example
        for row in range(n_plot):
            # Get single example
            input_ex = jnp.asarray(foreground[row, :, :, 0])
            output_ex = jnp.asarray(residual[row, :, :, 0])
            pred_ex = model(input_ex[None, :, :, None])[0, :, :, 0]
            residual_ex = pred_ex - output_ex

            # Set color limits for consistency
            vmin = min(jnp.min(input_ex), jnp.min(output_ex), jnp.min(pred_ex))
            vmax = max(jnp.max(input_ex), jnp.max(output_ex), jnp.max(pred_ex))
            res_max = max(abs(jnp.min(residual_ex)), abs(jnp.max(residual_ex)))
            res_vmin, res_vmax = -res_max, res_max

            # Plot input
            im0 = ax[row, 0].imshow(input_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=ax[row, 0], shrink=0.6)
            ax[row, 0].set_title(f"Input (Example {row + 1})")

            # Plot ground truth
            im1 = ax[row, 1].imshow(output_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=ax[row, 1], shrink=0.6)
            ax[row, 1].set_title(f"Ground Truth (Example {row + 1})")

            # Plot prediction
            im2 = ax[row, 2].imshow(pred_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im2, ax=ax[row, 2], shrink=0.6)
            ax[row, 2].set_title(f"Prediction (Example {row + 1})")

            # Plot residual
            im3 = ax[row, 3].imshow(residual_ex, vmin=res_vmin, vmax=res_vmax, cmap='RdBu_r')
            plt.colorbar(im3, ax=ax[row, 3], shrink=0.6)
            ax[row, 3].set_title(f"Residual (Example {row + 1})")

            # Plot distribution comparison
            ax[row, 4].hist(output_ex.flatten(), bins=30, alpha=0.6, color='red', density=True, label='Ground Truth')
            ax[row, 4].hist(pred_ex.flatten(), bins=30, alpha=0.6, color='blue', density=True, label='Prediction')
            ax[row, 4].set_title(f"Distribution (Example {row + 1})")
            ax[row, 4].set_xlabel("Pixel Value")
            ax[row, 4].set_ylabel("Density")
            ax[row, 4].grid(True, alpha=0.3)
            ax[row, 4].legend()

        # Add title and save
        total_epochs = len(metrics_history["train_loss"])
        fig.suptitle(f"Prediction Examples (Epoch {total_epochs})", fontsize=16)
        plt.tight_layout()

        outdir = os.path.join(self.model_dir, f"checkpoint_{total_epochs}")
        os.makedirs(outdir, exist_ok=True)
        plot_path = os.path.join(outdir, "prediction_examples.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[Plot] Prediction examples saved to {plot_path}")


# ========== Main Entry Point ==========
def main():
    """Main function for command-line execution."""
    # Disable JAX 64-bit mode for better performance
    jax.config.update("jax_enable_x64", False)
    print(f"[Init] JAX 64-bit mode: {jax.config.jax_enable_x64}")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train CMB-Free ILC S2_UNET model")

    # Core parameters
    parser.add_argument('--extract-comp', type=str, default="cmb", help='Component to extract (e.g., cmb)')
    parser.add_argument('--component', type=str, default="cfn", help='Components to use (cfn/cfne)')
    parser.add_argument('--frequencies', nargs='+', default=["030", "044", "070"], help='List of frequencies')
    parser.add_argument('--realisations', type=int, default=1000, help='Number of data realisations')
    parser.add_argument('--lmax', type=int, default=1023, help='Maximum multipole for wavelet transform')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--directory', type=str, default='/Scratch/matthew/data/', help='Base data directory')
    parser.add_argument('--run-id', type=str, default='test_run', help='Unique run ID for output files')
    parser.add_argument('--random', type=bool, default=False, help='Generate random test maps')
    parser.add_argument('--loss-tag', type=str, default='pixel', choices=['pixel', 'harmonic'], help='Loss type')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--resume-training', action='store_true', help='Resume from latest checkpoint')

    args = parser.parse_args()

    # Initialize trainer
    trainer = Train(
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=args.frequencies,
        realisations=args.realisations,
        lmax=args.lmax,
        epochs=args.epochs,
        batch_size=args.batch_size,
        directory=args.directory,
        run_id=args.run_id,
        random_generator=args.random,
        loss_tag=args.loss_tag,
        learning_rate=args.learning_rate,
        resume_training=args.resume_training
    )

    # Save run configuration (critical for reproducibility)
    trainer.save_run_config(vars(args))

    # Start training
    trainer.execute_training_procedure()


if __name__ == '__main__':
    main()