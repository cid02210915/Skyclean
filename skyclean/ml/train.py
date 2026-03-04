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
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = [x.strip() for x in vis.split(",") if x.strip()]
        return ids[min(jax_device_id, len(ids) - 1)]
    return str(jax_device_id)


def print_gpu_usage(stage_name: str, jax_device_id: int = 0):
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
        import gc
        print("[GPU Memory] Clearing cache...")
        jax.clear_caches()
        gc.collect()
        jax.block_until_ready(jnp.array([1.0]))
        print("[GPU Memory] Cache cleared")

    @staticmethod
    def _to_host_scalar(x):
        return float(np.asarray(x))

    # ========== 核心：极简 Checkpoint (无 Orbax) ==========
    def _get_ckpt_path(self, epoch: int) -> Path:
        return Path(self.model_dir) / f"checkpoint_epoch_{epoch}.msgpack"

    def save_model(self, model, optimizer, epoch) -> bool:
        """同时保存 Model 和 Optimizer，只用 flax.serialization"""
        try:
            # 1. 拆分获取 state
            _, (model_state, opt_state) = nnx.split((model, optimizer))

            # 2. 构建一个普通字典保存
            ckpt_data = {
                "model": model_state,
                "opt": opt_state,
                "epoch": epoch
            }

            # 3. 直接序列化保存
            bytes_data = serialization.to_bytes(ckpt_data)

            ckpt_path = self._get_ckpt_path(epoch)
            with open(ckpt_path, "wb") as f:
                f.write(bytes_data)

            # 清理旧的 checkpoint
            pat = re.compile(r"checkpoint_epoch_(\d+)\.msgpack$")
            for f in Path(self.model_dir).iterdir():
                m = pat.match(f.name)
                if m and int(m.group(1)) != epoch:
                    os.remove(f)

            print(f"[Checkpoint] Saved at epoch {epoch}")
            return True
        except Exception as e:
            print(f"[ERROR] Checkpoint save failed: {e}")
            return False

    def load_model_for_training(self, model, optimizer) -> int:
        """加载最新的 checkpoint"""
        model_dir = Path(self.model_dir)
        if not model_dir.exists():
            return 0

        # 找最新的 epoch
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
            return 0

        print(f"[Checkpoint] Loading from {latest_file}")
        try:
            # 1. 构建模板
            _, (empty_model_state, empty_opt_state) = nnx.split((model, optimizer))
            template = {
                "model": empty_model_state,
                "opt": empty_opt_state,
                "epoch": 0
            }

            # 2. 加载
            with open(latest_file, "rb") as f:
                bytes_data = f.read()

            restored = serialization.from_bytes(template, bytes_data)

            # 3. 更新回去
            nnx.update((model, optimizer), (restored["model"], restored["opt"]))

            print(f"[Checkpoint] Loaded successfully from epoch {restored['epoch']}")
            return restored["epoch"]

        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
            return 0

    # ========== Loss 函数 (已修复) ==========
    @staticmethod
    def pix_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss):
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

    # 修复点：补充 @staticmethod
    @staticmethod
    def harm_loss_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L: int = 1024):
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

        # 修复点：双层 vmap 适配 (B, T, P)
        forward = functools.partial(s2fft.forward, L=L, method="jax_cuda")
        forward_t = jax.vmap(forward, in_axes=0, out_axes=0)
        forward_b = jax.vmap(forward_t, in_axes=0, out_axes=0)

        pred_spec = forward_b(pred_maps)
        target_spec = forward_b(target_maps)
        losses = optax.l2_loss(jnp.abs(target_spec), jnp.abs(pred_spec))
        return jnp.mean(losses)

    @staticmethod
    def harm_acc_fn_from_pred(pred_residuals, residuals, norm_quad_weights, mask_mwss, L):
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

        # 修复点：双层 vmap
        forward = functools.partial(s2fft.forward, L=L, method="jax_cuda")
        forward_t = jax.vmap(forward, in_axes=0, out_axes=0)
        forward_batch = jax.vmap(forward_t, in_axes=0, out_axes=0)

        alm_ilc = forward_batch(delta_ilc_maps)
        alm_pred = forward_batch(pred_delta_ilc_maps)
        mse_ilc = jnp.mean(jnp.abs(alm_ilc) ** 2)
        mse_clean = jnp.mean((jnp.abs(alm_ilc) - jnp.abs(alm_pred)) ** 2)
        return 1.0 - mse_clean / (mse_ilc + 1e-24)

    def loss_and_acc_fn(self, model, images, residuals, norm_quad_weights, mask_mwss):
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

    @functools.partial(jax.jit, static_argnums=0)
    def train_step(self, graphdef, state, images, residuals, norm_quad_weights, mask_mwss):
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
        model, optimizer, metrics = nnx.merge(graphdef, state)
        model.eval()
        loss, accuracy = self.loss_and_acc_fn(model, images, residuals, norm_quad_weights, mask_mwss)
        metrics.update(loss=loss, accuracy=accuracy)
        _, state = nnx.split((model, optimizer, metrics))
        return state

    def execute_training_procedure(self, masked: bool = False, fsky: float = 0.7, apodization: int = 2):
        print("[train] Starting training...")
        print(f"[train] run_id={self.run_id}")
        Train.clear_gpu_cache()

        L = self.lmax + 1
        print_gpu_usage("Before dataset creation")
        print("Constructing the CMB-Free ILC dataset")
        train_ds, test_ds, n_train, n_test, drop_remainder_test = self.dataset.prepare_data()

        if self.prefetch:
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
            print("[Input Pipeline] Prefetch enabled.")
        else:
            print("[Input Pipeline] Prefetch disabled.")

        print(f"Data generators prepared. Train size: {n_train} Test size: {n_test}")
        print_gpu_usage("After dataset creation")

        if n_test == 0: raise ValueError("Test set is empty.")

        training_batches_per_epoch = n_train // self.batch_size
        testing_batches_per_epoch = n_test // self.batch_size if drop_remainder_test else (
                                                                                                      n_test + self.batch_size - 1) // self.batch_size
        train_iter, test_iter = iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))

        print("Constructing the model")
        print(self.chs)
        model = S2_UNET(L, len(self.frequencies), chs=self.chs, rngs=self.rngs)
        print_gpu_usage("After model creation")

        print("Configuring the optimizer")
        optimizer = nnx.Optimizer(model, optax.adam(self.learning_rate))
        print_gpu_usage("After optimizer creation")

        # Resume logic
        start_epoch = 1
        metrics_history = {
            "train_loss": [], "train_accuracy": [], "eval_loss": [], "eval_accuracy": [],
        }

        if self.resume_training:
            loaded_epoch = self.load_model_for_training(model, optimizer)
            start_epoch = loaded_epoch + 1

        print("Configuring the metrics")
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            accuracy=nnx.metrics.Average("accuracy"),
        )

        # Prepare data
        test_batch = next(iter(test_ds))
        norm_quad_weights = model.input_conv.conv.quad_weights.value / (4 * L)
        graphdef, state = nnx.split((model, optimizer, metrics))

        # Mask
        if not masked:
            _, by0 = next(iter(tfds.as_numpy(train_ds.take(1))))
            mask_mwss = jnp.ones_like(jnp.asarray(by0[0]), dtype=jnp.float32)
            print(f"Training WITHOUT mask (shape: {mask_mwss.shape})")
        else:
            mask_mwss = self.dataset.mask_mwss_beamed(fsky=fsky, apodization=apodization)
            mask_mwss = jnp.asarray(mask_mwss, dtype=jnp.float32)
            print(f"Training WITH mask (shape: {mask_mwss.shape})")

        print("Starting training")
        for epoch in range(start_epoch, self.epochs + 1):
            # Train
            for _ in range(training_batches_per_epoch):
                batch_x, batch_y = next(train_iter)
                state = self.train_step(graphdef, state, jnp.asarray(batch_x), jnp.asarray(batch_y),
                                        norm_quad_weights, mask_mwss)
                if epoch == 1 and _ == 0:
                    print_gpu_usage("After first training step")

            nnx.update((model, optimizer, metrics), state)
            train_iter = iter(tfds.as_numpy(train_ds))

            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(self._to_host_scalar(value))
            metrics.reset()

            # Eval
            for _ in range(testing_batches_per_epoch):
                batch_x, batch_y = next(test_iter)
                state = self.eval_step(graphdef, state, jnp.asarray(batch_x), jnp.asarray(batch_y),
                                       norm_quad_weights, mask_mwss)

            nnx.update((model, optimizer, metrics), state)
            test_iter = iter(tfds.as_numpy(test_ds))

            for metric, value in metrics.compute().items():
                metrics_history[f"eval_{metric}"].append(self._to_host_scalar(value))
            metrics.reset()

            print(f"[Train/Test] epoch = {epoch:03d}: "
                  f"train_loss = {metrics_history['train_loss'][-1]:.3f}, "
                  f"eval_loss = {metrics_history['eval_loss'][-1]:.3f}, "
                  f"train_acc = {metrics_history['train_accuracy'][-1]:.3f}, "
                  f"eval_acc = {metrics_history['eval_accuracy'][-1]:.3f}")
            print_gpu_usage(f"After epoch {epoch}")

            # 只在最后保存
            if epoch == self.epochs:
                self.save_model(model, optimizer, epoch)
                # 保存日志
                outdir = os.path.join(self.model_dir, f"checkpoint_{epoch}")
                os.makedirs(outdir, exist_ok=True)
                np.save(os.path.join(outdir, "training_log.npy"), metrics_history)
                # 画图
                self.plot_training_metrics(metrics_history)
                self.plot_examples(metrics_history, model, test_batch, n_examples=self.batch_size)

        print("[train] Done.")

    def plot_training_metrics(self, metrics_history: dict) -> None:
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4))
        total_epochs = len(metrics_history["train_loss"])
        epochs = np.arange(1, total_epochs + 1)

        train_loss = np.asarray(metrics_history["train_loss"], dtype=float)
        train_acc = np.asarray(metrics_history["train_accuracy"], dtype=float)
        eval_loss = np.asarray(metrics_history["eval_loss"], dtype=float)
        eval_acc = np.asarray(metrics_history["eval_accuracy"], dtype=float)

        eval_loss_valid = np.isfinite(eval_loss)
        eval_acc_valid = np.isfinite(eval_acc)

        ax1.plot(epochs, train_loss, 'b-', label='Training', linewidth=2)
        if np.any(eval_loss_valid):
            ax1.plot(epochs[eval_loss_valid], eval_loss[eval_loss_valid], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax3.plot(epochs, train_acc, 'b-', label='Training', linewidth=2)
        if np.any(eval_acc_valid):
            ax3.plot(epochs[eval_acc_valid], eval_acc[eval_acc_valid], 'r-', label='Validation', linewidth=2)
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        title = f"lmax={self.lmax}, Lam={self.lam}, nsamp={1200}, Realisations={self.realisations}, Batch Size={self.batch_size}, lr={self.learning_rate}, Momentum={self.momentum}, chs={self.chs}, loss_fc={self.loss_tag}"
        fig.suptitle(f"Epoch {total_epochs}\n{title}", fontsize=11, y=1.02)
        plt.tight_layout()

        outdir = os.path.join(self.model_dir, f"checkpoint_{total_epochs}")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "training_metrics.png"), bbox_inches='tight', dpi=150)
        plt.close()

    def plot_examples(self, metrics_history, model, test_batch, n_examples: int = 1):
        foreground, residual = test_batch
        batch_n = int(foreground.shape[0])
        n_plot = min(n_examples, batch_n)

        if n_plot == 0:
            print("[WARN] plot_examples: empty test batch; skipping plots.")
            return

        fig, ax = plt.subplots(n_plot, 5, figsize=(25, 5 * n_plot))
        if n_plot == 1:
            ax = ax.reshape(1, -1)

        for row in range(n_plot):
            input_ex = jnp.asarray(foreground[row, :, :, 0])
            output_ex = jnp.asarray(residual[row, :, :, 0])
            pred_ex = model(input_ex[None, :, :, None])[0, :, :, 0]
            residual_ex = pred_ex - output_ex

            vmin = min(jnp.min(input_ex), jnp.min(output_ex), jnp.min(pred_ex))
            vmax = max(jnp.max(input_ex), jnp.max(output_ex), jnp.max(pred_ex))
            res_max = max(abs(jnp.min(residual_ex)), abs(jnp.max(residual_ex)))
            res_vmin, res_vmax = -res_max, res_max

            im0 = ax[row, 0].imshow(input_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=ax[row, 0], shrink=0.6)
            ax[row, 0].set_title(f"Input (Ex {row + 1})")

            im1 = ax[row, 1].imshow(output_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=ax[row, 1], shrink=0.6)
            ax[row, 1].set_title(f"Output (Ex {row + 1})")

            im2 = ax[row, 2].imshow(pred_ex, vmin=vmin, vmax=vmax)
            plt.colorbar(im2, ax=ax[row, 2], shrink=0.6)
            ax[row, 2].set_title(f"Prediction (Ex {row + 1})")

            im3 = ax[row, 3].imshow(residual_ex, vmin=res_vmin, vmax=res_vmax, cmap='RdBu_r')
            plt.colorbar(im3, ax=ax[row, 3], shrink=0.6)
            ax[row, 3].set_title(f"Residual (Ex {row + 1})")

            ax[row, 4].hist(output_ex.flatten(), bins=30, alpha=0.6, color='red', density=True, label='Expected Output')
            ax[row, 4].hist(pred_ex.flatten(), bins=30, alpha=0.6, color='blue', density=True, label='Prediction')
            ax[row, 4].set_title(f"Distribution Comparison (Ex {row + 1})")
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

    parser = argparse.ArgumentParser()
    # 修复点：补充 extract_comp 和 component
    parser.add_argument('--extract-comp', type=str, default="cmb", help='Component to extract')
    parser.add_argument('--component', type=str, default="cfn", help='Components to use (cfn/cfne)')
    parser.add_argument('--frequencies', nargs='+', default=["030", "044", "070"])
    parser.add_argument('--realisations', type=int, default=1000)
    parser.add_argument('--lmax', type=int, default=1023)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--directory', type=str, default='/Scratch/matthew/data/')
    parser.add_argument('--run-id', type=str, default='test')
    # 修复点：--random 默认值改为布尔 False
    parser.add_argument('--random', type=bool, default=False, help='Generate test maps')
    args = parser.parse_args()

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
    )

    # 保存 config
    config_path = os.path.join(trainer.model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[train] config saved: {config_path}")

    trainer.execute_training_procedure()


if __name__ == '__main__':
    main()