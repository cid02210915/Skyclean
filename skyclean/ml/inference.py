"""
CMB-Free ILC Model Inference Class.

This module provides a class-based interface for loading trained models and applying them for inference,
with integrated FileTemplates support for organized data management.
"""

import os
import re
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx, serialization

from .model import S2_UNET
from .data import CMBFreeILC
from skyclean.silc.file_templates import FileTemplates
from skyclean.silc import SamplingConverters


class Inference:
    """Class for CMB prediction inference using trained models."""

    def __init__(self, extract_comp, component, frequencies, realisations, lmax, N_directions=1, lam=2.0, chs=None,
                 directory="data/", seed=0, model_path=None,
                 rn: int = 30, batch_size: int = 32, epochs: int = 120, learning_rate: float = 1e-3,
                 momentum: float = 0.9, nsamp: int = 1200, constraint: bool = False):
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
            model_path (str, optional): Specific path to model checkpoint (msgpack file or directory). If None, loads the latest model.
            nsamp (int)
            constraint (bool): Mode for the constrainted ILC method.
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
        self.constraint = constraint

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
            nsamp=self.nsamp,
            constraint=self.constraint,
            lam=self.lam,
            batch_size=1,  # Not used for inference
            directory=self.directory
        )

    @staticmethod
    def _is_valid_checkpoint_file(path: str) -> bool:
        """识别有效的 msgpack 格式 checkpoint 文件（state.msgpack 或 checkpoint_epoch_*.msgpack）"""
        if not os.path.isfile(path):
            return False
        filename = os.path.basename(path)
        return filename == "state.msgpack" or bool(re.match(r"checkpoint_epoch_\d+\.msgpack$", filename))

    def _find_latest_checkpoint_in_dir(self, dir_path: str) -> str:
        """在指定目录及其子目录下，递归查找最新的有效 msgpack checkpoint"""
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Checkpoint 目录不存在: {dir_path}")

        latest_epoch = -1
        latest_file = None

        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not self._is_valid_checkpoint_file(file_path):
                    continue

                # 尝试从路径/文件名提取 epoch
                epoch = None
                # 情况1: checkpoint_3/state.msgpack → 从目录名取 epoch
                parent_dir = os.path.basename(root)
                m = re.match(r"checkpoint_(\d+)$", parent_dir)
                if m:
                    epoch = int(m.group(1))
                # 情况2: checkpoint_epoch_3.msgpack → 从文件名取 epoch
                else:
                    m = re.match(r"checkpoint_epoch_(\d+)\.msgpack$", file)
                    if m:
                        epoch = int(m.group(1))
                # 兜底：用文件修改时间排序
                if epoch is None:
                    epoch = os.path.getmtime(file_path)

                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = file_path

        if latest_file is None:
            raise FileNotFoundError(f"在 {dir_path} 及其子目录下未找到任何有效的 msgpack checkpoint")
        return latest_file

    def load_model(self, force_load=False):
        """Load model weights for inference (无 Orbax，纯 flax.serialization)

        Parameters:
            force_load (bool): If True, skip compatibility check and force load.

        Returns:
            nnx.Module: The loaded model.
        """
        # 1) 兼容性检查（保留原有逻辑）
        if not force_load:
            compatibility = self.check_model_compatibility()
            if not compatibility.get('compatible', False):
                raise RuntimeError(
                    f"Model compatibility check failed: {compatibility.get('message', '')}. "
                    f"Pass force_load=True to bypass."
                )

        # 2) 解析 checkpoint 路径（支持目录/文件）
        if self.model_path is not None:
            checkpoint_path = os.path.abspath(self.model_path)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Specified model path does not exist: {checkpoint_path}")
            # 如果用户指定的是目录，尝试找目录下的 msgpack 文件
            if os.path.isdir(checkpoint_path):
                checkpoint_path = self._find_latest_checkpoint_in_dir(checkpoint_path)
            print(f"[Inference] Loading user-specified model from: {checkpoint_path}")
        else:
            # 自动找最新的 checkpoint
            checkpoint_path = self._find_latest_checkpoint_in_dir(self.file_templates.output_directories["ml_models"])
            print(f"[Inference] Loading latest checkpoint from: {checkpoint_path}")

        # 3) 构建和训练时完全一致的模型结构
        L = self.lmax + 1
        ch_in = len(self.frequencies)
        model = S2_UNET(L, ch_in, chs=self.chs, rngs=nnx.Rngs(self.seed))

        # 4) 拆分模型，获取目标 state 模板
        graphdef, target_state = nnx.split(model)

        # 5) 加载 msgpack 文件（pure dict + restore_int_paths 路径）
        with open(checkpoint_path, "rb") as f:
            bytes_data = f.read()

        restored_pure = serialization.msgpack_restore(bytes_data)
        # 兼容 {"model": ...} 包装格式
        if isinstance(restored_pure, dict) and "model" in restored_pure:
            restored_pure = restored_pure["model"]
        if not isinstance(restored_pure, dict):
            raise TypeError(
                f"Unsupported checkpoint payload type: {type(restored_pure).__name__}. "
                "Expected pure dict (or dict with key 'model')."
            )
        restored_pure = nnx.restore_int_paths(restored_pure)
        nnx.replace_by_pure_dict(target_state, restored_pure)
        restored_state = target_state

        # 6) 合并 state 到模型
        model = nnx.merge(graphdef, restored_state)

        # 7) 将参数放到设备上（加速推理）
        model = jax.tree.map(jax.device_put, model)

        self.model = model
        print(f"[Inference] Model loaded successfully from {checkpoint_path} ✅")
        return model

    def predict_cmb(self, realisation, save_result=True, masked=False):
        """Predict CMB for a specific realisation."""
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
        cmb_mw = SamplingConverters.mwss_map_2_mw_map(cmb_pred, L=self.lmax + 1)

        # Save result if requested
        if save_result:
            if masked:
                mask_mw = self.data_handler.mask_mw_beamed()
                cmb_mw *= mask_mw
                self._save_masked_cmb_prediction(cmb_mw, realisation, mask_mw)
            else:
                self._save_cmb_prediction(cmb_mw, realisation)

        print(f"CMB prediction completed for realisation {realisation}")
        print(f"Prediction shape: {cmb_mw.shape}")
        print(f"Value range: [{cmb_mw.min():.3e}, {cmb_mw.max():.3e}]")

        return cmb_mw

    def compute_mse(self, comp, realisation, save_result=True, masked=False):
        """Compute pixel-space MSE for a single realisation."""
        comp = comp.lower()
        if comp not in ("ilc", "nn"):
            raise ValueError("comp must be 'ilc' or 'nn'")

        # Get the data for this realisation
        F, R, _ = self.data_handler.create_residual_mwss_maps(realisation)
        # R has shape (H, W, 1); squeeze to (H, W)
        R = np.asarray(R)
        if R.ndim == 3 and R.shape[-1] == 1:
            R = R[..., 0]  # shape (H, W)
        elif R.ndim == 2:
            R = R  # already (H, W)
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

        else:  # comp == "nn":
            print(f"Calculating MSE(NN) for realisation {realisation}...")

            if self.model is None:  # Ensure model is loaded
                print("Loading model...")
                self.model = self.load_model()
                print("Loaded model.")

            # Prepare network input (same pipeline as in predict_cmb)
            F = self.data_handler.transform(F).astype(np.float32)
            F = jnp.expand_dims(F, axis=0)  # Add batch dimension

            # Predict normalised residual and inverse-transform
            R_pred_norm = self.model(F)
            R_pred = self.data_handler.inverse_transform(R_pred_norm)  # shape: ()
            R_pred = jnp.squeeze(R_pred, axis=(0, 3))  # Remove batch and channel dims

            # MSE(NN) = <(R_pred - R_true)^2>
            diff = R - np.asarray(R_pred)

        if mask is None:
            mse = float(np.mean(diff ** 2))  # in K
        else:
            w = mask
            num = np.sum(w * diff ** 2)
            denom = np.sum(w) + 1e-12
            mse = float(num / denom)

        print(f"MSE for realisation {realisation}: {mse:.6e}")
        return mse

    def _save_cmb_prediction(self, cmb_prediction, realisation):
        """Save CMB prediction using FileTemplates."""
        try:
            # Create a model configuration string for the filename
            chs = "_".join(str(n) for n in self.chs)

            if self.constraint:
                mode = "con"
            else:
                mode = "uncon"
            frequencies = '_'.join(self.frequencies)

            # Use FileTemplates to get the save path
            save_path = self.file_templates.file_templates["ilc_improved"].format(
                mode=mode,
                extract_comp=self.extract_comp,
                frequencies=frequencies,
                component=self.component,
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

            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save the prediction
            np.save(save_path, cmb_prediction)

            print(f"Saved CMB prediction to: {save_path}")

        except Exception as e:
            print(f"Warning: Failed to save CMB prediction: {str(e)}")

    def _save_masked_cmb_prediction(self, cmb_prediction, realisation, mask):
        """Save masked CMB prediction (修复参数错误)"""
        try:
            chs = "_".join(str(n) for n in self.chs)
            model_config = f"lmax{self.lmax}_lam{self.lam}_freq{'_'.join(self.frequencies)}_chs{chs}"
            save_path = self.file_templates.file_templates["ilc_improved_masked_map"].format(
                realisation=realisation,
                lmax=self.lmax,
                lam=self.lam,
                model_config=model_config
            )
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, cmb_prediction)
            print(f"Saved masked CMB prediction to: {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save masked CMB prediction: {str(e)}")

    def get_model_info(self):
        """Get information about the loaded model."""
        info = {
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'frequencies': self.frequencies,
            'lmax': self.lmax,
            'N_directions': self.N_directions,
            'lam': self.lam,
            'directory': self.directory,
            'model_dir': self.file_templates.output_directories["ml_models"],
            'checkpoint_format': "flax.serialization (msgpack, no Orbax)"  # 新增标识
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
        extract_comp="cmb",
        component="cfn",
        frequencies=frequencies,
        realisations=realisations,
        lmax=lmax,
        N_directions=N_directions,
        lam=lam,
        directory=directory
    )

    print("\n1. Model Information:")
    info = inference.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n2. Running Prediction for Realisation 0:")
    cmb_pred = inference.predict_cmb(realisation=0)
    print("Prediction successful.")

    print("\n3. Calculating MSE for Realisation 0:")
    mse_ilc = inference.compute_mse(comp="ilc", realisation=0)
    mse_nn = inference.compute_mse(comp="nn", realisation=0)
    print(f"MSE (ILC): {mse_ilc:.6e}")
    print(f"MSE (NN): {mse_nn:.6e}")
    print(f"Improvement: {(mse_ilc - mse_nn) / mse_ilc * 100:.2f}%")