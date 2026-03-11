"""
CMB-Free ILC Model Inference Class.
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

        self.extract_comp = extract_comp
        self.component = component
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.chs = chs if chs is not None else [512, 256, 128, 64]
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

        self.file_templates = FileTemplates(directory)
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
            batch_size=1,
            directory=self.directory
        )

    @staticmethod
    def _is_valid_checkpoint_file(path: str) -> bool:
        """识别有效的msgpack checkpoint文件"""
        if not os.path.isfile(path):
            return False
        filename = os.path.basename(path)
        return bool(re.match(r"checkpoint_epoch_\d+\.msgpack$", filename))

    def _find_latest_checkpoint_in_dir(self, dir_path: str) -> str:
        """递归查找目录下最新的msgpack checkpoint"""
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Checkpoint 目录不存在: {dir_path}")

        latest_epoch = -1
        latest_file = None

        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if not self._is_valid_checkpoint_file(file_path):
                    continue

                # 提取epoch
                m = re.match(r"checkpoint_epoch_(\d+)\.msgpack$", file)
                if m:
                    epoch = int(m.group(1))
                else:
                    epoch = os.path.getmtime(file_path)

                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = file_path

        if latest_file is None:
            raise FileNotFoundError(f"在 {dir_path} 及其子目录下未找到任何有效的 msgpack checkpoint")
        return latest_file

    def load_model(self, force_load=False):
        """加载模型，完全兼容flax 0.10.6"""
        if not force_load:
            compatibility = self.check_model_compatibility()
            if not compatibility.get('compatible', False):
                raise RuntimeError(
                    f"模型兼容性检查失败: {compatibility.get('message', '')}。"
                    f"传入 force_load=True 跳过检查。"
                )

        # 解析checkpoint路径
        if self.model_path is not None:
            checkpoint_path = os.path.abspath(self.model_path)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"指定路径不存在: {checkpoint_path}")
            # 如果是目录，自动找里面的msgpack
            if os.path.isdir(checkpoint_path):
                checkpoint_path = self._find_latest_checkpoint_in_dir(checkpoint_path)
            print(f"[Inference] Loading user-specified model from: {checkpoint_path}")
        else:
            # 自动找最新的checkpoint
            checkpoint_path = self._find_latest_checkpoint_in_dir(self.file_templates.output_directories["ml_models"])
            print(f"[Inference] Loading latest checkpoint from: {checkpoint_path}")

        # 构建和训练时完全一致的模型结构
        L = self.lmax + 1
        ch_in = len(self.frequencies)
        print(f"[Model] Building model with L={L}, ch_in={ch_in}, chs={self.chs}")
        model = S2_UNET(L, ch_in, chs=self.chs, rngs=nnx.Rngs(self.seed))

        # 拆分模型，获取空的state模板
        graphdef, empty_state = nnx.split(model)

        # 加载msgpack文件
        with open(checkpoint_path, "rb") as f:
            bytes_data = f.read()

        # 反序列化，兼容训练时的字典格式
        template = {
            "model": nnx.to_pure_dict(empty_state),
            "opt": {},
            "epoch": 0
        }
        restored_dict = serialization.from_bytes(template, bytes_data)
        #restored_state = restored_dict["model"]
        nnx.replace_by_pure_dict(empty_state, restored_dict["model"])

        # 合并state到模型
        model = nnx.merge(graphdef, empty_state)
        #model = jax.tree.map(jax.device_put, model)

        self.model = model
        print(f"[Inference] Model loaded successfully from {checkpoint_path} ✅")
        return model

    def check_model_compatibility(self):
        """检查模型兼容性"""
        expected_L = self.lmax + 1
        expected_ch_in = len(self.frequencies)

        result = {
            'compatible': True,
            'message': "Compatibility check passed (Orbax-free mode)",
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

        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                result['compatible'] = False
                result['message'] = f"Model path does not exist: {self.model_path}"
                return result
        else:
            try:
                self._find_latest_checkpoint_in_dir(self.file_templates.output_directories["ml_models"])
            except FileNotFoundError as e:
                result['compatible'] = False
                result['message'] = str(e)

        return result

    def predict_cmb(self, realisation, save_result=True, masked=False):
        """Predict CMB for a specific realisation."""
        if self.model is None:
            print("Loading model...")
            self.model = self.load_model()
            print("Loaded model.")

        print(f"Predicting CMB for realisation {realisation}...")

        F, _, ilc_mwss = self.data_handler.create_residual_mwss_maps(realisation)

        F = self.data_handler.transform(F).astype(np.float32)
        F = jnp.expand_dims(F, axis=0)

        R_pred_norm = self.model(F)

        R_pred = self.data_handler.inverse_transform(R_pred_norm)
        R_pred = jnp.squeeze(R_pred, axis=(0, 3))

        cmb_pred = ilc_mwss - R_pred

        cmb_mw = SamplingConverters.mwss_map_2_mw_map(cmb_pred, L=self.lmax + 1)

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

        F, R, _ = self.data_handler.create_residual_mwss_maps(realisation)
        R = np.asarray(R)
        if R.ndim == 3 and R.shape[-1] == 1:
            R = R[..., 0]
        elif R.ndim != 2:
            raise ValueError(f"Unexpected shape for R: {R.shape}")

        if masked:
            mask = self.data_handler.mask_mwss_beamed()
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

        else:
            print(f"Calculating MSE(NN) for realisation {realisation}...")

            if self.model is None:
                print("Loading model...")
                self.model = self.load_model()
                print("Loaded model.")

            F = self.data_handler.transform(F).astype(np.float32)
            F = jnp.expand_dims(F, axis=0)

            R_pred_norm = self.model(F)
            R_pred = self.data_handler.inverse_transform(R_pred_norm)
            R_pred = jnp.squeeze(R_pred, axis=(0, 3))

            diff = R - np.asarray(R_pred)

        if mask is None:
            mse = float(np.mean(diff ** 2))
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
            chs = "_".join(str(n) for n in self.chs)

            if self.constraint:
                mode = "con"
            else:
                mode = "uncon"
            frequencies = '_'.join(self.frequencies)

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

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, cmb_prediction)

            print(f"Saved CMB prediction to: {save_path}")

        except Exception as e:
            print(f"Warning: Failed to save CMB prediction: {str(e)}")

    def _save_masked_cmb_prediction(self, cmb_prediction, realisation, mask):
        """Save masked CMB prediction"""
        try:
            chs = "_".join(str(n) for n in self.chs)
            model_config = f"lmax{self.lmax}_lam{self.lam}_freq{'_'.join(self.frequencies)}_chs{chs}"
            save_path = self.file_templates.file_templates["ilc_improved_masked_map"].format(
                realisation=realisation,
                lmax=self.lmax,
                lam=self.lam,
                model_config=model_config
            )
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
            'checkpoint_format': "flax serialization (msgpack, no Orbax)"
        }

        compatibility = self.check_model_compatibility()
        info['model_compatibility'] = compatibility

        return info


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