# Contributed by Qixing Deng

"""
CMB-Free ILC Model Inference Class.
"""

import argparse
import csv
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx, serialization
from scipy.stats import kurtosis, skew

from .model import S2_UNET
from skyclean.silc.utils import ilc_mode_tag
from .data import CMBFreeILC
from .train import resolve_checkpoint_target, resolve_filter_type
from skyclean.silc.file_templates import FileTemplates, register_pixel_ps_component_template
from skyclean.silc import SamplingConverters


class Inference:
    """Class for CMB prediction inference using trained models."""

    def __init__(self, extract_comp, component, frequencies, realisations, lmax, N_directions=1, lam=2.0, chs=None,
                 directory="data/", seed=0, model_path=None,
                 rn: int = 30, batch_size: int = 32, epochs: int = 120, learning_rate: float = 1e-3,
                 momentum: float = 0.9, nsamp: int = 1200, constraint: bool = False,
                 pcilc: bool = False, pcilc_eps: float | None = None,
                 run_id: str | None = None, filter_type: str | None = "auto"):

        self.extract_comp = extract_comp
        self.component = component
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.chs = chs if chs is not None else [512, 256, 128, 64]
        self.filter_type = resolve_filter_type(filter_type, N_directions)
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
        self.pcilc = pcilc
        self.pcilc_eps = pcilc_eps
        self.run_id = (run_id or "").strip()
        if not self.run_id:
            raise ValueError("run_id must be provided for inference outputs.")

        self.file_templates = FileTemplates(directory)
        register_pixel_ps_component_template(
            self.file_templates.file_templates,
            self.file_templates.output_directories,
            self.component,
        )
        self.model = None
        self.config = None
        self.loaded_checkpoint_path = None
        self.loaded_checkpoint_epoch = None
        self.data_handler = CMBFreeILC(
            extract_comp=self.extract_comp,
            component=self.component,
            frequencies=self.frequencies,
            realisations=self.realisations,
            lmax=self.lmax,
            N_directions=self.N_directions,
            nsamp=self.nsamp,
            constraint=self.constraint,
            pcilc=self.pcilc,
            pcilc_eps=self.pcilc_eps,
            lam=self.lam,
            batch_size=1,
            directory=self.directory
        )

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
            checkpoint_dir, checkpoint_file, checkpoint_epoch = resolve_checkpoint_target(
                os.path.abspath(self.model_path)
            )
            checkpoint_path = str(checkpoint_file)
            print(f"[Inference] Loading model from: {checkpoint_path}")
        else:
            run_dir = os.path.join(self.file_templates.output_directories["ml_models"], self.run_id)
            checkpoint_dir, checkpoint_file, checkpoint_epoch = resolve_checkpoint_target(run_dir)
            checkpoint_path = str(checkpoint_file)
            print(f"[Inference] Loading latest checkpoint from: {checkpoint_path}")

        # 构建和训练时完全一致的模型结构
        L = self.lmax + 1
        ch_in = len(self.frequencies)
        model = S2_UNET(L, ch_in, chs=self.chs, filter_type=self.filter_type, rngs=nnx.Rngs(self.seed))

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
        self.loaded_checkpoint_path = checkpoint_path
        self.loaded_checkpoint_epoch = checkpoint_epoch
        print(f"[Inference] Model loaded successfully ✅")
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
                'lam': self.lam,
                'filter_type': self.filter_type
            }
        }

        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                result['compatible'] = False
                result['message'] = f"Model path does not exist: {self.model_path}"
                return result
        else:
            try:
                run_dir = os.path.join(self.file_templates.output_directories["ml_models"], self.run_id)
                resolve_checkpoint_target(run_dir)
            except FileNotFoundError as e:
                result['compatible'] = False
                result['message'] = str(e)

        result['compatible'] = True
        result['message'] = "Basic checkpoint path checks passed."
        return result

    def predict_cmb(self, realisation, save_result=True, masked=False):
        """Predict CMB for a specific realisation."""
        if self.model is None:
            print("Loading model...")
            self.model = self.load_model()
            print("Loaded model.")

        print(f"Predicting CMB for realisation {realisation}...")
        outputs = self._predict_realisation_outputs(realisation)
        cmb_mw = outputs["cmb_mw"]

        if save_result:
            if masked:
                mask_mw = self.data_handler.mask_mw_beamed()
                cmb_mw *= mask_mw
                self._save_masked_cmb_prediction(cmb_mw, realisation, mask_mw)
            else:
                self._save_cmb_prediction(cmb_mw, realisation)

        #print(f"CMB prediction completed for realisation {realisation}.")
        #print(f"Prediction shape: {cmb_mw.shape}")
        #print(f"Value range: [{cmb_mw.min():.3e}, {cmb_mw.max():.3e}]")

        return cmb_mw

    def _predict_realisation_outputs(self, realisation):
        """Run a single forward pass and return prediction artefacts."""
        if self.model is None:
            print("Loading model...")
            self.model = self.load_model()
            print("Loaded model.")

        F, R, ilc_mwss = self.data_handler.create_residual_mwss_maps(realisation)
        F_norm = self.data_handler.transform(F).astype(np.float32)
        F_norm = jnp.expand_dims(F_norm, axis=0)

        R_pred_norm = self.model(F_norm)
        R_pred = self.data_handler.inverse_transform(R_pred_norm)
        R_pred = jnp.squeeze(R_pred, axis=(0, 3))

        residual = np.asarray(R)
        ilc_mwss = np.asarray(ilc_mwss)
        if residual.ndim == 3 and residual.shape[-1] == 1:
            residual = residual[..., 0]
        if ilc_mwss.ndim == 3 and ilc_mwss.shape[-1] == 1:
            ilc_mwss = ilc_mwss[..., 0]

        pred_mwss = np.asarray(R_pred)
        cmb_pred_mwss = ilc_mwss - pred_mwss
        cmb_mw = SamplingConverters.mwss_map_2_mw_map(cmb_pred_mwss, L=self.lmax + 1)
        return {
            "residual": residual,
            "ilc_mwss": ilc_mwss,
            "pred_mwss": pred_mwss,
            "cmb_mw": cmb_mw,
        }

    def predict_test_set(self, save_result=True, masked=False):
        """Predict CMB for every held-out test realisation."""
        test_ids = self.data_handler.get_split_indices()["test"]
        outputs = {}
        for realisation in test_ids:
            outputs[int(realisation)] = self.predict_cmb(realisation=int(realisation), save_result=save_result, masked=masked)
        print(f"[Inference] Saved test-set predictions to: "
              f"{os.path.join(self.file_templates.output_directories['cmb_prediction'], self.run_id, 'ilc_improved_maps')}")
        return outputs

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
            diff = R
        else:
            pred_outputs = self._predict_realisation_outputs(realisation)
            diff = R - np.asarray(pred_outputs["pred_mwss"])

        if mask is None:
            mse = float(np.mean(diff ** 2))
        else:
            w = mask
            num = np.sum(w * diff ** 2)
            denom = np.sum(w) + 1e-12
            mse = float(num / denom)
            
        return mse

    def save_test_metrics_table(self, masked=False, save_predictions=True):
        """Save per-realisation metrics for the held-out test split."""
        if self.model is None:
            self.load_model()
        test_ids = self.data_handler.get_split_indices()["test"]
        checkpoint_tag = (
            f"checkpoint_{self.loaded_checkpoint_epoch}"
            if self.loaded_checkpoint_epoch is not None
            else "checkpoint_unknown"
        )
        out_dir = os.path.join(
            self.file_templates.output_directories["cmb_prediction"],
            self.run_id,
            "evaluation",
            checkpoint_tag,
        )
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "test_metrics.csv")

        def _moments(x):
            x = np.asarray(x, dtype=np.float64).ravel()
            x = x[np.isfinite(x)]
            std = float(np.std(x))
            skewness = float(skew(x, bias=False))
            kurt_excess = float(kurtosis(x, fisher=True, bias=False))
            return skewness, kurt_excess

        rows = []
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "realisation",
                "mse_ilc",
                "mse_ml",
                "skew_ilc",
                "skew_ml",
                "kurtosis_ilc",
                "kurtosis_ml",
            ])

            for realisation in test_ids:
                realisation = int(realisation)
                outputs = self._predict_realisation_outputs(realisation)
                ilc_mwss = outputs["ilc_mwss"]
                residual = outputs["residual"]
                pred_mwss = outputs["pred_mwss"]
                if masked:
                    mask = np.asarray(self.data_handler.mask_mwss_beamed())
                    if mask.ndim == 3 and mask.shape[-1] == 1:
                        mask = mask[..., 0]
                    mse_ilc = float(np.sum(mask * (residual ** 2)) / (np.sum(mask) + 1e-12))
                    mse_ml = float(np.sum(mask * ((residual - pred_mwss) ** 2)) / (np.sum(mask) + 1e-12))
                else:
                    mse_ilc = float(np.mean(residual ** 2))
                    mse_ml = float(np.mean((residual - pred_mwss) ** 2))
                skew_ilc, kurtosis_ilc = _moments(ilc_mwss)
                skew_ml, kurtosis_ml = _moments(pred_mwss)
                if save_predictions:
                    cmb_mw = outputs["cmb_mw"]
                    if masked:
                        mask_mw = self.data_handler.mask_mw_beamed()
                        cmb_mw = cmb_mw * mask_mw
                        self._save_masked_cmb_prediction(cmb_mw, realisation, mask_mw)
                    else:
                        self._save_cmb_prediction(cmb_mw, realisation)
                writer.writerow([
                    realisation,
                    mse_ilc,
                    mse_ml,
                    skew_ilc,
                    skew_ml,
                    kurtosis_ilc,
                    kurtosis_ml,
                ])
                rows.append({
                    "realisation": realisation,
                    "mse_ilc": mse_ilc,
                    "mse_ml": mse_ml,
                    "skew_ilc": skew_ilc,
                    "skew_ml": skew_ml,
                    "kurtosis_ilc": kurtosis_ilc,
                    "kurtosis_ml": kurtosis_ml,
                })

        print(f"[Inference] Saved test metrics table to: {csv_path}")
        return rows

    def save_test_scatter_plots(self, rows):
        """Save MSE and skewness scatter plots for the held-out test split."""
        if self.model is None:
            self.load_model()
        checkpoint_tag = (
            f"checkpoint_{self.loaded_checkpoint_epoch}"
            if self.loaded_checkpoint_epoch is not None
            else "checkpoint_unknown"
        )
        out_dir = os.path.join(
            self.file_templates.output_directories["cmb_prediction"],
            self.run_id,
            "evaluation",
            checkpoint_tag,
        )
        os.makedirs(out_dir, exist_ok=True)

        def _scatter(
            x_key,
            y_key,
            title,
            xlabel,
            ylabel,
            filename,
            origin_zero=False,
            double_max=False,
            figsize=(6, 6),
            diag_label=None,
        ):
            x = np.asarray([row[x_key] for row in rows], dtype=float)
            y = np.asarray([row[y_key] for row in rows], dtype=float)
            if x.size == 0:
                print(f"[Inference] No rows available for {filename}; skipping plot.")
                return
            x *= 1e12
            y *= 1e12

            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            if origin_zero:
                lo = 0.0
            if double_max:
                hi = max(0.0, hi) * 2.0
            if np.isclose(lo, hi):
                pad = 1e-12 if hi == 0.0 else abs(hi) * 0.05
                lo -= pad
                hi += pad

            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(x, y, s=36, alpha=0.5)
            hi_plot = hi * 1.1 if origin_zero else hi
            ax.plot([lo, hi_plot], [lo, hi_plot], "k--", linewidth=1, label=diag_label)
            ax.set_xlim(lo, hi_plot)
            ax.set_ylim(lo, hi_plot)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if diag_label is not None:
                ax.legend()
            fig.tight_layout()

            plot_path = os.path.join(out_dir, filename)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Inference] Saved plot to: {plot_path}")

        _scatter(
            "mse_ilc",
            "mse_ml",
            "MSE comparison",
            "ILC MSE [μK^2]",
            "ML MSE [μK^2]",
            "mse_scatter.png",
            origin_zero=True,
            figsize=(7, 4),
            diag_label="y=x",
        )
        _scatter(
            "skew_ilc",
            "skew_ml",
            "Test Skewness Scatter",
            "Skewness before ML (ILC)",
            "Skewness after ML",
            "skewness_scatter.png",
            origin_zero=True,
            double_max=True,
        )

    def _save_cmb_prediction(self, cmb_prediction, realisation):
        """Save CMB prediction using FileTemplates."""
        try:
            chs = "_".join(str(n) for n in self.chs)

            mode = ilc_mode_tag(constraint=self.constraint, pcilc=self.pcilc, pcilc_eps=self.pcilc_eps)
            frequencies = '_'.join(self.frequencies)
            checkpoint_tag = (
                f"checkpoint_{self.loaded_checkpoint_epoch}"
                if self.loaded_checkpoint_epoch is not None
                else "checkpoint_unknown"
            )
            save_dir = os.path.join(
                self.file_templates.output_directories["cmb_prediction"],
                self.run_id,
                "ilc_improved_maps",
                checkpoint_tag,
            )
            filename = os.path.basename(
                self.file_templates.file_templates["ilc_improved"].format(
                    mode=mode,
                    extract_comp=self.extract_comp,
                    frequencies=frequencies,
                    component=self.component,
                    realisation=realisation,
                    lmax=self.lmax,
                    N_directions=self.N_directions,
                    lam=self.lam,
                    nsamp=self.nsamp,
                    rn=self.rn,
                    batch=self.batch,
                    epochs=self.epochs,
                    lr=self.lr,
                    momentum=self.momentum,
                    chs=chs,
                )
            )
            if self.loaded_checkpoint_epoch is not None:
                stem, ext = os.path.splitext(filename)
                filename = f"{stem}_ckpt{self.loaded_checkpoint_epoch}{ext}"

            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_path, cmb_prediction)

            print(f"Saved CMB prediction to: {save_path}")

        except Exception as e:
            print(f"Warning: Failed to save CMB prediction: {str(e)}")

    def _save_masked_cmb_prediction(self, cmb_prediction, realisation, mask):
        """Save masked CMB prediction"""
        try:
            chs = "_".join(str(n) for n in self.chs)
            model_config = f"lmax{self.lmax}_lam{self.lam}_freq{'_'.join(self.frequencies)}_chs{chs}"
            if self.filter_type != "axisymmetric":
                model_config += f"_ft{self.filter_type}"
            checkpoint_tag = (
                f"checkpoint_{self.loaded_checkpoint_epoch}"
                if self.loaded_checkpoint_epoch is not None
                else "checkpoint_unknown"
            )
            checkpoint_suffix = (
                f"_ckpt{self.loaded_checkpoint_epoch}"
                if self.loaded_checkpoint_epoch is not None
                else ""
            )
            save_path = os.path.join(
                self.file_templates.output_directories["cmb_prediction"],
                self.run_id,
                "ilc_improved_maps",
                checkpoint_tag,
                f"masked_ilc_improved_r{int(realisation):04d}_{model_config}{checkpoint_suffix}.npy",
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
            'filter_type': self.filter_type,
            'directory': self.directory,
            'model_dir': self.file_templates.output_directories["ml_models"],
            'checkpoint_format': "flax serialization (msgpack, no Orbax)"
        }

        compatibility = self.check_model_compatibility()
        info['model_compatibility'] = compatibility

        return info
    


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Skyclean ML model.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example usage:\n"
            "  python -m skyclean.ml.inference \\\n"
            "    --run-id 20250101_120000 \\\n"
            "    --frequencies 030 100 353 \\\n"
            "    --realisations 1000 \\\n"
            "    --lmax 511 \\\n"
            "    --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ \\\n"
            "    --realisation 0 \\\n"
            "    --mse"
        ),
    )

    # ----- match Inference signature -----
    parser.add_argument("--extract-comp", type=str, default="cmb",
                        help="Component the model was trained to extract.")
    parser.add_argument("--component", type=str, default="cfn",
                        help="Input map product key, e.g. cfn, cfne, cfne_circ, or cfne_pix_N.")
    parser.add_argument("--frequencies", nargs="+", default=["030", "100", "353"],
                        help="Frequency channels the model was trained on.")
    parser.add_argument("--realisations", type=int, default=1000,
                        help="Total number of realisations in the dataset (defines the splits).")
    parser.add_argument("--lmax", type=int, default=511, help="Maximum multipole.")
    parser.add_argument("--N-directions", type=int, default=1, help="Number of wavelet directions.")
    parser.add_argument("--lam", type=float, default=2.0, help="Wavelet dilation parameter.")
    parser.add_argument("--nsamp", type=int, default=1200,
                        help="Number of Monte Carlo samples used by the ILC inputs.")
    parser.add_argument("--constraint", action="store_true",
                        help="Load constrained-ILC inputs instead of unconstrained.")
    parser.add_argument("--pcilc", action="store_true",
                        help="Load partially-constrained ILC (pcILC) inputs.")
    parser.add_argument("--pcilc-eps", type=float, default=None,
                        help="pcILC epsilon tolerance. Required with --pcilc; must match the SILC run.")
    parser.add_argument("--chs", nargs="+", type=int, default=[1, 16, 32, 32, 64],
                        help="Channel configuration. Must match the trained model.")
    parser.add_argument("--filter-type", type=str, default="auto",
                        choices=["auto", "axisymmetric", "directional", "square"],
                        help="DISCO filter type for S2_UNET conv blocks. 'auto' (default) follows\n"
                             "--N-directions: axisymmetric when it is 1, directional otherwise.\n"
                             "Must match the trained model: a mismatch fails at checkpoint restore.")
    parser.add_argument("--directory", type=str, default="data/", help="Base data directory.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used to build the model skeleton before restoring weights.")
    parser.add_argument("--run-id", type=str, required=True,
                        help="Run folder under ML/models, also used to lay out the prediction outputs.")
    parser.add_argument("--model-dir", type=str, default="",
                        help="Run directory or checkpoint_<epoch> directory to load. "
                             "If empty, the latest checkpoint under ML/models/<run-id> is used.")
    parser.add_argument("--checkpoint-epoch", type=int, default=None,
                        help="Specific epoch checkpoint to load. Only used with --model-dir "
                             "pointing at a run directory.")
    parser.add_argument("--rn", type=int, default=None,
                        help="Realisation count recorded in output filenames. Defaults to --realisations.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size recorded in output filenames.")
    parser.add_argument("--epochs", type=int, default=120, help="Epoch count recorded in output filenames.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate recorded in output filenames.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum recorded in output filenames.")

    # ----- what to run -----
    parser.add_argument("--realisation", type=int, default=None,
                        help="Predict this single realisation. Omit to predict the whole test split.")
    parser.add_argument("--masked", action="store_true",
                        help="Apply the Galactic mask to predictions and metrics.")
    parser.add_argument("--mse", action="store_true",
                        help="Also report ILC vs NN MSE for the selected realisation.")
    parser.add_argument("--metrics-table", action="store_true",
                        help="Write the test-set metrics table and scatter plots.")
    parser.add_argument("--force-load", action="store_true", default=True,
                        help="Skip the model compatibility check (default).")
    parser.add_argument("--no-force-load", dest="force_load", action="store_false",
                        help="Run the model compatibility check before loading.")

    args = parser.parse_args()

    model_path = None
    if args.model_dir.strip():
        model_path = os.path.abspath(args.model_dir.strip())
        if args.checkpoint_epoch is not None:
            ckpt_dir, _, _ = resolve_checkpoint_target(model_path, epoch=args.checkpoint_epoch)
            model_path = str(ckpt_dir)
    elif args.checkpoint_epoch is not None:
        parser.error("--checkpoint-epoch requires --model-dir.")

    inference = Inference(
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=args.frequencies,
        realisations=args.realisations,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        nsamp=args.nsamp,
        constraint=args.constraint,
        pcilc=args.pcilc,
        pcilc_eps=args.pcilc_eps,
        chs=args.chs,
        filter_type=args.filter_type,
        directory=args.directory,
        seed=args.seed,
        model_path=model_path,
        rn=args.rn if args.rn is not None else args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        run_id=args.run_id,
    )

    print("\n1. Model Information:")
    for key, value in inference.get_model_info().items():
        print(f"   {key}: {value}")

    inference.load_model(force_load=args.force_load)

    if args.realisation is None:
        print("\n2. Predicting the test split:")
        inference.predict_test_set(masked=args.masked)
    else:
        print(f"\n2. Predicting realisation {args.realisation}:")
        inference.predict_cmb(realisation=args.realisation, masked=args.masked)
        print("Prediction successful.")

        if args.mse:
            print(f"\n3. MSE for realisation {args.realisation}:")
            mse_ilc = inference.compute_mse(comp="ilc", realisation=args.realisation, masked=args.masked)
            mse_nn = inference.compute_mse(comp="nn", realisation=args.realisation, masked=args.masked)
            print(f"MSE (ILC): {mse_ilc:.6e}")
            print(f"MSE (NN): {mse_nn:.6e}")
            print(f"Improvement: {(mse_ilc - mse_nn) / mse_ilc * 100:.2f}%")

    if args.metrics_table:
        print("\nWriting test metrics table...")
        rows = inference.save_test_metrics_table(masked=args.masked)
        inference.save_test_scatter_plots(rows)


if __name__ == "__main__":
    main()
