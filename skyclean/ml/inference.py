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
import healpy as hp
import jax
import jax.numpy as jnp
from flax import nnx, serialization

from .model import S2_UNET
from skyclean.silc.utils import ilc_mode_tag
from .data import CMBFreeILC
from .train import resolve_checkpoint_target, resolve_filter_type
from skyclean.silc.file_templates import FileTemplates, register_pixel_ps_component_template
from skyclean.silc import SamplingConverters
from skyclean.silc.power_spec import MapAlmConverter, PowerSpectrumTT
import s2fft
from s2fft.sampling import s2_samples
from s2fft.utils import quadrature


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
            directory=self.directory,
            run_id=self.run_id,
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

    def predict_cmb(self, realisation, save_result=True, masked=False, component=None):
        """Predict CMB for a specific realisation.

        component="real" applies the model to the observed Planck sky (SILC run with --components real);
        realisation is then the SILC --start-realisation index used in the ilc_synth filename.
        """
        if self.model is None:
            print("Loading model...")
            self.model = self.load_model()
            print("Loaded model.")

        print(f"Predicting CMB for realisation {realisation}{' (observed sky)' if component == 'real' else ''}...")
        outputs = self._predict_realisation_outputs(realisation, component=component)
        cmb_mw = outputs["cmb_mw"]

        if save_result:
            # the full-sky map is always saved: it is what the evaluation (spectra, ratio plots) reads
            self._save_cmb_prediction(cmb_mw, realisation, component=component)
        if masked:
            # mask on the MW grid (L, 2L-1): bilinear interpolation of the HEALPix mask, values kept in [0, 1]
            L = self.data_handler.lmax + 1
            theta, phi = np.meshgrid(s2_samples.thetas(L, "mw"), s2_samples.phis_equiang(L, "mw"), indexing="ij")
            mask_mw = np.clip(hp.get_interp_val(self.data_handler.mask_hp(), theta.ravel(), phi.ravel()).reshape(theta.shape), 0.0, 1.0)
            cmb_mw = cmb_mw * mask_mw
            if save_result:
                self._save_masked_cmb_prediction(cmb_mw, realisation, mask_mw)

        #print(f"CMB prediction completed for realisation {realisation}.")
        #print(f"Prediction shape: {cmb_mw.shape}")
        #print(f"Value range: [{cmb_mw.min():.3e}, {cmb_mw.max():.3e}]")

        return cmb_mw

    def _predict_realisation_outputs(self, realisation, component=None):
        """Run a single forward pass and return prediction artefacts ("residual" is None for component="real")."""
        if self.model is None:
            print("Loading model...")
            self.model = self.load_model()
            print("Loaded model.")

        F, R, ilc_mwss = self.data_handler.create_residual_mwss_maps(realisation, component=component)
        F_norm = self.data_handler.transform(F).astype(np.float32)
        F_norm = jnp.expand_dims(F_norm, axis=0)

        R_pred_norm = self.model(F_norm)
        R_pred = self.data_handler.inverse_transform(R_pred_norm)
        R_pred = jnp.squeeze(R_pred, axis=(0, 3))

        residual = None if R is None else np.asarray(R)
        ilc_mwss = np.asarray(ilc_mwss)
        if residual is not None and residual.ndim == 3 and residual.shape[-1] == 1:
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

    def predict_and_visualise_real_sky(self, realisation: int = 0, masked: bool = False) -> dict:
        """
        Apply the model to the observed Planck sky, save the cleaned CMB map and its diagnostics.

        Needs the SILC pipeline run with --components real --wavelet-components real (same lmax/N/lam/nsamp);
        `realisation` is the SILC --start-realisation index in the from-real ilc_synth filename (default 0).
        Normalisation statistics come from the simulations the model was trained on (self.component).

        Saves, next to the prediction under ML/cmb_prediction/<run_id>/ilc_improved_maps/checkpoint_<epoch>/:
          <stem>.npy                  : cleaned CMB map (MW sampling)
          <stem>_maps.png             : mollviews of ILC, improved and their difference (the predicted
                                        foreground residual, expected to look like dust/tSZ, not CMB)
          <stem>_spectra.png          : ratio of TT D_ell, ILC / processed cmb and improved (ML) / processed cmb,
                                        linear axes; the reference is the processed simulated CMB ("true input
                                        CMB": processed_cmb, realisation `realisation`, same lmax, the same map as
                                        Pipeline.step_power_spec(source="processed", component="cmb",
                                        frequency="143")); all maps are in K with the same common 5' beam
          <stem>_spectra.npz          : ell, the three D_ell [µK^2] and the two ratios
        masked=True: the three maps are multiplied by the Planck common mask in HEALPix and the spectra are the
        pseudo-C_ell divided by f_sky2 = <mask^2> (the sky-fraction correction, exact for white spectra); the
        same mask on all three maps makes the ratios mask-independent to first order. The figure and npz then
        carry a `_masked` suffix and the npz also stores f_sky, f_sky2 and the mask-corrected pseudo-C_ell.
        Returns a dict with the maps, spectra and output paths.
        """
        lmax = self.lmax
        L = lmax + 1
        outputs = self._predict_realisation_outputs(realisation, component="real")
        cmb_mw = np.asarray(outputs["cmb_mw"], dtype=np.float64)
        ilc_mw = np.asarray(SamplingConverters.mwss_map_2_mw_map(outputs["ilc_mwss"], L=L), dtype=np.float64)
        resid_mw = ilc_mw - cmb_mw  # predicted foreground residual removed from the ILC map

        save_path = self._save_cmb_prediction(cmb_mw, realisation, component="real")
        if save_path is None:
            raise RuntimeError("[Inference] Failed to save the observed-sky CMB prediction.")
        stem = os.path.splitext(save_path)[0]

        # ---- maps ----
        panels = [("ILC", ilc_mw), ("Improved (ML)", cmb_mw), ("ILC - improved (predicted residual)", resid_mw)]
        fig = plt.figure(figsize=(18, 4.5))
        for i, (title, m_mw) in enumerate(panels, start=1):
            m_hp = SamplingConverters.mw_map_2_hp_map(m_mw, lmax=lmax) * 1e6
            lim = 300.0 if i < 3 else float(np.percentile(np.abs(m_hp), 99))
            hp.mollview(m_hp, sub=(1, 3, i), title=f"{title} [observed sky]", unit=r"$\mu$K",
                        min=-lim, max=lim, fig=fig.number)
        maps_png = stem + "_maps.png"
        plt.savefig(maps_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Inference] Saved maps figure to: {maps_png}")

        # ---- spectra ----
        # "True input CMB" reference: processed simulated CMB (HEALPix, K, common 5' beam), i.e. the same
        # map as Pipeline.step_power_spec(unit="K", source="processed", component="cmb", frequency="143",
        # realisation=realisation, lmax=lmax); the processed_cmb file is frequency independent.
        conv = MapAlmConverter(self.file_templates.file_templates)
        out_cmb = conv.to_alm(component="cmb", source="processed", frequency="143",
                              realisation=realisation, lmax=lmax)
        extra = {}
        if not masked:
            alm_ilc = np.asarray(s2fft.forward(np.ascontiguousarray(ilc_mw), L=L))
            alm_imp = np.asarray(s2fft.forward(np.ascontiguousarray(cmb_mw), L=L))
            ell, cl_ilc = PowerSpectrumTT.from_mw_alm(alm_ilc)
            _, cl_imp = PowerSpectrumTT.from_mw_alm(alm_imp)
            _, cl_cmb = PowerSpectrumTT.from_healpy_alm(out_cmb["alm"])
        else:
            # Masked pseudo-spectra: every map is multiplied by the same binary (non-apodised) HEALPix mask and
            # the pseudo-C_ell is divided by f_sky2 = <mask^2> (the sky-fraction correction, exact for a white spectrum).
            mask_hp = np.asarray(self.data_handler.mask_hp(apodisation_deg=0), dtype=np.float64)
            nside = hp.get_nside(mask_hp)
            f_sky = float(np.mean(mask_hp))
            f_sky2 = float(np.mean(mask_hp ** 2))
            hp_ilc = np.asarray(SamplingConverters.mw_map_2_hp_map(ilc_mw, lmax=lmax), dtype=np.float64) * mask_hp
            hp_imp = np.asarray(SamplingConverters.mw_map_2_hp_map(cmb_mw, lmax=lmax), dtype=np.float64) * mask_hp
            hp_cmb = hp.alm2map(np.ascontiguousarray(out_cmb["alm"], dtype=np.complex128), nside=nside) * mask_hp
            ell = np.arange(lmax + 1)
            cl_ilc = hp.anafast(hp_ilc, lmax=lmax) / f_sky2
            cl_imp = hp.anafast(hp_imp, lmax=lmax) / f_sky2
            cl_cmb = hp.anafast(hp_cmb, lmax=lmax) / f_sky2
            extra = {"f_sky": f_sky, "f_sky2": f_sky2, "cl_ilc": cl_ilc, "cl_improved": cl_imp, "cl_processed_cmb": cl_cmb}
            print(f"[Inference] Masked spectra: pseudo-C_ell / f_sky2 with f_sky={f_sky:.4f}, f_sky2={f_sky2:.4f}")
        # all maps are in K -> D_ell in µK^2
        Dl_ilc = PowerSpectrumTT.cl_to_Dl(ell, cl_ilc, input_unit="K")
        Dl_imp = PowerSpectrumTT.cl_to_Dl(ell, cl_imp, input_unit="K")
        Dl_cmb = PowerSpectrumTT.cl_to_Dl(ell, cl_cmb, input_unit="K")
        # ratios w.r.t. the processed cmb reference (both D_ell in µK^2, so the ratio is unitless)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_ilc = Dl_ilc / Dl_cmb
            ratio_imp = Dl_imp / Dl_cmb
        suffix = "_masked" if masked else ""
        ml_label = "Masked ML" if masked else "Improved (ML)"
        spectra_png = stem + f"_spectra{suffix}.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ell[2:], ratio_ilc[2:], "-", label="ILC / processed cmb")
        ax.plot(ell[2:], ratio_imp[2:], "-", label=f"{ml_label} / processed cmb")
        ax.axhline(1.0, ls=":", color="red")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"Ratio of $C_\ell$")
        ax.set_title("Observed Planck sky" + (f" (masked pseudo-$C_\\ell$ / $\\langle M^2 \\rangle$, $f_{{sky}}$={extra['f_sky']:.3f})" if masked else ""))
        ax.grid(True, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        plt.savefig(spectra_png, dpi=200)
        plt.close("all")
        spectra_npz = stem + f"_spectra{suffix}.npz"
        np.savez(spectra_npz, ell=ell, Dl_ilc=Dl_ilc, Dl_improved=Dl_imp, Dl_processed_cmb=Dl_cmb,
                 ratio_ilc=ratio_ilc, ratio_improved=ratio_imp, **extra)
        print(f"[Inference] Saved spectra to: {spectra_png} and {spectra_npz}")

        return {
            "cmb_mw": cmb_mw, "ilc_mw": ilc_mw, "residual_mw": resid_mw,
            "ell": ell, "Dl_ilc": Dl_ilc, "Dl_improved": Dl_imp, "Dl_processed_cmb": Dl_cmb,
            "ratio_ilc": ratio_ilc, "ratio_improved": ratio_imp, **extra,
            "save_path": save_path, "maps_png": maps_png, "spectra_png": spectra_png, "spectra_npz": spectra_npz,
        }

    def compute_mse(self, comp, realisation, save_result=True, masked=False):
        """Area-weighted pixel-space MSE (MWSS quadrature weights, as in the training loss) for a single realisation."""
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
            # mask on the MWSS grid as (H, W, 1) float32: bilinear interpolation of the HEALPix mask, values kept in [0, 1]
            L = self.data_handler.lmax + 1
            theta, phi = np.meshgrid(s2_samples.thetas(L, "mwss"), s2_samples.phis_equiang(L, "mwss"), indexing="ij")
            mask = np.clip(hp.get_interp_val(self.data_handler.mask_hp(), theta.ravel(), phi.ravel()).reshape(theta.shape), 0.0, 1.0)
            mask = mask[..., None].astype(np.float32)
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

        # area weights of the MWSS rings (equiangular grid: a plain pixel mean over-counts the poles), times the mask
        w = np.broadcast_to(np.asarray(quadrature.quad_weights(self.lmax + 1, sampling="mwss"))[:, None], diff.shape)
        if mask is not None:
            w = w * mask
        return float(np.sum(w * diff ** 2) / (np.sum(w) + 1e-12))

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
        csv_path = os.path.join(out_dir, "test_metrics_masked.csv" if masked else "test_metrics.csv")

        mask = mask_mw = None
        # area weights of the MWSS rings (as in the training loss): a plain pixel mean over-counts the poles
        area_w = np.broadcast_to(np.asarray(quadrature.quad_weights(self.lmax + 1, sampling="mwss"))[:, None],
                                 (self.data_handler.H, self.data_handler.W))
        if masked:
            # masks loaded once for all realisations: bilinear interpolation of the HEALPix mask onto the MWSS grid
            # (H, W, 1) float32 and the MW grid (L, 2L-1), values kept in [0, 1]
            L = self.data_handler.lmax + 1
            mask_hp = self.data_handler.mask_hp()
            theta, phi = np.meshgrid(s2_samples.thetas(L, "mwss"), s2_samples.phis_equiang(L, "mwss"), indexing="ij")
            mask = np.clip(hp.get_interp_val(mask_hp, theta.ravel(), phi.ravel()).reshape(theta.shape), 0.0, 1.0)
            mask = np.asarray(mask[..., None].astype(np.float32))[..., 0]
            theta, phi = np.meshgrid(s2_samples.thetas(L, "mw"), s2_samples.phis_equiang(L, "mw"), indexing="ij")
            mask_mw = np.clip(hp.get_interp_val(mask_hp, theta.ravel(), phi.ravel()).reshape(theta.shape), 0.0, 1.0)

        def _moments(x):
            """Area-weighted skewness and excess kurtosis over the MWSS grid (weights = ring area x mask, as for the MSE)."""
            x = np.asarray(x, dtype=np.float64)
            w = np.array(area_w * mask if mask is not None else area_w, dtype=np.float64)
            finite = np.isfinite(x)
            w[~finite] = 0.0
            x = np.where(finite, x, 0.0)
            w_sum = np.sum(w) + 1e-24
            mean = np.sum(w * x) / w_sum
            d = x - mean
            var = np.sum(w * d ** 2) / w_sum
            skewness = float(np.sum(w * d ** 3) / w_sum / (var ** 1.5 + 1e-24))
            kurt_excess = float(np.sum(w * d ** 4) / w_sum / (var ** 2 + 1e-24) - 3.0)
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
                w = area_w * mask if masked else area_w
                mse_ilc = float(np.sum(w * residual ** 2) / np.sum(w))
                mse_ml = float(np.sum(w * (residual - pred_mwss) ** 2) / np.sum(w))
                skew_ilc, kurtosis_ilc = _moments(ilc_mwss)
                skew_ml, kurtosis_ml = _moments(pred_mwss)
                if save_predictions:
                    cmb_mw = outputs["cmb_mw"]
                    self._save_cmb_prediction(cmb_mw, realisation)  # full-sky map, read by the evaluation spectra
                    if masked:
                        self._save_masked_cmb_prediction(cmb_mw * mask_mw, realisation, mask_mw)
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

    def save_test_scatter_plots(self, rows, masked=False):
        """Save MSE and skewness scatter plots for the held-out test split."""
        suffix = "_masked" if masked else ""
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
            f"mse_scatter{suffix}.png",
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
            f"skewness_scatter{suffix}.png",
            origin_zero=True,
            double_max=True,
        )

    def _save_cmb_prediction(self, cmb_prediction, realisation, component=None):
        """Save CMB prediction using FileTemplates. Returns the saved path (None on failure)."""
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
                    component=component or self.component,
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
            return save_path

        except Exception as e:
            print(f"Warning: Failed to save CMB prediction: {str(e)}")
            return None

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
    parser.add_argument("--real", action="store_true",
                        help="Apply the model to the observed Planck sky instead of a simulation. Needs the SILC "
                             "pipeline run with --components real --wavelet-components real (same lmax, N, lam, nsamp). "
                             "--realisation is then the SILC --start-realisation index (default 0). "
                             "Writes the MW .npy prediction, a HEALPix .fits copy, the map figure and the TT "
                             "spectra (see Inference.predict_and_visualise_real_sky).")
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

    if args.real:
        realisation = 0 if args.realisation is None else args.realisation
        print(f"\n2. Predicting the observed Planck sky (ilc_synth realisation index {realisation}):")
        inference.predict_and_visualise_real_sky(realisation=realisation, masked=args.masked)
        print("Prediction successful.")
    elif args.realisation is None:
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
        inference.save_test_scatter_plots(rows, masked=args.masked)


if __name__ == "__main__":
    main()
