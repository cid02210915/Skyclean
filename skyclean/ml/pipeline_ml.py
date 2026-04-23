# skyclean/ml/pipeline_ml.py
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # helps OOM/fragmentation
# use gpi1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX GPUs =", jax.devices("gpu"))

import time, re, argparse
from pathlib import Path
from flax import nnx
import numpy as np
import s2fft
import glob

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from skyclean.ml.train import Train, resolve_checkpoint_target
from skyclean.ml.inference import Inference
from skyclean.silc.file_templates import FileTemplates, register_pixel_ps_component_template
from skyclean.silc.power_spec import MapAlmConverter, PowerSpectrumCrossTT, PowerSpectrumTT


def resolve_evaluate_target(args) -> str:
    """
    Resolve evaluation target from CLI args.

    Accepted inputs:
    - `--model-dir` pointing to a run directory containing checkpoint_* folders
    - `--model-dir` pointing directly to a checkpoint_* directory
    - `--run-id` pointing to ML/models/<run_id>
    """
    if args.model_dir.strip():
        return os.path.abspath(args.model_dir.strip())

    files = FileTemplates(args.directory)
    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("Evaluation requires either --run-id or --model-dir.")
    base_model_dir = os.path.abspath(files.output_directories["ml_models"])
    run_dir = os.path.join(base_model_dir, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}. "
            "With --run-id, checkpoints are expected under ML/models/<run_id>/checkpoint_<epoch>.")
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline (train + evaluate) for Skyclean ML."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "train+evaluate"],
        help="Which stages to run."
    )

    # ----- match Train signature -----
    parser.add_argument("--extract-comp", type=str, default="cmb")
    parser.add_argument("--component", type=str, default="cfn",
                        help="Input map product key, e.g. cfn, cfne, cfne_circ, or cfne_pix_N.")

    parser.add_argument("--frequencies", nargs="+", default=["030", "100", "353"])
    parser.add_argument("--realisations", type=int, default=1000)

    parser.add_argument("--lmax", type=int, default=1023)
    parser.add_argument("--N-directions", type=int, default=1)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--nsamp", type=int, default=1200)

    parser.add_argument("--constraint", action="store_true", help="Enable constraint")


    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--split",
        nargs="+",
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Train/validation/test split ratios. Recommended: 0.8 0.1 0.1",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--chs", nargs="+", type=int, default=[1, 16, 32, 32, 64])

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--random", dest="random", action="store_true",
                        help="Enable random map generation for tests.")
    parser.add_argument("--no-random", dest="random", action="store_false",
                        help="Disable random map generation for tests.")

    parser.add_argument("--directory", type=str, default="data/")
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run folder under ML/models. If omitted in training, uses timestamp YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--loss-tag", type=str, default=None)

    # ----- inference controls -----
    parser.add_argument("--model-dir", type=str, default="",
            help="Override model directory containing checkpoint_<epoch> folders. "
             "If empty, derived from FileTemplates(directory) like Train.")
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Specific epoch checkpoint to use for evaluation. If omitted, use latest.",
    )

    # ----- evaluation controls -----
    parser.add_argument("--eval-every", type=int, default=1, 
            help="Run evaluation every N epochs (default: 1 = every epoch).")
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Number of non-improving validation checks allowed before stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-3,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--eval-batches",
        "--eval-steps",
        dest="eval_batches",
        type=int,
        default=-1,
        help="Number of validation batches per evaluation run. -1 means full test set.",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Enable tf.data prefetching in training. Default is disabled unless this flag is passed.",
    )
    parser.set_defaults(prefetch=False)
    parser.set_defaults(random=False)

    return parser.parse_args()


def step_train(args) -> str:
    """
    Train model and return the resolved model_dir where checkpoints were saved.
    """
    trainer = Train(
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=args.frequencies,
        realisations=args.realisations,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        nsamp=args.nsamp,
        constraint=args.constraint,
        batch_size=args.batch_size,
        split=args.split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        chs=args.chs,
        rngs=nnx.Rngs(args.seed),
        directory=args.directory,
        resume_training=args.resume_training,
        loss_tag=args.loss_tag,
        random_generator=args.random,
        eval_every=args.eval_every,
        eval_steps=args.eval_batches,
        prefetch=args.prefetch,
        run_id=(args.run_id.strip() or None),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    cfg_path = trainer.save_run_config(vars(args))
    print(f"[train] run_id={trainer.run_id}")
    print(f"[train] config saved: {cfg_path}")
    trainer.execute_training_procedure()
    return trainer.model_dir



def generate_spectrum_for_one(args=None, ckpt_dir: str | None = None):
    """
    Compute TT power spectra for:
      - processed_cmb (HEALPix)
      - ilc_synth (MW)
      - ilc_improved (MW)
    Also compute cross power spectra for:
      - ilc vs processed_cmb (MW x HEALPix)
      - ml vs processed_cmb (MW x HEALPix)

    Returns
    -------
    dict:
        {
          "processed_cmb": {"ell", "cl", "path"},
          "ilc_synth":     {"ell", "cl", "path"},
          "ilc_improved":  {"ell", "cl", "path"},
          "ilc-cmb":       {"ell", "cl", "path"},
          "ml-cmb":        {"ell", "cl", "path"},
        }
    """

    files = FileTemplates(args.directory)
    register_pixel_ps_component_template(
        files.file_templates,
        files.output_directories,
        args.component,
    )
    ft = files.file_templates
    conv = MapAlmConverter(ft)

    mode = "con" if args.constraint else "uncon"
    freq_tag = "_".join(str(x) for x in args.frequencies)
    chs = "_".join(str(n) for n in args.chs)
    lam_str = f"{float(args.lam):.1f}"
    if hasattr(args, "realisation") and args.realisation is not None:
        realisation = int(args.realisation)
    else:
        split = np.asarray(args.split, dtype=float)
        if len(split) == 2:
            split = np.asarray([split[0], split[1] / 2.0, split[1] / 2.0], dtype=float)
        n_train = int(split[0] * args.realisations)
        n_val = int(split[1] * args.realisations)
        test_ids = np.arange(args.realisations)[n_train + n_val:]
        if len(test_ids) == 0:
            raise ValueError("step_spec could not determine a test realisation because the test split is empty.")
        realisation = int(test_ids[0])
    lmax = int(args.lmax)
    nsamp = int(args.nsamp)

    results = {}

    # 1) auto-spectra: processed_cmb (HEALPix) -> alm -> C_ell
    out_proc = conv.to_alm(
        component="cmb",
        source="processed",
        realisation=realisation,
        lmax=lmax,
    )
    print(f"[Spectrum] loading processed_cmb map from: {out_proc['path']}")
    ell_proc, cl_proc = PowerSpectrumTT.from_healpy_alm(out_proc["alm"])
    results["processed_cmb"] = {
        "ell": ell_proc,
        "cl": cl_proc,
        "path": out_proc["path"],
    }

    # 2) auto-spectra: ilc_synth (MW) -> alm -> C_ell
    out_synth = conv.to_alm(
        component=args.component,
        source="ilc_synth",
        extract_comp=args.extract_comp,
        frequencies=args.frequencies,
        realisation=realisation,
        lmax=lmax,
        lam=lam_str,
        nsamp=nsamp,
        constraint=args.constraint,
    )
    print(f"[Spectrum] loading ilc_synth map from: {out_synth['path']}")
    ell_synth, cl_synth = PowerSpectrumTT.from_mw_alm(np.asarray(out_synth["alm"]))
    results["ilc_synth"] = {
        "ell": ell_synth,
        "cl": cl_synth,
        "path": out_synth["path"],
    }


    # 3) auto-spectra: ilc_improved (MW) -> alm -> C_ell
    improved_filename = os.path.basename(ft["ilc_improved"].format(
        mode=mode,
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=freq_tag,
        realisation=realisation,
        lmax=lmax,
        lam=lam_str,
        nsamp=nsamp,
        rn=int(args.realisations),
        batch=int(args.batch_size),
        epochs=int(args.epochs),
        lr=args.learning_rate,
        momentum=args.momentum,
        chs=chs,
    ))
    if args.checkpoint_epoch is None:
        raise ValueError("[spectrum] checkpoint_epoch is required for checkpoint-layered ML prediction paths.")
    improved_map_path = os.path.join(
        files.output_directories["cmb_prediction"],
        args.run_id,
        "ilc_improved_maps",
        f"checkpoint_{int(args.checkpoint_epoch)}",
        improved_filename.replace(".npy", f"_ckpt{int(args.checkpoint_epoch)}.npy"),
    )
    print(f"[Spectrum] loading ilc_improved map from: {improved_map_path}")
    if not os.path.exists(improved_map_path):
        if ckpt_dir is None:
            raise FileNotFoundError(f"[Spectrum] improved map not found: {improved_map_path}. ")

        print("[Spectrum] ML map missing. Generating it via existing inference pipeline...")
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
            chs=args.chs,
            directory=args.directory,
            model_path=ckpt_dir,
            rn=args.realisations,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            run_id=args.run_id,
        )
        model = inference.load_model(force_load=True)
        if model is None:
            raise RuntimeError(f"[spectrum] failed to load model from checkpoint: {ckpt_dir}")
        inference.predict_cmb(realisation=realisation)

        if not os.path.exists(improved_map_path):
            raise FileNotFoundError(
                "[spectrum] improved map still missing after inference generation: "
                f"{improved_map_path}"
            )

    improved_map = np.load(improved_map_path)
    L = lmax + 1
    arr = np.asarray(np.real(np.squeeze(improved_map)), dtype=np.float64, order="C")
    alm_mw = s2fft.forward(arr, L=L)
    ell_improved, cl_improved = PowerSpectrumTT.from_mw_alm(np.asarray(alm_mw))
    results["ilc_improved"] = {
        "ell": ell_improved,
        "cl": cl_improved,
        "path": improved_map_path,
    }

    # 4) cross-spectra: ilc vs processed_cmb (MW x HEALPix) -> C_ell
    ell_cross_ilc, cl_cross_ilc = PowerSpectrumCrossTT.from_any_alms(
        alm_X = out_synth["alm"],
        fmt_X = "mw",
        alm_Y = out_proc["alm"],
        fmt_Y = "healpy",
    )
    results["ilc-cmb"] = {
        "ell": ell_cross_ilc,
        "cl": cl_cross_ilc,
        "path": out_synth["path"],
    }

    # 5) cross-spectra: ml vs processed_cmb (MW x HEALPix) -> C_ell
    ell_cross_ml, cl_cross_ml = PowerSpectrumCrossTT.from_any_alms(
        alm_X = alm_mw,
        fmt_X = "mw",
        alm_Y = out_proc["alm"],
        fmt_Y = "healpy",
    )
    results["ml-cmb"] = {
        "ell": ell_cross_ml,
        "cl": cl_cross_ml,
        "path": improved_map_path,
    }

    return results

def step_evaluate(args, ckpt_dir: str | None = None):
    """
    Load latest checkpoint and save predictions for the held-out test split.
    """
    checkpoint_epoch = args.checkpoint_epoch
    if checkpoint_epoch is None and ckpt_dir is not None:
        ckpt_name = os.path.basename(os.path.normpath(ckpt_dir))
        ckpt_match = re.match(r"checkpoint_(\d+)$", ckpt_name)
        if ckpt_match is not None:
            checkpoint_epoch = int(ckpt_match.group(1))
        else:
            file_match = re.match(r"checkpoint_epoch_(\d+)\.msgpack$", ckpt_name)
            if file_match is not None:
                checkpoint_epoch = int(file_match.group(1))

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
        chs=args.chs,
        directory=args.directory,
        model_path=ckpt_dir,  # points directly to checkpoint_<epoch>
        rn=args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        run_id=args.run_id,
    )
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[Evaluate] Model failed to load (load_model returned falsy).")

    test_ids = inference.data_handler.get_split_indices()["test"]
    print(f"[Evaluate] Predicting CMB for {len(test_ids)} test realisations...")
    mode = "con" if args.constraint else "uncon"
    freq_tag = "_".join(str(x) for x in args.frequencies)
    chs = "_".join(str(n) for n in args.chs)
    lam_str = f"{float(args.lam):.1f}"
    if checkpoint_epoch is None:
        raise ValueError("[Evaluate] checkpoint_epoch is required for checkpoint-layered prediction outputs.")
    save_dir = os.path.join(
        inference.file_templates.output_directories["cmb_prediction"],
        args.run_id,
        "ilc_improved_maps",
        f"checkpoint_{checkpoint_epoch}",
    )
    missing_prediction = False
    for realisation in test_ids:
        filename = os.path.basename(inference.file_templates.file_templates["ilc_improved"].format(
            mode=mode,
            extract_comp=args.extract_comp,
            component=args.component,
            frequencies=freq_tag,
            realisation=int(realisation),
            lmax=int(args.lmax),
            lam=lam_str,
            nsamp=int(args.nsamp),
            rn=int(args.realisations),
            batch=int(args.batch_size),
            epochs=int(args.epochs),
            lr=args.learning_rate,
            momentum=args.momentum,
            chs=chs,
        ))
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}_ckpt{checkpoint_epoch}{ext}"
        if not os.path.exists(os.path.join(save_dir, filename)):
            missing_prediction = True
            break

    if missing_prediction:
        metrics_rows = inference.save_test_metrics_table(save_predictions=True)
        inference.save_test_scatter_plots(metrics_rows)
    else:
        print("[evaluate] Existing CMB prediction files found for all test realisations; skipping prediction generation.")

    ratio_ilc_all = []
    ratio_ml_all = []
    out_dir = os.path.join(
        inference.file_templates.output_directories["cmb_prediction"],
        args.run_id,
        "evaluation",
        f"checkpoint_{checkpoint_epoch}",
    )
    os.makedirs(out_dir, exist_ok=True)

    for realisation in test_ids:
        spec_args = argparse.Namespace(**vars(args))
        spec_args.realisation = int(realisation)
        spec_args.checkpoint_epoch = checkpoint_epoch
        spec = generate_spectrum_for_one(spec_args, ckpt_dir=ckpt_dir)
        bundle_path = os.path.join(out_dir, f"component_spectra_r{int(realisation):04d}.npz")
        np.savez(
            bundle_path,
            ell=np.asarray(spec["processed_cmb"]["ell"], dtype=float),
            processed_cmb_cl=np.asarray(spec["processed_cmb"]["cl"], dtype=float),
            ilc_synth_cl=np.asarray(spec["ilc_synth"]["cl"], dtype=float),
            ilc_improved_cl=np.asarray(spec["ilc_improved"]["cl"], dtype=float),
            ilc_cmb_cl=np.asarray(spec["ilc-cmb"]["cl"], dtype=float),
            ml_cmb_cl=np.asarray(spec["ml-cmb"]["cl"], dtype=float),
        )
        print(f"[Evaluate] wrote component spectra: {bundle_path}")

        ell = np.asarray(spec["processed_cmb"]["ell"], dtype=float)
        cl_processed = np.asarray(spec["processed_cmb"]["cl"], dtype=float)
        cl_ilc_synth = np.asarray(spec["ilc_synth"]["cl"], dtype=float)
        cl_ilc_improved = np.asarray(spec["ilc_improved"]["cl"], dtype=float)

        ratio_ilc = cl_ilc_synth / cl_processed
        ratio_ml = cl_ilc_improved / cl_processed
        ratio_ilc_all.append(ratio_ilc)
        ratio_ml_all.append(ratio_ml)

    ratio_ilc_all = np.asarray(ratio_ilc_all, dtype=float)
    ratio_ml_all = np.asarray(ratio_ml_all, dtype=float)
    ratio_ilc_mean = np.mean(ratio_ilc_all, axis=0)
    ratio_ilc_std = np.std(ratio_ilc_all, axis=0)
    ratio_ml_mean = np.mean(ratio_ml_all, axis=0)
    ratio_ml_std = np.std(ratio_ml_all, axis=0)

    ratio_npz = os.path.join(out_dir, "mean_ratio_spectra.npz")
    np.savez(
        ratio_npz,
        test_ids=np.asarray(test_ids, dtype=int),
        ell=ell,
        ratio_ilc_all=ratio_ilc_all,
        ratio_ml_all=ratio_ml_all,
        ratio_ilc_mean=ratio_ilc_mean,
        ratio_ilc_std=ratio_ilc_std,
        ratio_ml_mean=ratio_ml_mean,
        ratio_ml_std=ratio_ml_std,
    )
    print(f"[Evaluate] wrote ratio spectra over {len(test_ids)} test realisations: {ratio_npz}")

    import matplotlib.pyplot as plt

    plot_path = os.path.join(out_dir, "mean_ratio_spectra.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ell, ratio_ilc_mean, label="ILC Synth / Observed", color="blue")
    ax.fill_between(
        ell,
        ratio_ilc_mean - ratio_ilc_std,
        ratio_ilc_mean + ratio_ilc_std,
        color="blue",
        alpha=0.1,
    )
    ax.plot(ell, ratio_ml_mean, label="ML Improved / Observed", color="red")
    ax.fill_between(
        ell,
        ratio_ml_mean - ratio_ml_std,
        ratio_ml_mean + ratio_ml_std,
        color="red",
        alpha=0.1,
    )
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_ylim(0.95, 1.05)
    ax.set_xlim(2, ell.max())
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$C_\ell^{\mathrm{ratio}}$", fontsize=14)
    ax.set_title(f"Power Spectrum Ratio with Uncertainty ({args.run_id})", fontsize=15)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] wrote ratio plot for {len(test_ids)} test realisations: {plot_path}")
    
    return test_ids


def resolve_checkpoint_dir(args, model_dir: str) -> str:
    ckpt_dir, _, _ = resolve_checkpoint_target(model_dir, epoch=args.checkpoint_epoch)
    return str(ckpt_dir)


def main():
    start_time = time.perf_counter()
    jax.config.update("jax_enable_x64", False)
    print(f"JAX 64-bit mode: {jax.config.jax_enable_x64}")

    args = parse_args()
    if args.resume_training and args.mode not in {"train", "train+evaluate"}:
        raise ValueError(
            "--resume-training is only valid when --mode is 'train' or 'train+evaluate'."
        )
    if args.mode == "evaluate" and not args.run_id.strip() and not args.model_dir.strip():
        raise ValueError(
            "Evaluate mode requires either --run-id or --model-dir."
        )

    model_dir_from_train = None

    if args.mode == "train":
        print("[Train] Starting training...")
        model_dir_from_train = step_train(args)
        print(f"[Train] Done.")

    if args.mode == "train+evaluate":
        print("[Train] Starting training...")
        model_dir_from_train = step_train(args)
        print(f"[Train] Done.")
        print("[Evaluate] Starting evaluation...")
        ckpt_dir = resolve_checkpoint_dir(args, model_dir_from_train)
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[Evaluate] Done.")

    if args.mode == "evaluate":
        print("[Evaluate] Starting evaluation...")
        model_dir_from_train = resolve_evaluate_target(args)
        print(f"Model directory: {model_dir_from_train}")
        ckpt_dir = resolve_checkpoint_dir(args, model_dir_from_train)
        print(f'Loaded model from: {ckpt_dir}')
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[Evaluate] Done.")

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"[Time] Total pipeline time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()


# Example usage:
# 030 044 070 100 143 217 353 545 857
# python3 -m skyclean.ml.pipeline_ml --mode train+evaluate --extract-comp "cmb" --component "cfn" --frequencies 030 044 070 100 143 217 353 545 857 --realisations 7 --lmax 511 --N-directions 1 --lam 2.0 --batch-size 1 --split 0.3 0.3 0.4 --nsamp 1200 --epochs 2 --eval-every 1 --learning-rate 1e-3 --momentum 0.90 --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ --run-id test
# --resume-training
