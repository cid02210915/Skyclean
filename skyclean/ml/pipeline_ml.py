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

from skyclean.ml.train import Train
from skyclean.ml.inference import Inference
from skyclean.silc.file_templates import FileTemplates
from skyclean.silc.power_spec import MapAlmConverter, PowerSpectrumTT


def find_latest_checkpoint_dir(model_dir: str) -> str:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    pat = re.compile(r"^checkpoint_(\d+)$")
    candidates = []
    for p in model_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        # Accept Orbax or fallback serializer checkpoints, skip plot-only folders.
        names = {x.name for x in p.iterdir()}
        valid = (
            (p / "_METADATA").exists()
            or (p / "_CHECKPOINT_METADATA").exists()
            or (p / "state.msgpack").exists()
            or any(n not in {"prediction.png", "training_metrics.png", "training_log.npy", "spectrum.png"} for n in names)
        )
        if valid:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(
            f"No valid checkpoint_<epoch> folders found in {model_dir}. "
            f"Expected: {model_dir}/checkpoint_<epoch>"
        )

    candidates.sort(key=lambda t: t[0])
    epoch, ckpt_path = candidates[-1]
    print(f"[checkpoint] Latest checkpoint detected: epoch={epoch}, path={ckpt_path}")
    return str(ckpt_path)


def resolve_model_dir_from_directory(directory: str) -> str:
    files = FileTemplates(directory)
    return os.path.abspath(files.output_directories["ml_models"])


def resolve_evaluate_target(args) -> str:
    """
    Resolve evaluation target from CLI args.

    Accepted inputs:
    - `--model-dir` pointing to a run directory containing checkpoint_* folders
    - `--model-dir` pointing directly to a checkpoint_* directory
    - `--run-id` pointing to ML/models/<run_id>
    - no `--run-id`, in which case ML/models itself is searched
    """
    if args.model_dir.strip():
        return os.path.abspath(args.model_dir.strip())

    base_model_dir = resolve_model_dir_from_directory(args.directory)
    run_id = args.run_id.strip()
    if not run_id:
        return base_model_dir

    run_dir = os.path.join(base_model_dir, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}. "
            "With --run-id, checkpoints are expected under ML/models/<run_id>/checkpoint_<epoch>."
        )
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
                        help="Input map product key, e.g. cfn, cfne, or cfne_circ.")

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
    using the same core flow as SILC step_power_spec: map -> alm -> C_ell -> D_ell.

    Returns
    -------
    dict:
        {
          "processed_cmb": {"ell", "cl", "path"},
          "ilc_synth":     {"ell", "cl", "path"},
          "ilc_improved":  {"ell", "cl", "path"},
        }
    """

    files = FileTemplates(args.directory)
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

    # 1) processed_cmb (HEALPix) -> alm -> C_ell
    out_proc = conv.to_alm(
        component="cmb",
        source="processed",
        realisation=realisation,
        lmax=lmax,
    )
    print(f"[spectrum] loading processed_cmb map from: {out_proc['path']}")
    ell_proc, cl_proc = PowerSpectrumTT.from_healpy_alm(out_proc["alm"])
    results["processed_cmb"] = {
        "ell": ell_proc,
        "cl": cl_proc,
        "path": out_proc["path"],
    }

    # 2) ilc_synth (MW) -> alm -> C_ell
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
    print(f"[spectrum] loading ilc_synth map from: {out_synth['path']}")
    ell_synth, cl_synth = PowerSpectrumTT.from_mw_alm(np.asarray(out_synth["alm"]))
    results["ilc_synth"] = {
        "ell": ell_synth,
        "cl": cl_synth,
        "path": out_synth["path"],
    }


    # 3) ilc_improved (MW) -> alm -> C_ell
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
    improved_map_path = os.path.join(
        files.output_directories["cmb_prediction"], args.run_id, "ilc_improved_maps", improved_filename
    )
    print(f"[spectrum] loading ilc_improved map from: {improved_map_path}")
    if not os.path.exists(improved_map_path):
        if ckpt_dir is None:
            raise FileNotFoundError(f"[spectrum] improved map not found: {improved_map_path}. ")

        print("[spectrum] ML map missing. Generating it via existing inference pipeline...")
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

    return results

def step_evaluate(args, ckpt_dir: str | None = None):
    """
    Load latest checkpoint and save predictions for the held-out test split.
    """

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

    print("[evaluate] Loading model from latest checkpoint...")
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[evaluate] Model failed to load (load_model returned falsy).")

    test_ids = inference.data_handler.get_split_indices()["test"]
    print(f"[evaluate] Predicting CMB for {len(test_ids)} test realisations...")
    metrics_rows = inference.save_test_metrics_table(save_predictions=True)
    inference.save_test_scatter_plots(metrics_rows)

    ratio_ilc_all = []
    ratio_ml_all = []
    out_dir = os.path.join(inference.file_templates.output_directories["cmb_prediction"], args.run_id, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    for realisation in test_ids:
        spec_args = argparse.Namespace(**vars(args))
        spec_args.realisation = int(realisation)
        spec = generate_spectrum_for_one(spec_args, ckpt_dir=ckpt_dir)
        bundle_path = os.path.join(out_dir, f"component_spectra_r{int(realisation):04d}.npz")
        np.savez(
            bundle_path,
            ell=np.asarray(spec["processed_cmb"]["ell"], dtype=float),
            processed_cmb_cl=np.asarray(spec["processed_cmb"]["cl"], dtype=float),
            ilc_synth_cl=np.asarray(spec["ilc_synth"]["cl"], dtype=float),
            ilc_improved_cl=np.asarray(spec["ilc_improved"]["cl"], dtype=float),
        )
        print(f"[evaluate] wrote component spectra bundle: {bundle_path}")

        ell = np.asarray(spec["processed_cmb"]["ell"], dtype=float)
        cl_processed = np.asarray(spec["processed_cmb"]["cl"], dtype=float)
        print(f'range of processed_cmb C_ell: {cl_processed.min():.3e} to {cl_processed.max():.3e}')
        cl_ilc_synth = np.asarray(spec["ilc_synth"]["cl"], dtype=float)
        print(f'range of ilc_synth C_ell: {cl_ilc_synth.min():.3e} to {cl_ilc_synth.max():.3e}')
        cl_ilc_improved = np.asarray(spec["ilc_improved"]["cl"], dtype=float)
        print(f'range of ilc_improved C_ell: {cl_ilc_improved.min():.3e} to {cl_ilc_improved.max():.3e}')

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
    print(f"[evaluate] wrote ratio spectra over {len(test_ids)} test realisations: {ratio_npz}")

    import matplotlib.pyplot as plt

    plot_path = os.path.join(out_dir, "mean_ratio_spectra.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ell, ratio_ilc_mean, label="ILC mean", color="tab:orange")
    ax.fill_between(
        ell,
        ratio_ilc_mean - ratio_ilc_std,
        ratio_ilc_mean + ratio_ilc_std,
        color="tab:orange",
        alpha=0.2,
        label="ILC std",
    )
    ax.plot(ell, ratio_ml_mean, label="ML mean", color="tab:blue")
    ax.fill_between(
        ell,
        ratio_ml_mean - ratio_ml_std,
        ratio_ml_mean + ratio_ml_std,
        color="tab:blue",
        alpha=0.2,
        label="ML std",
    )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("Ratio to processed CMB")
    ax.set_title(f"Component Ratio Spectra Across Test Realisations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] wrote ratio plot for {len(test_ids)} test realisations: {plot_path}")
    
    return test_ids


def resolve_checkpoint_dir(args, model_dir: str) -> str:
    model_path = Path(model_dir)
    direct_ckpt_pat = re.compile(r"^checkpoint_(\d+)$")

    if model_path.is_dir() and direct_ckpt_pat.match(model_path.name):
        return str(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Run directory does not exist: {model_dir}. "
            "Evaluation with --run-id requires an existing ML/models/<run_id> directory."
        )

    if args.checkpoint_epoch is not None:
        ckpt_dir = os.path.join(model_dir, f"checkpoint_{args.checkpoint_epoch}")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Requested checkpoint does not exist: {ckpt_dir}")
        return ckpt_dir
    return find_latest_checkpoint_dir(model_dir)


def main():
    start_time = time.perf_counter()
    jax.config.update("jax_enable_x64", False)
    print(f"JAX 64-bit mode: {jax.config.jax_enable_x64}")

    args = parse_args()

    model_dir_from_train = None

    if args.mode == "train":
        print("[train] Starting training...")
        model_dir_from_train = step_train(args)
        print(f"[train] Done.")

    if args.mode == "train+evaluate":
        print("[train] Starting training...")
        model_dir_from_train = step_train(args)
        print(f"[train] Done.")
        print("[evaluate] Starting evaluation...")
        ckpt_dir = resolve_checkpoint_dir(args, model_dir_from_train)
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[evaluate] Done.")

    if args.mode == "evaluate":
        print("[evaluate] Starting evaluation...")
        model_dir_from_train = resolve_evaluate_target(args)
        print(f"Model directory: {model_dir_from_train}")
        ckpt_dir = resolve_checkpoint_dir(args, model_dir_from_train)
        print(f'loaded model from: {ckpt_dir}')
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[evaluate] Done.")

    print("[done] Pipeline complete.")
    elapsed_seconds = time.perf_counter() - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"[time] Total pipeline time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()


# Example usage:
# 030 044 070 100 143 217 353 545 857
# python3 -m skyclean.ml.pipeline_ml --mode evaluate --extract-comp "cmb" --component "cfn" --frequencies 030 044 070 100 143 217 353 545 857 --realisations 7 --lmax 511 --N-directions 1 --lam 2.0 --batch-size 10 --split 0.1 0.1 0.8 --nsamp 1200 --epochs 100 --eval-every 1 --learning-rate 1e-3 --momentum 0.90 --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ --run-id CFN_r500_511
