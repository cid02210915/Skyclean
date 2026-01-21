# skyclean/ml/pipeline_ml.py

import os

from skyclean.silc.visualise import Visualise

# ---- must happen before any TF/TFDS import (Train/Data imports TF) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # keep TF off GPU
except Exception as e:
    # If TF isn't installed in some envs, don't hard-fail here
    print(f"[warn] TensorFlow GPU disable skipped: {e}")
# ----------------------------------------------------------------------
import re
import argparse
from pathlib import Path

import jax
from flax import nnx

from skyclean.ml.train import Train
from skyclean.ml.inference import Inference
from skyclean.silc.file_templates import FileTemplates  # adjust if needed

"""
def find_latest_checkpoint_dir(model_dir: str) -> str:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    pat = re.compile(r"^checkpoint_(\d+)$")
    candidates = []
    for p in model_dir.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_<epoch> folders found in {model_dir}. "
            f"Expected: {model_dir}/checkpoint_<epoch>"
        )

    candidates.sort(key=lambda t: t[0])
    epoch, ckpt_path = candidates[-1]
    print(f"[infer] Latest checkpoint detected: epoch={epoch}, path={ckpt_path}")
    return str(ckpt_path)


def resolve_model_dir_from_directory(directory: str) -> str:
    files = FileTemplates(directory)
    return os.path.abspath(files.output_directories["ml_models"])
"""

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
    parser.add_argument("--component", type=str, default="cfn")

    parser.add_argument("--frequencies", nargs="+", default=["030", "100", "353"])
    parser.add_argument("--realisations", type=int, default=1000)

    parser.add_argument("--lmax", type=int, default=1023)
    parser.add_argument("--N-directions", type=int, default=1)
    parser.add_argument("--lam", type=float, default=2.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--split", nargs=2, type=float, default=[0.8, 0.2])

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--chs", nargs="+", type=int, default=[1, 16, 32, 32, 64])

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--random", type=bool, default=False,
                        help="Generate test maps or not: True/False")

    parser.add_argument("--directory", type=str, default="data/")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--loss-tag", type=str, default=None)

    # ----- inference controls -----
    parser.add_argument("--realisation-infer", type=int, default=0)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--nsamp", type=int, default=1200)

    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Override model directory containing checkpoint_<epoch> folders. "
             "If empty, derived from FileTemplates(directory) like Train."
    )

    return parser.parse_args()


def step_train(args) -> str:
    """
    Train model and return the resolved model_dir where checkpoints were saved.
    """
    shuffle = not args.no_shuffle

    trainer = Train(
        extract_comp=args.extract_comp,
        component=args.component,
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
        rngs=nnx.Rngs(args.seed),
        directory=args.directory,
        resume_training=args.resume_training,
        loss_tag=args.loss_tag,
        random_generator=args.random,
    )

    trainer.execute_training_procedure()
    return trainer.model_dir


def step_evaluate(args, ckpt_dir: str | None = None):
    """
    Load latest checkpoint and run evaluation.
    """
    #if model_dir is None:
    #    model_dir = args.model_dir.strip() or resolve_model_dir_from_directory(args.directory)

    #latest_ckpt = find_latest_checkpoint_dir(model_dir)
    inference = Inference(
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=args.frequencies,
        realisations=args.realisations,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        chs=args.chs,
        directory=args.directory,
        model_path=ckpt_dir,  # points directly to checkpoint_<epoch>
        rn=args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )

    print("[evaluate] Loading model from latest checkpoint...")
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[infer] Model failed to load (load_model returned falsy).")

    print(f"[evaluate] Predicting CMB for realisation={args.realisation_infer} ...")
    cmb_improved = inference.predict_cmb(realisation=args.realisation_infer)

    import matplotlib.pyplot as plt
    from skyclean.silc.map_tools import SamplingConverters
    if args.plot:
        import healpy as hp
        from skyclean.silc.map_tools import SamplingConverters

        hp_map = SamplingConverters.mw_map_2_hp_map(cmb_improved, lmax=args.lmax)
        hp.mollview(hp_map, unit="K", cbar=True)
        plt.title(f"Improved CMB (realisation {args.realisation_infer})")
        plt.show()
    
    print("Visualising power spectra...")
    visualiser = Visualise(
        inference = inference, 
        frequencies=args.frequencies,
        realisation=0,
        lmax=args.lmax,
        lam_list=[args.lam],
        directory=args.directory,
        rn=args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        chs=args.chs,
        nsamp=args.nsamp,
    )

    ell, results = visualiser.visualise_component_ratio_power_spectra(comp_a=['ilc_synth', 'ilc_improved'],comp_b='processed_cmb',ratio=True, all_freq=False, masked=False)

    import numpy as np
    np.savez(
        str(ckpt_dir)+"/component_ratio_spectra.npz",
        ell=ell,
        ilc_synth_over_processed_cmb=results['ilc_synth/processed_cmb'],
        ilc_improved_over_processed_cmb=results['ilc_improved/processed_cmb'],
    )

    fig, ax = plt.subplots(figsize=(8,6))
    data = np.load(f"{ckpt_dir}/component_ratio_spectra.npz")
    ax.plot(data["ell"], data["ilc_synth_over_processed_cmb"], label=f'ilc_synth / processed')
    ax.plot(data["ell"], data["ilc_improved_over_processed_cmb"], label=f'ml_improved / processed')
    ax.axhline(1, ls=":", color="red")
    ax.set_ylim(0.5, 1.5)
    ax.set_ylabel(r"Ratio of $D_{\ell}$", fontsize=14)
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_title('Ratio of ratio (processed vs final)\nML: L2, 50 epochs', fontsize=13)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig(f'{ckpt_dir}/spectrum.png', dpi=250)
    plt.close()

    return cmb_improved


def main():
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
        ckpt_dir = os.path.join(model_dir_from_train, f"checkpoint_{args.epochs}")
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[evaluate] Done.")

    if args.mode == "evaluate":
        print("[evaluate] Starting evaluation...")
        shuffle = not args.no_shuffle
        trainer = Train(
            extract_comp=args.extract_comp,
            component=args.component,
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
            rngs=nnx.Rngs(args.seed),
            directory=args.directory,
            resume_training=args.resume_training,
            loss_tag=args.loss_tag,
            )
        model_dir_from_train = trainer.model_dir
        print(f"Model directory: {model_dir_from_train}")
        ckpt_dir = os.path.join(model_dir_from_train, f"checkpoint_{args.epochs}")
        print(f'loaded model from : {ckpt_dir}')
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[evaluate] Done.")

    print("[done] Pipeline complete.")


if __name__ == "__main__":
    main()


# Example usage:
# python3 -m skyclean.ml.pipeline_ml --mode train+evaluate --frequencies 030 044 070 100 143 217 353 545 857 --realisations 3 --lmax 511 --lam 2.0 --batch-size 1 --epochs 3 --learning-rate 1e-3 --momentum 0.90 --directory /share/lustre/keir/Skyclean2026/Skyclean/skyclean/data/