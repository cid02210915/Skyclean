# skyclean/ml/pipeline_ml.py

import os
import re
import argparse
from pathlib import Path

import jax
from flax import nnx

from skyclean.ml.train import Train
from skyclean.ml.inference import Inference
from skyclean.silc.file_templates import FileTemplates  # adjust if needed


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline (train + inference) for Skyclean ML."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer", "train+infer"],
        help="Which stages to run."
    )

    # ----- match Train signature -----
    parser.add_argument("--extract-comp", type=str, default="cmb")
    parser.add_argument("--component", type=str, default="cfn")

    parser.add_argument("--frequencies", nargs="+", default=["030", "100", "353"])
    parser.add_argument("--realisations", type=int, default=1000)

    parser.add_argument("--lmax", type=int, default=1024)
    parser.add_argument("--N-directions", type=int, default=1)
    parser.add_argument("--lam", type=float, default=2.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--split", nargs=2, type=float, default=[0.8, 0.2])

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--chs", nargs="+", type=int, default=[1, 16, 32, 32, 64])

    # ✅ seed option (explicitly included + used in step_train)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument("--directory", type=str, default="data/")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--loss-tag", type=str, default=None)

    # ----- cluster controls -----
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mem-fraction", type=float, default=0.7)

    # ----- inference controls -----
    parser.add_argument("--realisation-infer", type=int, default=0)
    parser.add_argument("--save-npy", type=str, default="")
    parser.add_argument("--plot", action="store_true", default=False)

    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Override model directory containing checkpoint_<epoch> folders. "
             "If empty, derived from FileTemplates(directory) like Train."
    )

    return parser.parse_args()


def configure_gpu(gpu: int, mem_fraction: float):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    print(f"Using GPU {gpu} with {mem_fraction*100:.0f}% memory fraction.")


def step_train(args) -> str:
    """
    Train model and return the resolved model_dir where checkpoints were saved.
    """
    shuffle = not args.no_shuffle

    # ✅ seed used here
    rngs = nnx.Rngs(args.seed)
    print(f"[train] seed={args.seed}")

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
        rngs=rngs,
        directory=args.directory,
        resume_training=args.resume_training,
        loss_tag=args.loss_tag,
    )

    trainer.execute_training_procedure()
    return trainer.model_dir


def step_infer(args, model_dir: str | None = None):
    """
    Load latest checkpoint and run inference.
    """
    if model_dir is None:
        model_dir = args.model_dir.strip() or resolve_model_dir_from_directory(args.directory)

    latest_ckpt = find_latest_checkpoint_dir(model_dir)

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
        model_path=latest_ckpt,  # points directly to checkpoint_<epoch>
        rn=args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )

    print("[infer] Loading model from latest checkpoint...")
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[infer] Model failed to load (load_model returned falsy).")

    print(f"[infer] Predicting CMB for realisation={args.realisation_infer} ...")
    cmb_improved = inference.predict_cmb(realisation=args.realisation_infer)

    if args.save_npy:
        out = Path(args.save_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        np.save(str(out), cmb_improved)
        print(f"[infer] Saved inferred map to: {out}")

    if args.plot:
        import healpy as hp
        import matplotlib.pyplot as plt
        from skyclean.silc.map_tools import SamplingConverters

        hp_map = SamplingConverters.mw_map_2_hp_map(cmb_improved, lmax=args.lmax)
        hp.mollview(hp_map, unit="K", cbar=True)
        plt.title(f"Improved CMB (realisation {args.realisation_infer})")
        plt.show()

    return cmb_improved


def main():
    jax.config.update("jax_enable_x64", False)
    print(f"JAX 64-bit mode: {jax.config.jax_enable_x64}")

    args = parse_args()
    configure_gpu(args.gpu, args.mem_fraction)

    model_dir_from_train = None

    if args.mode in ("train", "train+infer"):
        print("[train] Starting training...")
        model_dir_from_train = step_train(args)
        print(f"[train] Done. model_dir={model_dir_from_train}")

    if args.mode in ("infer", "train+infer"):
        print("[infer] Starting inference...")
        step_infer(args, model_dir=model_dir_from_train)
        print("[infer] Done.")

    print("[done] Pipeline complete.")


if __name__ == "__main__":
    main()
