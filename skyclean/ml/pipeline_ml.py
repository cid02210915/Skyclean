# skyclean/ml/pipeline_ml.py
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # helps OOM/fragmentation

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
    parser.add_argument("--component", type=str, default="cfn")

    parser.add_argument("--frequencies", nargs="+", default=["030", "100", "353"])
    parser.add_argument("--realisations", type=int, default=1000)

    parser.add_argument("--lmax", type=int, default=1023)
    parser.add_argument("--N-directions", type=int, default=1)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--nsamp", type=int, default=1200)

    parser.add_argument("--constraint", action="store_true", help="Enable constraint")


    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-shuffle", action="store_true")
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
    # ziming: `--random` 是纯 flag（store_true），调用方应仅传 `--random`，不要传 `--random True`。
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
    parser.add_argument("--realisation-infer", type=int, default=0)
    parser.add_argument("--plot", action="store_true", default=False)

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
        "--eval-batches",
        "--eval-steps",
        dest="eval_batches",
        type=int,
        default=-1,
        help="Number of validation batches per evaluation run. -1 means full test set.",
    )
    parser.add_argument(
        "--prefetch",
        dest="prefetch",
        action="store_true",
        help="Enable tf.data prefetching in training.",
    )
    parser.add_argument(
        "--no-prefetch",
        dest="prefetch",
        action="store_false",
        help="Disable tf.data prefetching in training.",
    )
    parser.set_defaults(prefetch=False)
    parser.set_defaults(random=False)

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
        nsamp=args.nsamp,
        constraint=args.constraint,
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
        eval_every=args.eval_every,
        eval_steps=args.eval_batches,
        prefetch=args.prefetch,
        run_id=(args.run_id.strip() or None),
    )
    cfg_path = trainer.save_run_config(vars(args))
    print(f"[train] run_id={trainer.run_id}")
    print(f"[train] config saved: {cfg_path}")
    trainer.execute_training_procedure()
    return trainer.model_dir


def infer_args_from_checkpoint(
    ckpt_dir: str,
    *,
    realisation_infer: int | None = None,
    directory: str | None = None,
):
    """
    Infer pipeline args from checkpoint path by parsing ilc_improved map filenames.
    """
    ckpt_path = Path(ckpt_dir).resolve()
    ckpt_name = ckpt_path.name

    m_ckpt = re.search(r"checkpoint_(\d+)", ckpt_name)
    ckpt_epoch = int(m_ckpt.group(1)) if m_ckpt else None

    if directory is None:
        s = str(ckpt_path)
        if "/ML/models/" in s:
            directory = s.split("/ML/models/")[0]
        elif "/ML/model/" in s:
            directory = s.split("/ML/model/")[0]
        else:
            directory = str(ckpt_path.parent.parent.parent)

    files = FileTemplates(directory)
    improved_dir = files.output_directories["ilc_improved_maps"]
    candidates = sorted(glob.glob(os.path.join(improved_dir, "*.npy")))
    if not candidates:
        raise FileNotFoundError(f"No ilc_improved maps found in: {improved_dir}")

    pat = re.compile(
        r"^(?P<mode>con|uncon)_(?P<extract>[^_]+)_from-(?P<component>[^_]+)_improved_"
        r"f(?P<freqs>\d+(?:_\d+)*)_r(?P<real>\d+)_lmax(?P<lmax>\d+)_lam(?P<lam>[0-9.]+)_"
        r"nsamp(?P<nsamp>\d+)_rn(?P<rn>\d+)_batch(?P<batch>\d+)_epo(?P<epo>\d+)_"
        r"lr(?P<lr>[-+0-9.eE]+)_mom(?P<mom>[-+0-9.eE]+)_chs(?P<chs>\d+(?:_\d+)*)\.npy$"
    )

    parsed = []
    for fp in candidates:
        m = pat.match(os.path.basename(fp))
        if not m:
            continue
        d = m.groupdict()
        d["path"] = fp
        d["epo"] = int(d["epo"])
        parsed.append(d)

    if not parsed:
        raise RuntimeError(
            f"Found files in {improved_dir}, but none matched expected improved-map filename pattern."
        )

    if ckpt_epoch is not None:
        epoch_matches = [p for p in parsed if p["epo"] == ckpt_epoch]
    else:
        epoch_matches = parsed

    if not epoch_matches:
        raise RuntimeError(
            f"No improved-map filename matched checkpoint epoch={ckpt_epoch} in {improved_dir}."
        )

    chosen = max(epoch_matches, key=lambda p: os.path.getmtime(p["path"]))

    inferred_realisation = int(chosen["real"])
    args = argparse.Namespace(
        extract_comp=chosen["extract"],
        component=chosen["component"],
        frequencies=chosen["freqs"].split("_"),
        realisations=int(chosen["rn"]),
        realisation_infer=inferred_realisation if realisation_infer is None else int(realisation_infer),
        lmax=int(chosen["lmax"]),
        N_directions=1,
        lam=float(chosen["lam"]),
        nsamp=int(chosen["nsamp"]),
        constraint=(chosen["mode"] == "con"),
        batch_size=int(chosen["batch"]),
        epochs=int(chosen["epo"]),
        learning_rate=float(chosen["lr"]),
        momentum=float(chosen["mom"]),
        chs=[int(x) for x in chosen["chs"].split("_")],
        directory=directory,
        mode="evaluate",
        no_shuffle=False,
        split=[0.8, 0.2],
        seed=42,
        random=False,
        resume_training=False,
        loss_tag=None,
        model_dir="",
        eval_every=1,
        eval_steps=-1,
        prefetch=False,
        plot=False,
    )

    print(f"[infer_args_from_checkpoint] checkpoint: {ckpt_dir}")
    print(f"[infer_args_from_checkpoint] matched improved map: {chosen['path']}")
    return args


def step_spec(args=None, ckpt_dir: str | None = None):
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
          "processed_cmb": {"ell", "cl", "Dl", "path"},
          "ilc_synth":     {"ell", "cl", "Dl", "path"},
          "ilc_improved":  {"ell", "cl", "Dl", "path"},
        }
    """
    if args is None:
        if ckpt_dir is None:
            raise ValueError("step_spec requires either args or ckpt_dir.")
        args = infer_args_from_checkpoint(ckpt_dir)

    files = FileTemplates(args.directory)
    ft = files.file_templates
    conv = MapAlmConverter(ft)

    mode = "con" if args.constraint else "uncon"
    freq_tag = "_".join(str(x) for x in args.frequencies)
    chs = "_".join(str(n) for n in args.chs)
    lam_str = f"{float(args.lam):.1f}"
    realisation = int(args.realisation_infer)
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
    print(f"[step_spec] loading processed_cmb map from: {out_proc['path']}")
    ell_proc, cl_proc = PowerSpectrumTT.from_healpy_alm(out_proc["alm"])
    Dl_proc = PowerSpectrumTT.cl_to_Dl(ell_proc, cl_proc, input_unit="K")
    results["processed_cmb"] = {
        "ell": ell_proc,
        "cl": cl_proc,
        "Dl": Dl_proc,
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
    print(f"[step_spec] loading ilc_synth map from: {out_synth['path']}")
    ell_synth, cl_synth = PowerSpectrumTT.from_mw_alm(np.asarray(out_synth["alm"]))
    Dl_synth = PowerSpectrumTT.cl_to_Dl(ell_synth, cl_synth, input_unit="K")
    results["ilc_synth"] = {
        "ell": ell_synth,
        "cl": cl_synth,
        "Dl": Dl_synth,
        "path": out_synth["path"],
    }

    # 3) ilc_improved (MW) -> alm -> C_ell
    improved_map_path = ft["ilc_improved"].format(
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
    )
    print(f"[step_spec] loading ilc_improved map from: {improved_map_path}")
    if not os.path.exists(improved_map_path):
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"[step_spec] improved map not found: {improved_map_path}. "
                "Provide ckpt_dir so step_spec can generate it via inference."
            )

        print("[step_spec] improved map missing. Generating it via existing inference pipeline...")
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
        )
        model = inference.load_model(force_load=True)
        if model is None:
            raise RuntimeError(
                f"[step_spec] failed to load model from checkpoint: {ckpt_dir}"
            )
        inference.predict_cmb(realisation=realisation)

        if not os.path.exists(improved_map_path):
            raise FileNotFoundError(
                "[step_spec] improved map still missing after inference generation: "
                f"{improved_map_path}"
            )

    improved_map = np.load(improved_map_path)
    L = lmax + 1
    arr = np.asarray(np.real(np.squeeze(improved_map)), dtype=np.float64, order="C")
    alm_mw = s2fft.forward(arr, L=L)
    ell_improved, cl_improved = PowerSpectrumTT.from_mw_alm(np.asarray(alm_mw))
    Dl_improved = PowerSpectrumTT.cl_to_Dl(ell_improved, cl_improved, input_unit="K")
    results["ilc_improved"] = {
        "ell": ell_improved,
        "cl": cl_improved,
        "Dl": Dl_improved,
        "path": improved_map_path,
    }

    # Save improved spectrum in canonical SILC location (kept for compatibility).
    improved_spec_path = ft["ilc_improved_spectrum"].format(
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
    )
    os.makedirs(os.path.dirname(improved_spec_path), exist_ok=True)
    np.save(
        improved_spec_path,
        {"ell": ell_improved, "cl": cl_improved, "Dl": Dl_improved},
    )
    print(f"[step_spec] saved improved spectrum to: {improved_spec_path}")

    if ckpt_dir:
        ckpt_npz = os.path.join(ckpt_dir, "all_component_spectra.npz")
        np.savez(
            ckpt_npz,
            ell_processed=results["processed_cmb"]["ell"],
            cl_processed=results["processed_cmb"]["cl"],
            Dl_processed=results["processed_cmb"]["Dl"],
            ell_ilc_synth=results["ilc_synth"]["ell"],
            cl_ilc_synth=results["ilc_synth"]["cl"],
            Dl_ilc_synth=results["ilc_synth"]["Dl"],
            ell_ilc_improved=results["ilc_improved"]["ell"],
            cl_ilc_improved=results["ilc_improved"]["cl"],
            Dl_ilc_improved=results["ilc_improved"]["Dl"],
        )
        print(f"[step_spec] wrote checkpoint copy: {ckpt_npz}")

        plot_path = os.path.join(ckpt_dir, "all_component_spectra.png")
        PowerSpectrumTT.plot_Dl_series(
            [
                {
                    "ell": results["processed_cmb"]["ell"],
                    "Dl": results["processed_cmb"]["Dl"],
                    "label": "Processed CMB",
                    "source": "processed",
                },
                {
                    "ell": results["ilc_synth"]["ell"],
                    "Dl": results["ilc_synth"]["Dl"],
                    "label": "ILC-synth",
                    "source": "ilc_synth",
                },
                {
                    "ell": results["ilc_improved"]["ell"],
                    "Dl": results["ilc_improved"]["Dl"],
                    "label": "ILC-improved",
                    "source": "ilc_improved",
                },
            ],
            save_path=plot_path,
            show=False,
        )

    return results


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
    )

    print("[evaluate] Loading model from latest checkpoint...")
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[infer] Model failed to load (load_model returned falsy).")

    print(f"[evaluate] Predicting CMB for realisation={args.realisation_infer} ...")
    cmb_improved = inference.predict_cmb(realisation=args.realisation_infer)
    spec = step_spec(args, ckpt_dir=ckpt_dir)

    if ckpt_dir:
        ell = spec["processed_cmb"]["ell"]
        cl_processed = spec["processed_cmb"]["cl"]
        cl_ilc_synth = spec["ilc_synth"]["cl"]
        cl_ilc_improved = spec["ilc_improved"]["cl"]
        ratio_synth = cl_ilc_synth / cl_processed
        ratio_improved = cl_ilc_improved / cl_processed

        np.savez(
            os.path.join(ckpt_dir, "component_ratio_spectra.npz"),
            ell=ell,
            ilc_synth_over_processed_cmb=ratio_synth,
            ilc_improved_over_processed_cmb=ratio_improved,
        )
        print(f"[evaluate] wrote ratio spectra: {os.path.join(ckpt_dir, 'component_ratio_spectra.npz')}")

    import matplotlib.pyplot as plt
    from skyclean.silc.map_tools import SamplingConverters
    if args.plot:
        import healpy as hp
        from skyclean.silc.map_tools import SamplingConverters

        hp_map = SamplingConverters.mw_map_2_hp_map(cmb_improved, lmax=args.lmax)
        hp.mollview(hp_map, unit="K", cbar=True)
        if args.constraint == True:
            mode="cILC"
        else:
            mode="ILC"
        plt.title(f"Improved {mode} map\nlmax{args.lmax}, {args.extract_comp}, {args.frequencies}, r{args.realisation_infer}, lam{args.lam}, nsamp{args.nsamp})")
        plt.tight_layout()
        plt.savefig(f'{ckpt_dir}/ILC_improved.png', dpi=300)
        plt.show()
    
    return cmb_improved


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
# python3 -m skyclean.ml.pipeline_ml --mode train+evaluate --extract-comp "cmb" --component "cfn" --frequencies 030 044 070 100 143 217 353 545 857 --realisations 2 --lmax 511 --N-directions 1 --lam 2.0 --batch-size 1 --nsamp 1200 --epochs 10 --eval-every 5 --learning-rate 1e-3 --momentum 0.90 --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ --plot --run-id testing --random 


''' Example usage:
python3 -m skyclean.ml.pipeline_ml \
  --mode train+evaluate \
  --extract-comp cmb \
  --component cfn \
  --frequencies 030 044 070 100 143 217 353 545 857 \
  --realisations 100 \
  --lmax 511 \
  --N-directions 1 \
  --lam 2.0 \
  --nsamp 1200 \
  --batch-size 5 \
  --epochs 200 \
  --learning-rate 1e-3 \
  --momentum 0.90 \
  --eval-every 5 \
  --eval-batches 10 \
  --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ \
  --prefetch \
  --run-id 20260226_1
'''
