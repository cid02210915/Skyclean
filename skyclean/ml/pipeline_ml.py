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
try:
    print("JAX GPUs =", jax.devices("gpu"))
except RuntimeError:
    print("JAX GPUs = none (CPU only; sufficient for --mode prepare)")

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
from skyclean.ml.data import CMBFreeILC
from skyclean.silc.file_templates import FileTemplates, register_pixel_ps_component_template, register_mixture_component_template
from skyclean.silc.utils import ilc_mode, ml_tag
from skyclean.silc.power_spec import MapAlmConverter, PowerSpectrumCrossTT, PowerSpectrumTT
from skyclean.silc import HPTools
import healpy as hp


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


PIPELINE_STEPS = ("prepare", "train", "evaluate", "apply")


def parse_mode(mode: str) -> list[str]:
    """Helper function to parse the --mode argument, ensures that the specified steps are valid and returns them in execution order.
    Split a '+'-joined --mode string into pipeline steps, returned in execution order.
    """
    tokens = [t.strip().lower() for t in str(mode).split("+") if t.strip()]
    if not tokens:
        raise ValueError("--mode cannot be empty.")
    unknown = [t for t in tokens if t not in PIPELINE_STEPS]
    if unknown:
        raise ValueError(
            f"Unknown step(s) in --mode: {unknown}. Choose from {list(PIPELINE_STEPS)}, joined with '+'."
        )
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"--mode lists a step more than once: {mode!r}")
    return [step for step in PIPELINE_STEPS if step in tokens]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline (prepare + train + evaluate + apply) for Skyclean ML."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        metavar="STEP[+STEP...]",
        help="Which steps to run, joined with '+', from: "
             "'prepare' (data-preparation step: generate the model input/target maps from the SILC outputs), "
             "'train' (training step; reuses prepared maps if present, otherwise generates them first), "
             "'evaluate' (predict the held-out test split of --component: cleaned maps and the alm npz of the test split), "
             "'apply' (apply the trained model to the observed Planck sky; needs the SILC run with "
             "--components real --wavelet-components real; with --eval-component to the simulated maps of another "
             "foreground mixture instead). "
             "E.g. prepare, train, evaluate, apply, prepare+train, train+evaluate, prepare+train+evaluate. Default: train."
    )

    # ----- match Train signature -----
    parser.add_argument("--extract-comp", type=str, default="cmb")
    parser.add_argument("--component", type=str, default="cfn",
                        help="Input map product key, e.g. cfn, cfne, cfne_circ, cfne_pix_N, or a foreground-mixture tag such as "
                             "cfn-a1-d1-f1-s1. A '+'-joined tag (cfn-dust-ff-spindust-sync+cfn-a1-d1-f1-s1) trains on a mixed set: "
                             "the first half of --realisations from the first mixture, the second half from the second, with "
                             "--split applied inside each half; evaluate then writes one spectra npz per mixture.")

    parser.add_argument("--frequencies", nargs="+", default=["030", "100", "353"])
    parser.add_argument("--realisations", type=int, default=1000)

    parser.add_argument("--lmax", type=int, default=1023)
    parser.add_argument("--N-directions", type=int, default=1)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--nsamp", type=int, default=1200)

    parser.add_argument("--deproject", nargs="+", default=None, metavar="COMP",
                        help="SILC outputs that deproject these components (tsz, cib): cILC files tagged cilc-dp-<comps>, or with --pcilc-eps the pcILC files tagged pcilc-dp-<comp>-eps<eps>. Omit for the plain ILC. Must match the SILC run")
    parser.add_argument("--pcilc-eps", type=float, default=None,
                        help="pcILC tolerance on the single --deproject component (files tagged pcilc-dp-<comp>-eps<eps>); must match the SILC run. Omit for the cILC")


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
    parser.add_argument("--filter-type", type=str, default="auto",
                        choices=["auto", "axisymmetric", "directional", "square"],
                        help="DISCO filter type for S2_UNET conv blocks. 'auto' (default) follows "
                             "--N-directions: axisymmetric when it is 1, directional otherwise. "
                             "Changing this changes the weight structure; use a fresh --run-id.")

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
        help="Run folder under ML/models. Required with 'prepare'; if omitted in training, uses timestamp YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--loss-tag", type=str, default=None)
    parser.add_argument("--masked", action="store_true",
                        help="Masked ML: the training loss/accuracy are restricted to the Planck 2018 common CMB "
                             "intensity mask (fsky~0.78; another mask with --mask-index). Pass it again at evaluate/apply "
                             "for a run trained masked: the outputs are then tagged ml-masked and the pseudo-alm of "
                             "the maps inside the training mask are saved next to the full-sky alm.")
    parser.add_argument("--mask-index", type=int, default=None, choices=range(1, 31), metavar="1-30",
                        help="Masked ML with mask <index> of <directory>/masks_fsky_078_to_099 (1 = fsky 0.78 ... 30 = fsky 0.99) "
                             "instead of the Planck 2018 common mask; implies --masked. The run id gets the suffix "
                             "_mask<KK> and the outputs are tagged ml-masked-mask<KK>. Pass it again at evaluate/apply.")
    parser.add_argument("--eval-component", nargs="+", type=str, default=None, metavar="TAG",
                        help="apply: apply the trained model to the simulated SILC maps of these foreground mixtures "
                             "(e.g. cfn-dust-ff-spindust-sync, same lmax/N/lam/nsamp/mode), one after the other, instead of the "
                             "observed sky, for the realisations --eval-realisations; the normalisation statistics stay those "
                             "of --component (the training set). Per realisation: the cleaned map and a maps figure, plus "
                             "per mixture one npz with the full-sky alm of all its realisations (processed cmb, ILC, "
                             "ILC + ML) and, for a masked run, one with their pseudo-alm inside the training mask.")
    parser.add_argument("--eval-realisations", nargs=2, type=int, default=None, metavar=("FIRST", "LAST"),
                        help="apply with --eval-component: first and last realisation to apply to, inclusive (e.g. 100 199).")

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

    args = parser.parse_args()
    try:
        parse_mode(args.mode)
    except ValueError as e:
        parser.error(str(e))
    return args


def step_prepare_data(args) -> None:
    """
    Data-preparation step: generate the model input and target maps for every realisation from the SILC outputs,
    or random test maps with --random, then fit the train-split normalisation statistics and save them in the run
    folder ML/models/<run_id>/ (needs --run-id). Existing maps and statistics are kept. Runs on CPU; no model is created.
    """
    dataset = CMBFreeILC(
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=args.frequencies,
        realisations=args.realisations,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        nsamp=args.nsamp,
        deproject=args.deproject,
        pcilc_eps=args.pcilc_eps,
        batch_size=args.batch_size,
        split=args.split,
        directory=args.directory,
        random=args.random,
        prefetch=args.prefetch,
        produce_residuals=False,
        run_id=args.run_id.strip(),
        mask_index=args.mask_index,
    )
    dataset.produce_residuals()
    print(f"[Prepare] Input/target maps ready for {args.realisations} realisations.")
    if not args.random and dataset.load_normalization_stats() is None:
        dataset.find_dataset_mean_std()  # fits on the train split and saves norm_stats_*.npz


def step_train(args) -> str:
    """
    Training step: train the model and return the resolved model_dir where checkpoints were saved.
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
        deproject=args.deproject,
        pcilc_eps=args.pcilc_eps,
        batch_size=args.batch_size,
        split=args.split,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        chs=args.chs,
        filter_type=args.filter_type,
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
        mask_index=args.mask_index,
    )
    cfg_path = trainer.save_run_config(vars(args))
    print(f"[train] run_id={trainer.run_id}")
    print(f"[train] config saved: {cfg_path}")
    trainer.dataset.produce_residuals()  # data-preparation step; skips maps that already exist
    trainer.execute_training_procedure(masked=args.masked)  # training step
    return trainer.model_dir



def generate_spectrum_for_one(args=None, ckpt_dir: str | None = None, mask_hp: np.ndarray | None = None):
    """
    Compute TT power spectra for:
      - processed_cmb (HEALPix)
      - ilc_synth (MW)
      - ilc_ml (MW)
    Also compute cross power spectra for:
      - ilc vs processed_cmb (MW x HEALPix)
      - ml vs processed_cmb (MW x HEALPix)

    Returns
    -------
    dict:
        {
          "processed_cmb": {"ell", "cl", "path"},
          "ilc_synth":     {"ell", "cl", "path"},
          "ilc_ml":        {"ell", "cl", "path"},
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
    register_mixture_component_template(
        files.file_templates,
        files.output_directories,
        args.component,
    )
    ft = files.file_templates
    conv = MapAlmConverter(ft)

    mode = ilc_mode(deproject=args.deproject, pcilc_eps=args.pcilc_eps)
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
        N_directions=args.N_directions,
        lam=lam_str,
        nsamp=nsamp,
        deproject=args.deproject,
        mode=mode,
    )
    print(f"[Spectrum] loading ilc_synth map from: {out_synth['path']}")
    ell_synth, cl_synth = PowerSpectrumTT.from_mw_alm(np.asarray(out_synth["alm"]))
    results["ilc_synth"] = {
        "ell": ell_synth,
        "cl": cl_synth,
        "path": out_synth["path"],
    }


    # 3) auto-spectra: ilc_ml (MW) -> alm -> C_ell
    ilc_ml_filename = os.path.basename(ft["ilc_ml"].format(
        mode=mode,
        ml=ml_tag(args.masked, args.mask_index),
        extract_comp=args.extract_comp,
        component=args.component,
        frequencies=freq_tag,
        realisation=realisation,
        lmax=lmax,
        N_directions=args.N_directions,
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
    ilc_ml_map_path = os.path.join(
        files.output_directories["cmb_prediction"],
        args.run_id,
        "ilc_ml_maps",
        f"checkpoint_{int(args.checkpoint_epoch)}",
        ilc_ml_filename.replace(".npy", f"_ckpt{int(args.checkpoint_epoch)}.npy"),
    )
    print(f"[Spectrum] loading ilc_ml map from: {ilc_ml_map_path}")
    if not os.path.exists(ilc_ml_map_path):
        if ckpt_dir is None:
            raise FileNotFoundError(f"[Spectrum] ILC + ML map not found: {ilc_ml_map_path}. ")

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
            deproject=args.deproject,
            pcilc_eps=args.pcilc_eps,
            chs=args.chs,
            filter_type=args.filter_type,
            directory=args.directory,
            seed=args.seed,
            model_path=ckpt_dir,
            rn=args.realisations,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            run_id=args.run_id,
            masked=args.masked,
            split=args.split,
        )
        model = inference.load_model(force_load=True)
        if model is None:
            raise RuntimeError(f"[spectrum] failed to load model from checkpoint: {ckpt_dir}")
        inference.predict_cmb(realisation=realisation)

        if not os.path.exists(ilc_ml_map_path):
            raise FileNotFoundError(
                "[spectrum] ILC + ML map still missing after inference generation: "
                f"{ilc_ml_map_path}"
            )

    ilc_ml_map = np.load(ilc_ml_map_path)
    L = lmax + 1
    arr = np.asarray(np.real(np.squeeze(ilc_ml_map)), dtype=np.float64, order="C")
    alm_mw = s2fft.forward(arr, L=L)
    ell_ilc_ml, cl_ilc_ml = PowerSpectrumTT.from_mw_alm(np.asarray(alm_mw))
    results["ilc_ml"] = {
        "ell": ell_ilc_ml,
        "cl": cl_ilc_ml,
        "path": ilc_ml_map_path,
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
        "path": ilc_ml_map_path,
    }

    if mask_hp is not None:
        # Masked evaluation: pseudo-C_ell of the masked maps. The same mask multiplies every map, so the mode
        # coupling is common to the numerator and denominator of the ratios; dividing by <mask^2> puts the
        # spectra back on the full-sky scale (the ratios are unaffected).
        nside = HPTools.get_nside_from_lmax(lmax)
        mask_hp = np.asarray(mask_hp, dtype=np.float64)
        f_sky2 = float(np.mean(mask_hp ** 2))

        def masked_hp_map(alm, fmt):
            alm_hp = np.asarray(alm) if fmt == "healpy" else PowerSpectrumCrossTT.mw_to_healpy_alm(np.asarray(alm), lmax=lmax)
            return hp.alm2map(np.ascontiguousarray(alm_hp, dtype=np.complex128), nside=nside) * mask_hp

        maps = {
            "processed_cmb": masked_hp_map(out_proc["alm"], "healpy"),
            "ilc_synth": masked_hp_map(out_synth["alm"], "mw"),
            "ilc_ml": masked_hp_map(alm_mw, "mw"),
        }
        for key in ("processed_cmb", "ilc_synth", "ilc_ml"):
            results[key]["cl"] = hp.anafast(maps[key], lmax=lmax) / f_sky2
        results["ilc-cmb"]["cl"] = hp.anafast(maps["ilc_synth"], map2=maps["processed_cmb"], lmax=lmax) / f_sky2
        results["ml-cmb"]["cl"] = hp.anafast(maps["ilc_ml"], map2=maps["processed_cmb"], lmax=lmax) / f_sky2

    return results

def step_evaluate(args, ckpt_dir: str | None = None):
    """
    Predict the held-out test split of the training mixture (--component) with the checkpoint in ckpt_dir. Same
    outputs as `apply --eval-component` (see apply_to_realisations): per test realisation the cleaned CMB map and the
    maps figure, plus one npz with the full-sky alm of processed cmb / ILC / ILC + ML of every test realisation and,
    for a masked run, one with their pseudo-alm inside the training mask.
    Returns the test realisation ids.
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
        deproject=args.deproject,
        pcilc_eps=args.pcilc_eps,
        chs=args.chs,
        filter_type=args.filter_type,
        directory=args.directory,
        seed=args.seed,
        model_path=ckpt_dir,  # points directly to checkpoint_<epoch>
        rn=args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        run_id=args.run_id,
        masked=args.masked,
        split=args.split,
        mask_index=args.mask_index,
    )
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[Evaluate] Model failed to load (load_model returned falsy).")

    test_ids = [int(r) for r in inference.data_handler.get_split_indices()["test"]]
    if not test_ids:
        raise ValueError("[Evaluate] The test split is empty; adjust --split / --realisations.")
    # one group per part of a mixed training tag "A+B" (a single mixture gives one group), each with its own spectra npz
    groups = {}
    for r in test_ids:
        groups.setdefault(inference.data_handler.component_of(r), []).append(r)
    for component, ids in groups.items():
        print(f"[Evaluate] Predicting the {len(ids)} test realisations r{ids[0]:04d}-r{ids[-1]:04d} of '{component}'...")
        apply_to_realisations(args, inference, component, ids)
    return test_ids


def step_apply(args, ckpt_dir: str | None = None) -> str:
    """
    Apply the trained model to the observed Planck sky (Inference.predict_and_visualise_real_sky): saves the
    cleaned CMB map (MW .npy), the ILC / ILC + ML / difference map figure and the alm npz of the ILC and ILC + ML maps
    (see apply_to_realisations; the observed sky has no processed cmb) under <directory>/ML/cmb_prediction/<run_id>/ilc_ml_maps/checkpoint_<epoch>/.
    With --masked (model trained masked) every output is tagged ml-masked instead of ml (ml-masked-mask<KK> with --mask-index).
    Needs the SILC pipeline run with --components real --wavelet-components real (same lmax/N/lam/nsamp).
    Returns the path of the full-sky alm npz.
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
        deproject=args.deproject,
        pcilc_eps=args.pcilc_eps,
        chs=args.chs,
        filter_type=args.filter_type,
        directory=args.directory,
        seed=args.seed,
        model_path=ckpt_dir,  # points directly to checkpoint_<epoch>
        rn=args.realisations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        run_id=args.run_id,
        masked=args.masked,
        split=args.split,
        mask_index=args.mask_index,
    )
    model = inference.load_model(force_load=True)
    if not model:
        raise RuntimeError("[Apply] Model failed to load (load_model returned falsy).")

    if args.eval_component:
        return [step_apply_simulations(args, inference, component) for component in args.eval_component][-1]

    # The observed sky has no realisations; SILC still tags its from-real outputs with r0000
    # (its --start-realisation, assumed 0), so that index is used to locate the ilc_synth file.
    print("[Apply] Predicting the observed Planck sky...")
    return apply_to_realisations(args, inference, "real", [0])


def step_apply_simulations(args, inference: Inference, component: str) -> str:
    """
    Apply the trained model to the simulated SILC maps of another foreground mixture `component` (one of
    --eval-component) for the realisations --eval-realisations FIRST LAST (inclusive), with the normalisation
    statistics of the training set (--component). Outputs as apply_to_realisations, tagged from-<component>.
    Returns the combined npz path.
    """
    if args.eval_realisations is None:
        raise ValueError("--eval-component needs --eval-realisations FIRST LAST.")
    first, last = (int(x) for x in args.eval_realisations)
    if last < first:
        raise ValueError(f"--eval-realisations: LAST ({last}) is below FIRST ({first}).")
    # the file templates of the evaluation mixture (per-frequency maps, ilc_synth, cached model inputs) on both
    # template sets Inference reads from
    out_dirs = inference.file_templates.output_directories
    for templates in (inference.file_templates.file_templates, inference.data_handler.file_templates):
        register_pixel_ps_component_template(templates, out_dirs, component)
        register_mixture_component_template(templates, out_dirs, component)
    print(f"[Apply] Applying the model trained on '{args.component}' to the simulated realisations r{first:04d}-r{last:04d} "
          f"of '{component}' (normalisation statistics of '{args.component}')...")
    return apply_to_realisations(args, inference, component, list(range(first, last + 1)))


def apply_to_realisations(args, inference: Inference, component: str, realisations: list) -> str:
    """
    Run the loaded model on the given realisations of `component` (the training mixture for the test split, another
    mixture for --eval-component, "real" for the observed sky). Per realisation Inference.predict_and_visualise_real_sky
    saves, under <directory>/ML/cmb_prediction/<run_id>/ilc_ml_maps/checkpoint_<epoch>/ with the from-<component> tag,
    the cleaned CMB map (MW .npy) and the ILC / ILC + ML / difference figure. The alm of all realisations are then
    saved next to them (<stem> = the map name with r<FIRST>-<LAST>; it carries ml-masked-mask<KK> with --mask-index):
      <stem>_alm_fullsky.npz : always. alm_processed_cmb, alm_ilc, alm_ilc_ml of shape (n_realisations, n_alm),
                               complex128, K, healpy ordering (no alm_processed_cmb for the observed sky);
                               hp.alm2cl(alm[, alm_y]) is the full-sky auto / cross spectrum
      <stem>_alm_pseudo.npz  : masked run only. The same arrays for (map x training mask), plus apodisation_deg,
                               f_sky, f_sky2 of that mask; hp.alm2cl is the pseudo-C_ell (to decouple with NaMaster)
    Both hold realisations, component, stats_component, checkpoint and lmax, and for a masked run mask_index
    (0 = Planck 2018 common mask). Returns the path of the full-sky npz.
    """
    realisations = [int(r) for r in realisations]
    first, last = min(realisations), max(realisations)
    alms = {"fullsky": {}, "pseudo": {}}  # kind -> map name -> list of alm, one per realisation
    for realisation in realisations:
        out = inference.predict_and_visualise_real_sky(realisation=realisation, component=component)
        for kind, key in (("fullsky", "alm"), ("pseudo", "alm_pseudo")):
            for name, alm in out[key].items():
                alms[kind].setdefault(name, []).append(alm)

    stem = os.path.splitext(out["save_path"])[0]
    if first != last:
        stem = stem.replace(f"_r{realisations[-1]:04d}_", f"_r{first:04d}-{last:04d}_", 1)
    info = {"realisations": np.asarray(realisations, dtype=int), "component": component, "stats_component": args.component,
            "checkpoint": int(inference.loaded_checkpoint_epoch), "lmax": int(args.lmax)}
    mask_info = out["mask_info"]  # empty for an unmasked run
    for kind, extra in (("fullsky", {k: v for k, v in mask_info.items() if k == "mask_index"}), ("pseudo", mask_info)):
        if alms[kind]:
            path = f"{stem}_alm_{kind}.npz"
            np.savez(path, **info, **extra,
                     **{f"alm_{name}": np.asarray(v, dtype=np.complex128) for name, v in alms[kind].items()})
            print(f"[Alm] wrote the {kind} alm of {len(realisations)} realisation(s) of '{component}': {path}")
    return f"{stem}_alm_fullsky.npz"


def resolve_checkpoint_dir(args, model_dir: str) -> str:
    ckpt_dir, _, _ = resolve_checkpoint_target(model_dir, epoch=args.checkpoint_epoch)
    return str(ckpt_dir)


def main():
    start_time = time.perf_counter()
    jax.config.update("jax_enable_x64", False)
    print(f"JAX 64-bit mode: {jax.config.jax_enable_x64}")

    args = parse_args()
    steps = parse_mode(args.mode)
    if args.resume_training and "train" not in steps:
        raise ValueError(
            "--resume-training is only valid when --mode includes 'train'."
        )
    if "prepare" in steps and not args.run_id.strip():
        raise ValueError(
            "prepare mode requires --run-id: the normalisation statistics are saved in ML/models/<run_id>/."
        )
    if args.mask_index is not None:  # one of the masks_fsky_078_to_099 masks: masked ML, run id tagged with the mask
        if not args.run_id.strip():
            raise ValueError("--mask-index requires --run-id (the suffix _mask<KK> is appended to it).")
        args.masked = True
        args.run_id = f"{args.run_id.strip()}_mask{args.mask_index:02d}"
        print(f"[Mask] --mask-index {args.mask_index}: masked ML with mask {args.mask_index:02d}, run id {args.run_id}")
    needs_model = [s for s in ("evaluate", "apply") if s in steps]
    if needs_model and "train" not in steps and not args.run_id.strip() and not args.model_dir.strip():
        raise ValueError(
            f"{'/'.join(needs_model)} mode requires either --run-id or --model-dir."
        )

    model_dir_from_train = None

    if "prepare" in steps:
        print("[Prepare] Starting data preparation...")
        step_prepare_data(args)
        print("[Prepare] Done.")

    if "train" in steps:
        print("[Train] Starting training...")
        model_dir_from_train = step_train(args)
        print("[Train] Done.")

    if "evaluate" in steps:
        print("[Evaluate] Starting evaluation...")
        if model_dir_from_train is None:
            model_dir_from_train = resolve_evaluate_target(args)
            print(f"Model directory: {model_dir_from_train}")
        ckpt_dir = resolve_checkpoint_dir(args, model_dir_from_train)
        print(f'Loaded model from: {ckpt_dir}')
        step_evaluate(args, ckpt_dir=ckpt_dir)
        print("[Evaluate] Done.")

    if "apply" in steps:
        print("[Apply] Applying the trained model to "
              + (f"the simulated maps of {', '.join(args.eval_component)}..." if args.eval_component else "the observed Planck sky..."))
        if model_dir_from_train is None:
            model_dir_from_train = resolve_evaluate_target(args)
            print(f"Model directory: {model_dir_from_train}")
        ckpt_dir = resolve_checkpoint_dir(args, model_dir_from_train)
        print(f'Loaded model from: {ckpt_dir}')
        step_apply(args, ckpt_dir=ckpt_dir)
        print("[Apply] Done.")

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"[Time] Total pipeline time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()


''' example usage

# Steps (join with '+'):
#   prepare  = data-preparation step: generate model input/target maps from the SILC outputs and fit the normalisation
#              statistics into ML/models/<run-id>/ (CPU; needs --run-id)
#   train    = training step (GPU): reuses prepared maps/stats if they exist, otherwise generates them first
#   evaluate = predict the held-out test split and save the alm of its maps
#   apply    = apply the trained model to the observed Planck sky (SILC run with --components real --wavelet-components real)
# --deproject tsz reads the cILC SILC outputs (tagged cilc-dp-tsz) instead of the plain ILC ones (tagged ilc); the ML
#   maps, statistics and predictions carry the same tag, so ILC and cILC products never overwrite each other.
# Cached ML maps older than their SILC inputs (after a SILC --overwrite) are regenerated automatically; statistics of an
#   existing run that became stale raise an error on resume/evaluate/apply (train a new --run-id instead).
# CPU-only preparation (e.g. while the GPU is busy): JAX_PLATFORMS=cpu python3 -m skyclean.ml.pipeline_ml --mode prepare --run-id <run-id> ...

python3 -m skyclean.ml.pipeline_ml \
  --mode prepare+train \
  --extract-comp cmb \
  --component cfn \
  --frequencies 030 044 070 100 143 217 353 545 857 \
  --chs 1 16 16 32 64 \
  --realisations 20 \
  --lmax 511 \
  --N-directions 4 \
  --lam 2.0 \
  --nsamp 1200 \
  --batch-size 2 \
  --epochs 100 \
  --split 0.8 0.1 0.1 \
  --learning-rate 1e-3 \
  --momentum 0.90 \
  --eval-every 1 \
  --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ \
  --seed 42 \
  --early-stopping-min-delta 1e-4 \
  --run-id CFN_lmax511_N4_r20

  --resume-training \
  --masked            # masked ML: training loss restricted to the Planck 2018 common mask; outputs tagged ml-masked
  --mask-index 7      # masked ML with mask 07 of <directory>/masks_fsky_078_to_099 (1 = fsky 0.78 ... 30 = fsky 0.99) instead;
                      # implies --masked, run id becomes <run-id>_mask07, outputs tagged ml-masked-mask07

# Apply a trained run to the real Planck sky:
python3 -m skyclean.ml.pipeline_ml \
  --mode apply \
  --run-id CFN_lmax511_N4_r300 \
  --component cfn \
  --frequencies 030 044 070 100 143 217 353 545 857 \
  --chs 1 16 16 32 64 \
  --realisations 300 \
  --lmax 511 \
  --N-directions 4 \
  --lam 2.0 \
  --nsamp 1200 \
  --batch-size 2 \
  --epochs 100 \
  --split 0.8 0.1 0.1 \
  --learning-rate 1e-3 \
  --momentum 0.90 \
  --directory /Scratch/cindy/testing/Skyclean/skyclean/data/

  --masked            # the run was trained masked (or --mask-index <K> for a run trained with that mask)

# Apply a run trained on one foreground mixture to the simulated SILC maps of another mixture (same lmax/N/lam/nsamp;
# realisations FIRST..LAST inclusive, normalisation statistics of --component), e.g. trained on the PySM mixture
# cfn-a1-d1-f1-s1 (r0-199), applied to the FFP10 subset cfn-dust-ff-spindust-sync on r200-299 (the CMB / noise
# realisations are shared by the mixtures, so the apply range must not overlap the training range):
python3 -m skyclean.ml.pipeline_ml \
  --mode apply \
  --run-id CFN-a1-d1-f1-s1_ilc_lmax511_N2_r200_masked \
  --component cfn-a1-d1-f1-s1 \
  --eval-component cfn-dust-ff-spindust-sync \
  --eval-realisations 200 299 \
  --masked \
  ... (the training arguments: --frequencies, --realisations 200, --split, --lmax, --N-directions, --chs, --epochs, ...)
# Outputs per realisation as for the observed sky, tagged from-<eval component>_..._r<NNNN>_..., plus
#   ..._r0200-0299_..._ckpt<epoch>_alm_fullsky.npz   full-sky alm of processed cmb / ILC / ILC + ML for every realisation
#   ..._r0200-0299_..._ckpt<epoch>_alm_pseudo.npz    masked run: their pseudo-alm inside the training mask (2 deg apodised)
# 'evaluate' writes the same files for the test split of --component (tagged from-<component>, r<first>-<last> of the split).

# Mixed training set (option 3): a '+'-joined --component trains on the first half of --realisations from the first
# mixture and the second half from the second (r0-99 FFP10 subset, r100-199 PySM for 200), --split applied inside each
# half (80 / 10 / 10 of each). evaluate writes the alm npz per mixture (test splits r0090-0099 and r0190-0199);
# apply takes several --eval-component values and runs them one after the other (here both mixtures on r200-299):
python3 -m skyclean.ml.pipeline_ml \
  --mode prepare+train+evaluate+apply \
  --run-id CFN-dust-ff-spindust-sync+CFN-a1-d1-f1-s1_ilc_lmax511_N2_r200_masked \
  --component cfn-dust-ff-spindust-sync+cfn-a1-d1-f1-s1 \
  --eval-component cfn-dust-ff-spindust-sync cfn-a1-d1-f1-s1 \
  --eval-realisations 200 299 \
  --realisations 200 --split 0.8 0.1 0.1 \
  --masked \
  ... (the other training arguments as above)

# --checkpoint-epoch = optional; default: latest checkpoint in the run folder
# The SILC real run is assumed to use --start-realisation 0 (its outputs are tagged r0000).
# Outputs: <directory>/ML/cmb_prediction/<run-id>/ilc_ml_maps/checkpoint_<epoch>/
#   <mode>_ml[-masked[-mask<KK>]]_cmb_from-real_f..._ckpt<epoch>.npy   cleaned CMB map (MW sampling)
#   ..._ckpt<epoch>_maps.png                                ILC / ILC + ML / difference mollviews
#   ..._ckpt<epoch>_alm_fullsky.npz                         full-sky alm of the ILC and ILC + ML maps (healpy ordering, complex128, K)
#   ..._ckpt<epoch>_alm_pseudo.npz                          masked run: alm of (map x training mask), with apodisation_deg, f_sky, f_sky2
'''
