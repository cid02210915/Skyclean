import argparse
import os
import time
import numpy as np
import jax
import subprocess, sys

from .download import DownloadData
from .map_processing import ProcessMaps
from .file_templates import FileTemplates
from .ilc import ProduceSILC  
from .power_spec import MapAlmConverter, PowerSpectrumTT
from .mixing_matrix_constraint import SpectralVector


class Pipeline:
    """Run the SILC pipeline in discrete steps: download → process → wavelets → ilc."""
    def __init__(
        self,
        components: list,
        wavelet_components: list,
        ilc_components: list,
        frequencies: list,
        realisations: int,
        start_realisation: int = 0,
        lmax: int = 1024,
        N_directions: int = 1,
        lam: float = 2.0,
        method: str = "jax_cuda",
        visualise: bool = False,
        save_ilc_intermediates: bool = True,
        overwrite: bool = False,
        directory: str = "data/",
        constraint: bool = False,
        F = None,
        reference_vectors = None,
        scales: list | None = None,   # optional: let caller pin j-scales
    ):
        self.components = components
        self.wavelet_components = wavelet_components
        self.ilc_components = ilc_components
        self.frequencies = frequencies
        self.realisations = realisations
        self.start_realisation = start_realisation
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.method = method
        self.visualise = visualise
        self.save_ilc_intermediates = save_ilc_intermediates
        self.overwrite = overwrite
        self.directory = directory
        self.constraint = constraint
        self.F = F
        self.reference_vectors = reference_vectors
        self.scales = scales
        self.lam_str = f"{lam:.1f}" 

    # -------------------------
    # Steps
    # -------------------------
    def step_download(self):
        print(f"--- STARTING DATA DOWNLOAD ---")
        for d in jax.devices():
            ms = d.memory_stats()
            print(f"Device {d.id}:",
                  f"bytes_in_use={ms['bytes_in_use']}",
                  f"peak_bytes_in_use={ms['peak_bytes_in_use']}",
                  f"bytes_limit={ms['bytes_limit']}",
                  f"largest_free_chunk={ms.get('largest_free_chunk', 'n/a')}",
                  f"num_allocs={ms.get('num_allocs', 'n/a')}")
            
        downloader = DownloadData(
            self.components,
            self.frequencies,
            self.realisations,
            start_realisation=self.start_realisation,
            directory=self.directory,
        )
        downloader.download_all()

    def step_process(self):
        print("--- PROCESSING CFNs AND TOTAL MAP CFN ---")
        for d in jax.devices():
           ms = d.memory_stats()
           print(f"Device {d.id}:",
                 f"bytes_in_use={ms['bytes_in_use']}",
                 f"peak_bytes_in_use={ms['peak_bytes_in_use']}",
                 f"bytes_limit={ms['bytes_limit']}",
                 f"largest_free_chunk={ms.get('largest_free_chunk', 'n/a')}",
                 f"num_allocs={ms.get('num_allocs', 'n/a')}")
        processor = ProcessMaps(
            self.components,
            self.wavelet_components,
            self.frequencies,
            self.realisations,
            start_realisation=self.start_realisation,
            desired_lmax=self.lmax,
            directory=self.directory,
            method=self.method,
            overwrite=self.overwrite,
        )
        processor.produce_and_save_all_maps()

    def step_wavelets(self):
        print("--- PRODUCING WAVELET TRANSFORMS ---")
        for d in jax.devices():
            ms = d.memory_stats()
            print(f"Device {d.id}:",
                  f"bytes_in_use={ms['bytes_in_use']}",
                  f"peak_bytes_in_use={ms['peak_bytes_in_use']}",
                  f"bytes_limit={ms['bytes_limit']}",
                  f"largest_free_chunk={ms.get('largest_free_chunk', 'n/a')}",
                  f"num_allocs={ms.get('num_allocs', 'n/a')}")
        processor = ProcessMaps(
            self.components,
            self.wavelet_components,
            self.frequencies,
            self.realisations,
            start_realisation=self.start_realisation,
            desired_lmax=self.lmax,
            directory=self.directory,
            method=self.method,
            overwrite=self.overwrite,
        )
        processor.produce_and_save_wavelet_transforms(
            self.N_directions,
            self.lam,
            method=self.method,
            visualise=self.visualise,
        )

    # ---------- helper to infer available j-scales from disk ----------
    def _infer_scales_from_disk(self, file_template: str, comp_on_disk: str,
                                frequency: str, realisation: int) -> list[int]:
        """
        Probe j=0.63 and collect those that exist on disk for the given (comp, freq, realisation).
        Assumes the wavelet template uses {comp} and British {realisation:05d}.
        Stops after the first gap once at least one scale is found.
        """
        scales = []
        found_any = False

        real_brit = f"{realisation:05d}"
        for j in range(64):
            try:
                probe = file_template.format(
                    comp=comp_on_disk,
                    frequency=frequency,
                    scale=j,
                    realisation=int(real_brit),  # works with :05d
                    lmax=self.lmax,
                    lam=self.lam,
                )
            except KeyError:
                # If template ever switches to US spelling (unlikely here)
                probe = file_template.format(
                    comp=comp_on_disk,
                    frequency=frequency,
                    scale=j,
                    realisation=real_brit,
                    lmax=self.lmax,
                    lam=self.lam,
                )
            if os.path.exists(probe):
                scales.append(j)
                found_any = True
            elif found_any:
                break
        if not scales:
            raise FileNotFoundError(
                "Could not infer any wavelet scales from disk.\n"
                f"Checked with comp='{comp_on_disk}', frequency='{frequency}', realisation={realisation}.\n"
                f"Template example:\n{file_template}"
            )
        return scales

    def step_ilc(self):
        """Run ILC_wav_coeff_maps_MP with optional theory/empirical F."""
        print("--- RUNNING ILC (new functional API) ---")
        for d in jax.devices():
            ms = d.memory_stats()
            print(f"Device {d.id}:",
                  f"bytes_in_use={ms['bytes_in_use']}",
                  f"peak_bytes_in_use={ms['peak_bytes_in_use']}",
                  f"bytes_limit={ms['bytes_limit']}",
                  f"largest_free_chunk={ms.get('largest_free_chunk', 'n/a')}",
                  f"num_allocs={ms.get('num_allocs', 'n/a')}")
            
        ft = FileTemplates(self.directory).file_templates
    
        # Templates
        file_template = ft.get("wavelet_coeffs") or ft.get("wavelet_c_j")
        if file_template is None:
            raise KeyError("Missing wavelet template: expected 'wavelet_coeffs' or 'wavelet_c_j'.")
        output_templates = {
            "doubled_maps":           ft["doubled_maps"],
            "covariance_matrices":    ft["covariance_matrices"],
            "weight_vector_matrices": ft["weight_vector_matrices"],
            "ilc_maps":               ft["ilc_maps"],
            "trimmed_maps":           ft["trimmed_maps"],
            "ilc_synth":              ft["ilc_synth"],
            "ilc_spectrum":           ft.get("ilc_spectrum"),
            "scaling_coeffs":         ft["scaling_coeffs"],
        }
    
        # Realisations, freqs
        realisations = list(range(self.start_realisation, self.start_realisation + self.realisations))
        freqs = list(self.frequencies)
    
        # Input mixture and scales
        comp_in = self.wavelet_components[0]
        if getattr(self, "scales", None):
            scales = list(self.scales)
        else:
            first_real, first_freq = realisations[0], freqs[0]
            scales = self._infer_scales_from_disk(file_template, comp_in, first_freq, first_real)
    
        # Constraint inputs (build F/ref on demand; default empirical)
        do_constraint = getattr(self, "constraint", False)
        F = getattr(self, "F", None)
        reference_vectors = getattr(self, "reference_vectors", None)

        if do_constraint and (F is None or reference_vectors is None):
            source = getattr(self, "F_source", "theory")
            kwargs = dict(getattr(self, "F_kwargs", {}))
            freq_arg = kwargs.pop("frequencies", freqs)
            
            # keep only relevant keys for the chosen source
            if source == "theory":
                kwargs = {k: v for k, v in kwargs.items() if k in ("beta_s", "nu0")}
            else:  # empirical
                kwargs = {k: v for k, v in kwargs.items()
                          if k in ("base_dir", "file_templates", "realization", "mask_path", "normalize")}
            
            F_new, F_cols, ref_vecs, _ = SpectralVector.get_F(source, frequencies=freq_arg, **kwargs)

            self.F = F_new
            self.reference_vectors = ref_vecs
            F = self.F
            reference_vectors = self.reference_vectors
            
        # Run ILC for requested targets
        for extract_comp in self.ilc_components:
            print(f"--- ILC target='{extract_comp}'  input='{comp_in}'  lmax={self.lmax}  scales={scales} ---")
            _ = ProduceSILC.ILC_wav_coeff_maps_MP(
                file_template=file_template,
                frequencies=freqs,
                scales=scales,
                realisations=realisations,
                output_templates=output_templates,
                L_max=self.lmax + 1,
                N_directions=self.N_directions,
                comp=comp_in,
                constraint=do_constraint,
                F=F,
                extract_comp=extract_comp,
                reference_vectors=reference_vectors,
            )

    def step_power_spec(
        self,
        unit: str = "K",
        save_path: str | None = None,
        *,
        source: str = "auto",                  # "auto" | "ilc_synth" | "processed" | "downloaded"
        component: str | None = None,          # map component ("cmb","sync", "dust","noise","tsz") or "cfn" for ilc_synth {component}
        extract_comp: str | None = None,       # ilc_synth target (e.g. "cmb")
        frequencies: list[str] | None = None,  # ilc_synth band-set
        frequency: str | int | None = None,    # single band for processed/downloaded
        realisation: int | None = None,        # defaults to self.start_realisation
        lmax: int | None = None,               # defaults to self.lmax
        lam: str | float | int | None = None,  # defaults to self.lam_str
        field: int = 0
    ):
        """
        Compute TT C_ell (and plot D_ell). Returns (ell, cl).
        unit: "K" (input C_ell in K^2) or "uK"/"µK" (input already µK^2). Plot is in µK^2.
        """
       
        # defaults from pipeline
        r     = self.start_realisation if realisation is None else int(realisation)
        lmax_ = self.lmax if lmax is None else int(lmax)
        lam_  = self.lam_str if lam is None else (lam if isinstance(lam, str) else f"{float(lam):.1f}")
        freqs = self.frequencies if frequencies is None else list(frequencies)

        # choose source automatically (preserves your old behavior)
        if source == "auto":
            source = "ilc_synth" if self.ilc_components else "downloaded"

        # templates + converter
        ft = FileTemplates(self.directory).file_templates
        conv = MapAlmConverter(ft)

        # ---- load selected map ----
        if source == "ilc_synth":
            comp_in = component or (self.wavelet_components[0] if self.wavelet_components else "cfn")  # template {component}
            tgt     = extract_comp or (self.ilc_components[0] if self.ilc_components else "cmb")       # template {extract_comp}
            out = conv.to_alm(
                component=comp_in, source="ilc_synth",
                extract_comp=tgt, frequencies=freqs,
                realisation=r, lmax=lmax_, lam=lam_,
            )
            src = "ilc_synth"; label = f"ILC-synth ({tgt})"

        elif source == "processed":
            comp_use = component or "cmb"
            out = conv.to_alm(
                component=comp_use, source="processed",
                frequency=frequency, realisation=r, lmax=lmax_,
            )
            src = "processed"; label = f"Processed {comp_use}"

        elif source == "downloaded":
            comp_use = component or "cmb"
            out = conv.to_alm(
                component=comp_use, source="downloaded",
                frequency=frequency, realisation=r, lmax=lmax_, field=field,
            )
            src = "downloaded"; label = f"Downloaded {comp_use}"

        else:
            raise ValueError("source must be one of: 'auto', 'ilc_synth', 'processed', 'downloaded'.")

        # ---- alm -> C_ell ----
        if out["format"] == "mw":
            ell, cl = PowerSpectrumTT.from_mw_alm(np.asarray(out["alm"]))
        else:
            ell, cl = PowerSpectrumTT.from_healpy_alm(out["alm"])

        # ---- C_ell -> D_ell (plot in µK^2) ----
        Dl = PowerSpectrumTT.cl_to_Dl(ell, cl, input_unit=unit)  # "K" or "uK"

        # pick a simple style by source (works with tuple-plotter or dict-plotter)
        style = {"ilc_synth": "-", "processed": "--", "downloaded": "-."}.get(src, "-")

        try:
            # if your plotter accepts dicts
            PowerSpectrumTT.plot_Dl_series({"ell": ell, "Dl": Dl, "label": label, "source": src},
                                           save_path=save_path, show=True)
        except Exception:
            # fallback if your plotter only accepts tuples
            PowerSpectrumTT.plot_Dl_series([(ell, Dl, label, style)], save_path=save_path, show=True)

        return ell, cl

        
    def run(self, steps=None):
        """
        steps: list of steps to run, any of {'download','process','wavelets','ilc','all'}.
               If 'all' is included, runs the full pipeline.
        """
        if not steps:
            print("No steps selected. Use --steps with any of: download process wavelets ilc all")
            return

        if "all" in steps:
            steps = ["download", "process", "wavelets", "ilc"]

        start_time = time.perf_counter()
        print(f"=== RUN for lam={self.lam} ===")

        if "download" in steps:
            self.step_download()

        if "process" in steps:
            self.step_process()

        if "wavelets" in steps:
            self.step_wavelets()

        if "ilc" in steps:
            self.step_ilc()

        if "power_spec" in steps:
            self.step_power_spec()

        elapsed = time.perf_counter() - start_time
        print(f"SELECTED STEPS COMPLETED IN {elapsed:.2f} SECONDS (lam={self.lam}).")

def _spawn_gpu_run(gpu_id: int, start_real: int, n_real: int, base_argv: list[str]) -> subprocess.Popen:
    env = os.environ.copy()

    # isolate the GPU for this child and make x64 deterministic
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["JAX_ENABLE_X64"] = "True"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3" 

    # argv for the child run
    argv = [
        sys.executable,
        sys.argv[0],
        "--gpu", "0",                         # child's visible GPU is now device 0
        "--start-realisation", str(start_real),
        "--realisations", str(n_real),
    ] + base_argv

    return subprocess.Popen(argv, env=env)

# when not using multi-processing 
def main():
    parser = argparse.ArgumentParser(
        description="Run the SILC pipeline with configurable parameters and GPU selection."
    )
    parser.add_argument('--components', nargs='+', default=["cmb", "sync", "dust", "noise", 'tsz'])
    parser.add_argument('--wavelet-components', nargs='+', default=["cfn"])
    parser.add_argument('--ilc-components', nargs='+', default=["cmb"])
    parser.add_argument('--frequencies', nargs='+',
                        default=["030", "044", "070", "100", "143", "217", "353", "545", "857"])
    parser.add_argument('--realisations', type=int, default=1)
    parser.add_argument('--start-realisation', type=int, default=0)
    parser.add_argument('--lmax', type=int, default=512)
    parser.add_argument('--N-directions', type=int, default=1)
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--method', type=str, default='jax_cuda')
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--save-ilc-intermediates', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--directory', type=str, default='/Scratch/agnes/data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', nargs='+',
                        choices=['download', 'process', 'wavelets', 'ilc', 'all'],
                        help="Which steps to run (one or more). Examples: --steps download  or  --steps process wavelets  or  --steps all")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Using GPU {args.gpu} for computation.")

    pipeline = Pipeline(
        components=args.components,
        wavelet_components=args.wavelet_components,
        ilc_components=args.ilc_components,
        frequencies=args.frequencies,
        realisations=args.realisations,
        start_realisation=args.start_realisation,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        method=args.method,
        visualise=args.visualise,
        save_ilc_intermediates=args.save_ilc_intermediates,
        overwrite=args.overwrite,
        directory=args.directory,
    )
    pipeline.run(steps=args.steps)

'''
def main():
    parser = argparse.ArgumentParser(
        description="Run the SILC pipeline with configurable parameters and GPU selection."
    )
    parser.add_argument('--components', nargs='+', default=["cmb", "sync", "dust", "noise", 'tsz'])
    parser.add_argument('--wavelet-components', nargs='+', default=["cfn"])
    parser.add_argument('--ilc-components', nargs='+', default=["cmb"])
    parser.add_argument('--frequencies', nargs='+',
                        default=["030", "044", "070", "100", "143", "217", "353", "545", "857"])
    parser.add_argument('--realisations', type=int, default=1)
    parser.add_argument('--start-realisation', type=int, default=0)
    parser.add_argument('--lmax', type=int, default=512)
    parser.add_argument('--N-directions', type=int, default=1)
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--method', type=str, default='jax_cuda')
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--save-ilc-intermediates', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--directory', type=str, default='/Scratch/agnes/data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', nargs='+', type=int, help="Optional: list of GPU ids to use, e.g. --gpus 0 1")
    parser.add_argument('--steps', nargs='+',
                        choices=['download', 'process', 'wavelets', 'ilc', 'all'],
                        help="Which steps to run (one or more). Examples: --steps download  or  --steps process wavelets  or  --steps all")

    args = parser.parse_args()

    # -------- two-GPU launcher --------
    if args.gpus and len(args.gpus) > 1:
        total = int(args.realisations)
        start = int(args.start_realisation)
        if total < 2:
            print("Two-GPU run requested but realisations < 2; falling back to single-GPU.")
        else:
            # split realisations roughly in half between the first two GPUs provided
            half = total // 2
            r0 = half
            r1 = total - half
            g0, g1 = args.gpus[0], args.gpus[1]

            # reconstruct base argv for the child runs from current args
            base_argv = []
            def _extend(flag, val):
                if val is None: return
                if isinstance(val, bool):
                    if val: base_argv.extend([flag])
                elif isinstance(val, (list, tuple)):
                    base_argv.extend([flag] + list(map(str, val)))
                else:
                    base_argv.extend([flag, str(val)])

            _extend("--components", args.components)
            _extend("--wavelet-components", args.wavelet_components)
            _extend("--ilc-components", args.ilc_components)
            _extend("--frequencies", args.frequencies)
            _extend("--lmax", args.lmax)
            _extend("--N-directions", args.N_directions)
            _extend("--lam", args.lam)
            _extend("--method", args.method)
            if args.visualise: base_argv.append("--visualise")
            if args.save_ilc_intermediates: base_argv.append("--save-ilc-intermediates")
            if args.overwrite: base_argv.append("--overwrite")
            _extend("--directory", args.directory)
            if args.steps: _extend("--steps", args.steps)

            print(f"[launcher] Spawning GPU {g0} for realisations {start}..{start+r0-1} "
                  f"and GPU {g1} for {start+r0}..{start+total-1}")

            p0 = _spawn_gpu_run(gpu_id=g0, start_real=start,     n_real=r0, base_argv=base_argv)
            p1 = _spawn_gpu_run(gpu_id=g1, start_real=start+r0,  n_real=r1, base_argv=base_argv)

            rc0 = p0.wait()
            rc1 = p1.wait()
            sys.exit(0 if (rc0 == 0 and rc1 == 0) else 1)

    pipeline = Pipeline(
        components=args.components,
        wavelet_components=args.wavelet_components,
        ilc_components=args.ilc_components,
        frequencies=args.frequencies,
        realisations=args.realisations,
        start_realisation=args.start_realisation,
        lmax=args.lmax,
        N_directions=args.N_directions,
        lam=args.lam,
        method=args.method,
        visualise=args.visualise,
        save_ilc_intermediates=args.save_ilc_intermediates,
        overwrite=args.overwrite,
        directory=args.directory,
        # pass-throughs unchanged
    )
    pipeline.run(steps=args.steps)
'''

# Example usage:
#   python -m skyclean.silc.pipeline --gpu 0 --steps wavelets \
#     --directory /Scratch/agnes/data --wavelet-components cfn \
#     --frequencies 030 044 070 100 143 217 353 545 857 \
#     --realisations 1 --start-realisation 0 --lmax 512 --N-directions 1 --lam 2.0
#
#   python -m skyclean.silc.pipeline --gpu 0 --steps ilc \
#     --directory /Scratch/agnes/data --ilc-components cfn \
#     --frequencies 030 044 070 100 143 217 353 545 857 \
#     --realisations 1 --start-realisation 0 --lmax 512 --lam 2.0
#
#   python -m skyclean.silc.pipeline --gpu 0 --steps process wavelets ilc \
#     --directory /Scratch/agnes/data --components cmb sync dust noise tsz \
#     --wavelet-components cfn --ilc-components cfn \
#     --frequencies 100 143 217 --realisations 1 --start-realisation 0 \
#     --lmax 512 --N-directions 1 --lam 2.0 --method jax_cuda

if __name__ == '__main__':
    main()
