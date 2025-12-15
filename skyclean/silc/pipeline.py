import argparse
import os
import time
import numpy as np
import jax
import subprocess, sys
import s2wav.filters as filters

from .download import DownloadData
from .map_processing import ProcessMaps
from .file_templates import FileTemplates
from .ilc import ProduceSILC  
from .power_spec import MapAlmConverter, PowerSpectrumTT, PowerSpectrumCrossTT
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
        nsamp: float = 1200.0, 
        #scales: list | None = None,   # optional: let caller pin j-scales
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
        #self.scales = scales
        self.lam_str = f"{lam:.1f}" 
        self.nsamp = nsamp

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
        processor.produce_and_save_maps()

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
            "f_scal":                 ft["f_scal"], 
        }
    
        # Realisations, freqs
        realisations = list(range(self.start_realisation, self.start_realisation + self.realisations))
        freqs = list(self.frequencies)
    
        # Input mixture and scales
        comp_in = self.wavelet_components[0]

        if getattr(self, "scales", None) is not None:
            scales = list(self.scales)
        else:
            # derive number of wavelet bands from the filter bank (no disk probing)
            L = self.lmax + 1
            filt = filters.filters_directional_vectorised(L, self.N_directions, lam=self.lam)
            J = len(filt[0])              # number of wavelet bands (excludes scaling)
            scales = list(range(J))       # use only wavelet bands for ILC
    
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
                # build_F_theory(beta_s, nu0, frequencies, components_order)
                kwargs = {k: v for k, v in kwargs.items() if k in ("beta_s", "nu0")}
            else:  # empirical
                # build_F_empirical(base_dir, file_templates, frequencies, realization, mask_path, components_order)
                if "realisation" in kwargs and "realization" not in kwargs:
                    kwargs["realization"] = kwargs.pop("realisation")
                kwargs = {k: v for k, v in kwargs.items()
                          if k in ("base_dir", "file_templates", "realization", "mask_path")}

            # desired column order comes from your input component list; ignore extras like 'noise'
            components_order = [c.lower() for c in self.components if c.lower() in ("cmb", "tsz", "sync", "cib", "dust")]

            # build F with explicit column order 
            F_new, F_cols, ref_vecs, _ = SpectralVector.get_F(
                source=source,
                frequencies=freq_arg,
                components_order=components_order,
                **kwargs
            )

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
                lam=self.lam, 
                N_directions=self.N_directions,
                comp=comp_in,
                constraint=do_constraint,
                F=F,
                extract_comp=extract_comp,
                reference_vectors=reference_vectors,
                nsamp=self.nsamp, 
                overwrite=self.overwrite,
            )

    def step_power_spec(
        self,
        unit: str = "K",
        save_path: str | None = None,
        *,
        source: str = "auto",                  # "auto" | "ilc_synth" | "processed" | "downloaded"
        component: str | None = None,          # map component ("cmb","sync","dust","noise","tsz") or "cfn"
        extract_comp: str | None = None,       # ilc_synth target (e.g. "cmb")
        frequencies: list[str] | None = None,  # ilc_synth band-set
        frequency: str | int | None = None,    # single band for processed/downloaded
        realisation: int | None = None,        # defaults to self.start_realisation
        lmax: int | None = None,               # defaults to self.lmax
        lam: str | float | int | None = None,  # defaults to self.lam_str
        field: int = 0,
        nsamp: float | int | None = None,
        overwrite: bool | None = None,
    ):
        """
        Compute TT C_ell (and plot D_ell). Returns (ell, cl).
        unit: "K" (input C_ell in K^2) or "uK"/"µK" (input already µK^2). Plot is in µK^2.
        """
        # resolve overwrite
        overwrite = self.overwrite if overwrite is None else overwrite

        # defaults from pipeline
        r      = self.start_realisation if realisation is None else int(realisation)
        lmax_  = self.lmax if lmax is None else int(lmax)
        lam_   = self.lam_str if lam is None else (lam if isinstance(lam, str) else f"{float(lam):.1f}")
        freqs  = self.frequencies if frequencies is None else list(frequencies)
        nsamp_ = getattr(self, "nsamp", None) if nsamp is None else nsamp

        # --- templates + processed-CFN detection ---
        ft = FileTemplates(self.directory).file_templates
        has_processed_cfn = ("processed_cfn" in ft) or ("cfn" in ft)

        # --- choose source automatically (prefer processed CFN if present) ---
        if source == "auto":
            if has_processed_cfn:
                source = "processed"
            elif self.ilc_components:
                source = "ilc_synth"
            else:
                source = "downloaded"

        # --- make a local copy of templates and alias processed_cfn -> cfn if needed ---
        ft_local = dict(ft)
        if ("processed_cfn" not in ft_local) and ("cfn" in ft_local):
            ft_local["processed_cfn"] = ft_local["cfn"]

        conv = MapAlmConverter(ft_local)

        # ---- load selected map ----
        if source == "ilc_synth":
            comp_in = component or (self.wavelet_components[0] if self.wavelet_components else "cfn")
            tgt     = extract_comp or (self.ilc_components[0] if self.ilc_components else "cmb")
            out = conv.to_alm(
                component=comp_in, source="ilc_synth",
                extract_comp=tgt, frequencies=freqs,
                realisation=r, lmax=lmax_, lam=lam_,
                nsamp=nsamp_, constraint=self.constraint,
            )
            src = "ilc_synth"
            label = f"ILC-synth ({tgt})"

        elif source == "processed":
            if component is not None:
                comp_use = component
            else:
                comp_use = "cfn" if has_processed_cfn else "cmb"

            out = conv.to_alm(
                component=comp_use, source="processed",
                frequency=frequency, realisation=r, lmax=lmax_,
            )
            src = "processed"
            label = f"Processed {comp_use}"

        elif source == "downloaded":
            comp_use = component or "cmb"
            out = conv.to_alm(
                component=comp_use, source="downloaded",
                frequency=frequency, realisation=r, lmax=lmax_, field=field,
            )
            src = "downloaded"
            label = f"Downloaded {comp_use}"

        else:
            raise ValueError("source must be one of: 'auto', 'ilc_synth', 'processed', 'downloaded'.")

        # ---- alm -> C_ell ----
        if out["format"] == "mw":
            ell, cl = PowerSpectrumTT.from_mw_alm(np.asarray(out["alm"]))
        else:
            ell, cl = PowerSpectrumTT.from_healpy_alm(out["alm"])

        # ---- C_ell -> D_ell (plot in µK^2) ----
        Dl = PowerSpectrumTT.cl_to_Dl(ell, cl, input_unit=unit)

        # ---------- save spectrum arrays to .npy for ILC_synth ----------
        if source == "ilc_synth" and "ilc_spectrum" in ft:
            mode = "con" if getattr(self, "constraint", False) else "uncon"
            freq_tag = "_".join(freqs)

            spec_path = ft["ilc_spectrum"].format(
                mode=mode,
                extract_comp=tgt,
                component=comp_in,
                frequencies=freq_tag,
                realisation=r,
                lmax=lmax_,
                lam=lam_,
                nsamp=nsamp_,
            )
            os.makedirs(os.path.dirname(spec_path), exist_ok=True)
            if overwrite or not os.path.exists(spec_path):
                np.save(spec_path, {"ell": ell, "cl": cl, "Dl": Dl})

            if save_path is None:
                png_path = spec_path.replace(".npy", ".png")
            else:
                png_path = save_path
        else:
            png_path = save_path
        # ---------------------------------------------------------------------

        # pick a simple style
        style = {"ilc_synth": "-", "processed": "--", "downloaded": "-."}.get(src, "-")

        # respect overwrite for PNG
        if png_path is not None and (not overwrite) and os.path.exists(png_path):
            plot_save_path = None
        else:
            plot_save_path = png_path

        try:
            PowerSpectrumTT.plot_Dl_series(
                {"ell": ell, "Dl": Dl, "label": label, "source": src},
                save_path=plot_save_path, show=True
            )
        except Exception:
            PowerSpectrumTT.plot_Dl_series(
                [(ell, Dl, label, style)],
                save_path=plot_save_path, show=True
            )

        return ell, cl
   
        

    # --- step_cross_power_spec ---
    def step_cross_power_spec(
        self,
        unit: str = "K",
        save_path: str | None = None,
        *,
        # X side
        source_X: str = "auto",
        component_X: str | None = None,
        extract_comp_X: str | None = None,
        frequencies_X: list[str] | None = None,
        frequency_X: str | int | None = None,
        # Y side
        source_Y: str = "auto",
        component_Y: str | None = None,
        extract_comp_Y: str | None = None,
        frequencies_Y: list[str] | None = None,
        frequency_Y: str | int | None = None,
        # shared
        realisation: int | None = None,
        lmax: int | None = None,
        lam: str | float | int | None = None,
        field: int = 0,
        plot_r: bool = False, 
        overwrite: bool | None = None,  
    ):
        """
        Compute TT cross C_ell^{XY} (and plot D_ell^{XY}). Returns (ell, cl_xy).
        """
        # defaults (mirror step_power_spec)
        r     = self.start_realisation if realisation is None else int(realisation)
        lmax_ = self.lmax if lmax is None else int(lmax)
        lam_  = self.lam_str if lam is None else (lam if isinstance(lam, str) else f"{float(lam):.1f}")
        fX    = self.frequencies if frequencies_X is None else list(frequencies_X)
        fY    = self.frequencies if frequencies_Y is None else list(frequencies_Y)
        overwrite = self.overwrite if overwrite is None else overwrite

        # templates + processed-CFN detection (mirror)
        ft = FileTemplates(self.directory).file_templates
        has_processed_cfn = ("processed_cfn" in ft) or ("cfn" in ft)
        ft_local = dict(ft)
        if ("processed_cfn" not in ft_local) and ("cfn" in ft_local):
            ft_local["processed_cfn"] = ft_local["cfn"]
        conv = MapAlmConverter(ft_local)

        # auto source (mirror)
        def pick_source(s: str) -> str:
            if s != "auto":
                return s
            if has_processed_cfn:
                return "processed"
            elif self.ilc_components:
                return "ilc_synth"
            else:
                return "downloaded"

        # loader (single place)
        def load_one(source, component, extract_comp, frequencies, frequency):
            src = pick_source(source)
            if src == "ilc_synth":
                comp_in = component or (self.wavelet_components[0] if self.wavelet_components else "cfn")
                tgt     = extract_comp or (self.ilc_components[0] if self.ilc_components else "cmb")
                out = conv.to_alm(component=comp_in, source="ilc_synth",
                                   extract_comp=tgt, frequencies=frequencies,
                                   realisation=r, lmax=lmax_, lam=lam_)
                label = f"ILC-synth ({tgt})"; fmt = "mw" if out["format"] == "mw" else "hp"
            elif src == "processed":
                comp_use = component or ("cfn" if has_processed_cfn else "cmb")
                out = conv.to_alm(component=comp_use, source="processed",
                                   frequency=frequency, realisation=r, lmax=lmax_)
                label = f"Processed {comp_use}"; fmt = "mw" if out["format"] == "mw" else "hp"
            else:  # downloaded
                comp_use = component or "cmb"
                out = conv.to_alm(component=comp_use, source="downloaded",
                                   frequency=frequency, realisation=r, lmax=lmax_, field=field)
                label = f"Downloaded {comp_use}"; fmt = "mw" if out["format"] == "mw" else "hp"
            return out, label, fmt

        # load X and Y
        outX, labelX, fmtX = load_one(source_X, component_X, extract_comp_X, fX, frequency_X)
        outY, labelY, fmtY = load_one(source_Y, component_Y, extract_comp_Y, fY, frequency_Y)

        if fmtX != fmtY:
            raise ValueError("X and Y alm formats must match ('mw' vs 'healpy').")

        # alms -> C_ell^{XY}
        if outX["format"] == "mw":
            ell, cl_xy = PowerSpectrumCrossTT.from_mw_alm(np.asarray(outX["alm"]), np.asarray(outY["alm"]))
        else:
            ell, cl_xy = PowerSpectrumCrossTT.from_healpy_alm(outX["alm"], outY["alm"], lmax=lmax_)

        # C_ell -> D_ell^{XY} (µK^2) and plot
        Dl_xy = PowerSpectrumTT.cl_to_Dl(ell, cl_xy, input_unit=unit)
        style = "--"
        # respect overwrite for PNG
        if save_path is not None and (not overwrite) and os.path.exists(save_path):
            cross_save_path = None
        else:
            cross_save_path = save_path

        try:
            PowerSpectrumTT.plot_Dl_series(
                {"ell": ell, "Dl": Dl_xy,
                 "label": f"Cross: {labelX} × {labelY}",
                 "source": "processed"},
                save_path=cross_save_path, show=True
            )
        except Exception:
            PowerSpectrumTT.plot_Dl_series(
                [(ell, Dl_xy, f"Cross: {labelX} × {labelY}", style)],
                save_path=cross_save_path, show=True
            )
        if plot_r:
            if outX["format"] == "mw":
                _, cl_xx = PowerSpectrumTT.from_mw_alm(np.asarray(outX["alm"]))
                _, cl_yy = PowerSpectrumTT.from_mw_alm(np.asarray(outY["alm"]))
            else:
                _, cl_xx = PowerSpectrumTT.from_healpy_alm(outX["alm"])
                _, cl_yy = PowerSpectrumTT.from_healpy_alm(outY["alm"])

            PowerSpectrumCrossTT.plot_r_ell(
                ell, cl_xy, cl_xx, cl_yy,
                label=f"{labelX} × {labelY}"
            )

        return ell, cl_xy

    
    def step_power_spec_both(
        self,
        unit: str = "K",
        realisation: int | None = None,
        lmax: int | None = None,
        lam: str | float | int | None = None,
        field: int = 0,
        nsamp: float | int | None = None,
        overwrite: bool | None = None,
    ):
        """
        Convenience wrapper: compute power spectra for both the processed map
        and the ILC-synth map. Saves them to a single .npy file and returns
        (ell, cl_processed, cl_ilc).
        """

        # defaults from pipeline
        r       = self.start_realisation if realisation is None else int(realisation)
        lmax_   = self.lmax if lmax is None else int(lmax)
        overwrite_ = self.overwrite if overwrite is None else overwrite
        nsamp_  = getattr(self, "nsamp", None) if nsamp is None else nsamp
        lam_tag = self.lam_str if lam is None else (
            lam if isinstance(lam, str) else f"{float(lam):.1f}"
        )

        # 1) processed map spectrum
        ell_proc, cl_proc = self.step_power_spec(
            unit=unit,
            source="processed",
            component="cfn",     # or "cmb" depending on what you processed
            realisation=r,
            lmax=lmax_,
            lam=lam,
            field=field,
            nsamp=nsamp_,
            overwrite=overwrite_,
            save_path=None,      # avoid extra PNG here if you don't want plots
        )

        # 2) ILC map spectrum
        ell_ilc, cl_ilc = self.step_power_spec(
            unit=unit,
            source="ilc_synth",
            component="cfn",     # input comp to ILC
            extract_comp="cmb",  # extracted comp; adjust if needed
            realisation=r,
            lmax=lmax_,
            lam=lam,
            field=field,
            nsamp=nsamp_,
            overwrite=overwrite_,
            save_path=None,
        )

        # sanity check: same ell grid
        if not np.array_equal(ell_proc, ell_ilc):
            raise ValueError("ell grids for processed and ILC spectra do not match")

        ell = ell_proc

        # 3) save combined spectra
        mode = "con" if getattr(self, "constraint", False) else "uncon"

        out_dir = os.path.join(self.directory, "power_spectra")
        os.makedirs(out_dir, exist_ok=True)

        fname = f"Cl_both_{mode}_r{r}_lmax{lmax_}_lam{lam_tag}.npy"
        out_path = os.path.join(out_dir, fname)

        data = {
            "ell": ell,
            "cl_processed": cl_proc,
            "cl_ilc": cl_ilc,
        }

        if overwrite_ or (not os.path.exists(out_path)):
            np.save(out_path, data)
            print(f"[power_spec] Saved combined spectra to {out_path}")
        else:
            print(f"[power_spec] Skipped saving (exists and overwrite=False): {out_path}")

        
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
            #self.step_power_spec()
            self.step_power_spec_both()

        if "cross_power_spec" in steps:
            self.step_cross_power_spec()

        elapsed = time.perf_counter() - start_time
        print(f"SELECTED STEPS COMPLETED IN {elapsed:.2f} SECONDS (lam={self.lam}).")


# when not using multi-processing 
def main():
    parser = argparse.ArgumentParser(
        description="Run the SILC pipeline with configurable parameters and GPU selection."
    )
    parser.add_argument('--components', nargs='+',
                        default=["cmb", "sync", "dust", "noise", "tsz"],
                        help="Components to include in the pipeline.")
    parser.add_argument('--wavelet-components', nargs='+',
                        default=["cfn"],
                        help="Components to use as input for the wavelet transform.")
    parser.add_argument('--ilc-components', nargs='+',
                        default=["cmb"],
                        help="Components to extract with the ILC.")
    parser.add_argument('--frequencies', nargs='+',
                        default=["030", "044", "070", "100", "143", "217", "353", "545", "857"],
                        help="Frequency channels to use.")
    parser.add_argument('--realisations', type=int, default=1,
                        help="Number of realisations to process.")
    parser.add_argument('--start-realisation', type=int, default=0,
                        help="Index of the first realisation.")
    parser.add_argument('--lmax', type=int, default=512,
                        help="Maximum multipole L used in the analysis.")
    parser.add_argument('--N-directions', type=int, default=1,
                        help="Number of wavelet directions.")
    parser.add_argument('--lam', type=float, default=2.0,
                        help="Wavelet dilation parameter λ.")
    parser.add_argument('--method', type=str, default='jax_cuda',
                        help="Backend / method (e.g. 'jax_cuda', 'jax_cpu').")
    parser.add_argument('--visualise', action='store_true',
                        help="If set, produce diagnostic plots during wavelet steps.")
    parser.add_argument('--save-ilc-intermediates', action='store_true',
                        help="If set, save intermediate ILC products (covariances, weights, etc.).")
    parser.add_argument('--overwrite', action='store_true',
                        help="If set, overwrite existing files.")
    parser.add_argument('--directory', type=str, default='/Scratch/agnes/data',
                        help="Base directory for input/output data.")
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=0,
                        help="Index of the GPU to use (e.g. 0 or 1).")
    parser.add_argument(
        '--mem-fraction',
        type=float,
        default=0.7,
        help="GPU memory fraction to use (0–1, default: 0.7). "
             "Used to limit JAX/XLA GPU memory usage."
    )

    # Constrained ILC options (pipeline already supports these)
    parser.add_argument(
        '--constraint',
        action='store_true',
        help="Enable constrained ILC using a spectral mixing matrix F."
    )
    parser.add_argument(
        '--nsamp',
        type=float,
        default=1200.0,
        help="Number of Monte Carlo samples (nsamp) for constrained ILC."
    )

    # Which steps to run (now including power spectra)
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['download', 'process', 'wavelets', 'ilc',
                 'power_spec', 'cross_power_spec', 'all'],
        help=(
            "Which steps to run (one or more). Examples:\n"
            "  --steps download\n"
            "  --steps process wavelets\n"
            "  --steps ilc power_spec\n"
            "  --steps all"
        )
    )

    args = parser.parse_args()

    # GPU environment setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(args.mem_fraction)
    print(f"Using GPU {args.gpu} with memory fraction {args.mem_fraction} for computation.")

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
        constraint=args.constraint,
        nsamp=args.nsamp,
    )

    pipeline.run(steps=args.steps)

if __name__ == "__main__": # Run main() only when the file is executed as a script, e.g. python3 -m skyclean.silc.pipeline
    main()    