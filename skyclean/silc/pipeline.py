import argparse
import os
import time

from .download import DownloadData
from .map_processing import ProcessMaps
from .file_templates import FileTemplates
from .ilc import ProduceSILC  


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

    # -------------------------
    # Individual steps
    # -------------------------
    def step_download(self):
        print(f"--- STARTING DATA DOWNLOAD ---")
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
        Probe j=0..63 and collect those that exist on disk for the given (comp, freq, realisation).
        Assumes the wavelet template uses {comp} and British {realisation:05d}.
        Stops after the first gap once at least one scale is found.
        """
        scales = []
        found_any = False
        # realisation (british) is zero-padded to 5 in your wavelet template
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
        """
        Use the new functional ILC: ILC_wav_coeff_maps_MP(...)
        """
        print("--- RUNNING ILC (new functional API) ---")
        ft = FileTemplates(self.directory).file_templates

        # Template for loading original wavelet coeffs (your wavelets use {comp} & {realisation})
        file_template = ft.get("wavelet_coeffs") or ft.get("wavelet_c_j")
        if file_template is None:
            raise KeyError("Missing wavelet template: expected 'wavelet_coeffs' or 'wavelet_c_j'.")

        # Output templates expected by ILC_wav_coeff_maps_MP (US {realisation}, {component}, {extract_comp})
        output_templates = {
            "doubled_maps":         ft["doubled_maps"],
            "covariance_matrices":  ft["covariance_matrices"],
            "weight_vector_matrices": ft["weight_vector_matrices"],
            "ilc_maps":             ft["ilc_maps"],
            "trimmed_maps":         ft["trimmed_maps"],
            "ilc_synth":            ft["ilc_synth"],
            "ilc_spectrum":         ft.get("ilc_spectrum"),
        }

        # Realisations (ints). Frequencies: pass exactly what you used for wavelets (e.g., "030","100",...)
        realisations = list(range(self.start_realisation, self.start_realisation + self.realisations))
        freqs = list(self.frequencies)

        # Infer scales from the first freq/realisation for each ilc target
        for ilc_comp in self.ilc_components:
            # On-disk label for input wavelets ('cfn' was saved under 'cmb')
            comp_on_disk = "cmb" if ilc_comp == "cfn" else ilc_comp
            first_real = realisations[0]
            first_freq = freqs[0]
            scales = self._infer_scales_from_disk(file_template, comp_on_disk, first_freq, first_real)

            print(f"--- PRODUCING ILC FOR target='{ilc_comp}' (source on disk='{comp_on_disk}'), scales={scales} ---")

            # Map 'cfn' target to 'cmb' extracted label
            extract_label = "cmb" if ilc_comp == "cfn" else ilc_comp

            # Band tag is built inside the function from `frequencies`, so just pass `freqs`
            _ = ProduceSILC.ILC_wav_coeff_maps_MP(
                file_template=file_template,
                frequencies=freqs,        # keep same strings as saved wavelets
                scales=scales,
                realisations=realisations,
                output_templates=output_templates,
                L_max=self.lmax,
                N_directions=self.N_directions,
                comp=comp_on_disk,        # {component} in templates
                constraint=False,
                F=None,
                extract_comp=extract_label,  # {extract_comp} in templates
                reference_vectors=None,
            )

    # -------------------------
    # Orchestrator
    # -------------------------
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

        elapsed = time.perf_counter() - start_time
        print(f"SELECTED STEPS COMPLETED IN {elapsed:.2f} SECONDS (lam={self.lam}).")


def main():
    parser = argparse.ArgumentParser(
        description="Run the SILC pipeline with configurable parameters and GPU selection."
    )
    parser.add_argument('--components', nargs='+', default=["cmb", "sync", "dust", "noise", 'tsz'])
    parser.add_argument('--wavelet-components', nargs='+', default=["cfn"])
    parser.add_argument('--ilc-components', nargs='+', default=["cfn"])
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
    parser.add_argument('--directory', type=str, default='/Scratch/matthew/data')
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
#
if __name__ == '__main__':
    main()
