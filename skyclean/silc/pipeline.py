import argparse
import os
import time

from .download import DownloadData
from .map_processing import ProcessMaps
from .ilc import ProduceSILC

class Pipeline:
    """Run the entire SILC pipeline from downloading and processing data to wavelet transforms to producing the final ILC map."""
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

    def run(self):
        lam_list = [self.lam]
        for lam in lam_list:
            self.lam = lam
            start_time = time.perf_counter()
            print(f"---STARTING DATA DOWNLOAD---")
            downloader = DownloadData(
                self.components,
                self.frequencies,
                self.realisations,
                start_realisation=self.start_realisation,
                directory=self.directory,
            )
            downloader.download_all()

            print("---PROCESSING CFNs---")
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
            processor.produce_and_save_cfns()

            for wavelet_comp in self.wavelet_components:
                print(f"---PRODUCING WAVELET TRANSFORMS FOR {wavelet_comp}---")
                processor.produce_and_save_wavelet_transforms(
                    self.N_directions,
                    self.lam,
                    method=self.method,
                    visualise=self.visualise,
                )

            for ilc_comp in self.ilc_components:
                print(f"---PRODUCING ILC MAPS FOR {ilc_comp}---")
                ilc_producer = ProduceSILC(
                    self.ilc_components,
                    self.frequencies,
                    self.realisations,
                    start_realisation=self.start_realisation,
                    lmax=self.lmax,
                    N_directions=self.N_directions,
                    lam=self.lam,
                    directory=self.directory,
                    method=self.method,
                    overwrite=self.overwrite,
                )
                ilc_producer.process_wavelet_maps(
                    save_intermediates=self.save_ilc_intermediates,
                    visualise=self.visualise,
                )

            end_time = time.perf_counter()
            elapsed = end_time - start_time
            print(f"PIPELINE COMPLETED FOR lam={self.lam} IN {elapsed:.2f} SECONDS!")


def main():
    parser = argparse.ArgumentParser(
        description="Run the SILC pipeline with configurable parameters and GPU selection."
    )
    parser.add_argument(
        '--components',
        nargs='+',
        default=["cmb", "sync", "dust", "noise"],
        help='List of components in the CFN'
    )
    parser.add_argument(
        '--wavelet-components',
        nargs='+',
        default=["cfn"],
        help='Components to produce wavelet transforms for'
    )
    parser.add_argument(
        '--ilc-components',
        nargs='+',
        default=["cfn"],
        help='Components to produce ILC maps for'
    )
    parser.add_argument(
        '--frequencies',
        nargs='+',
        default=["030", "100", "353"],
        help='List of frequencies to process'
    )
    parser.add_argument(
        '--realisations',
        type=int,
        default=1,
        help='Total number of realisations to process'
    )
    parser.add_argument(
        '--start-realisation',
        type=int,
        default=0,
        help='Index of the first realisation to process'
    )
    parser.add_argument(
        '--lmax',
        type=int,
        default=512,
        help='Maximum multipole for wavelet transform'
    )
    parser.add_argument(
        '--N-directions',
        type=int,
        default=1,
        help='Number of directions for wavelet transform'
    )
    parser.add_argument(
        '--lam',
        type=float,
        default=2.0,
        help='Lambda factor (scaling) for wavelet transform'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='jax_cuda',
        help='Method for s2fft (e.g., jax_cuda)'
    )
    parser.add_argument(
        '--visualise',
        action='store_true',
        help='Visualise wavelet transforms and ILC outputs'
    )
    parser.add_argument(
        '--save-ilc-intermediates',
        action='store_true',
        help='Save intermediate ILC maps'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing data'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='/Scratch/matthew/data',
        help='Base directory for data'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='Index of the GPU to use (e.g., 0 or 1)'
    )
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
    pipeline.run()

# python3 pipeline.py \
#   --gpu 0 \
#   --components cmb sync dust noise\
#   --realisations 1 \
#   --start-realisation 0 \
#   --lmax 511 \
#   --lam 4.0 \
#   --frequencies 030 100 353 \



if __name__ == '__main__':
    main()
