from download import DownloadData
from map_processing import ProcessMaps
from ilc import ProduceSILC
import time

class Pipeline:
    """Run the entire SILC pipeline from downloading and processing data to wavelet transforms to producing the final ILC map."""
    def __init__(self, components: list, wavelet_components: list, ilc_components: list, frequencies: list, realisations: int, lmax:int = 1024,
                 N_directions: int = 1, lam: float = 2.0, noise = True, method: str = "jax_cuda", visualise: bool = False, save_ilc_intermediates: bool = True,
                  directory: str = "data/"):
        """
        Parameters:
            components (list): Components in the CFN.
            wavelet_components (list): List of components to produce wavelet transforms for.
            ilc_components (list): List of components to produce a wavelet transform / ILC for. 
            frequencies (list): Frequencies of maps to be processed.
            realisations (int): Number of realisations to process.
            lmax (int): Maximum multipole for the wavelet transform.
            N_directions (int): Number of directions for the wavelet transform.
            lam (float): lambda factor (scaling) for the wavelet transform.
            noise (bool): Whether to include noise in the processing.
            method (str): Method to use for s2fft (e.g., "jax_cuda)
            visualise (bool): Whether to visualise the wavelet transforms.
            directory (str): Directory where data is stored / saved to.
        """
        self.components = components
        self.wavelet_components = wavelet_components
        self.ilc_components = ilc_components
        self.frequencies = frequencies
        self.realisations = realisations
        self.lmax = lmax
        self.N_directions = N_directions
        self.lam = lam
        self.noise = noise
        self.method = method
        self.visualise = visualise
        self.save_ilc_intermediates = save_ilc_intermediates
        self.directory = directory

    def run(self): 
        """Run the entire pipeline."""
        # Step 1: Download data
        start_time = time.perf_counter()
        print("---STARTING DATA DOWNLOAD---")
        downloader = DownloadData(self.components, self.frequencies, self.realisations, directory=self.directory, noise=self.noise)
        downloader.download_all()
        
        # Step 2: Process CFNs
        print("---PROCESSING CFNs---")
        processor = ProcessMaps(self.components, self.wavelet_components, self.frequencies, self.realisations, self.lmax, directory=self.directory, method=self.method)
        processor.produce_and_save_cfns()
        
        # Step 3: Produce MW wavelet maps
        for wavelet_comp in self.wavelet_components:
            print(f"---PRODUCING WAVELET TRANSFORMS FOR {wavelet_comp}---")
            processor.produce_and_save_wavelet_transforms(self.N_directions, self.lam, method=self.method, visualise=self.visualise)
        # Step 3: Produce ILC map
        for ilc_comp in self.ilc_components:
            print(f"---PRODUCING ILC MAPS FOR {ilc_comp}---")
            ilc_producer = ProduceSILC(self.ilc_components, self.frequencies, self.realisations, self.lmax, self.N_directions, self.lam, directory=self.directory, method=self.method)
            ilc_producer.process_wavelet_maps(save_intermediates=self.save_ilc_intermediates, visualise=self.visualise)

        end_time = time.perf_counter()
        print(f"PIPELINE COMPLETED SUCCESFULLY IN {end_time - start_time} SECONDS!")

components = ["cmb", "sync", "dust"]
wavelet_components = ["cfn"]
ilc_components = ["cfn"]
frequencies = ["030", "044", "070"]
realisations = 1
N_directions = 1
lam = 4.0
lmax = 1024
N_directions = 1 
method = "jax_cuda"
noise = True
visualise = False
save_ilc_intermediates = False

directory = "/Scratch/matthew/data"

pipeline = Pipeline(components=components,
                    wavelet_components=wavelet_components,
                    ilc_components=ilc_components,
                    frequencies=frequencies,
                    realisations=realisations,
                    lmax=lmax,          
                    N_directions=N_directions,
                    lam=lam,
                    method=method,
                    visualise=visualise,
                    noise=noise,
                    directory=directory,
                    save_ilc_intermediates=save_ilc_intermediates)
pipeline.run()