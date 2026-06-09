

import os
import requests
import numpy as np
import healpy as hp
from astropy.io import fits
from .utils import *
from .file_templates import FileTemplates
from io import StringIO
import time

class DownloadData(): 
    """Download foreground data from Planck Legacy Archive (PLA) and generate CMB realisations."""
    def __init__(self, components: list, 
                 frequencies: list, 
                 realisations: int, 
                 start_realisation: int,
                 directory: str = "data/"): 
        """
        Parameters: 
            components (list): List of components to download. Includes: 'cmb', 'sync', 'dust' (synchrotron)
            frequencies (list): Frequencies of the data to be downloaded.
            realisations (int): Number of realisations to download.
            start_realisation (int): Starting realisation number for processing.
            directory (str): Directory to save the downloaded data.
        """
        self.components = components
        self.frequencies = frequencies
        self.realisations = realisations
        self.start_realisation = start_realisation
        self.directory = directory

        files = FileTemplates(directory)
        self.download_templates = files.download_templates
        self.file_templates = files.file_templates

    @staticmethod
    def _validate_fits_file(path: str) -> tuple[bool, str]:
        """Basic on-disk FITS integrity check."""
        if not os.path.exists(path):
            return False, "file does not exist"
        if os.path.getsize(path) == 0:
            return False, "file is empty (0 bytes)"
        try:
            with fits.open(path, memmap=False) as hdul:
                hdul.verify("exception")
                # Force table payload materialization to catch truncated binary-table data.
                if len(hdul) > 1 and hasattr(hdul[1], "data"):
                    data = hdul[1].data
                    if data is not None:
                        _ = data.shape
        except Exception as exc:
            return False, f"invalid FITS structure ({exc})"
        return True, ""

    
    def download_foreground_component(self, component: str, frequency: str, realisation: int = None):
            """
            Downloads the specified foreground component for a given frequency.

            Parameters:
                component (str): The foreground component to download.
                frequency (str): The frequency for which to download the component.
                realisation (int, optional): The realisation number to be downloaded (for noise).

            Returns:
                None
            """ 
            freq = str(frequency).zfill(3)

            # Skip unavailable CO channels
            if component == "co" and freq in {"030", "044", "070", "217"}:
                print(f"Skipping CO download at {freq} GHz.")
                return

            template = self.download_templates[component]
            file_template = self.file_templates[component]
            if realisation is None: 
                # foreground components same across realisations
                filename = file_template.format(frequency=frequency)
            else:
                filename = file_template.format(frequency=frequency, realisation=realisation)
            
            # Skip only if existing file is valid; otherwise force redownload.
            if os.path.exists(filename):
                valid, reason = self._validate_fits_file(filename)
                if valid:
                    print(f"File {filename} already exists and is valid. Skipping download.")
                    return
                print(f"Existing file {filename} is invalid ({reason}). Redownloading.")
                try:
                    os.remove(filename)
                except OSError as exc:
                    raise RuntimeError(f"Cannot remove invalid existing file {filename}: {exc}") from exc

            # PLA has a naming inconsistency for clusterirps: 44 GHz is published as 040.
            remote_frequency = "040" if (component == "clusterirps" and freq == "044") else frequency

            # PLA provides exactly 300 noise maps: mc_00000 ... mc_00299.
            # Reuse cyclically so r00300 -> mc_00000, r00301 -> mc_00001, etc.
            remote_realisation = realisation
            if component == "noise" and realisation is not None:
                remote_realisation = int(realisation) % 300

            url = template.format(frequency=remote_frequency, realisation=remote_realisation)

            
            # stronguchii @ 070 is "_full.f" on PLA, but we want to save as ".fits" locally
            if component == "stronguchii" and frequency == "070":
                url = url.replace("_full.fits", "_full.f") 

            # Send a GET request to the URL and stream the content to avoid loading large files into memory
            # Implement retry logic with exponential backoff to handle transient network issues during large downloads in parallel processing environments
            max_attempts = 5
            last_error = None

            for attempt in range(max_attempts):
                tmp_filename = f"{filename}.tmp.{os.getpid()}.{time.time_ns()}"
                try:
                    with requests.get(url, stream=True, timeout=120) as response:
                        response.raise_for_status()
                        content_length = response.headers.get("Content-Length")
                        if content_length is not None and int(content_length) == 0:
                            raise RuntimeError("Remote server responded with Content-Length: 0")

                        bytes_written = 0
                        with open(tmp_filename, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    bytes_written += len(chunk)

                    if bytes_written == 0:
                        raise RuntimeError("Downloaded 0 bytes.")

                    valid, reason = self._validate_fits_file(tmp_filename)
                    if not valid:
                        raise RuntimeError(f"Downloaded invalid FITS ({reason}).")

                    # Atomic rename (prevents partial-file poisoning)
                    os.replace(tmp_filename, filename)
                    # Quick post-write guard on the final path.
                    valid, reason = self._validate_fits_file(filename)
                    if not valid:
                        try:
                            os.remove(filename)
                        except OSError:
                            pass
                        raise RuntimeError(f"Final saved FITS failed validation ({reason}).")

                    print(f"Downloaded {component} data for frequency {frequency}.")
                    return

                except (requests.exceptions.RequestException, RuntimeError, OSError) as e:
                    last_error = e
                    print(f"Download failed ({attempt+1}/{max_attempts}): {e}")
                    time.sleep(10 * (attempt + 1))
                finally:
                    if os.path.exists(tmp_filename):
                        try:
                            os.remove(tmp_filename)
                        except OSError:
                            pass

            raise RuntimeError(
                f"Failed to download {component} at {frequency} after {max_attempts} attempts. "
                f"Last error: {last_error}"
            )
            
            
        

    def download_cmb_spectrum(self):
        """
        Download the Planck 2018 best-fit CMB TT spectrum
        and save it as cmb_spectrum.txt in the data directory.
        """

        url = "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt"
        path = os.path.join(self.directory, "cmb_spectrum.txt")

        if os.path.exists(path):
            print("cmb_spectrum.txt already exists. Skipping download.")
            return

        print("Downloading Planck TT spectrum...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        # Load directly into numpy
        data = np.loadtxt(StringIO(resp.text))
        ells = data[:, 0].astype(int)
        Dl   = data[:, 1]  # μK^2

        # Keep same 4-column format as before
        arr = np.column_stack([ells, Dl, np.zeros_like(ells), np.zeros_like(ells)])
        np.savetxt(path, arr, fmt="%.8e")
        print(f"Wrote Planck cmb_spectrum.txt at {path}")   


    def generate_and_save_cmb_realisation(self, realisation: int):
            """ Generates a CMB realisation based on the theoretical spectrum and downloads it.

            #### Notes: In future, will use CMB data with simulated beams; for now, use theoretical CMB.

            Parameters:
                realisation (int): The realisation number to be generated and downloaded.

            Returns:
                None
            """
            filename = self.file_templates["cmb"].format(realisation=realisation, lmax=1023)  # lmax is set to 1023 for now
            if os.path.exists(filename):
                valid, reason = self._validate_fits_file(filename)
                if valid:
                    print(f"CMB realisation {realisation} already exists and is valid. Skipping generation.")
                    return None
                print(f"Existing CMB realisation {realisation} is invalid ({reason}). Regenerating.")
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise RuntimeError(f"Cannot remove invalid existing CMB file {filename}: {exc}") from exc
            # load the theoretical spectrum, if not exists, load the Planck one
            if os.path.exists(os.path.join(self.directory, "cmb_spectrum_theory.txt")):
                l, dl = np.loadtxt(os.path.join(self.directory, "cmb_spectrum_theory.txt"),
                                    comments="#", usecols=(0,1), unpack=True)
                l = l.astype(int)
                dl *= (2.7255**2)
                lmax = 1023
                cl = np.zeros(lmax + 1)
                m = (l >= 2) & (l <= lmax)
                cl[l[m]] = dl[m] * (2.0 * np.pi) / (l[m] * (l[m] + 1.0))   # D_l -> C_l
                nside = 2048
                cmb_map = hp.synfast(cl, nside=nside, pixwin=False, lmax=1023) # convolve with pixel window
                tmp_filename = f"{filename}.tmp.{os.getpid()}.{time.time_ns()}"
                hp.write_map(tmp_filename, cmb_map)
                valid, reason = self._validate_fits_file(tmp_filename)
                if not valid:
                    try:
                        os.remove(tmp_filename)
                    except OSError:
                        pass
                    raise RuntimeError(f"Generated invalid CMB FITS at {tmp_filename} ({reason}).")
                os.replace(tmp_filename, filename)
                # Quick post-write guard on the final path.
                valid, reason = self._validate_fits_file(filename)
                if not valid:
                    try:
                        os.remove(filename)
                    except OSError:
                        pass
                    raise RuntimeError(f"Final saved CMB FITS failed validation ({reason}).")
                print(f"Downloaded CMB realisation {realisation}.")
            else:
                raise ValueError("No theoretical CMB spectrum found. Please provide cmb_spectrum_theory.txt in the data directory.")


    def download_all(self): 
        """
        Downloads all specified foreground components, noise and CMB realisations.

        Returns:
            None
        """
        # Set higher retry limit for requests to handle transient network issues during large downloads
        requests.adapters.DEFAULT_RETRIES = 5

        # Download foregrounds, which have only one realisation.
        print("Downloading foreground components...")
        if 'all' in self.components:
            for component in self.download_templates:
                if component == "cmb" or component == "noise" or component == "extra_feature" or component == "cib" or component == "mask":
                    continue
                else:
                    for frequency in self.frequencies:
                        self.download_foreground_component(component, frequency)

        else:
            for component in self.components:
                if component == "cmb" or component == "noise" or component == "extra_feature":
                    continue
                else:
                    for frequency in self.frequencies:

                        # minimal CIB guard
                        if component == "cib" and frequency not in {"353", "545", "857"}:
                            continue

                        self.download_foreground_component(component, frequency)

        # now download CMB and noise, which are realisation dependent.
        for realisation in range(self.realisations):
            realisation += self.start_realisation  # Adjust for starting realisation
            print(realisation)
            print(f"Downloading CMB for realisation {realisation}...")
            self.generate_and_save_cmb_realisation(realisation)
            if 'noise' in self.components or 'all' in self.components:
                print(f"Downloading noise for realisation {realisation}...")
                for frequency in self.frequencies:
                    self.download_foreground_component("noise", frequency, realisation)


# components = ["sync", "dust"]
# frequencies = ["030", "070", "143", "217", "353"]
# realisations = 2
# downloader = DownloadData(components, frequencies, realisations, directory = "/Scratch/matthew/data/", noise=True)
# downloader.download_all()