import os
import requests
import numpy as np
import healpy as hp
from utils import *

class DownloadData(): 
    """Download foreground data from Planck Legacy Archive (PLA) and generate CMB realisations."""
    def __init__(self, components: list, frequencies: list, realisations: int, directory: str = "data/", noise: bool = True): 
        """
        Parameters: 
            components (list): List of foreground components to download. Includes: 'sync' (synchrotron)
            directory (str): Directory to save the downloaded data.
            frequencies (list): Frequencies of the data to be downloaded.
            realisations (int): Number of realisations to download.
            noise (bool, optional): Whether to download noise realisations.
        """
        self.components = components
        self.frequencies = frequencies
        self.realisations = realisations
        self.directory = directory
        self.noise = noise
        
        output_dir = os.path.join(directory, "CMB_realisations/")
        create_dir(output_dir)
        base_url = "http://pla.esac.esa.int/pla/aio/"
        # create dictionary where key is component and value is a list of [template, realisation_digit, filename]
        self.component_templates = {
            "sync": [os.path.join(base_url, "product-action?SIMULATED_MAP.FILE_ID=COM_SimMap_synchrotron-ffp10-skyinbands-{frequency}_2048_R3.00_full.fits"),
                      None, #placeholder
                      os.path.join(output_dir, "sync_f{frequency}.fits")]

        }
        # realisation digit is the number of digits in the realisation number, e.g. 4 for 0001, 0002, etc.
        if noise:
            # treat noise like a foreground component for downloading
            self.component_templates["noise"] = [os.path.join(base_url, "product-action?SIMULATED_MAP.FILE_ID=ffp10_noise_{frequency}_full_map_mc_{realisation:05d}.fits"),
                                                  5,
                                                  os.path.join(output_dir, "noise_f{frequency}_r{realisation:05d}.fits")]
        # cmb is generated from its theory spectrum
        self.cmb_filepath = os.path.join(output_dir, "cmb_r{realisation:04d}.fits")

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
        template, realisation_digit, filename = self.component_templates[component]
        if realisation is None: 
            file_path = filename.format(frequency=frequency)
        else:
            file_path = filename.format(frequency=frequency, realisation=realisation)
        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. Skipping download.")
            return None

        # Format the URL with the current frequency and realisation
        url = template.format(frequency=frequency, realisation=realisation)
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Open the file in binary write mode and write the content
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {component} data for frequency {frequency}.")
        else:
            if realisation is None:
                print(f"Failed to download {component} data for frequency {frequency}. Status code: {response.status_code}")
            else:
                print(f"Failed to download {component} for frequency {frequency}. Status code: {response.status_code}")

    def download_cmb_spectrum(self):
        # UNFINISHED, right now, just use data/cmb_spectrum.txt
        pass

    def generate_and_save_cmb_realisation(self, realisation: int):
        """ Generates a CMB realisation based on the theoretical spectrum and downloads it.
        
        #### Notes: In future, will use CMB data with simulated beams; for now, use theoretical CMB.

        Parameters:
            realisation (int): The realisation number to be generated and downloaded.

        Returns:
            None
        """
        file_path = self.cmb_filepath.format(realisation=realisation)
        if os.path.exists(file_path):
            print(f"CMB realisation {realisation + 1} already exists. Skipping generation.")
            return None
        l, dl, _, _ = np.loadtxt("data/cmb_spectrum.txt").transpose()
        cl = (dl*2*np.pi)/(l*(l+1))
        cl *= 1E-12 # convert to K 
        nside = 2048
        cmb_map = hp.synfast(cl, nside=nside) 
        hp.write_map(file_path, cmb_map, overwrite=True)
        print(f"Downloaded CMB.")

    def download_all(self): 
        """
        Downloads all specified foreground components, noise and CMB realisations.

        Returns:
            None
        """
        # Download foreground, which have only one realisation.
        print("Downloading foreground components...")
        for component in self.components:
            if component == "noise":
                continue
            for frequency in self.frequencies:
                self.download_foreground_component(component, frequency)
                
        # now download CMB and noise
        for realisation in range(self.realisations):
            print(f"Downloading CMB & noise for realisation {realisation+1} of {self.realisations}...")
            self.generate_and_save_cmb_realisation(realisation)
            if self.noise:
                for frequency in self.frequencies:
                    self.download_foreground_component("noise", frequency, realisation)

# Test
components = ["sync"]
frequencies = ["030", "044"]
realisations = 3
downloader = DownloadData(components, frequencies, realisations)
downloader.download_all()









    