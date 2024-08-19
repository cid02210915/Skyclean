import numpy as np
import requests
import os

# Convert arcminutes to radians
def arcmin_to_radians(arcmin):
    '''
    Takes arcminutes and converts them to radians, 
    used for beam_fwhm = {30: 32.33, 44: 27.01, 70: 13.25},
    hp.sphtfunc.gauss_beam(fwhm_rad, lmax=lmax-1, pol=False),
    '''
    return np.radians(arcmin / 60)



def download_cmb_realizations(directory, frequencies, realizations, url_template, filename_template, realization_digit):
    """
    Downloads Cosmic Microwave Background (CMB) realizations for specified frequencies and realizations.

    This function will download files from a specified URL template, format them according to the frequency and 
    realization number, and save them in the specified directory. If the file already exists, it will skip the download.

    Parameters:
    directory (str): The directory where files will be saved.
    frequencies (list of str): A list of frequency identifiers to download.
    realizations (range or list of int): A range or list of realization numbers to download.
    url_template (str): The URL template that will be formatted with the frequency and realization.
    filename_template (str): The template for the filename that will be used when saving the files.
    realization_digit (int): The number of digits to use for formatting the realization number (e.g., 4 digits for "0001").

    Returns:
    None

    Example:# 
    directory = 'CMB_realizations'
    frequencies = ["030", "044", "070", "100", "143", "217", "353"]  # Example frequencies
    realizations = range(30)  # Example: 30 realizations from 0 to 29
    url_template = "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=febecop_ffp10_lensed_scl_cmb_{frequency}_mc_{realization}.fits"
    filename_template = "febecop_ffp10_lensed_scl_cmb_{frequency}_mc_{realization}.fits"
    realization_digit = 4 
    download_cmb_realizations(directory, frequencies, realizations, url_template, filename_template, realization_digit)
    
    # Note: 545 and 857 GHz have different urls
    frequencies = [ "545", "857"]  # Example frequencies
    filename_template = "symbeam_ffp10_lensed_scl_cmb_{frequency}_mc_{realization}.fits"
    url_template = "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=symbeam_ffp10_lensed_scl_cmb_{frequency}_mc_{realization}.fits"
    download_cmb_realizations(directory, frequencies, realizations, url_template, filename_template, realization_digit)
    
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Loop through each frequency and realization
    for realization in realizations:
        for frequency in frequencies: 
            realization_str = str(realization).zfill(realization_digit)
            filename = filename_template.format(frequency=frequency, realization=realization_str)
            file_path = os.path.join(directory, filename)

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"File {filename} already exists. Skipping download.")
                continue

            # Format the URL with the current frequency and realization
            url = url_template.format(frequency=frequency, realization=realization_str)

            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Open the file in binary write mode and write the content
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {file_path}")
            else:
                print(f"Failed to download data for frequency {frequency} and realization {realization_str}. Status code: {response.status_code}")


