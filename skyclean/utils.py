from pathlib import Path
import re
import os
import numpy as np
import healpy as hp

def create_dir(directory: str):
    """
    Create a directory if it does not exist.

    Parameters:
        dir (str): The path of the directory to create.

    Returns:
        None
    """
    if not os.path.exists(directory):
        print("Creating directory:", directory)
        os.makedirs(directory)
    else:
        pass

def detect_scales(directory: str, comp: str, realisation: int = 0, pad: int = 5) -> list[int]:
    """
    Detect the scales of wavelet coefficients for a given component in a specified directory.

    Parameters:
        directory (str): The directory to search for wavelet coefficient files.
        comp (str): The component for which to detect scales (e.g., 'sync', 'noise').
        realisation (int): The realisation number to filter files (default is 0).
        pad (int): The number of digits to pad the realisation number (default is 5).
    
    Returns:
        list[int]: A sorted list of detected scales.
    """
    prefix = f"{comp}_wavelet"
    p = Path(directory)
    # build a regex that enforces the right realisation, e.g. _r00000_
    real_str = f"{realisation:0{pad}d}"
    pattern = re.compile(
        rf"{re.escape(prefix)}.*_s(\d+)_r{real_str}_"
    )

    scales = set()
    for f in p.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            scales.add(int(m.group(1)))

    return sorted(scales)

def np_save_and_load(data, filename: str):
    """
    Save data to a file and then load it back.

    Parameters:
        data: The data to save.
        filename (str): The name of the file to save the data to.
    
    Returns:
        The loaded data.
    """
    np.save(filename, data)
    return np.load(filename)

def save_map(hp_map, filepath):
    """Save the processed map to the specified filepath."""
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping saving.")
    else:
        hp.write_map(filepath, hp_map)


def extract_file_parameters(filepath, file_type="auto"):
    """
    Extract frequency, realisation, and lmax from various file types.
    
    Parameters:
        filepath (str): Full path to the file
        file_type (str): Type of file to parse. Options: "auto", "cfn", "ilc", "wavelet", "noise", etc.
    
    Returns:
        dict: Dictionary containing extracted parameters. Keys may include:
              'frequency', 'realisation', 'lmax', 'scale', 'component'
              Values are None if not found in filename.
    """
    filename = os.path.basename(filepath)
    result = {
        'frequency': None,
        'realisation': None, 
        'lmax': None,
        'scale': None,
        'component': None
    }
    
    # Define regex patterns for different file types
    patterns = {
        'cfn': r'cfn_f(\d+)_r(\d+)_lmax(\d+)\.npy',
        'ilc': r'ILC_synthesised_Map_R(\d+)_lmax(\d+)\.npy',
        'wavelet': r'(\w+)_wavelet_f(\d+)_s(\d+)_r(\d+)_lmax(\d+)\.npy',
        'scaling': r'(\w+)_scaling_f(\d+)_r(\d+)_lmax(\d+)\.npy',
        'noise': r'processed_noise_f(\d+)_r(\d+)_lmax(\d+)\.npy',
        'cmb': r'processed_cmb_r(\d+)_lmax(\d+)\.npy',
        'sync': r'processed_sync_f(\d+)_lmax(\d+)\.npy',
        'dust': r'processed_dust_f(\d+)_lmax(\d+)\.npy',
        'doubled_maps': r'doubled_maps_F(\d+)_S(\d+)_R(\d+)_lmax(\d+)\.npy',
        'covariance': r'cov_MW_F(.+)_S(\d+)_R(\d+)_lmax(\d+)\.npy',
        'weight_vector': r'weight_vector_S(\d+)_R(\d+)_lmax(\d+)\.npy',
        'ilc_doubled': r'ILC_doubled_Map_S(\d+)_R(\d+)_lmax(\d+)\.npy',
        'ilc_trimmed': r'ILC_trimmed_wav_Map_S(\d+)_R(\d+)_lmax(\d+)\.npy'
    }
    
    # Auto-detect file type if not specified
    if file_type == "auto":
        for ftype, pattern in patterns.items():
            if re.search(pattern, filename):
                file_type = ftype
                break
        else:
            return result  # No pattern matched
    
    if file_type not in patterns:
        return result
    
    match = re.search(patterns[file_type], filename)
    if not match:
        return result
    
    # Extract parameters based on file type
    if file_type == 'cfn':
        result['frequency'] = match.group(1)
        result['realisation'] = int(match.group(2))
        result['lmax'] = int(match.group(3))
        
    elif file_type == 'ilc':
        result['realisation'] = int(match.group(1))
        result['lmax'] = int(match.group(2))
        
    elif file_type == 'wavelet':
        result['component'] = match.group(1)
        result['frequency'] = match.group(2)
        result['scale'] = int(match.group(3))
        result['realisation'] = int(match.group(4))
        result['lmax'] = int(match.group(5))
        
    elif file_type == 'scaling':
        result['component'] = match.group(1)
        result['frequency'] = match.group(2)
        result['realisation'] = int(match.group(3))
        result['lmax'] = int(match.group(4))
        
    elif file_type == 'noise':
        result['frequency'] = match.group(1)
        result['realisation'] = int(match.group(2))
        result['lmax'] = int(match.group(3))
        
    elif file_type == 'cmb':
        result['realisation'] = int(match.group(1))
        result['lmax'] = int(match.group(2))
        
    elif file_type in ['sync', 'dust']:
        result['frequency'] = match.group(1)
        result['lmax'] = int(match.group(2))
        
    elif file_type == 'doubled_maps':
        result['frequency'] = match.group(1)
        result['scale'] = int(match.group(2))
        result['realisation'] = int(match.group(3))
        result['lmax'] = int(match.group(4))
        
    elif file_type == 'covariance':
        result['frequency'] = match.group(1)  # Could be multiple frequencies joined by _
        result['scale'] = int(match.group(2))
        result['realisation'] = int(match.group(3))
        result['lmax'] = int(match.group(4))
        
    elif file_type in ['weight_vector', 'ilc_doubled', 'ilc_trimmed']:
        result['scale'] = int(match.group(1))
        result['realisation'] = int(match.group(2))
        result['lmax'] = int(match.group(3))
    
    return result