from pathlib import Path
import re
import os

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

