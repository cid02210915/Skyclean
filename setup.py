from pathlib import Path
from setuptools import setup, find_packages

import codecs
import os


this_directory = Path(__file__).parent

# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'A package that allows simple manipulation and noise reduction of CMB data using wavelet and component separation methods'

# Setting up
setup(
    name="skyclean",
    version=VERSION,
    author="Zhaoqi Wang",
    author_email="<maxwang0829@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['opencv-python', 'pyautogui', 'pyaudio'],
    keywords=['python', 'CMB', 'component separation methods', 'wavelets'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)