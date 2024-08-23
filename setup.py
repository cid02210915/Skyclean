from pathlib import Path
from setuptools import setup, find_packages

import codecs
import os


this_directory = Path(__file__).parent

# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = '0.0.4.1'
DESCRIPTION = 'A package that allows simple manipulation and noise reduction of CMB fits data using wavelet transform and Internal Linear Combination (ILC) methods.'

# Setting up
setup(
    name="skyclean",
    version=VERSION,
    author="Zhaoqi Wang",
    author_email="<maxwang0829@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[ 'numpy', 'matplotlib', 'healpy', 's2wav', 's2fft', 'mathplotlib'],
    keywords=['python', 'CMB', 'component separation methods', 'wavelet transform', "ILC"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)