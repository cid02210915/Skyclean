# skyclean 🌌
The observed Cosmic Microwave Background (CMB) is contaminated by a number of foreground astrophysical sources (e.g. synchrotron, dust). It is of great interest in cosmology to obtain a clear-view of the primordial light from the Big Bang - a cleaned CMB. For example, fluctuations in the CMB caused by dark matter can help us to deduce which particle(s) (e.g. WIMPs, axions) is/are the source of dark matter. `skyclean` provides functionality to perform this cleaning.

## SILC 🧹

One CMB cleaning method is the **internal linear combination (ILC)**, a 'blind' method that performs a weighted sum across frequency channels, subject to the constraint that the CMB is conserved. The weights are optimised such that the variance of the foreground & noise is minimised. An extension of this method is the **Scale-discretised, directional wavelet ILC** or **SILC** ([Rogers et al. (2016)](https://doi.org/10.1093/mnras/stw2128)) - the method employed by `skyclean`. This method performs the ILC on a **wavelet decomposition** of the signal, defined on the sphere. Wavelets extract information spatially and at different scales, with the scales allowing features at different multipoles to be cleaned. SILC additionally uses directional wavelets, which are non-axisymmetric and can be oriented in different directions.

`skyclean` makes use of the `s2wav` package to perform wavelet decompositions, and the `s2fft` package to perform harmonic space transformations (see the [astro-informatics](github.com/astro-informatics) GitHub). The code is GPU-accelerated on a `jax` backend. 

## Machine Learning 🤖

SILC is designed to minimise variance. However, many foregrounds have higher-order moments that SILC does not reduce. Recently, machine learning has been employed as a tool to learn more complex features as a post-processing stage to the ILC (see [McCarthy et al. (2025)](https://doi.org/10.1103/PhysRevD.111.063549)). In this method, the inputs to the network are CMB-free signals across frequency channels (observed map in frequency channel minus ILC), and the outputs are the ILC residual (ILC minus CMB, one channel). Current efforts have applied this to locally-flat patches of the sky.

`skyclean` aims to apply an ML stage to improve the SILC-processed maps [MORE INFORMATION NEEDED ON SPECIFIC METHOD CHOSEN], with the novelty of performing this on the entire sphere. This requires the use of geometric deep learning, a field which extends the idea of translational equivariance in traditional Euclidean convolutional networks to general group manifolds (see [Cohen et al. (2015)](https://proceedings.mlr.press/v48/cohenc16.html)); in this case, aiming to capture rotational equivariance on the $S^2$ and $\text{SO}(3)$ manifolds. The `s2ai` package used in `skyclean.ml` employs so-called discrete-continuous convolutions on the sphere, which capture equivaraince whilst avoiding expensive harmonic space transforms (see [Ocampo et al. (2022)](https://arxiv.org/abs/2209.13603)). In `skyclean.ml`, an $S^2$ UNET architecture is employed to map to the ILC residual, with the goal being to make the ILC power spectrum closer to the true CMB power spectrum. 

## Usage ♦️
Check the tutorials directory for a detailed walkthrough of the SILC and ML pipelines, with diagrams and examples. It is highly recommended to run the code on a GPU-based system, especially for larger multipole bandlimits.

The SILC pipeline and ML training can be ran directly from the terminal, for example:

```bash
python3 -m skyclean.silc.pipeline --gpu 0 --components cmb sync dust noise --realisations 1 --start-realisation 0 --lmax 511 --lam 4.0 --frequencies 030 100 353 --directory data/

python3 -m skyclean.ml.train   --gpu 1   --frequencies 030 100 353   --realisations 1000   --lmax 511   --lam 2.0   --batch-size 8   --epochs 100   --learning-rate 1e-3  --directory data/
```
See `train.py` and `pipeline.py` for more options.
## Directories 📞

``` bash
skyclean/  
├── silc/          # CMB map pre-processing, wavelet transforms and SILC.
│      ├─ utils.py        # - Utility tools for saving
│      ├─ file_templates.py          # - dictionaries containing data directories
│      ├─ download.py        # - Download CMB, foregrounds, noise (currently from Planck archive)
│      ├─ map_tools.py          # - healpy map tools, MW map tools, HP to MW converters
│      ├─ map_processing.py          # - convolve and downsample downloaded maps, convert to MW, wavelet transform 
│      ├─ ilc.py          # - GPU-accelerated ILC on MW wavelet maps
│      ├─ pipeline.py          # - Wrapper for running entire SILC process. CLI included.
│      ├─ power_spec.py          # TT power spectrum utilities
│      ├─ mixing_matrix_constraint.py      # Build spectral response F and cILC constraints    
│      ├─ run_ILC.py         # Example entry script to run the pipeline end-to-end
│
│
│
├── ml/          # ML post-processing stage (WIP)
│      ├─ data.py                # - Produce CMB-free and ML-ready transformed input and output datasets for training (using tf)
│      ├─ model.py               # - S2 UNET architectures
│      ├─ train.py              # - Run the training loops. CLI included.
       ├─ inference.py              # - apply trained model to improve ILC

tests/        # pytests (WIP)
├── test_ilc.py # tests for ILC pipeline
```

## Installation ⚙️
`skyclean` is not currently available as a package, and must be ran directly from the repo. This requires you to setup an environment matching the `skyclean` dependencies. A working environment can be reproduced as follows. 
First, git clone and enter this repo and run: 
```bash
conda env create -f environment.yml
```
This will install a conda environment `sc-gpu`. You will also need to install `s2ai`, a private repo, manually. First, request access from Jason McEwen: http://www.jasonmcewen.org/. Then, git clone and enter `s2ai` and run: 
```bash
pip install --no-deps .
```
in your `sc-gpu` environment. 




