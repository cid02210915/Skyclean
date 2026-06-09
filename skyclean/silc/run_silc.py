import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_ENABLE_X80", "True")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"
os.chdir("/Scratch/cindy/testing/Skyclean")       # repo root
sys.path.insert(0, os.getcwd())

from skyclean.silc.pipeline import Pipeline
import numpy as np

start_realisation_list = np.arange(0, 30, 1)
for start_realisation in start_realisation_list:
    pipe = Pipeline( 
        components=['cmb', 'dust', 'tsz', 'sync', 'noise'], 
        wavelet_components=["cfn"], 
        ilc_components=['cmb'], 
        frequencies=["030","044","070","100","143","217","353","545","857"], 
        realisations=1, 
        start_realisation=start_realisation, 
        lmax=511, 
        N_directions=1, 
        lam=2.0, 
        method="jax_cuda", 
        visualise=False, 
        save_ilc_intermediates=False,
        overwrite=False, 
        directory="/Scratch/cindy/testing/Skyclean/skyclean/data/" , 
        constraint=False,
        reference_vectors=None,
        nsamp=1200
        )
    #pipe.step_download()
    pipe.step_process()
    pipe.step_wavelets()
    pipe.step_ilc()