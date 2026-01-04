import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(f"Current directory: {current_dir}")
print(f"Added to Python path: {parent_dir}")

# Initialise parameters 
data_directory = "data/"

frequencies = ["030", "044", "070", "100", "143", "217", "353", "545", "857"]

#start_realisation = 0
desired_lmax = 511
realisations = 30 # 93
batch_size = 2
epochs = 120
lam = 2.0
N_directions = 1
L = desired_lmax + 1
lr = 1E-3
momentum = 0.9
chs = [1, 16, 32, 32, 64]
split = [0.8, 0.2]

from skyclean.ml.train import Train
trainer = Train(
    extract_comp="cmb",
    component="cfn",
    frequencies=frequencies,
    realisations=realisations,
    lmax=desired_lmax,
    N_directions=N_directions,
    lam=lam,
    batch_size=batch_size,
    epochs=epochs,
    learning_rate=lr,
    momentum=momentum,
    chs=chs,  # Use simple channel configuration for fast training
    directory=data_directory,
    resume_training=True
)

trainer.execute_training_procedure(masked=False)

ckpt_dir = os.path.join(trainer.model_dir, f"checkpoint_{epochs}")

from skyclean.ml.inference import Inference
from skyclean.silc.visualise import Visualise

inference = Inference(
    extract_comp="cmb",
    component="cfn",
    frequencies=frequencies,
    realisations=realisations,
    lmax=desired_lmax,
    N_directions=N_directions,
    lam=lam,
    chs=chs, 
    directory=data_directory,
    model_path=ckpt_dir,
    rn=realisations,
    batch_size=batch_size,
    epochs=epochs, 
    learning_rate=lr,
    momentum=momentum,
)

print("\nLoading trained model for inference...")
model = inference.load_model(force_load=True)  # Load the model
print("Model loaded successfully for inference.")
cmb_improved = inference.predict_cmb(realisation=0) # Apply ML inference to realisation 0

plot = False
if plot:
    hp_map = SamplingConverters.mw_map_2_hp_map(cmb_improved, lmax=desired_lmax)
    hp.mollview(hp_map,unit="K",cbar=True)
    plt.title("Improved CMB after ML Inference (Realisation 0)")
    plt.show()

print("Visualising power spectra...")
mask = False
visualiser = Visualise(
    inference = inference, 
    frequencies=frequencies,
    realisation=0,
    lmax=desired_lmax,
    lam_list=[lam],
    directory=data_directory,
    rn=realisations,
    batch_size=batch_size,
    epochs=epochs, 
    learning_rate=lr,
    momentum=momentum,
    chs=chs,
    nsamp=1200,
)

ell, results = visualiser.visualise_component_ratio_power_spectra(comp_a=['ilc_synth', 'ilc_improved'],comp_b='processed_cmb',ratio=True, all_freq=False, masked=mask)

np.savez(
    str(ckpt_dir)+"/component_ratio_spectra.npz",
    ell=ell,
    ilc_synth_over_processed_cmb=results['ilc_synth/processed_cmb'],
    ilc_improved_over_processed_cmb=results['ilc_improved/processed_cmb'],
)

fig, ax = plt.subplots(figsize=(8,6))
data = np.load(f"{ckpt_dir}/component_ratio_spectra.npz")
ax.plot(data["ell"], data["ilc_synth_over_processed_cmb"], label=f'ilc_synth / processed')
ax.plot(data["ell"], data["ilc_improved_over_processed_cmb"], label=f'ml_improved / processed')
ax.axhline(1, ls=":", color="red")
ax.set_ylim(0.5, 1.5)
ax.set_ylabel(r"Ratio of $D_{\ell}$", fontsize=14)
ax.set_xlabel(r"$\ell$", fontsize=14)
ax.set_title('Ratio of ratio (processed vs final)\nML: L2, 50 epochs', fontsize=13)
ax.grid(True, which="both", linestyle=":", linewidth=0.5)
ax.legend(fontsize=14)
fig.tight_layout()
plt.savefig('spectrum.png', dpi=250)
plt.close()