from utils import * 
from map_tools import *
import healpy as hp
import matplotlib.pyplot as plt


### WIP: need to implement file templates. Plotting functions work otherwise.


class Visualise(): 
    def __init__(self, frequencies: list, realisation: int, lmax: int, directory: str = "data/"):
        """
        Parameters:
            frequencies (list): List of frequencies to visualise.
            realisation (int): The realisation number to visualise.
            lmax (int): Maximum multipole for the visualisation.
            directory (str): Directory where data is stored / saved to.
        """
        self.frequencies = frequencies
        self.realisation = realisation
        self.lmax = lmax
        self.directory = directory

        self.saved_directories = {
            "cfn": os.path.join(self.directory, "CFN_realisations"),
            "processed_maps": os.path.join(self.directory, "processed_maps"),
            "wavelet_coeffs": os.path.join(self.directory, "wavelet_transforms/wavelet_coeffs"),
            "scaling_coeffs": os.path.join(self.directory, "wavelet_transforms/scaling_coeffs"),
            "ilc_synthesised": os.path.join(self.directory, "SILC/ilc_synthesised_maps"),
        }

        self.file_templates = {
            "cfn": os.path.join(self.saved_directories["cfn"], "cfn_f{frequency}_r{realisation:04d}_lmax{lmax}.npy"),
            "cmb": os.path.join(self.saved_directories["processed_maps"], "processed_cmb_r{realisation:04d}_lmax{lmax}.npy"),
            "sync": os.path.join(self.saved_directories["processed_maps"], "processed_sync_f{frequency}_lmax{lmax}.npy"),
            "dust": os.path.join(self.saved_directories["processed_maps"], "processed_dust_f{frequency}_lmax{lmax}.npy"),
            "noise": os.path.join(self.saved_directories["processed_maps"], "processed_noise_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
            "wavelet_coeffs": os.path.join(self.saved_directories["wavelet_coeffs"], "{comp}_wavelet_f{frequency}_s{scale}_r{realisation:05d}_lmax{lmax}.npy"),
            "scaling_coeffs": os.path.join(self.saved_directories["scaling_coeffs"], "{comp}_scaling_f{frequency}_r{realisation:05d}_lmax{lmax}.npy"),
            "ilc_synthesised": os.path.join(self.saved_directories["ilc_synthesised"], "ilc_synthesised_map_r{realisation:04d}_lmax{lmax}.npy"),
        }

        self.ilc_spectrum_output_path = os.path.join(self.saved_directories["ilc_synthesised"], "ilc_power_spectrum_R{realisation:04d}_lmax{lmax}.npy")

    def visualise_maps(self, comps: list):
        """
        visualise maps for each frequency and component.

        Parameters: 
            comps (list): List of components to visualise. e.g. ['sync', 'cmb', 'noise'].
        """
        file_templates = {}
        for comp in comps: 
            file_templates[comp] = self.file_templates[comp]
        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        # setup figure
        nrows = len(list(file_templates.keys()))
        ncols = len(frequencies)
        fig = plt.figure(figsize=(5*ncols, 5*nrows))
        for i,(comp,file_template) in enumerate(file_templates.items()):
            for j, freq in enumerate(frequencies):
                fp = file_template.format(frequency=freq, realisation = realisation, lmax=lmax)
                if comp == "ilc_synthesised":
                    mw_map = np.load(fp)
                    hp_map = SamplingConverters.mw_map_2_hp_map(mw_map, lmax=lmax)
                else:
                    hp_map, _ = hp.read_map(fp, h=True)
                panel = j + i*ncols + 1
                if comp == "sync_map": # plot log scale for sync
                    hp.mollview(
                        hp_map,
                        fig=fig.number,
                        sub=(nrows, ncols, panel),
                        title=f"{comp} @ {freq} GHz",
                        unit="K",
                        cbar=True       
                    )
                else:
                    hp.mollview(
                    hp_map,
                    fig=fig.number,
                    sub=(nrows, ncols, panel),
                    title=f"{comp} @ {freq} GHz",
                    unit="K",
                    cbar=True      
                    )
            #cb = fig.colorbar(hp.graticule(), ax=fig.axes, orientation="horizontal", fraction=0.05, pad=0.07)
        plt.tight_layout()
        #plt.savefig('components.pdf', dpi=1400)
        plt.show()

    def visualise_wavelet_maps(self, comps: list = ['cmb', 'cfn', 'ilc']):
        """
        visualise wavelet maps for each frequency and scale.

        CURRENTLY HARDCODED SO THAT COMPS SHOULD BE ['cmb', 'cfn', 'ilc']

        Parameters:
            comps (list): List of components to visualise. e.g. ['sync', 'cmb', 'noise'].

        """
        file_templates = {}
        for comp in comps:
            file_templates[comp] = self.file_templates['wavelet_coeffs'].format(comp=comp)
        scales = self.scales
        frequencies = self.frequencies
        realisation = self.realisation
        n_freq   = len(frequencies)
        n_scales = len(scales)
        # four blocks: CMB_total (n_freq), ILC (1), CMB–ILC (n_freq), CMB-only–ILC (1)
        nrows = 2*n_freq + 2
        ncols = n_scales

        fig = plt.figure(figsize=(5*ncols, 5*nrows))
        fig.subplots_adjust(top=0.90, left=0.12, right=0.98, hspace=0.3)

        # 1) CMB_total block
        csn_dict = {}
        for i, freq in enumerate(frequencies):
            coeffs = MWTools.load_wavelet_scaling_coeffs(
                n_scales, realisation,
                file_templates['cfn'], freq
            )
            csn_dict[freq] = coeffs
            for j, scale in enumerate(scales):
                panel = i*ncols + j + 1
                MWTools.visualise_mw_map(
                    coeffs[scale],
                    title="",   # per‐cell titles cleared
                    fig=fig, nrows=nrows, ncols=ncols, panel=panel
                )

        # 2) ILC row
        ilc_coeffs = MWTools.load_wavelet_scaling_coeffs(
            n_scales, realisation,
            file_templates['ilc'], None
        )
        ilc_row = n_freq
        for j, scale in enumerate(scales):
            panel = ilc_row*ncols + j + 1
            MWTools.visualise_mw_map(
                ilc_coeffs[scale],
                title="", fig=fig, nrows=nrows, ncols=ncols, panel=panel
            )

        # 3) CMB_total–ILC block
        for i, freq in enumerate(frequencies):
            diff_row   = n_freq + 1 + i
            coeffs     = csn_dict[freq]
            for j, scale in enumerate(scales):
                panel = diff_row*ncols + j + 1
                diff_map = coeffs[scale] - ilc_coeffs[scale]
                MWTools.visualise_mw_map(
                    diff_map,
                    title="", fig=fig, nrows=nrows, ncols=ncols, panel=panel
                )

        # 4) CMB-only–ILC bottom row (first freq)
        cmb_only = MWTools.load_wavelet_scaling_coeffs(
            n_scales, realisation,
            file_templates['cmb'], frequencies[0]
        )
        bottom = 2*n_freq + 1
        for j, scale in enumerate(scales):
            panel = bottom*ncols + j + 1
            diff  = cmb_only[scale] - ilc_coeffs[scale]
            MWTools.visualise_mw_map(
                diff,
                title="", fig=fig, nrows=nrows, ncols=ncols, panel=panel
            )

        

        # — Column titles at top —
        for j, scale in enumerate(scales):
            x = (j + 0.5)/ncols
            fig.text(
                x, 0.96,
                f"Scale {scale}",
                ha='center', va='bottom',
                fontsize=14, fontweight='bold'
            )

        # — titles at top of each block —
        blocks = [
            (0,              "CSN(f)"),
            (n_freq,         "ILC"),
            (n_freq+1,       "CSN(f) – ILC"),
            (2*n_freq+1,     "CMB-only – ILC"),
        ]
        pad = 1.0/nrows * 0.2  # small vertical padding
        for start, title in blocks:
            y_edge  = 1 - start/nrows
            y_title = y_edge - pad
            fig.text(
                0.5, y_title,
                title,
                ha='center', va='bottom',
                fontsize=30, fontweight='bold'
            )

        # — Horizontal separators —
        for boundary in [n_freq-1, n_freq, 2*n_freq]:
            y = 1 - (boundary+1)/nrows
            line = Line2D([0.05, 0.95], [y, y],
                        transform=fig.transFigure,
                        linestyle='--', linewidth=1)
            fig.add_artist(line)

        # — Vertical frequency labels for blocks 1, 3 —
        # Block 1 (CMB Total)
        for i, freq in enumerate(frequencies):
            y = 1.0 - (i + 0.5)/nrows
            fig.text(
                0, y,
                f"{freq} GHz",
                va='center', ha='left',
                rotation='vertical', fontsize=30
            )
        # Block 3 (CMB–ILC)
        for i, freq in enumerate(frequencies):
            row = n_freq + 1 + i
            y   = 1.0 - (row + 0.5)/nrows
            fig.text(
                0, y,
                f"{freq} GHz",
                va='center', ha='left',
                rotation='vertical', fontsize=30
            )

        plt.show()
    
    def compute_and_save_ilc_power_spec(self, fp: str):
        """
        Compute and save the ILC power spectrum from the ILC map. 
        This takes longer than hp maps since ILC  is in MW form and must be converted.
        Parameters:
            fp (str): File path to the ILC map.
        
        Returns:
            cl (np.ndarray): The computed power spectrum.
        """
        ilc_path = self.ilc_spectrum_output_path.format(realisation=self.realisation, lmax=self.lmax)
        # Recommended to run this first if computing ilc
        if os.path.exists(ilc_path):
            cl = np.load(ilc_path)
        else:
            print(f"Existing ILC power spec not found at {ilc_path}, computing now...")
            ilc_mw = np.load(fp)
            hp_ilc = SamplingConverters.mw_map_2_hp_map(ilc_mw,self.lmax)*1E6  #
            cl = hp.sphtfunc.anafast(hp_ilc, lmax=self.lmax)
            nside = self.lmax // 2
            cl /= hp.pixwin(nside, lmax=self.lmax)**2
            np.save(self.ilc_spectrum_output_path.format(realisation=self.realisation, lmax = self.lmax), cl)
        return cl

    def visualise_power_spectra(self, comps = ['cmb', 'cfn', 'ilc']):
        """
        Arrange one subplot per frequency (max 3 columns per row) in a single figure,
        plotting C_ell for every component (including frequency-independent ILC).
        The ILC map is loaded only once (at the first frequency) via `if idx==0`.

        Parameters:
            comps (list): List of components to visualise. e.g. ['sync', 'cmb', 'noise'].
        """
        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        file_templates = {}
        for comp in comps:
            file_templates[comp] = self.file_templates[comp]
        # build color map for components
        colours   = plt.rcParams['axes.prop_cycle'].by_key()['color']
        components = list(file_templates.keys())
        comp_color = {
            comp: colours[i % len(colours)]
            for i, comp in enumerate(components)
        }

        # layout: at most 3 columns
        n_freq = len(frequencies)
        ncols  = min(3, n_freq)
        nrows  = int(np.ceil(n_freq / 3))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            squeeze=False
        )
        fig.suptitle(
            f"Power spectra",
            fontsize=20, fontweight='bold')
        axes_flat = axes.flatten()

        # loop on frequencies
        for i, freq in enumerate(frequencies):
            ax = axes_flat[i]
            #ax.set_xlim(10, lmax)
            #ax.set_ylim(1E4, 1E5)
            #ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\ell$', fontsize = 14)
            ax.set_ylabel(r'$(\ell^2 C_\ell)/(2\pi)$ ($\mu K^2$)', fontsize = 14)
            ax.set_title(f"{freq} GHz", fontsize=14)

            for comp, template in file_templates.items():
                color = comp_color[comp]
                fp = template.format(
                        frequency=freq,
                        lmax=lmax,
                        realisation=realisation
                    )

                if comp == 'ilc_synthesised':
                    # load ILC only once at the first frequency
                    cl = self.compute_and_save_ilc_power_spec(self.file_templates['ilc_synthesised'].format(lmax = lmax, realisation=realisation))
                else:
                    hp_map, _ = hp.read_map(fp, h=True)
                    hp_map*=1E6
                    cl  = hp.sphtfunc.anafast(hp_map, lmax=lmax)
                    cl /= hp.pixwin(hp.get_nside(hp_map), lmax=lmax)**2  # correct for pixel window function
                ell = np.arange(len(cl))
                ell_sq_cl = (ell*(ell+1)*cl)/2*np.pi
                if freq == "030":
                    cl_copy = np.copy(cl)
                ax.plot(
                    ell, ell_sq_cl,
                    label=comp,
                    color=color,
                    linewidth=1.5
                )
                ax.set_ylim(5E3,1E5)

            ax.legend(title="Component", fontsize=10)
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # hide any unused subplots
        for ax in axes_flat[n_freq:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig('power_spectra.pdf')
        plt.show()

    def visualise_component_ratio_power_spectra(self, comp_a: str, comp_b: str):
        """
        Visualise the ratio of power spectra between two components across frequencies.

        Parameters:
            file_templates (dict): Mapping from component name to a Python format-string for its filename.
            comp_a (str): The first component to compare.
            comp_b (str): The second component to compare.
            ilc_output_path (str, optional): Path to save the ILC map if needed
        """
        file_templates = {}
        for comp in [comp_a, comp_b]:
            file_templates[comp] = self.file_templates[comp]
        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        ell = np.arange(lmax+1)

        # Figure layout
        n_freq = len(frequencies)
        ncols  = min(3, n_freq)
        nrows  = int(np.ceil(n_freq / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5*ncols, 4*nrows),
            squeeze=False
        )
        fig.suptitle(
            f"Power Spectrum Ratio: {comp_a} / {comp_b}",
            fontsize=20, fontweight='bold')
        axes_flat = axes.flatten()

        for idx, freq in enumerate(frequencies):
            ax = axes_flat[idx]
            fp_a = file_templates[comp_a].format(
                    frequency=freq,
                    lmax=lmax,
                    realisation=realisation
                )
            # --- load and compute spectrum for comp_a ---
            if comp_a == 'ilc_synthesised':
                # load ILC only once at the first frequency
                cl_a = self.compute_and_save_ilc_power_spec(fp_a.format(lmax = lmax, realisation=realisation))
            else:
                map_a = hp.read_map(fp_a)*1E6
                cl_a = hp.sphtfunc.anafast(map_a, lmax=lmax)

            # --- load and compute spectrum for comp_b ---
            fp_b = file_templates[comp_b].format(
                    frequency=freq,
                    lmax=lmax,
                    realisation=realisation
                )
            if comp_b == 'ilc_synthesised':
                cl_b = self.compute_and_save_ilc_power_spec(fp_b.format(lmax = lmax, realisation=realisation))
            else:
                map_b = hp.read_map(fp_b, verbose=False)*1E6
                cl_b = hp.sphtfunc.anafast(map_b, lmax=lmax)
            # --- ratio and plot ---
            ratio = cl_a / cl_b
            ax.plot(ell, ratio, linewidth=1.5)
            ax.set_xlim(1, lmax)
            ax.set_ylim(0.5,1.5)
            ax.set_xlabel(r'$\ell$', fontsize = 16)
            ax.set_ylabel("Ratio", fontsize = 16)
            ax.set_title(f"{freq} GHz")
            ax.axhline(1, ls=':', color='red')
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # hide any unused subplots
        for ax in axes_flat[n_freq:]:
            ax.set_visible(False)

        plt.tight_layout()
        #plt.savefig('plots/component_ratio_power_spectra.pdf')
        plt.show()

    def visualise_freq_ratio_spectra(self, comps: str, freq1: str, freq2: str):
        """
        Visualise the ratio of power spectra between two frequencies for specified components.

        Parameters:
            comps (list): List of components to visualise.
            freq1 (str): The first frequency to compare.
            freq2 (str): The second frequency to compare.
            file_templates (dict): Mapping from component name to a Python format-string for its filename.
        """
        file_templates = {}
        for comp in comps:
            file_templates[comp] = self.file_templates[comp]
        realisation = self.realisation
        lmax = self.lmax
        ell = np.arange(lmax+1)
        n_comp = len(comps)
        ncols  = min(3, n_comp)
        nrows  = int(np.ceil(n_comp / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5*ncols, 4*nrows),
            squeeze=False
        )
        fig.suptitle(
            f"Power spectrum frequency ratio: {freq1} GHz / {freq2} GHz",
            fontsize=20, fontweight='bold')
        axes_flat = axes.flatten()

        for idx, comp in enumerate(comps):
            ax = axes_flat[idx]

            # Load map at freq1
            fp1 = file_templates[comp].format(
                frequency=freq1,
                lmax=lmax,
                realisation=realisation
            )
            map1 = hp.read_map(fp1, verbose=False)
            cl1 = hp.sphtfunc.anafast(map1, lmax=lmax)

            # Load map at freq2
            fp2 = file_templates[comp].format(
                frequency=freq2,
                lmax=lmax,
                realisation=realisation
            )
            if fp2.endswith('.npy'):
                map2 = np.load(fp2)
            else:
                map2 = hp.read_map(fp2, verbose=False)
            cl2 = hp.sphtfunc.anafast(map2, lmax=lmax)

            # Ratio and plotting
            ratio = cl1 / cl2
            ax.plot(ell, ratio, linewidth=1.5)
            ax.set_xscale('log')
            ax.set_xlim(1, lmax)
            ax.set_ylim(ratio[1:].min()*0.9, ratio[1:].max()*1.1)
            ax.set_xlabel(r'$\ell$', fontsize = 16)
            ax.set_ylabel(
                f"Ratio", fontsize = 16
            )
            ax.set_title(comp.upper())
            ax.axhline(1, ls=':', color='red')
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # hide unused axes
        for ax in axes_flat[n_comp:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


frequencies = ["030", "044"]
realisation = 0
lmax = 1024
directory = "/Scratch/matthew/data/"
map_comps = ["cmb", "cfn", "ilc_synthesised"]

visualiser = Visualise(
    frequencies=frequencies,
    realisation=realisation,
    lmax=lmax,
    directory=directory
)

#visualiser.visualise_maps(map_comps)
visualiser.visualise_power_spectra(map_comps)
visualiser.visualise_component_ratio_power_spectra("cmb", "ilc_synthesised")

