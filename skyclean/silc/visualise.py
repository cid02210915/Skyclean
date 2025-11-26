import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from .map_tools import *
from .utils import *
from .file_templates import *
from .power_spec import *
from ..ml.inference import Inference

class Visualise(): 
    def __init__(self, inference: Inference, frequencies: list, realisation: int, lmax: int, lam_list: float = [2.0], directory: str = "data/",
                 rn: int = 30, batch_size: int = 32, epochs: int = 120, 
                 learning_rate: float = 1e-3, momentum: float = 0.9, chs: list = None, 
                 ):
        """
        Parameters:
            frequencies (list): List of frequencies to visualise.
            realisation (int): The realisation number to visualise.
            lmax (int): Maximum multipole for the visualisation.
            lam_list (list): List of lambda values for the ILC. 
            directory (str): Directory where data is stored / saved to.
        """
        self.inference = inference
        self.frequencies = frequencies
        self.realisation = realisation
        self.lmax = lmax
        self.directory = directory
        self.lam_list = lam_list
        self.lr = learning_rate
        self.rn = rn
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.chs = chs

        files = FileTemplates(directory)
        self.file_templates = files.file_templates

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
                if comp == "ilc_synth":
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
    
    def compute_and_save_mw_power_spec(self, map_path: str, component: str, lam: float = 2.0):
        """
        Compute and save the power spectrum from any MW-format map. 
        This is a generalized version that works with any component type.
        
        Parameters:
            map_path (str): File path to the MW map.
            component (str): Component name (e.g., 'ilc_synth', 'ilc_improved', etc.).
            lam (float): Lambda value for the processing.
        
        Returns:
            cl (np.ndarray): The computed power spectrum.
        """
        # Create spectrum template key by joining component name with 'spectrum'
        spectrum_template_key = f"{component}_spectrum"
        
        # Check if the specific spectrum template exists, otherwise use a default path
        if spectrum_template_key in self.file_templates:
            if component == 'ilc_improved':
                chs = "_".join(str(n) for n in self.chs)
                spectrum_path = self.file_templates[spectrum_template_key].format(
                                        realisation=self.realisation, 
                                        lmax=self.lmax,
                                        lam=lam,
                                        rn=self.rn,
                                        batch=self.batch_size,
                                        epochs=self.epochs,
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        chs=chs,)
            
            else:
                spectrum_path = self.file_templates[spectrum_template_key].format(
                    realisation=self.realisation, 
                    lmax=self.lmax, 
                    lam=lam,
                )
        else:
            if component == 'ilc_improved':
                chs = "_".join(str(n) for n in self.chs)
                map_dir = os.path.dirname(map_path)
                spectrum_filename = f"{component}_power_spectrum_r{self.realisation:04d}_lmax{self.lmax}_lam{lam}_rn{self.rn}_batch{self.batch_size}_epo{self.epochs}_lr{self.lr}_mom{self.momentum}_chs{chs}.npy"
                spectrum_path = os.path.join(map_dir, spectrum_filename)
            else:
                # Fallback: create path in the same directory as the map
                map_dir = os.path.dirname(map_path)
                spectrum_filename = f"{component}_power_spectrum_r{self.realisation:04d}_lmax{self.lmax}_lam{lam}.npy"
                spectrum_path = os.path.join(map_dir, spectrum_filename)
            
        # Check if spectrum already exists
        if os.path.exists(spectrum_path):
            cl = np.load(spectrum_path)
            print(f"Loaded existing {component} power spectrum from {spectrum_path}")
        else:
            print(f"Computing {component} power spectrum from {map_path}...")
            
            # Load MW map and convert to HEALPix
            mw_map = np.load(map_path)
            hp_map = SamplingConverters.mw_map_2_hp_map(mw_map, self.lmax) * 1E6
            
            # Compute power spectrum
            cl = hp.sphtfunc.anafast(hp_map, lmax=self.lmax)
            
            # Apply pixel window function correction
            #nside = HPTools.get_nside_from_lmax(self.lmax)
            #cl /= hp.pixwin(nside, lmax=self.lmax)**2
            
            # Save the computed spectrum
            np.save(spectrum_path, cl)
            print(f"Saved {component} power spectrum to {spectrum_path}")

        return cl

    def compute_power_spectrum_for_component(self, component: str, lam: float = 2.0):
        """
        Compute power spectrum for any component type.
        Determines whether to use MW or HEALPix processing.
        In MW case, spectrum is saved since computing it is computationally heavy.
        
        Parameters:
            component (str): Component name ('ilc_synth', 'ilc_improved', 'cmb', etc.)
            lam (float): Lambda value for MW components.
            
        Returns:
            cl (np.ndarray): The computed power spectrum.
        """
        mw_components = ['ilc_synth', 'ilc_improved']

        frequencies = self.frequencies
        
        if component in mw_components:
            # MW-format component - use the general MW power spectrum function
            map_path = self.file_templates[component].format(
                extract_comp='cmb',
                component='cfn',
                frequencies="_".join(str(x) for x in frequencies),
                realisation=self.realisation, 
                lmax=self.lmax, 
                lam=lam
            )
            return self.compute_and_save_mw_power_spec(map_path, component, lam)
        else:
            # HEALPix component - direct computation (no caching, it's fast)
            map_path = self.file_templates[component].format(
                realisation=self.realisation, 
                lmax=self.lmax,
                frequency=self.frequencies[0] if 'frequency' in self.file_templates[component] else None
            )
            
            print(f"Computing {component} power spectrum from {map_path}...")
            hp_map = hp.read_map(map_path, verbose=False) * 1E6
            cl = hp.sphtfunc.anafast(hp_map, lmax=self.lmax)
            #nside = hp.get_nside(hp_map)
            #cl /= hp.pixwin(nside, lmax=self.lmax)**2
                
            return cl


    def visualise_power_spectra(self, comps=['processed_cmb', 'cfn', 'ilc_synth'], all_freq = None, cross_correlation_indices=None):
        """
        Arrange one subplot per frequency (3 columns per row) in a single figure,
        plotting C_ell for every component (including frequency-independent ILC).
        Optionally includes cross-correlations between specified components.

        Parameters:
            comps (list): List of components to visualise. e.g. ['sync', 'cmb', 'noise'].
            cross_correlation_indices (list): List of pairs [i, j] where i and j are indices 
                                             into comps to compute cross-correlations between.
                                             E.g. [[0, 2]] for cross-correlation between comps[0] and comps[2].
        """
        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        
        lam = self.lam_list[0]
        
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        all_plot_items = comps.copy()
        
        # Add cross-correlation labels to the plots
        if cross_correlation_indices:
            for i, j in cross_correlation_indices:
                cross_label = f"{comps[i]} × {comps[j]}"
                all_plot_items.append(cross_label)
        
        comp_color = {
            item: colours[i % len(colours)]
            for i, item in enumerate(all_plot_items)
        }
        if all_freq: 
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
                #ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel(r'$\ell$', fontsize = 14)
                ax.set_ylabel(r'$D_{\ell}$ ($\mu K^2$)', fontsize = 14)
                ax.set_title(f"{freq} GHz", fontsize=14)
                ax.set_xlim(0,lmax)
                ax.set_ylim(10, 7000)
                #ax.set_ylim(1E4,1E5)

                # Store loaded maps for cross-correlation computation
                loaded_maps = {}

                # Plot auto-correlations for each component
                for comp in comps:
                    template = self.file_templates[comp]
                    color = comp_color[comp]
                    
                    # Define MW-format components
                    mw_components = ['ilc_synth', 'ilc_improved']
                    
                    # Handle MW-format components (ILC types)
                    if comp in mw_components:
                        # Loop over lambda values for MW components
                        for j, lam in enumerate(self.lam_list):
                            # Use the generalized convenience method
                            cl = self.compute_power_spectrum_for_component(comp, lam=lam)
                            ell = np.arange(len(cl))
                            D_ell = (ell*(ell+1)*cl)/(2*np.pi)
                            
                            # Use different line styles or colors for different lambda values
                            label = f"{comp} (λ={lam})" if len(self.lam_list) > 1 else comp
                            
                            ax.plot(
                                ell, D_ell,
                                label=label,
                                color=color,
                                linewidth=1.5
                            )
                            
                            # Store the map for potential cross-correlation
                            if cross_correlation_indices:
                                fp = template.format(
                                    realisation=realisation,
                                    lmax=lmax,
                                    lam=lam
                                )
                                ilc_mw = np.load(fp)
                                hp_ilc = SamplingConverters.mw_map_2_hp_map(ilc_mw, lmax)*1E6
                                # Only store the map for the first lambda value for cross-correlation
                                if j == 0:  # j is the index in the lambda loop
                                    loaded_maps[comp] = hp_ilc
                    else:
                        # Use the generalized convenience method for HEALPix components
                        cl = self.compute_power_spectrum_for_component(comp, lam=lam)
                        ell = np.arange(len(cl))
                        D_ell = (ell*(ell+1)*cl)/(2*np.pi)
                        
                        ax.plot(
                            ell, D_ell,
                            label=comp,
                            color=color,
                            linewidth=1.5
                        )
                        
                        # Store the map for potential cross-correlation
                        if cross_correlation_indices:
                            fp = template.format(
                                frequency=freq,
                                realisation=realisation,
                                lmax=lmax
                            )
                            hp_map, _ = hp.read_map(fp, h=True)
                            hp_map *= 1E6
                            loaded_maps[comp] = hp_map

                # Plot cross-correlations if requested
                if cross_correlation_indices:
                    for idx_i, idx_j in cross_correlation_indices:
                        comp_i = comps[idx_i]
                        comp_j = comps[idx_j]
                        cross_label = f"{comp_i} × {comp_j}"
                        color = comp_color[cross_label]
                        
                        # Cross-correlation between components (using first lambda for ILC)
                        map_i = loaded_maps[comp_i]
                        map_j = loaded_maps[comp_j]
                        
                        cl_cross = hp.sphtfunc.anafast(map_i, map2=map_j, lmax=lmax)
                        #cl_cross /= (hp.pixwin(hp.get_nside(map_i), lmax=lmax) * 
                        #        hp.pixwin(hp.get_nside(map_j), lmax=lmax))
                        
                        ell = np.arange(len(cl_cross))
                        D_ell = (ell*(ell+1)*cl_cross)/2*np.pi
                        
                        ax.plot(
                            ell, D_ell,
                            label=cross_label,
                            color=color,
                            linestyle='--',
                            linewidth=1.5
                        )

                ax.legend(title="Component", fontsize=10)
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)

            # hide any unused subplots
            for ax in axes_flat[n_freq:]:
                ax.set_visible(False)

            plt.tight_layout()
            #plt.savefig('spec.png')
            plt.show()

        else:
            lmax = self.lmax
            mw_components = ['ilc_synth', 'ilc_improved']

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r'$\ell$', fontsize=14)
            ax.set_ylabel(r'$D_{\ell}$ ($\mu K^2$)', fontsize=14)
            ax.set_xlim(0, lmax)
            #ax.set_yscale('log')  # optional
            ax.set_ylim(1e-1, 7000)  # optional, set if you need

            for comp in comps:
                color = comp_color[comp]

                if comp in mw_components:
                    # MW-format components: one curve per lambda
                    for lam in self.lam_list:
                        cl = self.compute_power_spectrum_for_component(comp, lam=lam)
                        ell = np.arange(len(cl))
                        D_ell = (ell * (ell + 1) * cl) / (2 * np.pi)

                        label = f"{comp} (λ={lam})" if len(self.lam_list) > 1 else comp
                        ax.plot(
                            ell, D_ell,
                            label=label,
                            color=color,
                            linewidth=1.5,
                        )

                else:
                    # HEALPix components: lam is ignored inside helper if not needed
                    cl = self.compute_power_spectrum_for_component(comp)
                    ell = np.arange(len(cl))
                    D_ell = (ell * (ell + 1) * cl) / (2 * np.pi)

                    ax.plot(
                        ell, D_ell,
                        label=comp,
                        color=color,
                        linewidth=1.5,
                    )

            ax.legend(title="Component", fontsize=14)
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)
            fig.tight_layout()
            plt.show()



    def visualise_component_ratio_power_spectra(
        self,
        comp_a,
        comp_b: str,
        include_cross_correlation: bool = False,
        ratio: bool = True,
        all_freq: bool = False,
    ):
        """
        Visualise the ratio or residual of power spectra between components and a reference
        component across frequencies, with two modes:

        - all_freq=True  (default): one subplot per frequency (current behaviour, single comp_a).
        - all_freq=False: a single figure, no subplots. Supports multiple comp_a values,
                        e.g. ['ilc_synth', 'ilc_improved'] vs 'cmb', plotted as multiple lines.

        Parameters:
            comp_a: str or list/tuple of str
                Component(s) to compare against comp_b.
                - If all_freq=True, only a single string is supported.
                - If all_freq=False, can be a list/tuple of components, each plotted as a line.
            comp_b (str): Reference component to compare with. e.g. 'processed_cmb'
            include_cross_correlation (bool): Only used when all_freq=True.
            ratio (bool): If True, compute D_ell(comp_a)/D_ell(comp_b).
                        If False, compute D_ell(comp_b) - D_ell(comp_a).
            all_freq (bool): 
                - True  → subplots per frequency (original behaviour).
                - False → single figure (no subplots), single frequency.
        """
        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        ell = np.arange(lmax + 1)
        rn = self.rn
        batch = self.batch_size
        epochs = self.epochs
        lr = self.lr
        momentum = self.momentum
        chs = chs="_".join(str(n) for n in self.chs)

        # Normalise comp_a into a list
        if isinstance(comp_a, str):
            comp_list = [comp_a]
        else:
            comp_list = list(comp_a)

        # MW-format components that require special handling
        mw_components = ['ilc_synth', 'ilc_improved']

        # -------------------------------------------------------------------------
        # MODE 1: all_freq = True  → one subplot per frequency (original behaviour)
        # -------------------------------------------------------------------------
        if all_freq:
            if len(comp_list) != 1:
                raise ValueError(
                    "When all_freq=True, only a single comp_a is supported. "
                    "For multiple components (e.g. ['ilc_synth', 'ilc_improved'] vs 'cmb'), "
                    "use all_freq=False."
                )

            comp_a_single = comp_list[0]
            lam = self.lam_list[0]  # used for cross-correlations in MW case

            # Figure layout
            n_freq = len(frequencies)
            ncols = min(3, n_freq)
            nrows = int(np.ceil(n_freq / ncols))
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(5 * ncols, 4 * nrows),
                squeeze=False
            )
            axes_flat = axes.flatten()

            for idx, freq in enumerate(frequencies):
                ax = axes_flat[idx]

                # Store loaded maps for cross-correlation computation
                loaded_maps = {}

                # Decide if we need lambda loop (ILC/MW components)
                if (comp_a_single in mw_components) or (comp_b in mw_components):
                    # Loop over lambda values when MW components are involved
                    for j, lam_val in enumerate(self.lam_list):
                        # --- load/compute spectrum for comp_a_single ---
                        if comp_a_single in mw_components:
                            fp_a = self.file_templates[comp_a_single].format(
                                frequency=freq,
                                lmax=lmax,
                                realisation=realisation,
                                lam=lam_val
                            )
                            cl_a = self.compute_and_save_mw_power_spec(
                                fp_a, component=comp_a_single, lam=lam_val
                            )

                            # Store map for cross-correlation (only first lambda)
                            if include_cross_correlation and j == 0:
                                mw_map = np.load(fp_a)
                                hp_map = SamplingConverters.mw_map_2_hp_map(
                                    mw_map, lmax
                                ) * 1E6
                                loaded_maps[comp_a_single] = hp_map
                        else:
                            fp_a = self.file_templates[comp_a_single].format(
                                frequency=freq,
                                lmax=lmax,
                                realisation=realisation
                            )
                            map_a = hp.read_map(fp_a) * 1E6
                            cl_a = hp.sphtfunc.anafast(map_a, lmax=lmax)

                            if include_cross_correlation and j == 0:
                                loaded_maps[comp_a_single] = map_a

                        # --- load/compute spectrum for comp_b ---
                        if comp_b in mw_components:
                            fp_b = self.file_templates[comp_b].format(
                                frequency=freq,
                                lmax=lmax,
                                realisation=realisation,
                                lam=lam_val
                            )
                            cl_b = self.compute_and_save_mw_power_spec(
                                fp_b, component=comp_b, lam=lam_val
                            )

                            if include_cross_correlation and j == 0:
                                mw_map = np.load(fp_b)
                                hp_map = SamplingConverters.mw_map_2_hp_map(
                                    mw_map, lmax
                                ) * 1E6
                                loaded_maps[comp_b] = hp_map
                        else:
                            fp_b = self.file_templates[comp_b].format(
                                frequency=freq,
                                lmax=lmax,
                                realisation=realisation
                            )
                            map_b = hp.read_map(fp_b, verbose=False) * 1E6
                            cl_b = hp.sphtfunc.anafast(map_b, lmax=lmax)

                            if include_cross_correlation and j == 0:
                                loaded_maps[comp_b] = map_b

                        # --- Auto-correlation ratio/residual ---
                        D_ell_a = (ell * (ell + 1) * cl_a) / (2 * np.pi)
                        D_ell_b = (ell * (ell + 1) * cl_b) / (2 * np.pi)

                        if ratio:
                            result = D_ell_a / D_ell_b
                            if len(self.lam_list) > 1:
                                label = f"Auto: λ={lam_val}"
                            else:
                                label = f"Auto: ({comp_a_single}×{comp_a_single})/({comp_b}×{comp_b})"
                        else:
                            result = D_ell_b - D_ell_a
                            if len(self.lam_list) > 1:
                                label = f"Residual: λ={lam_val}"
                            else:
                                label = f"({comp_b}-{comp_a_single})"

                        ax.plot(ell, result, linewidth=2.0, label=label, linestyle='-')

                else:
                    # Neither component is in MW: no lambda loop needed
                    freq = frequencies[idx]

                    fp_a = self.file_templates[comp_a_single].format(
                        frequency=freq,
                        lmax=lmax,
                        realisation=realisation
                    )
                    map_a = hp.read_map(fp_a) * 1E6
                    cl_a = hp.sphtfunc.anafast(map_a, lmax=lmax)

                    fp_b = self.file_templates[comp_b].format(
                        frequency=freq,
                        lmax=lmax,
                        realisation=realisation
                    )
                    map_b = hp.read_map(fp_b, verbose=False) * 1E6
                    cl_b = hp.sphtfunc.anafast(map_b, lmax=lmax)

                    D_ell_a = (ell * (ell + 1) * cl_a) / (2 * np.pi)
                    D_ell_b = (ell * (ell + 1) * cl_b) / (2 * np.pi)

                    if ratio:
                        result = D_ell_a / D_ell_b
                        label = f"Auto: ({comp_a_single}×{comp_a_single})/({comp_b}×{comp_b})"
                    else:
                        result = D_ell_b - D_ell_a
                        label = f"({comp_b}-{comp_a_single})"

                    ax.plot(ell, result, linewidth=2.0, label=label, linestyle='-')

                    # Store maps for cross-correlation
                    if include_cross_correlation:
                        loaded_maps[comp_a_single] = map_a
                        loaded_maps[comp_b] = map_b

                # Cross-correlation ratio/residual, if requested
                if include_cross_correlation:
                    map_a_cross = loaded_maps[comp_a_single]
                    map_b_cross = loaded_maps[comp_b]

                    cl_cross = hp.sphtfunc.anafast(
                        map_a_cross, map2=map_b_cross, lmax=lmax
                    )

                    # Auto-correlation of comp_b
                    if comp_b in mw_components:
                        fp_b_auto = self.file_templates[comp_b].format(
                            frequency=freq,
                            lmax=lmax,
                            realisation=realisation,
                            lam=lam
                        )
                        cl_b_auto = self.compute_and_save_mw_power_spec(
                            fp_b_auto, component=comp_b, lam=lam
                        )
                    else:
                        cl_b_auto = hp.sphtfunc.anafast(map_b_cross, lmax=lmax)

                    # Auto-correlation of comp_a_single
                    if comp_a_single in mw_components:
                        fp_a_auto = self.file_templates[comp_a_single].format(
                            frequency=freq,
                            lmax=lmax,
                            realisation=realisation,
                            lam=lam
                        )
                        cl_a_auto = self.compute_and_save_mw_power_spec(
                            fp_a_auto, component=comp_a_single, lam=lam
                        )
                    else:
                        cl_a_auto = hp.sphtfunc.anafast(map_a_cross, lmax=lmax)

                    D_ell_cross = (ell * (ell + 1) * cl_cross) / (2 * np.pi)
                    D_ell_b_auto = (ell * (ell + 1) * cl_b_auto) / (2 * np.pi)
                    D_ell_a_auto = (ell * (ell + 1) * cl_a_auto) / (2 * np.pi)

                    if ratio:
                        cross_result = D_ell_cross / D_ell_b_auto
                        cross_label = f"Cross: ({comp_a_single}×{comp_b})/({comp_b}×{comp_b})"
                    else:
                        cross_result = D_ell_cross - D_ell_a_auto
                        cross_label = f"Cross residual: ({comp_a_single}×{comp_b})-({comp_a_single}×{comp_a_single})"

                    ax.plot(
                        ell, cross_result, linewidth=2.0,
                        label=cross_label, linestyle='--', color='red'
                    )

                ax.set_xlim(1, lmax)
                if ratio:
                    ax.set_ylim(0.7, 1.2)
                    ax.set_ylabel("Ratio of $D_{\\ell}$", fontsize=14)
                    ax.axhline(1, ls=':', color='red')
                else:
                    ax.set_ylabel(r'$\Delta D_{\ell}$ ($\mu K^2$)', fontsize=14)
                    ax.axhline(0, ls=':', color='black', alpha=0.7)

                ax.set_xlabel(r'$\ell$', fontsize=14)
                ax.set_title(f"{freq} GHz")
                ax.grid(True, which='both', linestyle=':', linewidth=0.5)
                ax.legend(fontsize=9)

            # Hide any unused subplots
            for ax in axes_flat[n_freq:]:
                ax.set_visible(False)

            plt.tight_layout()
            filename = 'component_ratio_power_spectra.pdf' if ratio else 'component_residual_power_spectra.pdf'
            plt.savefig(filename)
            plt.show()
            return

        # -------------------------------------------------------------------------
        # MODE 2: all_freq = False  → single figure, multiple comp_a vs comp_b
        # -------------------------------------------------------------------------

        # Use a single representative frequency (you can change this if you want)
        freq = frequencies[0]
        lam0 = self.lam_list[0]

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, comp_ai in enumerate(comp_list):
            color = colours[i % len(colours)]

            # --- compute cl for comp_ai ---
            if comp_ai in mw_components:
                if comp_ai == 'ilc_improved':
                    fp_a = self.file_templates[comp_ai].format(
                        rn=rn,
                        batch=batch,
                        epochs=epochs,
                        lr=lr,
                        momentum=momentum,
                        chs=chs,
                        lmax=lmax,
                        realisation=realisation,
                        lam=lam0,
                    )
                elif comp_ai == 'ilc_synth':
                    frequencies = self.frequencies

                    fp_a = self.file_templates[comp_ai].format(
                        extract_comp="cmb",
                        component="cfn",
                        frequencies="_".join(str(x) for x in frequencies),
                        lmax=lmax,
                        realisation=realisation,
                        lam=lam0,)

                cl_a = self.compute_and_save_mw_power_spec(
                    fp_a, component=comp_ai, lam=lam0
                )
            else: # hp
                if comp_ai == 'processed_cmb':
                    fp_a = self.file_templates[comp_ai].format(
                        lmax=lmax,
                        realisation=realisation,
                    )
                map_a = hp.read_map(fp_a) * 1E6
                cl_a = hp.sphtfunc.anafast(map_a, lmax=lmax)

            # --- compute cl for comp_b ---
            if comp_b in mw_components:
                if comp_b == 'ilc_improved':
                    fp_b = self.file_templates[comp_b].format(
                        rn=rn,
                        batch=batch,
                        epochs=epochs,
                        lr=lr,
                        momentum=momentum,
                        chs=chs,
                        lmax=lmax,
                        realisation=realisation,
                        lam=lam0,
                        )
                elif comp_b == 'ilc_synth':
                    frequencies = self.frequencies

                    fp_b = self.file_templates[comp_b].format(
                        extract_comp="cmb",
                        component="cfn",
                        frequencies="_".join(str(x) for x in frequencies),
                        lmax=lmax,
                        realisation=realisation,
                        lam=lam0,)
                
                cl_b = self.compute_and_save_mw_power_spec(
                    fp_b, component=comp_b, lam=lam0
                )
            else:
                if comp_b == 'processed_cmb':
                    fp_b = self.file_templates[comp_b].format(
                        lmax=lmax,
                        realisation=realisation,)
                map_b = hp.read_map(fp_b, verbose=False) * 1E6
                cl_b = hp.sphtfunc.anafast(map_b, lmax=lmax)

            D_ell_a = (ell * (ell + 1) * cl_a) / (2 * np.pi)
            D_ell_b = (ell * (ell + 1) * cl_b) / (2 * np.pi)

            if ratio:
                result = D_ell_a / D_ell_b
                label = f"{comp_ai}/{comp_b}"
            else:
                result = D_ell_b - D_ell_a
                label = f"{comp_b}-{comp_ai}"

            ax.plot(
                ell, result,
                linewidth=2.0,
                label=label,
                linestyle='-',
                color=color,
            )

        ax.set_xlim(1, lmax)
        if ratio:
            ax.set_ylim(0.5, 1.5)
            ax.set_ylabel("Ratio of $D_{\\ell}$", fontsize=14)
            ax.axhline(1, ls=':', color='red')
        else:
            ax.set_ylabel(r'$\Delta D_{\ell}$ ($\mu K^2$)', fontsize=14)
            ax.axhline(0, ls=':', color='black', alpha=0.7)

        ax.set_xlabel(r'$\ell$', fontsize=14)
        ax.set_title(f"Processed CMB vs cILC-synth ratio (lam={lam0})")
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.legend(fontsize=14)
        fig.tight_layout()
        plt.show()
    


    def visualise_cross_spectrum(
        self,
        comp_pairs,
        use_Dl: bool = True,
        logy: bool = False,
        loglog: bool = False,
        input_unit: str = "uK",
        ):
        """
        Visualise cross power spectra C_ell^{XY} (or D_ell^{XY}) for one or more
        component pairs.

        Parameters
        ----------
        comp_pairs : tuple(str, str) or list[tuple(str, str)]
            One or more (comp_x, comp_y) pairs to cross-correlate.
            Example:
                ('ilc_synth', 'processed_cmb')
                [('ilc_synth', 'processed_cmb'),
                 ('ilc_improved', 'processed_cmb')]
        use_Dl : bool
            If True, plot D_ell = ell(ell+1) C_ell / (2π) (µK² if input_unit="K").
            If False, plot C_ell.
        logy : bool
            If True, use log scale in y axis.
        loglog : bool
            If True, use log–log axes.
        input_unit : {"K", ...}
            Passed to PowerSpectrumCrossTT.cl_to_Dl.
        """

        # normalise comp_pairs to a list of pairs
        if isinstance(comp_pairs, tuple) and len(comp_pairs) == 2:
            pair_list = [comp_pairs]
        else:
            pair_list = list(comp_pairs)

        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        rn = self.rn
        batch = self.batch_size
        epochs = self.epochs
        lr = self.lr
        momentum = self.momentum
        chs = "_".join(str(n) for n in self.chs)
        lam = self.lam_list[0]

        mw_components = ["ilc_synth", "ilc_improved"]
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, ax = plt.subplots(figsize=(8, 6))

        # ---- helpers: return HEALPix map in µK for any component ----
        def get_mw_hp_map(component: str):
            """Load MW-format map from disk and convert to HEALPix, in µK."""
            if component == "ilc_improved":
                map_path = self.file_templates[component].format(
                    rn=rn,
                    batch=batch,
                    epochs=epochs,
                    lr=lr,
                    momentum=momentum,
                    chs=chs,
                    realisation=realisation,
                    lmax=lmax,
                    lam=lam,
                )
            elif component == "ilc_synth":
                map_path = self.file_templates[component].format(
                    extract_comp="cmb",
                    component="cfn",
                    frequencies="_".join(str(x) for x in frequencies),
                    realisation=realisation,
                    lmax=lmax,
                    lam=lam,
                )
            else:
                raise ValueError(f"Unknown MW component '{component}'")

            mw_map = np.load(map_path)
            hp_map = SamplingConverters.mw_map_2_hp_map(mw_map, lmax) * 1e6
            return hp_map

        def get_hp_map(component: str):
            """Load HEALPix map for a component, in µK."""
            template = self.file_templates[component]
            # crude but matches your current compute_power_spectrum_for_component
            if "frequency" in template:
                map_path = template.format(
                    realisation=realisation,
                    lmax=lmax,
                    frequency=frequencies[0],
                )
            else:
                map_path = template.format(
                    realisation=realisation,
                    lmax=lmax,
                )

            hp_map = hp.read_map(map_path, verbose=False) * 1e6
            return hp_map

        # ---- loop over pairs and plot cross spectra ----
        for i, (comp_x, comp_y) in enumerate(pair_list):
            color = colours[i % len(colours)]

            # pick the right loader for each component independently
            if comp_x in mw_components:
                hp_map_x = get_mw_hp_map(comp_x)
            else:
                hp_map_x = get_hp_map(comp_x)

            if comp_y in mw_components:
                hp_map_y = get_mw_hp_map(comp_y)
            else:
                hp_map_y = get_hp_map(comp_y)

            # cross-spectrum C_ell^{XY}
            Cl_xy = hp.sphtfunc.anafast(hp_map_x, map2=hp_map_y, lmax=lmax)
            ell = np.arange(Cl_xy.size)

            if use_Dl:
                yvals = PowerSpectrumCrossTT.cl_to_Dl(ell, Cl_xy, input_unit=input_unit)
                ylabel = (
                    r"$\ell(\ell+1)C_\ell^{XY}/2\pi\,[\mu\mathrm{K}^2]$"
                    if str(input_unit).lower() in ("k", "kelvin")
                    else r"$\ell(\ell+1)C_\ell^{XY}/2\pi ($\mu K^2$)$"
                )
            else:
                yvals = Cl_xy
                ylabel = r"$C_\ell^{XY} (\mu{K^2})$"

            ax.plot(
                ell,
                yvals,
                linestyle="-",
                linewidth=1.0,
                color=color,
                label=f"{comp_x} × {comp_y}",
            )

        # ---- formatting ----
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e2, 1e4)
        if logy:
            ax.set_yscale("log")

        ax.set_xlabel(r"$\ell$", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(f"power spectrum (lam={lam})", fontsize=16)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(fontsize=12)
        fig.tight_layout()

        plt.show()


    def visualise_mse_scatter(self, n_realisations: int, split=(0.8, 0.2)):
        """
        Plot MSE(NN) vs MSE(ILC) for an arbitrary number of realisations,
        with a train/validation split.

        Parameters
        ----------
        n_realisations : int
            Total number of realisations to evaluate (0..n_realisations-1).
        split : float or tuple/list
            Train/validation split fraction.
            - If float, interpreted as train fraction (e.g. 0.8).
            - If tuple/list, uses split[0] as train fraction.
              Example: split=(0.8, 0.2).
        """

        # --- interpret split argument ---
        if isinstance(split, (list, tuple, np.ndarray)):
            train_frac = float(split[0])
        else:
            train_frac = float(split)

        train_frac = max(0.0, min(1.0, train_frac))  # clamp to [0,1]

        # --- build index arrays ---
        idx = np.arange(n_realisations)
        cut = int(train_frac * n_realisations)
        train_idx = idx[:cut]
        val_idx = idx[cut:]

        print(f"Using {len(train_idx)} train and {len(val_idx)} validation realisations.")

        # --- compute MSE(ILC) and MSE(NN) for all realisations ---
        mse_ilc = np.empty(n_realisations, dtype=float)
        mse_nn  = np.empty(n_realisations, dtype=float)

        for r in idx:
            mse_ilc[r] = self.inference.compute_mse("ilc", r) * 1E12
            mse_nn[r]  = self.inference.compute_mse("nn",  r) * 1E12 

        # --- prepare scatter plot ---
        fig, ax = plt.subplots(figsize=(7, 6))

        # train points
        if len(train_idx) > 0:
            ax.loglog(
                mse_ilc[train_idx],
                mse_nn[train_idx],
                'o',
                color='blue',
                alpha=0.6,
                label="train",
            )

        # validation points
        if len(val_idx) > 0:
            ax.loglog(
                mse_ilc[val_idx],
                mse_nn[val_idx],
                's',
                color='red',
                alpha=0.8,
                label="validation",
            )

        # y = x line
        #xmin = mse_ilc.min()
        #xmax = mse_ilc.max()
        #ymin = mse_nn.min()
        #ymax = mse_nn.max()
        #lo = min(xmin, ymin)
        #hi = max(xmax, ymax)
        lo = 5E1
        hi = 0.5E4

        xline = np.logspace(np.log10(lo), np.log10(hi), 100)
        ax.loglog(xline, xline, 'k--', linewidth=1.0, label=r"$y=x$")

        ax.set_xlabel(r"MSE(ILC) $[\mu K^2]$", fontsize=14)
        ax.set_ylabel(r"MSE(NN) $[\mu K^2]$", fontsize=14)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        try:
            title_lam = self.lam
        except AttributeError:
            title_lam = None

        if title_lam is not None:
            ax.set_title(f"MSE(NN) vs MSE(ILC) (lam={title_lam})", fontsize=14)
        else:
            ax.set_title("MSE(NN) vs MSE(ILC)", fontsize=14)

        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(fontsize=14)
        fig.tight_layout()
        plt.show()
        

# frequencies = ["030"]
# realisation = 0
# lmax = 511
# lam_list = [2.0]
# directory = "/Scratch/matthew/data/"
# map_comps = ["ilc_synth", "ilc_improved", "cmb"]

# visualiser = Visualise(
#     frequencies=frequencies,
#     realisation=realisation,
#     lmax=lmax,
#     lam_list=lam_list,
#     directory=directory
# )

# #visualiser.visualise_maps(map_comps)
# #visualiser.visualise_power_spectra(map_comps, cross_correlation_indices=[[0, 1]])  
# visualiser.visualise_component_ratio_power_spectra("cmb","ilc_synth", include_cross_correlation=True, ratio=False)
