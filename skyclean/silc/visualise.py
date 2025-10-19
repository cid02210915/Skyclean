import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from .map_tools import *
from .utils import *
from .file_templates import *

class Visualise(): 
    def __init__(self, frequencies: list, realisation: int, lmax: int, lam_list: float = [2.0], directory: str = "data/"):
        """
        Parameters:
            frequencies (list): List of frequencies to visualise.
            realisation (int): The realisation number to visualise.
            lmax (int): Maximum multipole for the visualisation.
            lam_list (list): List of lambda values for the ILC. 
            directory (str): Directory where data is stored / saved to.
        """
        self.frequencies = frequencies
        self.realisation = realisation
        self.lmax = lmax
        self.directory = directory
        self.lam_list = lam_list

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
            spectrum_path = self.file_templates[spectrum_template_key].format(
                realisation=self.realisation, 
                lmax=self.lmax, 
                lam=lam
            )
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
            nside = HPTools.get_nside_from_lmax(self.lmax)
            cl /= hp.pixwin(nside, lmax=self.lmax)**2
            
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
        
        if component in mw_components:
            # MW-format component - use the general MW power spectrum function
            map_path = self.file_templates[component].format(
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
            nside = hp.get_nside(hp_map)
            cl /= hp.pixwin(nside, lmax=self.lmax)**2
                
            return cl


    def visualise_power_spectra(self, comps=['cmb', 'cfn', 'ilc_synth'], cross_correlation_indices=None):
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
                            linewidth=2.0
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
                        linewidth=2.0
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
                    cl_cross /= (hp.pixwin(hp.get_nside(map_i), lmax=lmax) * 
                               hp.pixwin(hp.get_nside(map_j), lmax=lmax))
                    
                    ell = np.arange(len(cl_cross))
                    D_ell = (ell*(ell+1)*cl_cross)/2*np.pi
                    
                    ax.plot(
                        ell, D_ell,
                        label=cross_label,
                        color=color,
                        linestyle='--',
                        linewidth=2.0
                    )

            ax.legend(title="Component", fontsize=10)
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # hide any unused subplots
        for ax in axes_flat[n_freq:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig('spec.png')
        plt.show()

    def visualise_component_ratio_power_spectra(self, comp_a: str, comp_b: str, include_cross_correlation: bool = True, ratio: bool = True):
        """
        Visualise the ratio or residual of power spectra between two components across frequencies.
        Optionally includes cross-correlation analysis.

        Parameters:
            comp_a (str): The first component to compare.
            comp_b (str): The second component to compare.
            include_cross_correlation (bool): Whether to include cross-correlation analysis.
            ratio (bool): If True, calculate ratios (comp_a/comp_b). If False, calculate residuals (comp_b - comp_a).
        """
        frequencies = self.frequencies
        realisation = self.realisation
        lmax = self.lmax
        ell = np.arange(lmax+1)
        
        # Define MW-format components that require special handling
        mw_components = ['ilc_synth', 'ilc_improved']
        
        # Use only the first lambda value for cross-correlations
        lam = self.lam_list[0]

        # Figure layout
        n_freq = len(frequencies)
        ncols  = min(3, n_freq)
        nrows  = int(np.ceil(n_freq / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5*ncols, 4*nrows),
            squeeze=False
        )
        axes_flat = axes.flatten()

        for idx, freq in enumerate(frequencies):
            ax = axes_flat[idx]
            
            # Store loaded maps for cross-correlation computation
            loaded_maps = {}
            
            # Check if either component is a MW-format component to determine if we need to loop over lambda
            mw_components = ['ilc_synth', 'ilc_improved']
            if comp_a in mw_components or comp_b in mw_components:
                # Loop over lambda values when MW components are involved (for auto-correlation ratios)
                for j, lam_val in enumerate(self.lam_list):
                    # --- load and compute spectrum for comp_a ---
                    if comp_a in mw_components:
                        fp_a = self.file_templates[comp_a].format(
                            frequency=freq,
                            lmax=lmax,
                            realisation=realisation,
                            lam=lam_val
                        )
                        cl_a = self.compute_and_save_mw_power_spec(fp_a, component=comp_a, lam=lam_val)
                        
                        # Store map for cross-correlation (only first lambda)
                        if include_cross_correlation and j == 0:
                            mw_map = np.load(fp_a)
                            hp_map = SamplingConverters.mw_map_2_hp_map(mw_map, lmax)*1E6
                            loaded_maps[comp_a] = hp_map
                    else:
                        fp_a = self.file_templates[comp_a].format(
                            frequency=freq,
                            lmax=lmax,
                            realisation=realisation
                        )
                        map_a = hp.read_map(fp_a)*1E6
                        cl_a = hp.sphtfunc.anafast(map_a, lmax=lmax)
                        
                        # Store map for cross-correlation (only once)
                        if include_cross_correlation and j == 0:
                            loaded_maps[comp_a] = map_a

                    # --- load and compute spectrum for comp_b ---
                    if comp_b in mw_components:
                        fp_b = self.file_templates[comp_b].format(
                            frequency=freq,
                            lmax=lmax,
                            realisation=realisation,
                            lam=lam_val
                        )
                        cl_b = self.compute_and_save_mw_power_spec(fp_b, component=comp_b, lam=lam_val)
                        
                        # Store map for cross-correlation (only first lambda)
                        if include_cross_correlation and j == 0:
                            mw_map = np.load(fp_b)
                            hp_map = SamplingConverters.mw_map_2_hp_map(mw_map, lmax)*1E6
                            loaded_maps[comp_b] = hp_map
                    else:
                        fp_b = self.file_templates[comp_b].format(
                            frequency=freq,
                            lmax=lmax,
                            realisation=realisation
                        )
                        map_b = hp.read_map(fp_b, verbose=False)*1E6
                        cl_b = hp.sphtfunc.anafast(map_b, lmax=lmax)
                        
                        # Store map for cross-correlation (only once)
                        if include_cross_correlation and j == 0:
                            loaded_maps[comp_b] = map_b
                    
                    # --- Auto-correlation ratio/residual and plot ---
                    D_ell_a = (ell * (ell + 1) * cl_a) / (2 * np.pi)
                    D_ell_b = (ell * (ell + 1) * cl_b) / (2 * np.pi)
                    
                    if ratio:
                        result = D_ell_a / D_ell_b
                        label = f"Auto: $\\lambda$={lam_val}" if len(self.lam_list) > 1 else f"Auto: ({comp_a}×{comp_a})/({comp_b}×{comp_b})"
                    else:
                        result = D_ell_b - D_ell_a
                        label = f"Residual: $\\lambda$={lam_val}" if len(self.lam_list) > 1 else f"({comp_b}-{comp_a})"
                    
                    ax.plot(ell, result, linewidth=2.0, label=label, linestyle='-')
            else:
                # Neither component is ILC, so no lambda loop needed
                fp_a = self.file_templates[comp_a].format(
                    frequency=freq,
                    lmax=lmax,
                    realisation=realisation
                )
                map_a = hp.read_map(fp_a)*1E6
                cl_a = hp.sphtfunc.anafast(map_a, lmax=lmax)

                fp_b = self.file_templates[comp_b].format(
                    frequency=freq,
                    lmax=lmax,
                    realisation=realisation
                )
                map_b = hp.read_map(fp_b, verbose=False)*1E6
                cl_b = hp.sphtfunc.anafast(map_b, lmax=lmax)
                
                # --- Auto-correlation ratio/residual and plot ---
                D_ell_a = (ell * (ell + 1) * cl_a) / (2 * np.pi)
                D_ell_b = (ell * (ell + 1) * cl_b) / (2 * np.pi)
                
                if ratio:
                    result = D_ell_a / D_ell_b
                    label = f"Auto: ({comp_a}×{comp_a})/({comp_b}×{comp_b})"
                else:
                    result = D_ell_b - D_ell_a
                    label = f"({comp_b}-{comp_a})"
                    
                ax.plot(ell, result, linewidth=2.0, label=label, linestyle='-')
                
                # Store maps for cross-correlation
                if include_cross_correlation:
                    loaded_maps[comp_a] = map_a
                    loaded_maps[comp_b] = map_b
            
            # Plot cross-correlation ratio if requested
            if include_cross_correlation:
                # Cross-correlation between comp_a and comp_b
                map_a_cross = loaded_maps[comp_a]
                map_b_cross = loaded_maps[comp_b]
                
                cl_cross = hp.sphtfunc.anafast(map_a_cross, map2=map_b_cross, lmax=lmax)
                cl_cross /= (hp.pixwin(hp.get_nside(map_a_cross), lmax=lmax) * 
                           hp.pixwin(hp.get_nside(map_b_cross), lmax=lmax))
                
                # Auto-correlation of comp_b (using first lambda if MW component)
                if comp_b in mw_components:
                    fp_b_auto = self.file_templates[comp_b].format(
                        frequency=freq,
                        lmax=lmax,
                        realisation=realisation,
                        lam=lam
                    )
                    cl_b_auto = self.compute_and_save_mw_power_spec(fp_b_auto, component=comp_b, lam=lam)
                else:
                    cl_b_auto = hp.sphtfunc.anafast(map_b_cross, lmax=lmax)
                    cl_b_auto /= hp.pixwin(hp.get_nside(map_b_cross), lmax=lmax)**2
                
                # Auto-correlation of comp_a (using first lambda if MW component)
                if comp_a in mw_components:
                    fp_a_auto = self.file_templates[comp_a].format(
                        frequency=freq,
                        lmax=lmax,
                        realisation=realisation,
                        lam=lam
                    )
                    cl_a_auto = self.compute_and_save_mw_power_spec(fp_a_auto, component=comp_a, lam=lam)
                else:
                    cl_a_auto = hp.sphtfunc.anafast(map_a_cross, lmax=lmax)
                    cl_a_auto /= hp.pixwin(hp.get_nside(map_a_cross), lmax=lmax)**2
                
                # Cross-correlation / auto-correlation ratio or residual
                D_ell_cross = (ell * (ell + 1) * cl_cross) / (2 * np.pi)
                D_ell_b_auto = (ell * (ell + 1) * cl_b_auto) / (2 * np.pi)
                D_ell_a_auto = (ell * (ell + 1) * cl_a_auto) / (2 * np.pi)
                
                if ratio:
                    cross_result = D_ell_cross / D_ell_b_auto
                    cross_label = f"Cross: ({comp_a}×{comp_b})/({comp_b}×{comp_b})"
                else:
                    # Cross-correlation residual: cross_correlation - comp_a_auto
                    cross_result = D_ell_cross - D_ell_a_auto
                    cross_label = f"Cross residual: ({comp_a}×{comp_b})-({comp_a}×{comp_a})"
                
                ax.plot(ell, cross_result, linewidth=2.0, 
                       label=cross_label, 
                       linestyle='--', color='red')
            
            ax.set_xlim(1, lmax)
            if ratio:
                ax.set_ylim(0.7, 1.2)
                ax.set_ylabel("Ratio of $D_{\\ell}$", fontsize=16)
                ax.axhline(1, ls=':', color='red')
            else:
                ax.set_ylabel(r'$\Delta D_{\ell}$ ($\mu K^2$)', fontsize=16)
                ax.axhline(0, ls=':', color='black', alpha=0.7)
            
            ax.set_xlabel(r'$\ell$', fontsize=16)
            ax.set_title(f"{freq} GHz")
            ax.grid(True, which='both', linestyle=':', linewidth=0.5)
            
            # Add legend if multiple lambda values are plotted or cross-correlation is included
            if ((comp_a == 'ilc_synth' or comp_b == 'ilc_synth') and len(self.lam_list) > 1) or include_cross_correlation:
                ax.legend(fontsize=10)

        # hide any unused subplots
        for ax in axes_flat[n_freq:]:
            ax.set_visible(False)
        plt.legend(loc = 'upper right')
        plt.tight_layout()
        filename = 'component_ratio_power_spectra.pdf' if ratio else 'component_residual_power_spectra.pdf'
        plt.savefig(filename)
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
