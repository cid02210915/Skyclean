import numpy as np
import matplotlib.pyplot as plt

class AxisymmetricGenerators:
    """
    Object-oriented version of the original generator builder.
    Keeps the same content/logic, just organized into a class.

    Methods exposed:
      - s(t):      C∞ bump on [-1, 1]           (@staticmethod)
      - s_lam(t):  shifted/scaled bump on [1/lam, 1]
      - k_lam(t):  smooth taper (1 below 1/lam, 0 above 1), monotone on [1/lam, 1]
      - kappa(t):  sqrt( k_lam(t/lam) - k_lam(t) )
      - eta(t):    sqrt( k_lam(t) )
    """
    def __init__(self, lam: float):
        assert lam > 1.0, "λ must be > 1"
        self.lam = float(lam)

        # Affine map coefficients for s_lam
        self.a = 2.0 * self.lam / (self.lam - 1.0)
        self.b = - (self.lam + 1.0) / (self.lam - 1.0)

        # Precompute k_λ(t) ingredients on a dense grid in [1/λ, 1]
        self.t_grid = np.linspace(1.0/self.lam, 1.0, 80001)
        w = self.s_lam(self.t_grid)**2 / self.t_grid
        dt = self.t_grid[1] - self.t_grid[0]
        # Trapezoidal cumulative integral from left
        self.integ_left = np.zeros_like(w)
        self.integ_left[1:] = np.cumsum((w[:-1] + w[1:]) * 0.5 * dt)
        # ∫_{1/λ}^1 (s_λ(t')^2 / t') dt'
        self.total = float(self.integ_left[-1])

    @staticmethod
    def s(t):
        """Base C∞ bump s(t) with support [-1, 1]."""
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        mask = (t >= -1.0) & (t <= 1.0)
        tm = t[mask]
        out[mask] = np.exp(-1.0 / (1.0 - tm * tm))
        return out

    def s_lam(self, t):
        """Affine map: t ∈ [1/λ, 1] → u ∈ [-1, 1]; then s(u)."""
        t = np.asarray(t, dtype=float)
        u = self.a * t + self.b
        return self.s(u)

    def k_lam(self, t):
        """
        k_λ(t) = ∫_t^1 (s_λ(t')^2 / t') dt'  /  ∫_{1/λ}^1 (s_λ(t')^2 / t') dt'
        with clamping:
          k_λ(t) = 1 for t < 1/λ
          k_λ(t) = 0 for t > 1
        """
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        # Regions
        left = t < (1.0/self.lam)
        right = t > 1.0
        mid = ~(left | right)

        out[left] = 1.0
        out[right] = 0.0
        if np.any(mid):
            # integral from 1/λ to t
            integ_to_t = np.interp(t[mid], self.t_grid, self.integ_left)
            # integral from t to 1 is total - integ_to_t
            num = self.total - integ_to_t
            out[mid] = num / self.total if self.total > 0 else 0.0
        return out

    def kappa(self, t):
        t = np.asarray(t)
        x = self.k_lam(t/self.lam) - self.k_lam(t)
        return np.sqrt(np.maximum(x, 0.0))
    
    def eta(self, t):
        t = np.asarray(t)
        return np.sqrt(np.maximum(self.k_lam(t), 0.0))


class HarmonicWindows:
    """
    Build windows on demand from AxisymmetricGenerators.
    Inputs:
      L   : band-limit (ℓ = 0..L-1)
      lam : same λ used to build the generators
      J0  : lowest wavelet scale index (0 ≤ J0 ≤ J)
      g   : AxisymmetricGenerators(lam) instance
    """
    def __init__(self, L, lam=2.0, J0=0):
        self.L   = int(L)
        self.lam = float(lam)
        self.J0  = int(J0)
        self.g   = AxisymmetricGenerators(self.lam)
        self.ells = np.arange(self.L)
        self.J = int(np.ceil(np.log(max(1, self.L - 1)) / np.log(self.lam)))

    def scaling(self):
        # Φ_{ℓ0} = sqrt((2ℓ+1)/(8π**2)) * η_λ(ℓ / λ^{J0})
        return np.sqrt((2.0*self.ells + 1.0)/(8.0*np.pi**2)) * \
               self.g.eta(self.ells / (self.lam**self.J0))

    def wavelet(self, j):
        # Ψ_{j;ℓ0} = sqrt((2ℓ+1)/(8π**2)) * κ_λ(ℓ / λ^{j})
        return np.sqrt((2.0*self.ells + 1.0)/(8.0*np.pi**2)) * \
               self.g.kappa(self.ells / (self.lam**j))


def admissibility(Phi_l0, Psi_j_l0, ells, tol=1e-6):
    """
    Compute S_ell = (4π/(2ℓ+1)) ( |Φ_{ℓ0}|^2 + Σ_j |Ψ_{j;ℓ0}|^2 ) for ℓ=0..L-1
    and check admissibility: |S_ell - 1| < tol for all ℓ ≥ 1.

    Returns
    -------
    S : np.ndarray, shape (L,)
        The admissibility sum over ℓ.
    ok : bool
        True if admissibility holds (excluding ℓ=0), else False.
    """
    ells = np.asarray(ells)
    S = np.abs(Phi_l0)**2
    for W in Psi_j_l0.values():
        S += np.abs(W)**2
    S = (8.0*np.pi**2)/(2.0*ells + 1.0) * S

    err = np.abs(S - 1.0)
    ok = bool(np.all(err < tol))
    if not ok:
        i = int(np.argmax(err))
        print(f"[admissibility] FAIL: max |S-1|={err[i]:.3e} at ℓ={int(ells[i])}")
        for k in np.where(err >= tol)[0][:10]:
            print(f"  ℓ={int(ells[k])}  S={S[k]:.8f}  |S-1|={err[k]:.3e}")
    return S, ok


class HRFigures:
    """
    Plot the generator curves using existing content.
    """
    def __init__(self, generators: AxisymmetricGenerators):
        self.g = generators

    def generators(self, linspace_t=(0.0, 1.5, 2000)):
        t = np.linspace(*linspace_t)

        # k_λ(t)
        plt.figure()
        plt.title(f"k_lam(t)   (λ={self.g.lam})")
        plt.plot(t, self.g.k_lam(t))   # <-- was k_lam
        plt.xlabel("t"); plt.ylabel("k_lam(t)")
        plt.grid(True)
        plt.show()

        # κ_λ(t)
        plt.figure()
        plt.title(f"kappa_lambda(t)   (λ={self.g.lam})")
        plt.plot(t, self.g.kappa(t))
        plt.xlabel("t"); plt.ylabel("kappa_lambda(t)")
        plt.grid(True)
        plt.show()

        # η_λ(t)
        plt.figure()
        plt.title(f"eta_lambda(t)   (λ={self.g.lam})")
        plt.plot(t, self.g.eta(t))
        plt.xlabel("t"); plt.ylabel("eta_lambda(t)")
        plt.grid(True)
        plt.show()


    @staticmethod
    def visualise_generating_functions(L: int, lam: float | None = None, J0: int = 0):
        ells = np.arange(L)
        hw   = HarmonicWindows(L=L, lam=lam, J0=J0)

        bands = [] 

        # scaling generating windows
        eta = hw.g.eta(ells / (lam**J0))
        plt.plot(ells, eta, '--', label='Scal.')

        # band-limits for scaling
        if np.max(np.abs(eta)) != 0:
            w = np.abs(eta)
            thr = 1e-3 * w.max()
            supp = np.where(w > thr)[0]
            ell_min = int(supp[0])
            ell_max = int(supp[-1])
            ell_peak = int(ells[np.argmax(w)])
            bands.append(("Scal.", ell_min, ell_peak, ell_max))

        # wavelet generating windows
        for j in range(J0, hw.J + 1):
            kappa = hw.g.kappa(ells / (lam**j))
            plt.plot(ells, kappa, label=f'j={j}')

            # band-limits for this wavelet
            if np.max(np.abs(kappa)) == 0:
                continue
            w = np.abs(kappa)
            thr = 1e-3 * w.max()
            supp = np.where(w > thr)[0]
            ell_min = int(supp[0])
            ell_max = int(supp[-1])
            ell_peak = int(ells[np.argmax(w)])
            bands.append((j, ell_min, ell_peak, ell_max))

        plt.xlabel('ℓ')
        plt.ylabel('Generating response')
        plt.grid(ls=':')
        plt.legend()
        plt.title(f'Generating functions (λ={lam}, J0={J0}, L={L})')
        plt.show()

        # print table 
        print("Generating fn   ell_min^j   ell_peak^j   ell_max^j")
        for label, ell_min, ell_peak, ell_max in bands:
            print(f"{str(label):>13s}   {ell_min:9d}   {ell_peak:11d}   {ell_max:9d}")


    @staticmethod
    def visualise_harmonic_generators(L: int, lam: float | None = None, J0: int = 0):
        hw   = HarmonicWindows(L=L, lam=lam, J0=J0)
        ells = hw.ells  # 0..L-1

        # scaling Φ_{ℓ0}
        plt.plot(ells, hw.scaling(), '--', label='Scal.')

        # wavelets Ψ_{j;ℓ0}
        for j in range(J0, hw.J + 1):
            plt.plot(ells, hw.wavelet(j), label=f'j={j}')

        plt.xlabel('ℓ'); plt.ylabel('Response (with √((2ℓ+1)/(4π)))')
        plt.grid(ls=':'); plt.legend()
        plt.title(f'Harmonic Response (λ={lam}, J0={J0}, L={L})')
        plt.xlim(0, 10)   
        plt.ylim(0, 1)  
        plt.show()

def cosine_taper(L, ell_taper, ell_max=None):
    """
    Raised-cosine taper in multipole l.

    Define the window w(l) by
        w(l) = 1
            for  l ≤ l_taper

        w(l) = ½ [ 1 + cos( π (l - l_taper) / (l_max - l_taper) ) ]
            for  l_taper < l < l_max

        w(l) = 0
            for  l ≥ l_max

    Parameters
    ----------
    L : int
        Band-limit (l runs from 0 to L-1).
    ell_taper : int
        Multipole at which tapering starts.
    ell_max : int, optional
        Multipole at which the window reaches zero. Defaults to L-1.

    Returns
    -------
    w : ndarray, shape (L,)
        Taper window w(l).
    """
    if ell_max is None:
        ell_max = L - 1

    ells = np.arange(L, dtype=float)
    w = np.ones(L, dtype=float)

    mid = (ells > ell_taper) & (ells < ell_max)
    w[ells >= ell_max] = 0.0
    w[mid] = 0.5 * (1.0 + np.cos(np.pi * (ells[mid] - ell_taper) /
                                 (ell_max - ell_taper)))
    return w



def build_axisym_filter_bank(L, lam, J0=0):
    """
    Format to match `filters.filters_directional_vectorised(L, N_directions, lam)`:
      - psi: (J, L, 2L-1)   complex, only m=0 (col L-1) nonzero
      - phi: (L,)           complex, radial (m=0 column only)
    Returns [psi, phi].
    """
    # --- scale count: J = ceil(log_lam(L)) so that L=129, lam=2 -> J=8 -> J-J0+1 = 9 bands
    J_hi = int(np.ceil(np.log(L) / np.log(lam)))
    J = max(1, J_hi - J0 + 1)  # number of wavelet bands

    M   = 2*L - 1
    mid = L - 1                # m = 0 column

    # Wavelets Ψ^{(j)}_{ℓ0} placed at m=0
    hw  = HarmonicWindows(L=L, lam=lam, J0=J0)
    psi = np.zeros((J, L, M), dtype=np.complex64)
    for j in range(J):
        idx   = J0 + j
        psi_l = hw.wavelet(idx).astype(np.float64)          # (L,)
        psi[j, :, mid] = psi_l.astype(np.complex64)

    # Scaling Φ_{ℓ0}: return *radial vector* (L,), match the input format of s2wav.analysis()
    phi_l = hw.scaling().astype(np.float64).astype(np.complex64)   # (L,)

    # Optional: remove monopole & dipole
    lmin = min(2, L)
    if lmin > 0:
        psi[:, :lmin, :] = 0.0
        phi_l[:lmin]     = 0.0
    '''
    # ---- automatic taper: 94% of lmax ----
    lmax = L - 1
    ell_taper = int(0.94 * lmax)
    w = cosine_taper(L, ell_taper, ell_max=lmax)

    psi = psi * w[None, :, None]
    phi_l = phi_l * w
    '''
    return [psi, phi_l]


class SimpleHarmonicWindows:
    """
    Build scale-discretised wavelet *and scaling* windows.

    For each band j you provide:
      - ell_peaks[j] : desired wavelet peak ℓ_peak^j
      - lam_list[j]  : λ_j used in that band

    Wavelets use kappa; scaling uses eta.
    """

    def __init__(self, L, ell_peaks, lam_list, scal_ell_cut=None, scal_lam=None):
        self.L    = int(L)
        self.ells = np.arange(self.L, dtype=float)

        self.ell_peaks = np.asarray(ell_peaks, dtype=float)
        self.lam_list  = np.asarray(lam_list,  dtype=float)
        assert self.ell_peaks.shape == self.lam_list.shape

        self.J = len(self.ell_peaks)

        # one generator per λ_j (for wavelets)
        self.g_per_j = [AxisymmetricGenerators(lam_j)
                        for lam_j in self.lam_list]

        # --- scaling parameters (you choose these) ---
        # scal_ell_cut: multipole up to which scaling has support/peak
        # scal_lam:     λ used for scaling generator
        self.scal_ell_cut = scal_ell_cut if scal_ell_cut is not None else 64.0
        self.scal_lam     = scal_lam     if scal_lam     is not None else 2.0

        # ----- scaling window settings -----
        # use the same λ as the lowest band (or just 2.0)
        self.scal_lam   = float(self.lam_list[0])
        self.g_scal     = AxisymmetricGenerators(self.scal_lam)
        self.scal_band  = (0, 64, 64)   # (ℓ_min, ℓ_peak, ℓ_max) 

    # ---------- scaling (eta) ----------

    def scaling_raw(self):
        """
        Scaling window using eta, centred at ℓ_peak = 64.
        """
        ell_min, ell_peak, ell_max = self.scal_band

        t   = self.ells / float(ell_peak)
        phi = self.g_scal.eta(t)           # η: scaling kernel

        # normalise so max = 1
        m = phi.max()
        if m > 0:
            phi = phi / m
        return phi

    def scaling_band(self, truncate=True):
        """
        Scaling window, optionally truncated to [0, 64]
        like the 'Scal.' row in Table 1.
        """
        phi = self.scaling_raw()
        if truncate:
            ell_min, ell_peak, ell_max = self.scal_band
            mask_outside = (self.ells < ell_min) | (self.ells > ell_max)
            phi[mask_outside] = 0.0
        return phi


    # ---------- wavelets (kappa) ----------

    def wavelet_raw(self, j):
        ell_peak = self.ell_peaks[j]
        gen      = self.g_per_j[j]

        t   = self.ells / ell_peak
        psi = gen.kappa(t)        # <-- κ: wavelet

        m = psi.max()
        if m > 0:
            psi = psi / m
        return psi

    def band_edges(self, j):
        ell_peak = float(self.ell_peaks[j])
    
        # 1) choose which λ to use for the edges  (UNCHANGED idea)
        if j > 0 and (ell_peak == 512 or ell_peak == 2015):
            lam_edges = float(self.lam_list[j - 1])
        else:
            lam_edges = float(self.lam_list[j])
    
        # 2) compute edges from that λ  (same formulas as before)
        ell_min = ell_peak / lam_edges
        ell_max = ell_peak * lam_edges
    
        # 3) EXTRA STEP: at transition peaks, chop the TOP edge
        #    at the *next* peak, so j=3 uses ell_max ≈ 705, and
        #    j=8 uses ell_max ≈ 2539.
        if (ell_peak == 512 or ell_peak == 2015) and j < self.J - 1:
            ell_max = min(ell_max, float(self.ell_peaks[j + 1]))
    
        return int(ell_min), int(ell_peak), int(ell_max)

    
    def wavelet_band(self, j, truncate=True):
        psi = self.wavelet_raw(j)
        if truncate:
            ell_min, ell_peak, ell_max = self.band_edges(j)
            mask_outside = (self.ells < ell_min) | (self.ells > ell_max)
            psi[mask_outside] = 0.0
        return psi

    # ---------- scaling (eta) ----------

    def scaling_raw(self):
        """
        Scaling window using eta:

            Phi_raw[ell] = eta( ell / scal_ell_cut )

        (you control scal_ell_cut and scal_lam).
        """
        t   = self.ells / float(self.scal_ell_cut)
        phi = self.g_scal.eta(t)      # <-- η: scaling

        m = phi.max()
        if m > 0:
            phi = phi / m
        return phi

    # ---------- plotting helper ----------
    def plot_all_wavelets(self, truncate=True):
        """
        Plot all bands j on one figure.
        Set truncate=False to see the raw κ-wavelets.
        Also print (ℓ_min^j, ℓ_peak^j, ℓ_max^j) for each band.
        """
        plt.figure(figsize=(10, 6))
    
        for j in range(self.J):
            psi_j = self.wavelet_band(j, truncate=truncate)
            ell_min, ell_peak, ell_max = self.band_edges(j)
    
            # print band edges for this j
            print(f"j = {j:2d}  ->  ell_min = {ell_min:4d}, "
                  f"ell_peak = {ell_peak:4d}, ell_max = {ell_max:4d}")
    
            plt.plot(self.ells, psi_j, label=f"j={j}, peak={ell_peak}")
    
            # show theoretical band edges
            plt.axvline(ell_min,  color='k', linestyle='--', alpha=0.15)
            plt.axvline(ell_peak, color='k', linestyle=':',  alpha=0.3)
            plt.axvline(ell_max,  color='k', linestyle='--', alpha=0.15)
    
        plt.xlim(0, self.L - 1)
        plt.ylim(0, 1.1)
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$\Psi_{\ell j}$")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.title(
            "Scale-discretised wavelet bands "
            f"({'truncated' if truncate else 'raw'})"
        )
        plt.tight_layout()
        plt.show()

    def plot_scaling_and_wavelets(self, truncate=True):
         """
         Plot scaling + all wavelet bands on one figure.
         Legend on the right, larger axis/legend fonts.
         """
         plt.figure(figsize=(12, 6))
     
         axis_label_size = 18
         tick_size = 14
         legend_size = 12  # larger legend / "j" labels
     
         # ----- scaling -----
         phi = self.scaling_band(truncate=truncate)
         ell_min_s, ell_peak_s, ell_max_s = self.scal_band
     
         # print scaling band (keep full info)
         print(f"scal -> ell_min = {ell_min_s:4d}, ell_peak = {ell_peak_s:4d}, ell_max = {ell_max_s:4d}")
     
         plt.plot(self.ells, phi, label="scal", linewidth=2)
         plt.axvline(ell_min_s,  color='k', linestyle='--', alpha=0.15)
         plt.axvline(ell_peak_s, color='k', linestyle=':',  alpha=0.3)
         plt.axvline(ell_max_s,  color='k', linestyle='--', alpha=0.15)
     
         # ----- wavelets -----
         for j in range(self.J):
             psi_j = self.wavelet_band(j, truncate=truncate)
             ell_min, ell_peak, ell_max = self.band_edges(j)
     
             # no need to print peak value separately any more
             print(f"j = {j:2d}  ->  ell_min = {ell_min:4d}, ell_max = {ell_max:4d}")
     
             # legend label is just j=...
             plt.plot(self.ells, psi_j, label=f"j={j}")
             plt.axvline(ell_min,  color='k', linestyle='--', alpha=0.15)
             plt.axvline(ell_peak, color='k', linestyle=':',  alpha=0.3)
             plt.axvline(ell_max,  color='k', linestyle='--', alpha=0.15)
     
         plt.xlim(0, self.L - 1)
         plt.ylim(0, 1.1)
     
         plt.xlabel(r"$\ell$", fontsize=axis_label_size)
         plt.ylabel("window", fontsize=axis_label_size)
         plt.xticks(fontsize=tick_size)
         plt.yticks(fontsize=tick_size)
         plt.grid(True, alpha=0.3)
         plt.title("Scaling + wavelet harmonic windows", fontsize=axis_label_size)
     
         # legend to the right, with larger font
         plt.legend(
             loc="center left",
             bbox_to_anchor=(1.02, 0.5),
             fontsize=legend_size,
             frameon=False
         )
     
         plt.tight_layout()
         plt.show()
