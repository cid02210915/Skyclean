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
        self.t_grid = np.linspace(1.0/self.lam, 1.0, 40001)
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
    Build windows on demand from your AxisymmetricGenerators.
    Inputs:
      L   : band-limit (ℓ = 0..L-1)
      lam : same λ you used to build the generators
      J0  : lowest wavelet scale index (0 ≤ J0 ≤ J)
      g   : AxisymmetricGenerators(lam) instance
    """
    def __init__(self, L, lam=2.0, J0=0):
        self.L   = int(L)
        self.lam = float(lam)
        self.J0  = int(J0)
        self.g   = AxisymmetricGenerators(self.lam)
        self.ells = np.arange(self.L)
        self.J    = int(np.floor(np.log(self.L - 1) / np.log(self.lam)))

    def scaling(self):
        # Φ_{ℓ0} = sqrt((2ℓ+1)/(4π)) * η_λ(ℓ / λ^{J0})
        return np.sqrt((2.0*self.ells + 1.0)/(4.0*np.pi)) * \
               self.g.eta(self.ells / (self.lam**self.J0))

    def wavelet(self, j):
        # Ψ_{j;ℓ0} = sqrt((2ℓ+1)/(4π)) * κ_λ(ℓ / λ^{j})
        return np.sqrt((2.0*self.ells + 1.0)/(4.0*np.pi)) * \
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
    S = np.abs(Phi_l0)**2
    for W in Psi_j_l0.values():
        S = S + np.abs(W)**2
    S = (4.0*np.pi) / (2.0*ells + 1.0) * S

    ok = np.all(np.abs(S[1:] - 1.0) < tol)
    return S, bool(ok)

class HRFigures:
    """
    Plot the generator curves using your existing content.
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
   