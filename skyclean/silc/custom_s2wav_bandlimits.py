import math
import numpy as np

"""
How to patch s2wav to use the SILC fixed band edges (ELL_MIN / ELL_MAX)
======================================================================

Goal
----
Replace s2wav's default lambda-geometric band-limit logic with a fixed
SILC-style filter bank by overriding the following functions in:

    s2wav/samples.py

Functions to override (as referenced in our analysis notes)
----------------------------------------------------------
- j_max(L, lam)
- scal_bandlimit(L, J_min, lam, multiresolution)
- LN_j(L, j, N, lam, multiresolution)
(Optional, only if you prefer to route everything explicitly)
- wav_j_bandlimit(L, j, lam, multiresolution)
- L0_j(j, lam)

Patch workflow (terminal)
-------------------------
1) Use a local editable checkout of s2wav so edits affect imports:

    pip uninstall -y s2wav
    cd /path/to/local/s2wav/repo
    pip install -e .

2) Confirm Python imports s2wav from your local repo (not site-packages):

    python -c "import s2wav, inspect; print(inspect.getfile(s2wav))"

   Expected: a path inside your repo, e.g. /path/to/local/s2wav/s2wav/__init__.py

3) Edit the s2wav sampling helpers:

    nano s2wav/samples.py

4) Paste the SILC code block into `samples.py`:
   - ELL_MIN / ELL_MAX arrays
   - SILC implementations: j_max_silc, scal_bandlimit_silc, wav_j_bandlimit_silc,
     L0_j_silc, LN_j_silc

5) Rename the SILC implementations to the ORIGINAL public names that s2wav expects:

    j_max_silc            -> j_max
    scal_bandlimit_silc   -> scal_bandlimit
    LN_j_silc             -> LN_j
    (optional)
    wav_j_bandlimit_silc  -> wav_j_bandlimit
    L0_j_silc             -> L0_j

6) Save and verify behaviour:

    python -c "from s2wav.samples import LN_j; print(LN_j(4096, j=3, N=5, multiresolution=True))"
 etc.

"""

# ----------------------------------------------------------------------
# Fixed band edges from SimpleHarmonicWindows / SILC design.
# j = 0, 1, ..., J-1 correspond to these rows in the printed table:
#
# j =  0  ->  ell_min =   32, ell_max =   128
# j =  1  ->  ell_min =   64, ell_max =   256
# j =  2  ->  ell_min =  128, ell_max =   512
# j =  3  ->  ell_min =  256, ell_max =   705
# j =  4  ->  ell_min =  542, ell_max =   916
# j =  5  ->  ell_min =  705, ell_max =  1192
# j =  6  ->  ell_min =  916, ell_max =  1549
# j =  7  ->  ell_min = 1192, ell_max =  2015
# j =  8  ->  ell_min = 1550, ell_max =  2539
# j =  9  ->  ell_min = 2115, ell_max =  3047
# j = 10  ->  ell_min = 2539, ell_max =  3656
# j = 11  ->  ell_min = 3046, ell_max =  4253
#
# ----------------------------------------------------------------------

ELL_MIN = np.array(
    [32,  64, 128, 256, 542, 705,
     916, 1192, 1550, 2115, 2539, 3046],
    dtype=int,
)

ELL_MAX = np.array(
    [128, 256, 512, 705, 916, 1192,
     1549, 2015, 2539, 3047, 3656, 4253],
    dtype=int,
)


# ----------------------------------------------------------------------
# j_max: how many wavelet bands to use
# ----------------------------------------------------------------------

def j_max_silc(L: int, lam: float = 2.0) -> int:
    """
    Custom maximum wavelet scale index j_max for the SILC filter bank.

    Rule: include all bands whose LOWER edge ell_min[j] is below the
    global harmonic band-limit L, so that every multipole ell < L is
    covered by at least one filter.

    The parameter lam is kept only for API compatibility and ignored.
    """
    valid = np.where(ELL_MIN < L)[0]
    if valid.size == 0:
        raise ValueError("Band-limit L is too small for this filter bank.")
    return int(valid[-1])


# ----------------------------------------------------------------------
# scal_bandlimit: harmonic support of the scaling function
# ----------------------------------------------------------------------

def scal_bandlimit_silc(
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> int:
    """
    Custom scaling band-limit L_s.

    In the SILC design, the scaling window occupies only the very
    low-ell range up to ell_max_scal = 64.  s2wav expects the *number*
    of ell modes, so we return

        L_s = min(64 + 1, L) = min(65, L)

    when multiresolution=True.  For multiresolution=False we keep the
    original behaviour (full band-limit L).
    """
    if multiresolution:
        return int(min(65, L))
    else:
        return int(L)


# ----------------------------------------------------------------------
# wav_j_bandlimit & L0_j: upper / lower edges for band j
# ----------------------------------------------------------------------

def wav_j_bandlimit_silc(
    L: int,
    j: int,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> int:
    """
    Custom upper ell band-limit L_j for wavelet band j.

    If multiresolution is True, use the fixed ELL_MAX[j], clipped by L.
    Otherwise, return the full band-limit L (same as original s2wav).
    """
    if not multiresolution:
        return int(L)

    if j < 0 or j >= len(ELL_MAX):
        raise IndexError(f"j={j} out of range for ELL_MAX")
    return int(min(ELL_MAX[j], L))


def L0_j_silc(j: int, lam: float = 2.0) -> int:
    """
    Custom lower ell cut L0_j for wavelet band j, using fixed ELL_MIN[j].

    For completeness we mirror the original function signature, but lam
    is ignored here.
    """
    if j < 0 or j >= len(ELL_MIN):
        raise IndexError(f"j={j} out of range for ELL_MIN")
    return int(ELL_MIN[j])


# ----------------------------------------------------------------------
# LN_j: main helper returning (L_j, N_j, L0_j)
# ----------------------------------------------------------------------

def LN_j_silc(
    L: int,
    j: int = 0,
    N: int = 1,
    lam: float = 2.0,
    multiresolution: bool = False,
):
    """
    Custom version of LN_j(L, j, N, lam, multiresolution).

    Returns
    -------
    Lj  : int
        Upper harmonic band-limit for scale j.
    Nj  : int
        Orientational band-limit for scale j.
    L0j : int
        Lower harmonic multipole supported by scale j.
    """
    if multiresolution:
        Lj = wav_j_bandlimit_silc(L, j, lam, multiresolution=True)
        L0j = L0_j_silc(j, lam)
    else:
        # No multiresolution: filters and coefficients are stored at full L
        Lj = int(L)
        L0j = 0

    Nj = int(N)
    if multiresolution:
        # Same parity trick as original s2wav implementation
        Nj = min(N, Lj)
        Nj += (Nj + N) % 2

    return Lj, Nj, L0j

