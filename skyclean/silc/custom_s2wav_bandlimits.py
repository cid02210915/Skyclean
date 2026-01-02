import math
import numpy as np

"""
How to patch s2wav to use the SILC fixed band edges (ELL_MIN / ELL_MAX)
=====================================================================

Goal
----
Replace s2wav’s default λ-geometric band construction with a fixed,
SILC-style wavelet filter bank, **without generating duplicate or
phantom wavelet blocks at the highest band**.

The SILC filter bank is defined by fixed harmonic support intervals
(ELL_MIN[j], ELL_MAX[j]) and must satisfy:

  • Each wavelet band has non-empty harmonic support within ℓ < L
  • No two wavelet bands produce identical effective band-limits
  • The partition of unity is enforced only on *existing* ℓ modes

This patch ensures those conditions are met.

Files to modify
---------------
All changes are made in:

    s2wav/samples.py

Functions that MUST be overridden
---------------------------------
These functions control how many wavelet bands exist and how their
harmonic support is defined:

  • j_max(L, lam)
  • scal_bandlimit(L, J_min, lam, multiresolution)
  • LN_j(L, j, N, lam, multiresolution)

Optional (only if you want full explicit control):
  • wav_j_bandlimit(L, j, lam, multiresolution)
  • L0_j(j, lam)

------------------------
**Never rely on `range(j_max + 1)` internally.**

Instead, the wavelet indices must be selected by a function that:
  • removes empty bands
  • removes duplicate clipped bands
  • preserves strict ordering in effective Lj

This is what prevents the duplicate “last wavelet” bug.

Implementation strategy
-----------------------
1. Define the fixed SILC band edges:

       ELL_MIN[j], ELL_MAX[j]

2. Select valid wavelet indices using:

       wavelet_js_silc(L)

   This function:
     • keeps j only if ELL_MIN[j] < L
     • clips Lj = min(ELL_MAX[j], L)
     • discards bands with Lj ≤ L0j
     • discards bands whose Lj duplicates the previous band

3. Redefine j_max(L) as:

       j_max(L) = max(wavelet_js_silc(L))

   (Compatibility only — do NOT loop with range(j_max+1))

4. Override LN_j so that:

       Lj  = min(ELL_MAX[j], L)
       L0j = ELL_MIN[j]

   with multiresolution=True

5. Ensure downstream code loops as:

       for j in wavelet_js_silc(L):

   NOT:

       for j in range(j_max(L)+1)

Patch workflow (terminal)
------------------------
1) Use a local editable checkout of s2wav:

       pip uninstall -y s2wav
       cd /path/to/local/s2wav
       pip install -e .

2) Confirm imports resolve to the local repo:

       python - << EOF
       import s2wav, inspect
       print(inspect.getfile(s2wav))
       EOF

   Expected: path inside your local repo.

3) Edit sampling helpers:

       nano s2wav/samples.py

4) Insert:
   • ELL_MIN / ELL_MAX arrays
   • wavelet_js_silc(L)
   • patched versions of j_max, scal_bandlimit, LN_j

5) Rename patched functions to the public API names s2wav expects:

       j_max_silc          → j_max
       scal_bandlimit_silc → scal_bandlimit
       LN_j_silc           → LN_j

   (Optional)
       wav_j_bandlimit_silc → wav_j_bandlimit
       L0_j_silc            → L0_j
       wavelet_js_silc     → wavelet_js

6) Verify behaviour:

       from s2wav.samples import LN_j
       from s2wav.samples import j_max
       print("j_max:", j_max(256))
       for j in range(j_max(256)+1):
           print(j, LN_j(256, j=j, N=1, multiresolution=True))

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
# ----------------------------------------------------------------------

import numpy as np

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
# wavelet_js: which wavelet bands to build (NO duplicates after clipping)
# ----------------------------------------------------------------------

def wavelet_js_silc(L: int) -> list[int]:
    """
    Return wavelet indices j to build, avoiding duplicated multiresolution blocks.

    We keep j only if:
      - ELL_MIN[j] < L  (band starts before data ends)
      - Lj > L0j        (band has non-empty harmonic support)
      - Lj is strictly increasing vs previous kept band (prevents duplicates)
    """
    L = int(L)
    js: list[int] = []
    prev_Lj: int | None = None

    for j in range(len(ELL_MIN)):
        if int(ELL_MIN[j]) >= L:
            break

        L0j = int(ELL_MIN[j])
        Lj  = int(min(ELL_MAX[j], L))  # multiresolution clip

        if Lj <= L0j:
            continue

        if (prev_Lj is not None) and (Lj == prev_Lj):
            # this is exactly the duplicate you are seeing (e.g. L=256 gives Lj=256 twice)
            continue

        js.append(j)
        prev_Lj = Lj

    if len(js) == 0:
        raise ValueError("Band-limit L is too small for this filter bank.")

    return js


# ----------------------------------------------------------------------
# j_max: keep for compatibility, but make it consistent with wavelet_js_silc
# ----------------------------------------------------------------------

def j_max_silc(L: int, lam: float = 2.0) -> int:
    """
    Compatibility helper: returns the largest j that will be USED.

    NOTE: If your code currently does `for j in range(j_max+1)`,
    you MUST change that loop to `for j in wavelet_js_silc(L)`,
    otherwise you can still generate duplicates.
    """
    return int(wavelet_js_silc(int(L))[-1])


# ----------------------------------------------------------------------
# scaling bandlimit (unchanged)
# ----------------------------------------------------------------------

def scal_bandlimit_silc(
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> int:
    if multiresolution:
        return int(min(65, int(L)))
    else:
        return int(L)


# ----------------------------------------------------------------------
# wav_j_bandlimit & L0_j (unchanged)
# ----------------------------------------------------------------------

def wav_j_bandlimit_silc(
    L: int,
    j: int,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> int:
    if not multiresolution:
        return int(L)

    if j < 0 or j >= len(ELL_MAX):
        raise IndexError(f"j={j} out of range for ELL_MAX")
    return int(min(int(ELL_MAX[j]), int(L)))


def L0_j_silc(j: int, lam: float = 2.0) -> int:
    if j < 0 or j >= len(ELL_MIN):
        raise IndexError(f"j={j} out of range for ELL_MIN")
    return int(ELL_MIN[j])


# ----------------------------------------------------------------------
# LN_j (unchanged)
# ----------------------------------------------------------------------

def LN_j_silc(
    L: int,
    j: int = 0,
    N: int = 1,
    lam: float = 2.0,
    multiresolution: bool = False,
):
    if multiresolution:
        Lj = wav_j_bandlimit_silc(L, j, lam, multiresolution=True)
        L0j = L0_j_silc(j, lam)
    else:
        Lj = int(L)
        L0j = 0

    Nj = int(N)
    if multiresolution:
        Nj = min(N, Lj)
        Nj += (Nj + N) % 2

    return Lj, Nj, L0j
