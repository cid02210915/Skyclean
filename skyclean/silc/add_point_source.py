import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from collections.abc import Sequence
from typing import Optional, Tuple, Literal

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .file_templates import FileTemplates


# ----------------------------
# Fast connected-component precompute (per map)
# ----------------------------
def precompute_component_sums(
    hp_map: np.ndarray,
    *,
    nest: bool = False,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute connected components on the set {pix | hp_map[pix] > threshold and finite},
    using 8-neighbour adjacency, then compute sum of values per component.

    Returns
    -------
    label_of_pix : (npix,) int32
        label_of_pix[p] = component_id for active pixels, else -1
    sum_per : (n_comp,) float64
        sum_per[c] = sum of hp_map over all pixels in component c
    """
    npix = hp_map.size
    nside = hp.get_nside(hp_map)

    active = np.isfinite(hp_map) & (hp_map > threshold)
    pix_active = np.where(active)[0]
    n_active = pix_active.size

    if n_active == 0:
        return np.full(npix, -1, dtype=np.int32), np.array([], dtype=np.float64)

    # map global pixel id -> active index
    idx_of_pix = -np.ones(npix, dtype=np.int32)
    idx_of_pix[pix_active] = np.arange(n_active, dtype=np.int32)

    # neighbours for each active pixel (n_active, 8); -1 for missing
    neigh = hp.get_all_neighbours(nside, pix_active, nest=nest).T
    neigh_idx = idx_of_pix[neigh]  # -1 where neighbour not active

    # edges (row -> col) for active-active neighbours
    row = np.repeat(np.arange(n_active, dtype=np.int32), 8)
    col = neigh_idx.reshape(-1)

    good = col >= 0
    row = row[good]
    col = col[good]

    # build sparse adjacency (undirected)
    A = coo_matrix(
        (np.ones(row.size, dtype=np.uint8), (row, col)),
        shape=(n_active, n_active),
    )
    A = A + A.T

    n_comp, labels = connected_components(A, directed=False, return_labels=True)

    vals = hp_map[pix_active]
    sum_per = np.bincount(labels, weights=vals, minlength=n_comp).astype(np.float64)

    label_of_pix = np.full(npix, -1, dtype=np.int32)
    label_of_pix[pix_active] = labels.astype(np.int32)

    return label_of_pix, sum_per


def build_catalog_from_reference_map(
    hp_map: np.ndarray,
    *,
    nest: bool = False,
    threshold: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Build a source catalogue from a reference map by connected components.

    Returns a dict with:
      label, npix, val(sum), peak, peak_pix, lon_deg, lat_deg, rad_deg
    """
    nside = hp.get_nside(hp_map)
    pixarea_sr = hp.nside2pixarea(nside)

    active = np.isfinite(hp_map) & (hp_map > threshold)
    pix_active = np.where(active)[0]
    n_active = pix_active.size
    if n_active == 0:
        # empty catalogue
        return {
            "label": np.array([], dtype=np.int32),
            "npix": np.array([], dtype=np.int64),
            "val": np.array([], dtype=np.float64),
            "peak": np.array([], dtype=np.float64),
            "peak_pix": np.array([], dtype=np.int64),
            "lon_deg": np.array([], dtype=np.float64),
            "lat_deg": np.array([], dtype=np.float64),
            "rad_deg": np.array([], dtype=np.float64),
        }

    # reuse precompute to get labels for active pixels
    label_of_pix, sum_per = precompute_component_sums(hp_map, nest=nest, threshold=threshold)
    labels = label_of_pix[pix_active]
    n_comp = int(labels.max()) + 1

    vals = hp_map[pix_active]

    # npix per component
    npix_per = np.bincount(labels, minlength=n_comp)
    # sum per component (already computed globally)
    # but sum_per corresponds to all active pixels too; consistent.
    sum_per = sum_per  # (n_comp,)

    # find peak pixel per component: sort by (label asc, val asc) and take last per label
    order = np.lexsort((vals, labels))
    labels_s = labels[order]
    pix_s = pix_active[order]
    vals_s = vals[order]

    last = np.r_[np.where(np.diff(labels_s) != 0)[0], labels_s.size - 1]
    peak_pix = pix_s[last]
    peak_val = vals_s[last]

    theta, phi = hp.pix2ang(nside, peak_pix, nest=nest)
    lon_deg = np.degrees(phi)
    lat_deg = 90.0 - np.degrees(theta)

    area_sr = npix_per * pixarea_sr
    rad_deg = np.degrees(np.sqrt(area_sr / np.pi))

    return {
        "label": np.arange(n_comp, dtype=np.int32),
        "npix": npix_per.astype(np.int64),
        "val": sum_per.astype(np.float64),
        "peak": peak_val.astype(np.float64),
        "peak_pix": peak_pix.astype(np.int64),
        "lon_deg": lon_deg.astype(np.float64),
        "lat_deg": lat_deg.astype(np.float64),
        "rad_deg": rad_deg.astype(np.float64),
    }


def select_sources(
    cat: dict[str, np.ndarray],
    *,
    N: int = 10,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    brightness_percentile: Optional[Tuple[float, float]] = (75.0, 100.0),
    mode: Literal["brightest", "random"] = "brightest",
    seed: int = 0,
) -> np.ndarray:
    """
    Returns indices into cat arrays.
    """
    n = len(cat["label"])
    if n == 0:
        return np.array([], dtype=np.int64)

    mask = np.ones(n, dtype=bool)

    if lon_range is not None:
        lo, hi = lon_range
        if lo is not None:
            mask &= (cat["lon_deg"] >= lo)
        if hi is not None:
            mask &= (cat["lon_deg"] <= hi)

    if lat_range is not None:
        lo, hi = lat_range
        if lo is not None:
            mask &= (cat["lat_deg"] >= lo)
        if hi is not None:
            mask &= (cat["lat_deg"] <= hi)

    if brightness_percentile is not None:
        lo, hi = brightness_percentile
        p_lo = np.percentile(cat["val"], lo)
        p_hi = np.percentile(cat["val"], hi)
        mask &= (cat["val"] >= p_lo) & (cat["val"] <= p_hi)

    idx = np.where(mask)[0]
    if idx.size == 0:
        return idx.astype(np.int64)

    if mode == "brightest":
        idx = idx[np.argsort(cat["val"][idx])[::-1]]
        return idx[:N].astype(np.int64)

    # mode == "random"
    rng = np.random.default_rng(seed)
    if N >= idx.size:
        return idx.astype(np.int64)
    return rng.choice(idx, size=N, replace=False).astype(np.int64)


def increase_size_deg(rad_deg: np.ndarray, factor: int | float) -> np.ndarray:
    """
    Your original logic:
      ell ~ 180 / rad_deg / factor
      rad_new = 180 / ell = rad_deg * factor
    """
    return np.asarray(rad_deg, dtype=np.float64) * float(factor)


def create_sed_lists_from_results(results_df: pd.DataFrame, frequencies: Sequence[str]) -> list[list[float]]:
    sed_df = results_df[[f"val_{fre}" for fre in frequencies]].copy()
    sed_lists = sed_df.to_numpy(dtype=float).tolist()
    print(f"Created SED lists for {len(sed_lists)} sources across {len(frequencies)} channels")
    return sed_lists


def print_results_table(results_df: pd.DataFrame, comp: str, frequencies: Sequence[str]) -> None:
    if results_df.empty:
        print(f"\n=== {comp} ===")
        print("(no results)")
        return

    base_cols = ["label", "lon_deg", "lat_deg", "rad_deg"]
    val_cols = [f"val_{fre}" for fre in frequencies]
    cols = base_cols + val_cols

    print(f"\n=== {comp} ===")
    header = "  i  " + "  ".join([f"{c:>12s}" for c in cols])
    print(header)
    print("-" * len(header))

    for i, row in results_df.reset_index(drop=True).iterrows():
        line = f"{i:3d}  "
        for c in cols:
            v = row[c]
            if c in ("lon_deg", "lat_deg", "rad_deg"):
                line += f"{float(v):12.4f}  "
            elif c.startswith("val_"):
                line += f"{float(v):12.3e}  "
            else:
                line += f"{int(v):12d}  "
        print(line)



class PointSource:
    """Output point source catalogue."""

    def __init__(
        self,
        ps_component: str,
        frequencies: Sequence[str],
        n_points: int = 10,
        lon_range: Optional[Tuple[float, float]] = None,
        lat_range: Tuple[float, float] = (20.0, 90.0),
        brightness_percentile: Optional[Tuple[float, float]] = (75.0, 100.0),
        mode: Literal["random", "brightest"] = "random",
        random_seed: int = 1,
        factor: int | float = 50.0,
        file_templates: FileTemplates | None = None,
        directory: str = "data/",
        nest: bool = False,
        threshold: float = 0.0,
    ):
        self.ps_component = ps_component
        self.frequencies = list(frequencies)
        self.n_points = int(n_points)
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.brightness_percentile = brightness_percentile
        self.mode = mode
        self.random_seed = int(random_seed)
        self.factor = factor
        self.directory = directory
        self.nest = nest
        self.threshold = float(threshold)

        self.templates = file_templates if file_templates is not None else FileTemplates(directory=directory)
        # expects: self.templates.file_templates[ps_component] -> format string taking (freq)
        self.file_templates = self.templates.file_templates

    def _path_for_freq(self, fre: str) -> str:
        tmpl = self.file_templates[self.ps_component]
        return tmpl.format(frequency=fre)

    def _unit_convert(self, path: str, fre: str) -> np.ndarray:
        m = hp.read_map(path, field=0, verbose=False)
        # your existing conversion factors
        if fre == "545":
            m = m / 57.117072864249856
        if fre == "857":
            m = m / 1.4357233820474276
        return m

    def create_and_output_catalogue(self):
        """
        Returns:
          lon_list, lat_list, rad_list, val_list, sed_lists
        where val_list is from the reference frequency (frequencies[0]).
        """
        if len(self.frequencies) == 0:
            raise ValueError("frequencies is empty")

        frequencies = self.frequencies
        ref_fre = frequencies[0]

        # --- reference map -> catalogue ---
        ref_path = self._path_for_freq(ref_fre)
        ref_map = self._unit_convert(ref_path, ref_fre)

        cat = build_catalog_from_reference_map(ref_map, nest=self.nest, threshold=self.threshold)

        idx_sel = select_sources(
            cat,
            N=self.n_points,
            lon_range=self.lon_range,
            lat_range=self.lat_range,
            brightness_percentile=self.brightness_percentile,
            mode=("brightest" if self.mode == "brightest" else "random"),
            seed=self.random_seed,
        )

        if idx_sel.size == 0:
            empty = pd.DataFrame(columns=["label", "lon_deg", "lat_deg", "rad_deg"] + [f"val_{f}" for f in frequencies])
            print_results_table(empty, self.ps_component, frequencies)
            return np.array([]), np.array([]), np.array([]), np.array([]), []

        # fixed per-source geometry from reference catalogue
        lon_sel = cat["lon_deg"][idx_sel].astype(np.float64)
        lat_sel = cat["lat_deg"][idx_sel].astype(np.float64)
        rad_sel = cat["rad_deg"][idx_sel].astype(np.float64)
        lab_sel = cat["label"][idx_sel].astype(np.int32)

        # --- per-frequency precompute once, then O(1) lookup per source ---
        out = {
            "label": lab_sel,
            "lon_deg": lon_sel,
            "lat_deg": lat_sel,
            "rad_deg": rad_sel,
        }

        for fre in frequencies:
            path = self._path_for_freq(fre)
            m = self._unit_convert(path, fre)

            label_of_pix, sum_per = precompute_component_sums(m, nest=self.nest, threshold=self.threshold)
            nside = hp.get_nside(m)

            # vectorised seed pixels
            theta = np.deg2rad(90.0 - lat_sel)
            phi = np.deg2rad(lon_sel)
            seed_pix = hp.ang2pix(nside, theta, phi, nest=self.nest).astype(np.int64)

            labs = label_of_pix[seed_pix]
            vals = np.zeros(labs.size, dtype=np.float64)
            good = labs >= 0
            vals[good] = sum_per[labs[good]]

            out[f"val_{fre}"] = vals

        results = pd.DataFrame(out)
        
        #print_results_table(results, self.ps_component, frequencies)

        # outputs matching your original return
        sed_lists = create_sed_lists_from_results(results, frequencies)
        val_list = results[f"val_{ref_fre}"].to_numpy(dtype=np.float64)
        lon_list = results["lon_deg"].to_numpy(dtype=np.float64)
        lat_list = results["lat_deg"].to_numpy(dtype=np.float64)
        rad_list = increase_size_deg(results["rad_deg"].to_numpy(dtype=np.float64), self.factor)

        return lon_list, lat_list, rad_list, val_list, sed_lists
