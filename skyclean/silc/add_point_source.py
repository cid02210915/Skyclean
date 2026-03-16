import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from collections.abc import Sequence
from typing import Optional, Tuple, Literal

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .file_templates import FileTemplates
from .map_tools import HPTools


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
    Build a point source catalogue from a reference map by connected components.

    Returns a dict with:
      label, npix, val(sum), peak, peak_pix, theta, phi, lon_deg, lat_deg, rad_deg
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
            "theta": np.array([], dtype=np.float64),
            "phi": np.array([], dtype=np.float64),
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
        "theta": theta.astype(np.float64),
        "phi": phi.astype(np.float64),
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


def sum_selected_pixels_by_udgrade_region(
    hp_map_in: np.ndarray,
    selected_pix_in: np.ndarray,
    nside_out: int,
    *,
    nest: bool = False,
) -> np.ndarray:
    """
    Sum high-resolution map values over selected input pixels, grouped by the
    low-resolution parent pixels reached by ud_grade-style grouping.
    """
    nside_in = hp.get_nside(hp_map_in)
    selected_pix_in = np.asarray(selected_pix_in, dtype=np.int64)

    if nside_out > nside_in:
        raise ValueError("nside_out must be <= nside_in")

    theta, phi = hp.pix2ang(nside_in, selected_pix_in, nest=nest)
    pix_out = hp.ang2pix(nside_out, theta, phi, nest=nest)

    sums_out = np.zeros(hp.nside2npix(nside_out), dtype=np.float64)
    np.add.at(sums_out, pix_out, hp_map_in[selected_pix_in])
    return sums_out


def measure_source_values(
    hp_map: np.ndarray,
    center_theta: np.ndarray,
    center_phi: np.ndarray,
    *,
    nest: bool = False,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Measure per-source summed values from a map by connected-component lookup.
    """
    label_of_pix, sum_per = precompute_component_sums(hp_map, nest=nest, threshold=threshold)
    nside = hp.get_nside(hp_map)
    theta = np.asarray(center_theta, dtype=np.float64)
    phi = np.asarray(center_phi, dtype=np.float64)
    seed_pix = hp.ang2pix(nside, theta, phi, nest=nest).astype(np.int64)
    labs = label_of_pix[seed_pix]
    vals = np.zeros(labs.size, dtype=np.float64)
    good = labs >= 0
    vals[good] = sum_per[labs[good]]
    return vals


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

    print(f"\n=== {comp} === after radius and brightness scaling === ")
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
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        brightness_percentile: Optional[Tuple[float, float]] = (75.0, 100.0),
        mode: Literal["random", "brightest"] = "random",
        random_seed: int = 1,
        ps_radius_range: Tuple[float, float] = (1.0, 1.0),
        ps_brightness_scale: float = 1.0,
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
        self.ps_radius_range = tuple(float(x) for x in ps_radius_range)
        self.ps_brightness_scale = float(ps_brightness_scale)
        self.directory = directory
        self.nest = nest
        self.threshold = float(threshold)

        self.templates = file_templates if file_templates is not None else FileTemplates(directory=directory)
        # expects: self.templates.file_templates[ps_component] -> format string taking (freq)
        self.file_templates = self.templates.file_templates

    def _path_for_freq(self, fre: str) -> str:
        tmpl = self.file_templates[self.ps_component]
        return tmpl.format(frequency=fre)

    def _load_and_convert_map(self, path: str, fre: str) -> np.ndarray:
        m = hp.read_map(path, field=0, verbose=False)
        return HPTools.unit_convert(m, fre)

    def create_and_output_catalogue(self):
        """
        Returns:
          theta_list, phi_list, rad_list, val_list, sed_lists
        where val_list is from the reference frequency (frequencies[0]).
        """
        if len(self.frequencies) == 0:
            raise ValueError("frequencies is empty")

        frequencies = self.frequencies
        ref_fre = frequencies[2]

        # --- reference map -> catalogue ---
        ref_path = self._path_for_freq(ref_fre)
        ref_map = self._load_and_convert_map(ref_path, ref_fre)

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

        rmin, rmax = self.ps_radius_range
        if rmin <= 0 or rmax <= 0:
            raise ValueError("ps_radius_range values must be > 0.")
        if rmin > rmax:
            raise ValueError("ps_radius_range must satisfy min <= max.")

        # fixed per-source geometry from reference catalogue
        theta_sel = cat["theta"][idx_sel].astype(np.float64)
        phi_sel = cat["phi"][idx_sel].astype(np.float64)
        lon_sel = cat["lon_deg"][idx_sel].astype(np.float64)
        lat_sel = cat["lat_deg"][idx_sel].astype(np.float64)
        rad_sel = cat["rad_deg"][idx_sel].astype(np.float64)
        lab_sel = cat["label"][idx_sel].astype(np.int32)
        rng = np.random.default_rng(self.random_seed)
        radius_scales = rng.uniform(rmin, rmax, size=rad_sel.size)

        # --- per-frequency precompute once, then O(1) lookup per source ---
        out = {
            "label": lab_sel,
            "lon_deg": lon_sel,
            "lat_deg": lat_sel,
            "rad_deg": rad_sel,
        }

        for fre in frequencies:
            path = self._path_for_freq(fre)
            m = self._load_and_convert_map(path, fre)

            label_of_pix, sum_per = precompute_component_sums(m, nest=self.nest, threshold=self.threshold)
            nside = hp.get_nside(m)

            seed_pix = hp.ang2pix(nside, theta_sel, phi_sel, nest=self.nest).astype(np.int64)

            labs = label_of_pix[seed_pix]
            vals = np.zeros(labs.size, dtype=np.float64)
            good = labs >= 0
            vals[good] = sum_per[labs[good]]

            out[f"val_{fre}"] = vals

        results = pd.DataFrame(out)

        # apply same brightness scaling to all ps across all frequencies, to preserve SED shapes
        for fre in frequencies:
            col = f"val_{fre}"
            results[col] = (results[col].to_numpy(dtype=np.float64) * self.ps_brightness_scale)
        # apply random radius scaling to each ps
        results["rad_deg"] = (results["rad_deg"].to_numpy(dtype=np.float64) * radius_scales)

        # print the resulting catalogue
        print_results_table(results, self.ps_component, frequencies) 

        # output arrays for injection
        theta_list = theta_sel
        phi_list = phi_sel
        rad_list = results["rad_deg"].to_numpy(dtype=np.float64)
        #sed_lists = create_sed_lists_from_results(results, frequencies)
        sed_lists = results[[f"val_{fre}" for fre in frequencies]].to_numpy(dtype=float).tolist()

        return theta_list, phi_list, rad_list, sed_lists


class CircularPSInjector:
    """
    Build extra-feature maps as filled discs centred on the selected source positions.
    """

    def __init__(
        self,
        *,
        ps_component: str,
        map_loader,
    ):
        self.ps_component = ps_component
        self.map_loader = map_loader

    def build_map(
        self,
        *,
        frequency: str,
        realisation: int,
        center_theta: np.ndarray,
        center_phi: np.ndarray,
        radius_deg: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        print(f"[circular_ps] using ps_component='{self.ps_component}' at {frequency} GHz")
        src_map = self.map_loader(self.ps_component, frequency, realisation)
        nside = hp.get_nside(src_map)
        extra_feature_map = np.zeros_like(src_map, dtype=np.float64)
        for theta, phi, rad_deg, value in zip(
            np.asarray(center_theta, dtype=np.float64),
            np.asarray(center_phi, dtype=np.float64),
            np.asarray(radius_deg, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        ):
            vec = hp.ang2vec(theta, phi)
            pix = hp.query_disc(nside, vec, np.deg2rad(rad_deg))
            extra_feature_map[pix] = value
        return extra_feature_map


class PixelPSInjector:
    """
    Build low-resolution extra-feature maps using a fixed reference-frequency
    ratio pattern and per-frequency source-flux rescaling.
    """

    def __init__(
        self,
        *,
        components: Sequence[str],
        frequencies: Sequence[str],
        ps_component: str,
        desired_lmax: int,
        map_loader,
        reference_frequency: str = "030",
        nest: bool = False,
        threshold: float = 0.0,
    ):
        self.components = list(components)
        self.frequencies = list(frequencies)
        self.ps_component = ps_component
        self.desired_lmax = int(desired_lmax)
        self.map_loader = map_loader
        self.reference_frequency = str(reference_frequency).zfill(3)
        self.nest = nest
        self.threshold = float(threshold)

    def _foreground_components(self) -> list[str]:
        excluded = {"cmb", "noise"}
        resolved = []
        for comp in self.components:
            if comp in excluded:
                continue
            if comp == "extra_feature":
                resolved.append(self.ps_component)
            else:
                resolved.append(comp)
        return list(dict.fromkeys(resolved))

    def _compute_inputs_for_frequency(
        self,
        *,
        frequency: str,
        realisation: int,
        center_theta: np.ndarray,
        center_phi: np.ndarray,
        nside_out: int,
    ) -> dict[str, np.ndarray]:
        print(f"[pixel_ps] using ps_component='{self.ps_component}' at {frequency} GHz")
        ps_map = self.map_loader(self.ps_component, frequency, realisation)
        foreground_components = self._foreground_components()
        if not foreground_components:
            raise ValueError(
                "pixel_ps injection requires at least one foreground component in `components` "
                "besides 'cmb' and 'noise'."
            )

        foreground_maps = [
            self.map_loader(comp, frequency, realisation)
            for comp in foreground_components
        ]
        working_nside = min([hp.get_nside(ps_map)] + [hp.get_nside(m) for m in foreground_maps])
        if nside_out > working_nside:
            raise ValueError(
                f"desired low-resolution nside_out={nside_out} exceeds working_nside={working_nside} "
                f"for frequency {frequency}"
            )

        if hp.get_nside(ps_map) != working_nside:
            ps_map = hp.ud_grade(ps_map, nside_out=working_nside, order_in="RING", order_out="RING", power=0)

        foreground_sum = np.zeros(hp.nside2npix(working_nside), dtype=np.float64)
        for comp, fg_map in zip(foreground_components, foreground_maps):
            if hp.get_nside(fg_map) != working_nside:
                fg_map = hp.ud_grade(fg_map, nside_out=working_nside, order_in="RING", order_out="RING", power=0)
            foreground_sum += fg_map
            print(f"[extra_feature:{frequency}] added foreground component '{comp}'")

        label_of_pix, _ = precompute_component_sums(ps_map, nest=self.nest, threshold=self.threshold)
        theta = np.asarray(center_theta, dtype=np.float64)
        phi = np.asarray(center_phi, dtype=np.float64)
        seed_pix_hi = hp.ang2pix(working_nside, theta, phi, nest=self.nest).astype(np.int64)

        labs = label_of_pix[seed_pix_hi]
        keep_labs = np.unique(labs[labs >= 0])
        if keep_labs.size == 0:
            zeros = np.zeros(hp.nside2npix(nside_out), dtype=np.float64)
            return {
                "sums_out": zeros.copy(),
                "average_ratio": np.full_like(zeros, np.nan, dtype=np.float64),
                "fg_lowres": zeros.copy(),
            }

        selected_pix_hi = np.where(np.isin(label_of_pix, keep_labs))[0]
        sums_out = sum_selected_pixels_by_udgrade_region(ps_map, selected_pix_hi, nside_out, nest=self.nest)
        counts = sum_selected_pixels_by_udgrade_region(np.ones_like(ps_map), selected_pix_hi, nside_out, nest=self.nest)

        ratio_map_hi = np.full_like(ps_map, np.nan, dtype=np.float64)
        valid_fg = np.isfinite(foreground_sum) & (foreground_sum != 0)
        ratio_map_hi[selected_pix_hi] = np.divide(
            ps_map[selected_pix_hi],
            foreground_sum[selected_pix_hi],
            out=np.full(selected_pix_hi.shape, np.nan, dtype=np.float64),
            where=valid_fg[selected_pix_hi],
        )
        sums_ratio = sum_selected_pixels_by_udgrade_region(
            np.nan_to_num(ratio_map_hi, nan=0.0, posinf=0.0, neginf=0.0),
            selected_pix_hi,
            nside_out,
            nest=self.nest,
        )

        average_ratio = np.full_like(sums_ratio, np.nan, dtype=np.float64)
        np.divide(sums_ratio, counts, out=average_ratio, where=counts > 0)
        fg_lowres = hp.ud_grade(foreground_sum, nside_out=nside_out, order_in="RING", order_out="RING", power=0)

        # Diagnostic: visualise the low-resolution PS/FG contamination ratio map.
        #hp.mollview(
        #    np.nan_to_num(average_ratio, nan=0.0, posinf=0.0, neginf=0.0),
        #    title=f"[pixel_ps] average PS/FG ratio @ {frequency} GHz",
        #    unit="",
        #)
        #plt.show()

        return {
            "sums_out": sums_out,
            "average_ratio": average_ratio,
            "fg_lowres": fg_lowres,
        }

    def build_maps(
        self,
        *,
        realisation: int,
        center_theta: np.ndarray,
        center_phi: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        nside_out = HPTools.get_nside_from_lmax(self.desired_lmax)
        if self.reference_frequency not in self.frequencies:
            raise ValueError(
                f"Reference frequency '{self.reference_frequency}' is not in frequencies {self.frequencies}"
            )

        per_freq = {}
        for frequency in self.frequencies:
            print(f"[extra_feature] computing ratio-rescaled inputs for {frequency} GHz")
            per_freq[frequency] = self._compute_inputs_for_frequency(
                frequency=frequency,
                realisation=realisation,
                center_theta=center_theta,
                center_phi=center_phi,
                nside_out=nside_out,
            ) 

        ref_average_ratio = np.nan_to_num(
            per_freq[self.reference_frequency]["average_ratio"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        ref_sums_out = per_freq[self.reference_frequency]["sums_out"]
        ref_fg_lowres = per_freq[self.reference_frequency]["fg_lowres"]

        extra_feature_maps = {}
        sed_matrix = np.zeros((len(center_theta), len(self.frequencies)), dtype=np.float64)
        for i, frequency in enumerate(self.frequencies):
            sums_out = per_freq[frequency]["sums_out"]
            scale = np.divide(
                sums_out,
                ref_sums_out,
                out=np.zeros_like(sums_out, dtype=np.float64),
                where=ref_sums_out != 0,
            )
            extra_feature_map = ref_average_ratio * scale * ref_fg_lowres
            extra_feature_maps[frequency] = np.asarray(extra_feature_map, dtype=np.float64)
            sed_matrix[:, i] = measure_source_values(
                extra_feature_maps[frequency],
                center_theta=center_theta,
                center_phi=center_phi,
                nest=self.nest,
                threshold=self.threshold,
            )
        return extra_feature_maps, sed_matrix
