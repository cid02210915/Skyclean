#!/usr/bin/env python3
"""Generate individual PySM3 foreground-temperature maps (no summation).

Initial batch: d1, s1, a1, f1, cib1, rg1, tsz1 at Planck nominal frequencies.
All outputs are Galactic HEALPix RING maps in K_CMB at NSIDE=512,
with harmonic lmax=512. Monochromatic, foreground-only pilot products:
NOT Planck bandpass/beam integrated maps. For high-resolution input
templates, requesting nside=512 can cause aliasing from PySM downgrading;
validate before scientific use.
"""
import argparse
import gc
import os
import time
from importlib.metadata import version
from pathlib import Path

import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u

PLANCK = (30, 44, 70, 100, 143, 217, 353, 545, 857)
COMPONENTS = (
    # Thermal dust
    "d0", "d1", "d2", "d3", "d4", "d5", "d6",
    "d7", "d8", "d9", "d10", "d11", "d12",

    # Synchrotron
    "s0", "s1", "s2", "s3",
    "s4", "s5", "s6", "s7",

    # AME
    "a1", "a2",

    # Free-free
    "f1",
)


def path_for(root, preset, freq, nside, lmax):
    return root / preset / f"PySM_{preset}_{freq:03d}GHz_nside{nside}_lmax{lmax}_I_Kcmb.fits"


def write_atomic(path, data, *, freq, nside, lmax, preset):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".partial.fits")
    hdr = [
        ("BUNIT", "K_CMB"),
        ("FREQGHZ", freq),
        ("LMAX", lmax),
        ("PRESET", preset[:60]),
        ("FREQTYPE", "MONO"),
    ]
    try:
        hp.write_map(str(temp), data.astype(np.float32), nest=False,
                     dtype=np.float32, coord="G", overwrite=True, extra_header=hdr)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def generate(args):
    preset = args.component
    high_res_models = (
    "d9", "d10", "d11", "d12",
    "s4", "s5", "s6", "s7"
    )

    work_nside = (
        max(2048, args.nside)
        if preset in high_res_models
        else args.nside
    )
    print(f"PySM3 {version('pysm3')}; component={preset}; NSIDE={args.nside}; LMAX={args.lmax}", flush=True)
    print("Monochromatic Planck nominal frequencies; NOT integrated instrument bandpasses.", flush=True)
    if preset in ("cib1", "rg1", "tsz1"):
        print("[CAUTION] Extragalactic WebSky source maps are high-resolution;", flush=True)
        print("          PySM can downgrade these templates to output NSIDE via ud_grade.", flush=True)
        print("          Pilot only: investigate aliasing and beam/bandpass before using", flush=True)
        print("          the outputs in Planck-equivalent scientific simulations.", flush=True)
    start = time.perf_counter()
    sky = pysm3.Sky(nside=work_nside, preset_strings=[preset])
    print(f"Loaded {preset} at NSIDE={work_nside} in {time.perf_counter()-start:.1f} s", flush=True)

    for freq in args.frequencies:
        outfile = path_for(args.output, preset, freq, args.nside, args.lmax)
        if outfile.exists() and not args.overwrite:
            print(f"[SKIP] {outfile}", flush=True)
            continue
        tic = time.perf_counter()
        print(f"[START] {preset}, {freq} GHz", flush=True)
        e_rj = sky.get_emission(freq * u.GHz)
        e_cmb = e_rj.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq * u.GHz))
        vals = np.asarray(e_cmb.value)
        intensity = vals if vals.ndim == 1 else vals[0]
        intensity = np.asarray(intensity, dtype=np.float64) * 1.e-6
        if intensity.shape != (hp.nside2npix(work_nside),):
            raise ValueError(f"Unexpected output shape: {vals.shape}")
        if not np.isfinite(intensity).all():
            raise ValueError(f"Nonfinite foreground values for {preset} {freq} GHz")
        alm = hp.map2alm(intensity, lmax=args.lmax, iter=3)
        bandlimited = hp.alm2map(alm, nside=args.nside, lmax=args.lmax)
        if not np.isfinite(bandlimited).all():
            raise ValueError("Nonfinite values after harmonic filtering")
        write_atomic(outfile, bandlimited, freq=freq, nside=args.nside,
                     lmax=args.lmax, preset=preset)
        print(f"[DONE] {outfile} | {time.perf_counter()-tic:.1f} s | "
              f"{outfile.stat().st_size/1024**2:.1f} MiB", flush=True)
        del e_rj, e_cmb, vals, intensity, alm, bandlimited
        # Limit accumulated source-map caching for interpolated extragalactic templates.
        for component in sky.components:
            cache = getattr(component, "cached_maps", None)
            if cache is not None and hasattr(cache, "clear"):
                cache.clear()
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component", required=True, choices=COMPONENTS, help="Generate one individual foreground preset")
    ap.add_argument("--frequencies", nargs="+", type=int, default=list(PLANCK))
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--lmax", type=int, default=512)
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not hp.isnsideok(args.nside) or not 0 <= args.lmax <= 3*args.nside - 1:
        ap.error("Invalid nside or lmax")
    if not args.frequencies or any(f <= 0 for f in args.frequencies):
        ap.error("Frequencies must be positive GHz values")
    generate(args)


if __name__ == "__main__":
    main()
