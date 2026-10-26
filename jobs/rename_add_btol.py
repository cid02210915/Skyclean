#!/usr/bin/env python3
"""
One-off migration: add the b_tol filename tag to SILC / ML products written before the tag existed.

Since 2026-10-25 every file that depends on the covariance tolerance b_tol carries "_btol<value>" right after the
"_nsamp<nsamp>" field (see FileTemplates.BTOL_TAGGED and utils.b_tol_tag), e.g.
    ilc_cmb_from-cfn_f030_..._r0000_lmax511_N4_lam2.0_nsamp1200.npy
 -> ilc_cmb_from-cfn_f030_..._r0000_lmax511_N4_lam2.0_nsamp1200_btol0.02.npy
Files without the tag are invisible to the pipeline, so rename the existing ones with the b_tol they were made with.

Usage (dry run by default, prints every rename; add --apply to do it):
    python jobs/rename_add_btol.py --data-dir /share/lustre/keir/Skyclean2026/Skyclean/skyclean/data --lmax 511 --N 4 --btol 0.02
    python jobs/rename_add_btol.py --data-dir ... --lmax 511 --N 4 --btol 0.02 --apply
Only files whose name contains "_lmax<lmax>_N<N>_" and an "_nsamp<n>" field and no "_btol" yet are touched. Wavelet
coefficients and the doubled wavelet maps (SILC/doubled_maps) do not depend on b_tol and are left alone.
Safe to run from several jobs at once (jobs/silc_phased.sbatch does): a file that another process renamed first is
skipped, and re-running after completion renames nothing. --quiet prints only the summary (and problems).
"""
import argparse
import os
import re
import sys

# directories holding b_tol-dependent files; ML/cmb_prediction is searched recursively (ilc_ml maps/spectra per run id)
FLAT_DIRS = [
    "SILC/covariance_matrix", "SILC/weight_vector_data", "SILC/ilc_doubled_wavelet_maps", "SILC/ilc_trimmed_maps",
    "SILC/ilc_synthesised_maps", "SILC/ilc_synthesised_spec", "SILC/ilc_synthesised_cross_spec",
    "SILC/ilc_ml_maps", "SILC/ilc_ml_spec", "ML/maps",
]
RECURSIVE_DIRS = ["ML/models", "ML/cmb_prediction"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="skyclean data directory (the pipeline's --directory)")
    ap.add_argument("--lmax", type=int, required=True)
    ap.add_argument("--N", type=int, required=True, help="N_directions of the files to rename")
    ap.add_argument("--btol", type=float, required=True, help="b_tol the files were made with, e.g. 0.02")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    ap.add_argument("--quiet", action="store_true", help="print only the summary and problems")
    a = ap.parse_args()

    tag = f"_btol{a.btol:g}"
    sel = re.compile(rf"_lmax{a.lmax}_N{a.N}_")
    nsamp = re.compile(r"(_nsamp\d+)")

    def candidates():
        for d in FLAT_DIRS:
            full = os.path.join(a.data_dir, d)
            if os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    yield full, f
        for d in RECURSIVE_DIRS:
            full = os.path.join(a.data_dir, d)
            for root, _, files in os.walk(full):
                for f in sorted(files):
                    yield root, f

    n_done = n_skip = n_raced = 0
    for root, f in candidates():
        if not sel.search(f) or "_btol" in f or not nsamp.search(f):
            continue
        new = nsamp.sub(rf"\1{tag}", f, count=1)
        src, dst = os.path.join(root, f), os.path.join(root, new)
        if os.path.exists(dst):
            print(f"SKIP (target exists): {dst}"); n_skip += 1; continue
        if not a.quiet:
            print(f"{'RENAME' if a.apply else 'would rename'}: {os.path.relpath(src, a.data_dir)}\n{'':>14}-> {new}")
        if a.apply:
            try:
                os.rename(src, dst)
            except FileNotFoundError:
                n_raced += 1      # another process renamed it between the listing and now
                continue
        n_done += 1
    print(f"\n{'renamed' if a.apply else 'would rename'} {n_done} file(s), skipped {n_skip}, already renamed by another process {n_raced}"
          + ("" if a.apply else "   (dry run: add --apply)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
