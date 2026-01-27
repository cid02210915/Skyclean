"""
Probe the maximum number of realisations that the ML pipeline can handle
without an OOM by running the pipeline in a subprocess and doing a
geometric + bisection search.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
from typing import Tuple


OOM_MARKERS = (
    "out of memory",
    "oom",
    "resource_exhausted",
    "cuda_error_out_of_memory",
)

MEMORY_LINE_RE = re.compile(
    r"\[GPU Memory\]\s+After epoch\s+(\d+):\s+([0-9.]+)%\s+\((\d+)/(\d+)\s+MB\)"
)


def repo_root() -> Path:
    # This file lives at Skyclean/skyclean/ml/oom_probe.py
    return Path(__file__).resolve().parents[2]


def run_once(
    realisations: int,
    args: argparse.Namespace,
    lmax: int,
    batch_size: int,
    chs: list[int] | None,
) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "skyclean.ml.pipeline_ml",
        "--mode",
        args.mode,
        "--lmax",
        str(lmax),
        "--realisations",
        str(realisations),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(args.epochs),
    ]

    if args.frequencies:
        cmd += ["--frequencies"] + args.frequencies
    if args.lam is not None:
        cmd += ["--lam", str(args.lam)]
    if args.learning_rate is not None:
        cmd += ["--learning-rate", str(args.learning_rate)]
    if args.momentum is not None:
        cmd += ["--momentum", str(args.momentum)]
    if args.directory:
        cmd += ["--directory", args.directory]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.nsamp is not None:
        cmd += ["--nsamp", str(args.nsamp)]

    if args.random:
        cmd += ["--random", "True"]
    if args.no_shuffle:
        cmd.append("--no-shuffle")

    if chs:
        cmd += ["--chs"] + [str(c) for c in chs]

    if args.pipeline_args:
        cmd += args.pipeline_args

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        capture_output=True,
        text=True,
    )

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ok = proc.returncode == 0 and not has_oom_marker(combined)
    return ok, combined


def has_oom_marker(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in OOM_MARKERS)


def parse_epoch_memory(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        match = MEMORY_LINE_RE.search(line)
        if not match:
            continue
        epoch, percent, mb_used, mb_total = match.groups()
        records.append(
            {
                "epoch": int(epoch),
                "percent": float(percent),
                "mb_used": float(mb_used),
                "mb_total": float(mb_total),
            }
        )
    records.sort(key=lambda r: r["epoch"])
    return records


def quantify_memory_growth(records: list[dict]) -> list[dict]:
    if not records:
        return []
    base = records[0]["mb_used"]
    prev = base
    enriched = []
    for rec in records:
        delta = rec["mb_used"] - prev
        delta_from_start = rec["mb_used"] - base
        enriched.append({**rec, "delta_mb": delta, "delta_from_start_mb": delta_from_start})
        prev = rec["mb_used"]
    return enriched


def write_memory_log(
    log_dir: Path,
    lmax: int,
    realisations: int,
    batch_size: int,
    chs: list[int] | None,
    records: list[dict],
) -> None:
    if not records:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    chs_label = "none" if not chs else "-".join(str(c) for c in chs)
    log_path = log_dir / f"mem_lmax{lmax}_reals{realisations}_bs{batch_size}_chs{chs_label}.csv"
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "percent",
                "mb_used",
                "mb_total",
                "delta_mb",
                "delta_from_start_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def format_memory_summary(records: list[dict]) -> str:
    if not records:
        return ""
    start = records[0]["mb_used"]
    end = records[-1]["mb_used"]
    peak = max(r["mb_used"] for r in records)
    growth = end - start
    return (
        f"[probe] Memory growth: start={start:.0f}MB end={end:.0f}MB "
        f"delta={growth:.0f}MB peak={peak:.0f}MB epochs={len(records)}"
    )


def write_log(log_dir: Path, lmax: int, realisations: int, text: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"probe_lmax{lmax}_r{realisations}.log"
    log_path.write_text(text)


def probe(
    args: argparse.Namespace,
    lmax: int,
    batch_size: int,
    chs: list[int] | None,
) -> int:
    min_r = args.min_realisations
    max_r = args.max_realisations
    start_r = args.start_realisations or min_r
    if start_r < min_r:
        start_r = min_r

    print(f"[probe] lmax={lmax} batch_size={batch_size} chs={chs}")
    print(f"[probe] Starting at {start_r} realisations")

    last_success = None
    last_failure = None

    def log_attempt(real: int, text: str, ok: bool) -> None:
        nonlocal last_success, last_failure
        if ok:
            last_success = (real, text)
        else:
            last_failure = (real, text)
        mem_records = quantify_memory_growth(parse_epoch_memory(text))
        if mem_records:
            summary = format_memory_summary(mem_records)
            if summary:
                print(summary)
            if args.mem_log_dir and not args.log_final_only:
                write_memory_log(
                    Path(args.mem_log_dir),
                    lmax,
                    real,
                    batch_size,
                    chs,
                    mem_records,
                )
        if args.log_dir and not args.log_final_only:
            write_log(Path(args.log_dir), lmax, real, text)

    ok, out = run_once(start_r, args, lmax, batch_size, chs)
    print(f"[probe] Result at {start_r}: {'SUCCESS' if ok else 'FAIL'}")
    log_attempt(start_r, out, ok)
    if not ok:
        print("[probe] Failed at minimum realisations; cannot proceed.")
        if args.log_dir and args.log_final_only:
            real, text = last_failure
            write_log(Path(args.log_dir), lmax, real, text)
        if args.mem_log_dir and args.log_final_only:
            real, text = last_failure
            mem_records = quantify_memory_growth(parse_epoch_memory(text))
            write_memory_log(
                Path(args.mem_log_dir),
                lmax,
                real,
                batch_size,
                chs,
                mem_records,
            )
        return 0

    low = start_r
    high = None

    # Geometric growth to find first failure.
    while low < max_r:
        candidate = min(low * 2, max_r)
        print(f"[probe] Trying {candidate} realisations")
        ok, out = run_once(candidate, args, lmax, batch_size, chs)
        print(f"[probe] Result at {candidate}: {'SUCCESS' if ok else 'FAIL'}")
        log_attempt(candidate, out, ok)
        if ok:
            low = candidate
            if low == max_r:
                print("[probe] Reached max bound without failure.")
                if args.log_dir and args.log_final_only:
                    real, text = last_success
                    write_log(Path(args.log_dir), lmax, real, text)
                if args.mem_log_dir and args.log_final_only:
                    real, text = last_success
                    mem_records = quantify_memory_growth(parse_epoch_memory(text))
                    write_memory_log(
                        Path(args.mem_log_dir),
                        lmax,
                        real,
                        batch_size,
                        chs,
                        mem_records,
                    )
                return low
        else:
            high = candidate
            break

    if high is None:
        return low

    # Bisection between last success (low) and first failure (high).
    while high - low > 1:
        mid = (low + high) // 2
        print(f"[probe] Trying {mid} realisations (bisection)")
        ok, out = run_once(mid, args, lmax, batch_size, chs)
        print(f"[probe] Result at {mid}: {'SUCCESS' if ok else 'FAIL'}")
        log_attempt(mid, out, ok)
        if ok:
            low = mid
        else:
            high = mid

    if args.log_dir and args.log_final_only:
        if last_success:
            real, text = last_success
            write_log(Path(args.log_dir), lmax, real, text)
        elif last_failure:
            real, text = last_failure
            write_log(Path(args.log_dir), lmax, real, text)
    if args.mem_log_dir and args.log_final_only:
        if last_success:
            real, text = last_success
        elif last_failure:
            real, text = last_failure
        else:
            real = None
            text = ""
        if real is not None:
            mem_records = quantify_memory_growth(parse_epoch_memory(text))
            write_memory_log(
                Path(args.mem_log_dir),
                lmax,
                real,
                batch_size,
                chs,
                mem_records,
            )

    return low


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find max realisations without OOM by probing the ML pipeline."
    )
    parser.add_argument("--lmax", type=int, nargs="+", required=True)
    parser.add_argument("--min-realisations", type=int, default=1)
    parser.add_argument("--start-realisations", type=int, default=0,
                        help="Start probing from this value (>= min-realisations).")
    parser.add_argument("--max-realisations", type=int, default=4096)
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "evaluate", "train+evaluate"])
    parser.add_argument("--frequencies", nargs="+", default=[],
                        help="Frequency list passed to pipeline_ml.py.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[],
                        help="Optional grid of batch sizes to sweep.")
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--directory", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nsamp", type=int, default=None)
    parser.add_argument("--chs-set", action="append", default=[],
                        help="Comma-separated list of channels, e.g. 1,16,32,32,64. "
                             "Repeat flag to provide multiple sets.")
    parser.add_argument("--random", action="store_true",
                        help="Use random test maps to avoid large I/O.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable shuffling to reduce host memory use.")
    parser.add_argument("--log-dir", type=str, default="",
                        help="Optional directory to write subprocess logs.")
    parser.add_argument("--log-final-only", action="store_true",
                        help="Only save the final probe log for each combo.")
    parser.add_argument("--mem-log-dir", type=str, default="",
                        help="Optional directory to write per-epoch memory logs as CSV.")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER,
                        help="Extra args passed to pipeline_ml.py after '--'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmax_list = args.lmax
    batch_sizes = args.batch_sizes or [args.batch_size]
    chs_sets = [parse_chs_set(s) for s in args.chs_set] if args.chs_set else [None]

    results = []
    for lmax in lmax_list:
        for batch_size in batch_sizes:
            for chs in chs_sets:
                max_ok = probe(args, lmax, batch_size, chs)
                if max_ok <= 0:
                    sys.exit(2)
                results.append((lmax, batch_size, chs, max_ok))
                print(f"[probe] Max realisations without OOM: {max_ok}")

    if len(results) > 1:
        print("\n[probe] Summary:")
        for lmax, batch_size, chs, max_ok in results:
            print(f"  lmax={lmax} batch_size={batch_size} chs={chs} -> {max_ok}")


def parse_chs_set(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


if __name__ == "__main__":
    main()

"""Usage:
python -m skyclean.ml.oom_probe \
  --lmax 127 \
  --epochs 10 \
  --min-realisations 3 \
  --max-realisations 4 \
  --batch-sizes 2 \
  --chs-set 1,16,32,32,64 \
  --log-dir skyclean/data/ML/oom_logs \
  --mode train \
  --frequencies 030 044 070 100 143 217 353 545 857 \
  --lam 2.0 \
  --learning-rate 1e-3 \
  --momentum 0.90 \
  --directory /Scratch/cindy/testing/Skyclean/skyclean/data/ \
  --seed 42 \
  --nsamp 1200 \
  --random \
  --mem-log-dir skyclean/data/ML/oom_mem_logs

"""
