#!/usr/bin/env python3
"""Rerun only the files of a finished session that did not complete.

Example (Windows, one line):

    python experiments\\rerun_failed.py --session experiments\\experiment_runs\\<session>
        --variant anytime-smt-jointlb-lp --deadline 20 --smt-after 5 --observation-timeout 60 --online-timeout 0 --on-observation-timeout continue

* Files are selected from ``<session>/all_runs_summary.csv``: every file whose
  ``online_status`` is not ``ONLINE_OK`` (or only the statuses given with
  ``--status``, comma-separated, e.g. ``OBSERVATION_TIMEOUT``).
* They are rerun suite by suite with the given variant and time limits
  (same options as the experiment runners: ``--observation-timeout``,
  ``--online-timeout``, ``--deadline``, ``--smt-after``; 0 disables a limit).
* The rerun is written to ``<session>_rerun_<timestamp>[_limits]``; its
  ``merged`` subfolder contains ``all_runs_summary.csv`` and
  ``all_runs_prefix_timings.csv`` of the original session with the rerun files
  replaced, so ``tools/compare_sessions.py <rerun>/merged ...`` works directly.
* ``--runner smt`` reruns with the SMT runner instead of the symbolic one.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows):
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="finished session folder")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--runner", choices=["symbolic", "smt"], default="symbolic")
    ap.add_argument("--status", default="", help="statuses to rerun (default: all but ONLINE_OK)")
    # time-limit options are read by the runner module from sys.argv
    ap.add_argument("--observation-timeout", type=float)
    ap.add_argument("--online-timeout", type=float)
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--smt-after", type=float)
    ap.add_argument("--on-observation-timeout", choices=["stop", "continue"])
    args = ap.parse_args()

    session = Path(args.session).resolve()
    summary = session / "all_runs_summary.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} not found (is the session complete?)")
    statuses = {s.strip() for s in args.status.split(",") if s.strip()}
    rows = read_csv(summary)
    failed = [(r["experiment"], r["file"]) for r in rows
              if (r.get("online_status") in statuses if statuses
                  else r.get("online_status") != "ONLINE_OK")]
    if not failed:
        print("No files to rerun.")
        return 0

    sys.path.insert(0, str(HERE))
    runner = importlib.import_module("run_online_symbolic" if args.runner == "symbolic"
                                     else "run_online_smt")
    if args.variant not in runner.VARIANTS:
        raise SystemExit(f"unknown variant {args.variant!r}; choose from {sorted(runner.VARIANTS)}")
    runner.VARIANT = args.variant
    by_name = {e["name"]: e for e in runner.EXPERIMENTS}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = session.parent / (session.name + "_rerun_" + stamp + runner.time_suffix())
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(runner, "check_time_margin"):
        _, sym = runner.apply_time_overrides([], runner.VARIANTS.get(args.variant) or None)
        if (runner.TIME_OVERRIDES.get("observation") or 1) > 0:
            runner.check_time_margin(sym)

    grouped = {}
    for exp_name, file in failed:
        grouped.setdefault(exp_name, []).append(file)
    print(f"Rerunning {len(failed)} files from {len(grouped)} suites into {out}")

    records, used = [], []
    for exp_name, files in grouped.items():
        exp = by_name.get(exp_name)
        if exp is None:
            print(f"  skipping unknown suite {exp_name}")
            continue
        src = Path(exp["folder"])
        inputs = out / "_inputs" / exp_name
        inputs.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(src / f, inputs / f)
        exp2 = dict(exp, folder=inputs)
        used.append(exp2)
        for combo_name in exp["backend_sets"]:
            backend, offline = runner.BACKEND_SETS[combo_name]
            rd = out / exp_name / combo_name / "run_001"
            runner.run_one(exp2, combo_name, backend, offline, rd)
            records.append({"experiment": exp_name, "combo": combo_name, "repetition": 1, "dir": rd})
    (out / "experiment_config.json").write_text(json.dumps(used, default=str, indent=2), encoding="utf-8")
    runner.aggregate(out, records)

    # merged view: original session with the rerun files replaced
    merged = out / "merged"
    merged.mkdir(exist_ok=True)
    rerun_keys = {(r["experiment"], r["file"]) for r in read_csv(out / "all_runs_summary.csv")}
    for name in ("all_runs_summary.csv", "all_runs_prefix_timings.csv"):
        orig = session / name
        new = out / name
        if not orig.exists() or not new.exists():
            continue
        kept = [r for r in read_csv(orig) if (r["experiment"], r["file"]) not in rerun_keys]
        write_csv(merged / name, kept + read_csv(new))
    shutil.copy2(session / "experiment_config.json", merged / "experiment_config.json")
    print(f"\nRerun session: {out}")
    print(f"Merged results (original with rerun files replaced): {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
