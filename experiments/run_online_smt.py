#!/usr/bin/env python3
"""Reproducible ODACC experiment runner.

The configurations below reproduce the experiment protocol used in the paper.
Each measured run receives its own output directory, and session-level CSV/JSON
files are generated for subsequent analysis.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "batch_folder.py"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# EDIT THIS SECTION
# ---------------------------------------------------------------------------
COCOMOT_ROOT = Path("../cocomot-main")
RESULT_ROOT = ROOT / "experiment_runs"

BACKEND_SETS = {
    "symbolic_symbolic": ("symbolic", "symbolic"),
    "symbolic_smt": ("symbolic", "smt"),
    "smt_smt": ("smt", "smt"),
}

EXPERIMENTS = [
    # comprehensive
    {
        "name": "comprehensive_online",
        "folder": ROOT / "examples/comprehensive",
        "model": ROOT / "examples/comprehensive/net.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # object_attributes
    {
        "name": "object_attributes_online",
        "folder": ROOT / "examples/object_attributes",
        "model": ROOT / "examples/object_attributes/net_object_attrs.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - component_scale
    {
        "name": "component_scale_online",
        "folder": ROOT / "examples/scalability/component_scale",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - deviation_scale
    {
        "name": "deviation_scale_online",
        "folder": ROOT / "examples/scalability/deviation_scale",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - deviation_scale_extended
    {
        "name": "deviation_scale_extended_online",
        "folder": ROOT / "examples/scalability/deviation_scale_extended",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - length_scale
    {
        "name": "length_scale_online",
        "folder": ROOT / "examples/scalability/length_scale",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - merge_scale
    {
        "name": "merge_scale_online",
        "folder": ROOT / "examples/scalability/merge_scale",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - merge_scale_extended
    {
        "name": "merge_scale_extended_online",
        "folder": ROOT / "examples/scalability/merge_scale_extended",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # scalability - guard_complexity_simple
    {
        "name": "guard_complexity_simple_online",
        "folder": ROOT / "examples/scalability/guard_complexity",
        "model": ROOT / "examples/scalability/models/net_guard_simple.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
	# scalability - guard_complexity_medium
    {
        "name": "guard_complexity_medium_online",
        "folder": ROOT / "examples/scalability/guard_complexity",
        "model": ROOT / "examples/scalability/models/net_guard_medium.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
	# scalability - guard_complexity_complex
    {
        "name": "guard_complexity_complex_online",
        "folder": ROOT / "examples/scalability/guard_complexity",
        "model": ROOT / "examples/scalability/models/net_guard_complex.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - components
    {
        "name": "components_online",
        "folder": ROOT / "examples/order_management/components",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - fit
    {
        "name": "controlled_mutations_fit_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/fit",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - cf_extra_pick
    {
        "name": "controlled_mutations_cf_extra_pick_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/cf_extra_pick",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - cf_missing_pick
    {
        "name": "controlled_mutations_cf_missing_pick_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/cf_missing_pick",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - data_order_price
    {
        "name": "controlled_mutations_data_order_price_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/data_order_price",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - data_package_weight
    {
        "name": "controlled_mutations_data_package_weight_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/data_package_weight",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - object_extra_item_relation
    {
        "name": "controlled_mutations_object_extra_item_relation_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/object_extra_item_relation",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - controlled_mutations - object_missing_item_relation
    {
        "name": "controlled_mutations_object_missing_item_relation_online",
        "folder": ROOT / "examples/order_management/controlled_mutations/object_missing_item_relation",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - multi_deviation_benchmark - two_same_dimension
    {
        "name": "multi_deviation_benchmark_two_same_dimension_online",
        "folder": ROOT / "examples/order_management/multi_deviation_benchmark/two_same_dimension",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - multi_deviation_benchmark - two_mixed
    {
        "name": "multi_deviation_benchmark_two_mixed_online",
        "folder": ROOT / "examples/order_management/multi_deviation_benchmark/two_mixed",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - multi_deviation_benchmark - three_mixed
    {
        "name": "multi_deviation_benchmark_three_mixed_online",
        "folder": ROOT / "examples/order_management/multi_deviation_benchmark/three_mixed",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
    # order_management - multi_deviation_benchmark - stress_4_5
    {
        "name": "multi_deviation_benchmark_stress_4_5_online",
        "folder": ROOT / "examples/order_management/multi_deviation_benchmark/stress_4_5",
        "model": ROOT / "examples/order_management/order_management_dopid.pnml",
        "backend_sets": ["smt_smt"],
        "mode": "online",
        "warmups": 0,
        "repetitions": 1,
        "offline_timeout": 600,
        "online_timeout": 600,
        "extra_args": [
            "--observation-timeout", "60",
            "--native-crash-retries", "1",
        ],
    },
]

# ---------------------------------------------------------------------------


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def stats(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {k: None for k in ("n", "min", "max", "mean", "median", "stdev", "q1", "q3")}
    ordered = sorted(vals)
    if len(ordered) == 1:
        q1 = q3 = ordered[0]
    else:
        qs = statistics.quantiles(ordered, n=4, method="inclusive")
        q1, q3 = qs[0], qs[2]
    return {
        "n": len(vals), "min": min(vals), "max": max(vals),
        "mean": statistics.fmean(vals), "median": statistics.median(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "q1": q1, "q3": q3,
    }


def product_count(filename):
    m = re.search(r"products_(\d+)", filename)
    return int(m.group(1)) if m else None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if fields:
            w=csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)


def run_one(exp, combo_name, backend, offline_backend, run_dir, warmup=False):
    cmd = [PYTHON, str(BATCH),
           "--cocomot-root", str(COCOMOT_ROOT),
           "--folder", str(exp["folder"]), "--model", str(exp["model"]),
           "--out", str(run_dir), "--mode", exp.get("mode", "both"),
           "--backend", backend, "--offline-backend", offline_backend,
           "--online-timeout", str(exp.get("online_timeout", 120)),
           "--offline-timeout", str(exp.get("offline_timeout", 120))]
    cmd += list(exp.get("extra_args", []))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(map(str,cmd))+"\n", encoding="utf-8")
    print(f"\n[{exp['name']} / {combo_name}] {'warmup' if warmup else run_dir.name}")
    print(" ".join(map(str,cmd)))
    cp = subprocess.run(cmd, cwd=ROOT, text=True)
    (run_dir / "runner_status.json").write_text(json.dumps({"returncode": cp.returncode}, indent=2), encoding="utf-8")
    return cp.returncode


def aggregate(session_dir: Path, run_records):
    summary_rows=[]; prefix_rows=[]
    for rr in run_records:
        rows=read_csv(rr["dir"] / "summary.csv")
        for r in rows:
            r.update({"experiment":rr["experiment"], "combo":rr["combo"], "repetition":rr["repetition"]})
            summary_rows.append(r)
        rows=read_csv(rr["dir"] / "prefix_timings.csv")
        for r in rows:
            r.update({"experiment":rr["experiment"], "combo":rr["combo"], "repetition":rr["repetition"]})
            prefix_rows.append(r)

    write_csv(session_dir / "all_runs_summary.csv", summary_rows)
    write_csv(session_dir / "all_runs_prefix_timings.csv", prefix_rows)

    grouped={}
    for r in summary_rows:
        key=(r["experiment"],r["combo"],r["file"])
        grouped.setdefault(key,[]).append(r)
    aggs=[]
    metrics=["online_total_seconds","online_avg_seconds","online_max_seconds","offline_total_seconds","conformance_total_seconds","wall_seconds"]
    for (exp,combo,file), rows in sorted(grouped.items()):
        a={"experiment":exp,"combo":combo,"file":file,"product_count":product_count(file),
           "runs":len(rows),
           "online_ok_runs":sum(r.get("online_status")=="ONLINE_OK" for r in rows),
           "offline_ok_runs":sum(r.get("offline_status") in ("OFFLINE_OK",) for r in rows),
           "offline_timeout_runs":sum(r.get("offline_status")=="OFFLINE_TIMEOUT" for r in rows),
           "online_oom_runs":sum(r.get("online_status")=="ONLINE_OOM" for r in rows),
           "offline_oom_runs":sum(r.get("offline_status")=="OFFLINE_OOM" for r in rows),
           "prefix_cost_values":";".join(sorted({str(r.get('prefix_total')) for r in rows})),
           "offline_cost_values":";".join(sorted({str(r.get('offline_total')) for r in rows})),
           "offline_semantics":";".join(sorted({str(r.get('offline_semantics')) for r in rows})),
        }
        for metric in metrics:
            st=stats([fnum(r.get(metric)) for r in rows])
            for name,val in st.items(): a[f"{metric}_{name}"]=val
        aggs.append(a)
    write_csv(session_dir / "aggregate_summary.csv", aggs)
    (session_dir / "aggregate_summary.json").write_text(json.dumps(aggs,indent=2),encoding="utf-8")

    pgroup={}
    for r in prefix_rows:
        key=(r["experiment"],r["combo"],r["file"],r.get("update_index"))
        pgroup.setdefault(key,[]).append(r)
    paggs=[]
    for (exp,combo,file,upd), rows in sorted(pgroup.items(), key=lambda x:(x[0][0],x[0][1],x[0][2],int(x[0][3] or 0))):
        a={"experiment":exp,"combo":combo,"file":file,"product_count":product_count(file),
           "update_index":upd,"activity":rows[0].get("activity"),"merged":rows[0].get("merged"),
           "component_object_count":rows[0].get("component_object_count"),
           "event_object_count":rows[0].get("event_object_count")}
        st=stats([fnum(r.get("wall_seconds")) for r in rows])
        for name,val in st.items(): a[f"wall_seconds_{name}"]=val
        paggs.append(a)
    write_csv(session_dir / "aggregate_prefix_timings.csv", paggs)

    # Small, tidy subsets intended as stable inputs to paper plotting scripts.
    plot_scale=[r for r in aggs if r.get("product_count") is not None]
    write_csv(session_dir / "plot_scalability.csv", plot_scale)
    write_csv(session_dir / "plot_prefix_latency.csv", paggs)
    return aggs


def main():
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    session=RESULT_ROOT / stamp
    session.mkdir(parents=True,exist_ok=True)
    (session/"experiment_config.json").write_text(json.dumps(EXPERIMENTS,default=str,indent=2),encoding="utf-8")
    run_records=[]
    for exp in EXPERIMENTS:
        for combo_name in exp["backend_sets"]:
            backend,offline=BACKEND_SETS[combo_name]
            for i in range(1,exp.get("warmups",0)+1):
                run_one(exp,combo_name,backend,offline,session/exp["name"]/combo_name/f"warmup_{i:02d}",warmup=True)
            for i in range(1,exp.get("repetitions",1)+1):
                rd=session/exp["name"]/combo_name/f"run_{i:03d}"
                run_one(exp,combo_name,backend,offline,rd)
                run_records.append({"experiment":exp["name"],"combo":combo_name,"repetition":i,"dir":rd})
    aggregate(session,run_records)
    print(f"\nExperiment session: {session}")
    print(f"Aggregate summary: {session/'aggregate_summary.csv'}")
    print(f"Aggregate prefix timings: {session/'aggregate_prefix_timings.csv'}")
    print(f"Plot-ready scalability: {session/'plot_scalability.csv'}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
