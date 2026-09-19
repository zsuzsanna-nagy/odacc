#!/usr/bin/env python3
"""One-command builder for the order-management ODACC benchmark."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cpn", required=True, type=Path)
    ap.add_argument("--ocel", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-events", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=120)
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model=args.out/"order_management_dopid.pnml"
    run([sys.executable, HERE/"cpn_to_dopid.py", "--cpn", args.cpn,
         "--profile-json", HERE/"order_management_profile.json",
         "--out-model", model, "--out-report", args.out/"cpn_mapping_report.json"])
    run([sys.executable, HERE/"ocel2_to_odacc.py", "--input", args.ocel,
         "--profile-json", HERE/"order_management_profile.json",
         "--mode", "components", "--out", args.out/"components",
         "--min-events", args.min_events, "--max-events", args.max_events])
    run([sys.executable, HERE/"ocel2_to_odacc.py", "--input", args.ocel,
         "--profile-json", HERE/"order_management_profile.json",
         "--mode", "stream", "--out", args.out/"order_management.stream.jsonl"])
    print("\nReady model:", model)
    print("Ready component folder:", args.out/"components")
    print("Use batch_folder.py on the component folder, preferably online first.")
if __name__=="__main__": raise SystemExit(main())
