#!/usr/bin/env python3
"""Per-prefix comparison of two ODACC online runs (e.g. symbolic vs SMT).

The paper-level evaluation compares only the *final* prefix cost.  An online
method must, however, return an optimal alignment after every observable unit,
so this tool compares costs update by update and also checks that every
returned prefix alignment is well formed (its log/sync moves explain exactly
the events observed so far in that component).

Usage
-----
Compare two existing output files::

    python tools/compare_prefix_costs.py A.json B.json

Run both backends on one log and compare::

    python tools/compare_prefix_costs.py --run --model M.pnml --log L.jsonocel \\
        [--cocomot-root ../cocomot] [--timeout 600] [--keep-dir out/]

Run on every log of a folder (one model)::

    python tools/compare_prefix_costs.py --run --model M.pnml --log-dir DIR ...

Exit status is 1 if any per-prefix cost differs or any result is malformed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


def load_prefixes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("prefix", []))


def malformed(prefixes: List[Dict[str, Any]]) -> List[str]:
    """Check that each result's log projection is the component's observed prefix."""
    problems = []
    seen_by_component: Dict[Any, set] = {}
    merged_into: Dict[Any, Any] = {}
    for r in prefixes:
        if not r.get("feasible", False):
            problems.append(f"prefix {r.get('prefix_position')}: infeasible "
                            f"({r.get('failure_reason', '')})")
            continue
        events = [m.get("event_id") for m in r.get("moves", []) if m.get("kind") in ("log", "sync")]
        if len(events) != len(set(events)):
            problems.append(f"prefix {r.get('prefix_position')}: event explained twice")
        comp = r.get("component")
        prev = seen_by_component.get(comp, set())
        if not prev <= set(events):
            problems.append(f"prefix {r.get('prefix_position')}: previously observed events "
                            f"{sorted(prev - set(events))} missing from component {comp}")
        seen_by_component[comp] = set(events)
        for m in r.get("moves", []):
            if m.get("kind") == "sync" and m.get("object_match") is False:
                problems.append(f"prefix {r.get('prefix_position')}: sync move "
                                f"{m.get('event_id')} with object mismatch")
    return problems


def compare(a_path: Path, b_path: Path, label_a: str = "A", label_b: str = "B") -> Tuple[bool, List[str]]:
    a, b = load_prefixes(a_path), load_prefixes(b_path)
    lines = []
    ok = True
    for name, pref in ((label_a, a), (label_b, b)):
        for p in malformed(pref):
            ok = False
            lines.append(f"  [{name}] MALFORMED {p}")
    bpos = {r.get("prefix_position"): r for r in b}
    compared = 0
    for ra in a:
        pos = ra.get("prefix_position")
        rb = bpos.get(pos)
        if rb is None:
            continue
        compared += 1
        uncertified = ra.get("certified") is False or rb.get("certified") is False
        if uncertified:
            lb_ok = all((r.get("lower_bound") is None or r.get("cost") is None or
                         other.get("cost") is None or
                         (r.get("certified") is not False or r["lower_bound"] <= other["cost"] <= r["cost"]))
                        for r, other in ((ra, rb), (rb, ra)))
            if not lb_ok:
                ok = False
                lines.append(f"  prefix {pos}: uncertified bounds inconsistent: {label_a}="
                             f"[{ra.get('lower_bound')},{ra.get('cost')}] {label_b}="
                             f"[{rb.get('lower_bound')},{rb.get('cost')}]")
            continue
        if ra.get("cost") != rb.get("cost"):
            ok = False
            lines.append(f"  prefix {pos}: {label_a}={ra.get('cost')} "
                         f"({ra.get('certification_route_detail') or ra.get('certification_route')}) "
                         f"{label_b}={rb.get('cost')}")
    lines.insert(0, f"  compared {compared} prefixes "
                    f"({label_a}: {len(a)}, {label_b}: {len(b)})")
    return ok, lines


def run_backend(model: Path, log: Path, backend: str, out: Path, cocomot_root: str,
                timeout: int) -> Optional[str]:
    cmd = [sys.executable, str(ROOT / "odacc.py"), "--cocomot-root", cocomot_root,
           "--model", str(model), "--log", str(log), "--mode", "online",
           "--backend", backend, "--quiet-solver", "--quiet", "--output", str(out)]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    if proc.returncode != 0 or not out.exists():
        return f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="two output JSON files to compare")
    ap.add_argument("--run", action="store_true", help="run symbolic and smt backends first")
    ap.add_argument("--model")
    ap.add_argument("--log")
    ap.add_argument("--log-dir")
    ap.add_argument("--cocomot-root", default=str(ROOT.parent / "cocomot"))
    ap.add_argument("--timeout", type=int, default=600, help="per backend run, seconds")
    ap.add_argument("--keep-dir", help="directory for the output JSON files")
    args = ap.parse_args()

    if not args.run:
        if len(args.files) != 2:
            ap.error("give two output files or use --run")
        ok, lines = compare(Path(args.files[0]), Path(args.files[1]))
        print(("OK" if ok else "DIFF") + f" {args.files[0]} vs {args.files[1]}")
        print("\n".join(lines))
        return 0 if ok else 1

    if not args.model or not (args.log or args.log_dir):
        ap.error("--run needs --model and --log or --log-dir")
    logs = [Path(args.log)] if args.log else sorted(
        p for p in Path(args.log_dir).iterdir() if p.suffix in (".jsonocel", ".json"))
    out_dir = Path(args.keep_dir) if args.keep_dir else Path(tempfile.mkdtemp(prefix="odacc_cmp_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for log in logs:
        outs = {}
        status = {}
        for backend in ("symbolic", "smt"):
            out = out_dir / f"{log.stem}.{backend}.json"
            status[backend] = run_backend(Path(args.model), log, backend, out,
                                          args.cocomot_root, args.timeout)
            outs[backend] = out
        if status["symbolic"] or status["smt"]:
            print(f"SKIP {log.name}: symbolic={status['symbolic'] or 'ok'} smt={status['smt'] or 'ok'}")
            if status["symbolic"] is None:
                problems = malformed(load_prefixes(outs["symbolic"]))
                for p in problems:
                    print(f"  [symbolic] MALFORMED {p}")
                all_ok = all_ok and not problems
            continue
        ok, lines = compare(outs["symbolic"], outs["smt"], "symbolic", "smt")
        all_ok = all_ok and ok
        print(("OK  " if ok else "DIFF") + f" {log.name}")
        if not ok:
            print("\n".join(lines))
    print(f"outputs in {out_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
