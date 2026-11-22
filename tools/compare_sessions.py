#!/usr/bin/env python3
"""Compare two experiment sessions (e.g. symbolic vs SMT, or two SMT variants).

A session is a directory written by ``experiments/run_online_*.py``
(``experiment_runs/<stamp>[_variant]``).  Complete sessions contain
``all_runs_summary.csv`` and ``all_runs_prefix_timings.csv``; for interrupted
sessions the per-run ``summary.csv``/``prefix_timings.csv`` files are read.

For every experiment (suite) the report gives: cases, completed runs (online
status ONLINE_OK) in A and B, jointly completed cases, agreement of the final
cost, agreement of the cost at *every* prefix reported by both, median total
online time on jointly completed cost-agreeing cases, median and geometric-mean
paired speedup (B time / A time, i.e. > 1 means A is faster), and the
certification-route distribution of A.

Usage::

    python tools/compare_sessions.py SESSION_A SESSION_B \\
        [--label-a symbolic] [--label-b smt] [--out-dir report/]

``--out-dir`` additionally writes ``suite_summary.csv``,
``case_comparison.csv`` and ``prefix_disagreements.csv``.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_session(root: Path):
    root = Path(root)
    summary, prefixes = [], []
    if (root / "all_runs_summary.csv").exists():
        summary = _read(root / "all_runs_summary.csv")
        if (root / "all_runs_prefix_timings.csv").exists():
            prefixes = _read(root / "all_runs_prefix_timings.csv")
    else:
        for f in sorted(root.glob("*/*/run_*/summary.csv")):
            exp = f.parts[-4]
            for r in _read(f):
                r["experiment"] = exp
                summary.append(r)
            pf = f.with_name("prefix_timings.csv")
            if pf.exists() and pf.stat().st_size > 0:
                for r in _read(pf):
                    r["experiment"] = exp
                    prefixes.append(r)
    # one row per (experiment, file): first repetition
    cases = {}
    for r in summary:
        cases.setdefault((r["experiment"], r["file"]), r)
    pref = {}
    for r in prefixes:
        key = (r["experiment"], r["file"], str(r.get("update_index")))
        pref.setdefault(key, r)
    return cases, pref


def _uncertified(row) -> bool:
    return str(row.get("certified", "")).strip().lower() in ("false", "0")


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_a")
    ap.add_argument("session_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    A, B = args.label_a, args.label_b
    ca, pa = load_session(Path(args.session_a))
    cb, pb = load_session(Path(args.session_b))
    if not ca or not cb:
        print("no runs found in one of the sessions", file=sys.stderr)
        return 2

    by_prefix_case = defaultdict(list)
    for key, r in pa.items():
        if key in pb:
            by_prefix_case[key[:2]].append((key[2], r, pb[key]))

    suites = defaultdict(lambda: Counter())
    times = defaultdict(list)
    routes = defaultdict(Counter)
    case_rows, prefix_rows = [], []
    for key in sorted(set(ca) | set(cb)):
        exp = key[0]
        ra, rb = ca.get(key), cb.get(key)
        st = suites[exp]
        st["cases"] += 1
        st[f"{A}_runs"] += ra is not None
        st[f"{B}_runs"] += rb is not None
        oka = ra is not None and ra.get("online_status") == "ONLINE_OK"
        okb = rb is not None and rb.get("online_status") == "ONLINE_OK"
        st[f"{A}_ok"] += oka
        st[f"{B}_ok"] += okb
        pairs = by_prefix_case.get(key, [])
        n_pref = len(pairs)
        n_pref_agree = 0
        for upd, x, y in pairs:
            if (x.get("skipped") or y.get("skipped")
                    or str(x.get("cost", "")).strip() in ("", "None")
                    or str(y.get("cost", "")).strip() in ("", "None")):
                # skipped after a component timeout (SMT): no cost to compare
                st["prefixes_skipped"] = st.get("prefixes_skipped", 0) + 1
                continue
            if _uncertified(x) or _uncertified(y):
                # Anytime-mode result: its cost is only an upper bound.
                st["prefixes_uncertified"] += 1
                for r in (x, y):
                    if _uncertified(r):
                        c, lb = fnum(r.get("cost")), fnum(r.get("lower_bound"))
                        if c is not None and lb is not None:
                            st["max_gap"] = max(st["max_gap"], int(c - lb))
                continue
            cx, cy = fnum(x.get("cost")), fnum(y.get("cost"))
            if cx is not None and cy is not None and cx == cy:
                n_pref_agree += 1
            else:
                prefix_rows.append({"experiment": exp, "file": key[1], "update_index": upd,
                                    "activity": x.get("activity"), f"{A}_cost": x.get("cost"),
                                    f"{B}_cost": y.get("cost"),
                                    f"{A}_route": x.get("certification_route_detail") or x.get("certification_route")})
        st["prefixes_compared"] += n_pref
        st["prefixes_agree"] += n_pref_agree
        final_agree = None
        speed = None
        final_uncertified = False
        if pairs:
            _upd, lx, ly = max(pairs, key=lambda t: fnum(t[0]) or 0)
            final_uncertified = _uncertified(lx) or _uncertified(ly)
        if oka and okb and final_uncertified:
            st["both_ok"] += 1
            st["final_uncertified"] += 1
        elif oka and okb:
            st["both_ok"] += 1
            final_agree = fnum(ra.get("prefix_total")) == fnum(rb.get("prefix_total"))
            st["final_agree"] += final_agree
            ta, tb = fnum(ra.get("online_total_seconds")), fnum(rb.get("online_total_seconds"))
            if final_agree and ta and tb:
                times[exp].append((ta, tb))
                speed = tb / ta
        case_rows.append({"experiment": exp, "file": key[1],
                          f"{A}_status": ra.get("online_status") if ra else None,
                          f"{B}_status": rb.get("online_status") if rb else None,
                          f"{A}_cost": ra.get("prefix_total") if ra else None,
                          f"{B}_cost": rb.get("prefix_total") if rb else None,
                          f"{A}_seconds": ra.get("online_total_seconds") if ra else None,
                          f"{B}_seconds": rb.get("online_total_seconds") if rb else None,
                          "final_cost_agree": final_agree, "prefixes_compared": n_pref,
                          "prefixes_agree": n_pref_agree, f"speedup_{B}_over_{A}": speed})
    for key, r in pa.items():
        route = r.get("certification_route")
        if route:
            routes[key[0]][route] += 1

    suite_rows = []
    total = Counter()
    all_times = []
    for exp in sorted(suites):
        st = suites[exp]
        tl = times.get(exp, [])
        all_times += tl
        for k, v in st.items():
            if k == "max_gap":
                total[k] = max(total[k], v)
            else:
                total[k] += v
        suite_rows.append(_suite_row(exp, st, tl, routes.get(exp, Counter()), A, B))
    total_routes = Counter()
    for c in routes.values():
        total_routes.update(c)
    suite_rows.append(_suite_row("OVERALL", total, all_times, total_routes, A, B))

    cols = list(suite_rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in suite_rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in suite_rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"\nPrefix disagreements: {len(prefix_rows)} (cost differs at an update reported by both)")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("suite_summary.csv", suite_rows), ("case_comparison.csv", case_rows),
                           ("prefix_disagreements.csv", prefix_rows)):
            with (out / name).open("w", newline="", encoding="utf-8") as fh:
                if rows:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
        print(f"CSV files written to {out}")
    return 0


def _suite_row(exp, st, tl, routes, A, B):
    med_a = statistics.median([a for a, _ in tl]) if tl else None
    med_b = statistics.median([b for _, b in tl]) if tl else None
    ratios = [b / a for a, b in tl if a > 0 and b > 0]
    n_routes = sum(routes.values())
    route_txt = ", ".join(f"{k} {100.0 * v / n_routes:.1f}%" for k, v in routes.most_common()) if n_routes else ""
    return {
        "suite": exp, "cases": st["cases"], f"{A}_runs": st[f"{A}_runs"], f"{B}_runs": st[f"{B}_runs"],
        f"{A}_ok": st[f"{A}_ok"], f"{B}_ok": st[f"{B}_ok"],
        "both_ok": st["both_ok"], "final_agree": f"{st['final_agree']}/{st['both_ok'] - st['final_uncertified']}",
        "prefix_agree": f"{st['prefixes_agree']}/{st['prefixes_compared'] - st['prefixes_uncertified']}",
        "uncertified": st["prefixes_uncertified"],
        "max_gap": st["max_gap"],
        f"{A}_median_s": None if med_a is None else round(med_a, 2),
        f"{B}_median_s": None if med_b is None else round(med_b, 2),
        "median_speedup": round(statistics.median(ratios), 2) if ratios else None,
        "geomean_speedup": round(math.exp(statistics.fmean(math.log(x) for x in ratios)), 2) if ratios else None,
        f"{A}_routes": route_txt,
    }


if __name__ == "__main__":
    sys.exit(main())
