#!/usr/bin/env python3
"""Create the result tables of the paper from experiment sessions.

Three methods are compared:

* SMT       -- the SMT reference (re-solves every prefix)          --smt SESSION
* Exact     -- the proposed method, exact (pure symbolic) mode     --exact SESSION
* Anytime   -- the proposed method, anytime mode with SMT check    --anytime SESSION

Optionally ``--smt-long SESSION`` (an SMT run with a larger per-update limit)
adds reference costs for checking correctness.  A session is a folder written
by the experiment runners (``all_runs_summary.csv`` and
``all_runs_prefix_timings.csv``); for a rerun, pass its ``merged`` subfolder.

Output (``--out DIR``):
  table_completion.tex / .csv   completion and certification per family
  table_runtime.tex / .csv      per-update time and paired speed-ups vs SMT
  table_correctness.tex / .csv  agreement with SMT, anytime gaps
  table_routes.tex / .csv       how updates were certified (Exact, Anytime)
  per_suite.csv                 the same figures per suite
  summary.md                    all tables as Markdown (for a quick look)

Example (Windows, one line):
  python tools\\make_results_tables.py --smt <anytime-smt-jointlb-lp> --exact <pure-jointlb-lp> --anytime <rerun>\\merged
         --smt-long <smt600> --out report\\tables
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = [
    ("Comprehensive", ("comprehensive",)),
    ("Object attributes", ("object_attributes",)),
    ("Scalability", ("component_scale", "deviation_scale", "length_scale", "merge_scale",
                     "guard_complexity")),
    ("Order components", ("components_online",)),
    ("Controlled mutations", ("controlled_mutations",)),
    ("Multi-deviation", ("multi_deviation",)),
]
ROUTES = [("direct", "Direct continuation"), ("local_repair", "Local repair"),
          ("joint_lb", "Joint lower bound"), ("checkpoint_exact", "Checkpoint-relative search"),
          ("full_exact", "Full-prefix search"), ("smt_exact", "SMT check"),
          ("uncertified", "Uncertified (deadline)")]


def family_of(experiment: str) -> str:
    for name, keys in FAMILIES:
        if any(k in experiment for k in keys):
            return name
    return "Other"


def suite_of(experiment: str) -> str:
    return experiment.replace("_online", "")


# ---------------------------------------------------------------------------
def load(session):
    if not session:
        return None
    root = Path(session)
    cases, updates = {}, defaultdict(list)
    summ = root / "all_runs_summary.csv"
    pref = root / "all_runs_prefix_timings.csv"
    if summ.exists():
        with summ.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                cases.setdefault((r["experiment"], r["file"]), r)
        if pref.exists():
            with pref.open(newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    updates[(r["experiment"], r["file"])].append(r)
    else:  # interrupted session: per-run files
        for f in glob.glob(str(root / "*" / "*" / "run_*" / "summary.csv")):
            exp = Path(f).parts[-4]
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    r["experiment"] = exp
                    cases.setdefault((exp, r["file"]), r)
            pf = Path(f).with_name("prefix_timings.csv")
            if pf.exists() and pf.stat().st_size:
                with pf.open(newline="", encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        r["experiment"] = exp
                        updates[(exp, r["file"])].append(r)
    for rows in updates.values():
        rows.sort(key=lambda r: int(r.get("update_index") or 0))
    return {"cases": cases, "updates": updates}


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def ok(sess, key):
    return sess is not None and sess["cases"].get(key, {}).get("online_status") == "ONLINE_OK"


def pct(a, b):
    return None if not b else 100.0 * a / b


def med(xs):
    return statistics.median(xs) if xs else None


def quant(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else None


def route_of(r):
    if str(r.get("certified", "")).lower() == "false":
        return "uncertified"
    detail = r.get("certification_route_detail") or ""
    if "joint_lb" in detail:
        return "joint_lb"
    route = r.get("certification_route") or detail.split(":")[0]
    return route if route in dict(ROUTES) else "full_exact" if route else "direct"


# ---------------------------------------------------------------------------
def compute(groups, S, X, A, L):
    """Figures per group; groups: {name: [case keys]}."""
    out = {}
    for name, keys in groups.items():
        g = {}
        g["cases"] = len(keys)
        g["updates"] = sum(max((len(s["updates"].get(k, [])) for s in (S, X, A, L) if s), default=0)
                           for k in keys)
        for label, sess in (("smt", S), ("exact", X), ("any", A)):
            if sess is None:
                continue
            all_rows = [r for k in keys for r in sess["updates"].get(k, [])]
            # updates skipped after a component timeout (SMT) are not solved
            rows = [r for r in all_rows if not r.get("skipped")]
            g[f"{label}_ok"] = sum(ok(sess, k) for k in keys)
            g[f"{label}_partial"] = sum(sess["cases"].get(k, {}).get("online_status") == "ONLINE_PARTIAL"
                                        for k in keys)
            g[f"{label}_skipped"] = len(all_rows) - len(rows)
            g[f"{label}_updates"] = len(rows)
            w = [num(r.get("wall_seconds")) for r in rows if num(r.get("wall_seconds")) is not None]
            g[f"{label}_med"] = med(w)
            g[f"{label}_p95"] = quant(w, 0.95)
            g[f"{label}_max"] = max(w) if w else None
            g[f"{label}_total_h"] = sum(w) / 3600.0 if w else None
            g[f"{label}_over_limit"] = sum(str(r.get("over_observation_limit", "")).lower() == "true"
                                           for r in rows)
            if label in ("exact", "any"):
                cert = sum(str(r.get("certified", "")).lower() == "true" for r in rows)
                g[f"{label}_cert"] = cert
                g[f"{label}_cert_pct"] = pct(cert, len(rows))
                routes = Counter(route_of(r) for r in rows)
                for rk, _ in ROUTES:
                    g[f"{label}_route_{rk}"] = pct(routes.get(rk, 0), len(rows))
        # paired speed-ups on cases completed by both with equal final cost
        for label, sess in (("exact", X), ("any", A)):
            if sess is None or S is None:
                continue
            ratios = []
            for k in keys:
                if ok(sess, k) and ok(S, k):
                    a, b = sess["cases"][k], S["cases"][k]
                    if num(a.get("prefix_total")) == num(b.get("prefix_total")):
                        ta, tb = num(a.get("online_total_seconds")), num(b.get("online_total_seconds"))
                        if ta and tb:
                            ratios.append(tb / ta)
            g[f"{label}_speedup_n"] = len(ratios)
            g[f"{label}_speedup_med"] = med(ratios)
            g[f"{label}_speedup_geo"] = geomean(ratios)
        # correctness: costs at prefixes reached by the method and a reference
        refs = [s for s in (S, L) if s is not None]
        for label, sess in (("exact", X), ("any", A)):
            if sess is None or not refs:
                continue
            checked = disagree = 0
            for k in keys:
                own = sess["updates"].get(k, [])
                for i, r in enumerate(own):
                    if str(r.get("certified", "")).lower() == "false":
                        continue
                    ref_costs = {num(ref["updates"][k][i].get("cost")) for ref in refs
                                 if i < len(ref["updates"].get(k, []))}
                    ref_costs.discard(None)
                    if not ref_costs:
                        continue
                    checked += 1
                    disagree += num(r.get("cost")) not in ref_costs
            g[f"{label}_checked"] = checked
            g[f"{label}_disagree"] = disagree
        # anytime gaps; optimum known from a reference?
        if A is not None:
            gaps, known, opt_cost, lb_opt = [], 0, 0, 0
            for k in keys:
                for i, r in enumerate(A["updates"].get(k, [])):
                    if str(r.get("certified", "")).lower() != "false":
                        continue
                    c, lb = num(r.get("cost")), num(r.get("lower_bound"))
                    if c is not None and lb is not None:
                        gaps.append(c - lb)
                    for ref in refs:
                        rows = ref["updates"].get(k, [])
                        if i < len(rows) and num(rows[i].get("cost")) is not None:
                            known += 1
                            o = num(rows[i].get("cost"))
                            opt_cost += c == o
                            lb_opt += lb == o
                            break
            g["any_uncert"] = len(gaps)
            g["any_gap_med"] = med(gaps)
            g["any_gap_max"] = max(gaps) if gaps else None
            g["any_uncert_ref"] = known
            g["any_uncert_cost_opt"] = opt_cost
            g["any_uncert_lb_opt"] = lb_opt
        out[name] = g
    return out


# ---------------------------------------------------------------------------
def fmt(v, kind="int"):
    if v is None:
        return "--"
    if kind == "int":
        return f"{int(round(v)):,}".replace(",", "{,}")
    if kind == "pct":
        return f"{v:.1f}"
    if kind == "sec":
        return f"{v:.2f}" if v < 10 else f"{v:.1f}"
    if kind == "x":
        return f"{v:.1f}$\\times$"
    return str(v)


def latex_table(path, caption, label, header, rows, colspec):
    lines = ["% generated by tools/make_results_tables.py", "\\begin{table}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize",
             "% shrink to the text width only if wider (needs graphicx)",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    lines += header
    lines.append("\\midrule")
    for i, r in enumerate(rows):
        if i == len(rows) - 1 and r[0] == "Overall":
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plain(v):
    return str(v).replace("{,}", ",").replace("$\\times$", "x") if isinstance(v, str) else v


def write_csv(path, rows, fields):
    rows = [{k: plain(v) for k, v in r.items()} for r in rows]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(plain(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smt", help="SMT reference session (same limits as the others)")
    ap.add_argument("--exact", help="proposed method, exact (pure symbolic) session")
    ap.add_argument("--anytime", help="proposed method, anytime session (rerun: its merged folder)")
    ap.add_argument("--smt-long", help="optional SMT session with a larger limit (verification)")
    ap.add_argument("--out", default="report/tables")
    args = ap.parse_args()
    S, X, A, L = load(args.smt), load(args.exact), load(args.anytime), load(args.smt_long)
    if not any((S, X, A)):
        raise SystemExit("give at least one of --smt, --exact, --anytime")
    keys = set()
    for s in (S, X, A):
        if s:
            keys |= set(s["cases"])
    by_family, by_suite = defaultdict(list), defaultdict(list)
    for k in sorted(keys):
        by_family[family_of(k[0])].append(k)
        by_suite[suite_of(k[0])].append(k)
    fam_order = [n for n, _ in FAMILIES if n in by_family] + [n for n in by_family if n not in dict(FAMILIES)]
    groups = {n: by_family[n] for n in fam_order}
    groups["Overall"] = sorted(keys)
    F = compute(groups, S, X, A, L)
    P = compute(dict(sorted(by_suite.items())), S, X, A, L)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = []

    # ---- Table 1: completion and certification
    show_over = any(F[n].get("exact_over_limit") for n in groups)
    rows = []
    for n in groups:
        r = [n, fmt(F[n]["cases"]), fmt(F[n]["updates"]), fmt(F[n].get("smt_ok")),
             fmt(F[n].get("smt_partial")), fmt(F[n].get("smt_updates")), fmt(F[n].get("exact_ok"))]
        if show_over:
            r.append(fmt(F[n].get("exact_over_limit")))
        r += [fmt(F[n].get("any_ok")), fmt(F[n].get("any_cert_pct"), "pct")]
        rows.append(r)
    hdr = ["Family", "Cases", "Updates", "SMT done", "SMT partial", "SMT solved upd.", "Exact done"]
    hdr += (["Exact upd. > limit"] if show_over else []) + ["Anytime done", "Anytime cert. %"]
    ex_cols = 2 if show_over else 1
    caption = ("Completion and certification. ``done'': cases processed to the last event "
               "with all updates solved; ``partial'': cases with given-up SMT components; "
               "``solved upd.'': updates solved by SMT; "
               + ("``upd.\\ $>$ limit'': exact-mode updates over the per-update limit; "
                  if show_over else "")
               + "``cert.\\ \\%'': share of certified anytime updates.")
    sub = " & & & done & partial & solved upd. & done" + (" & upd.\\ $>$ limit" if show_over else "") \
        + " & done & cert.\\ \\% \\\\"
    e0 = 7
    latex_table(out / "table_completion.tex", caption, "tab:completion",
                ["Family & Cases & Updates & \\multicolumn{3}{c}{SMT} & "
                 f"\\multicolumn{{{ex_cols}}}{{c}}{{Exact}} & \\multicolumn{{2}}{{c}}{{Anytime}} \\\\",
                 f"\\cmidrule(lr){{4-6}}\\cmidrule(lr){{{e0}-{e0 + ex_cols - 1}}}"
                 f"\\cmidrule(lr){{{e0 + ex_cols}-{e0 + ex_cols + 1}}}",
                 sub],
                rows, "l" + "r" * (len(hdr) - 1))
    write_csv(out / "table_completion.csv", [dict(zip(hdr, r)) for r in rows], hdr)
    md += ["## Completion and certification", md_table(hdr, rows)]

    # ---- Table 2: running time
    rows = [[n, fmt(F[n].get("smt_med"), "sec"), fmt(F[n].get("exact_med"), "sec"),
             fmt(F[n].get("any_med"), "sec"), fmt(F[n].get("exact_p95"), "sec"),
             fmt(F[n].get("any_p95"), "sec"),
             fmt(F[n].get("exact_speedup_med"), "x"), fmt(F[n].get("exact_speedup_n")),
             fmt(F[n].get("any_speedup_med"), "x"), fmt(F[n].get("any_speedup_n"))] for n in groups]
    hdr = ["Family", "SMT med. s", "Exact med. s", "Anytime med. s", "Exact p95 s", "Anytime p95 s",
           "Exact speed-up", "n", "Anytime speed-up", "n"]
    latex_table(out / "table_runtime.tex",
                "Time per update (median, 95th percentile; s) and median paired speed-up "
                "over SMT on the $n$ cases completed by both with equal final cost.",
                "tab:runtime",
                ["& \\multicolumn{3}{c}{Median s / update} & \\multicolumn{2}{c}{p95 s / update} & "
                 "\\multicolumn{4}{c}{Speed-up vs.\\ SMT} \\\\",
                 "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-10}",
                 "Family & SMT & Exact & Anytime & Exact & Anytime & Exact & $n$ & Anytime & $n$ \\\\"],
                rows, "lrrrrrrrrr")
    write_csv(out / "table_runtime.csv", [dict(zip(hdr, r)) for r in rows], hdr)
    md += ["## Running time", md_table(hdr, rows)]

    # ---- Table 3: correctness and anytime gaps
    show_dis = any(F[n].get("exact_disagree") or F[n].get("any_disagree") for n in groups)
    rows = []
    for n in groups:
        r = [n, fmt(F[n].get("exact_checked"))]
        if show_dis:
            r.append(fmt(F[n].get("exact_disagree")))
        r.append(fmt(F[n].get("any_checked")))
        if show_dis:
            r.append(fmt(F[n].get("any_disagree")))
        r += [fmt(F[n].get("any_uncert")), fmt(F[n].get("any_gap_med")), fmt(F[n].get("any_gap_max")),
              fmt(F[n].get("any_uncert_ref")), fmt(F[n].get("any_uncert_cost_opt"))]
        rows.append(r)
    hdr = ["Family", "Exact checked"] + (["Exact disagree"] if show_dis else []) + ["Anytime checked"] \
        + (["Anytime disagree"] if show_dis else []) \
        + ["Uncertified", "Median gap", "Max gap", "Optimum known", "Cost optimal"]
    if show_dis:
        head = ["& \\multicolumn{2}{c}{Exact} & \\multicolumn{2}{c}{Anytime} & "
                "\\multicolumn{5}{c}{Anytime, uncertified updates} \\\\",
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-10}",
                "Family & checked & disagree & checked & disagree & count & med.\\ gap & "
                "max gap & opt.\\ known & cost opt. \\\\"]
        cap_dis = ("``disagree'': of these, prefixes with a different cost. ")
    else:
        head = ["& Exact & Anytime & \\multicolumn{5}{c}{Anytime, uncertified updates} \\\\",
                "\\cmidrule(lr){2-2}\\cmidrule(lr){3-3}\\cmidrule(lr){4-8}",
                "Family & checked & checked & count & med.\\ gap & max gap & opt.\\ known & cost opt. \\\\"]
        cap_dis = "None differs. "
    latex_table(out / "table_correctness.tex",
                "Correctness and anytime gaps. ``checked'': certified prefixes compared with an "
                "SMT result. " + cap_dis + "Gap: returned cost $-$ proven bound; ``opt.\\ "
                "known'': uncertified updates with an SMT optimum; ``cost opt.'': of these, "
                "returned cost optimal.",
                "tab:correctness", head, rows, "l" + "r" * (len(hdr) - 1))
    write_csv(out / "table_correctness.csv", [dict(zip(hdr, r)) for r in rows], hdr)
    md += ["## Correctness and gaps", md_table(hdr, rows)]

    # ---- Table 4: certification routes (overall)
    rows = []
    for rk, rname in ROUTES:
        rows.append([rname, fmt(F["Overall"].get(f"exact_route_{rk}"), "pct"),
                     fmt(F["Overall"].get(f"any_route_{rk}"), "pct")])
    hdr = ["Route", "Exact %", "Anytime %"]
    latex_table(out / "table_routes.tex",
                "How updates were settled (share of all updates, \\%).",
                "tab:routes", ["Route & Exact & Anytime \\\\"], rows, "lrr")
    write_csv(out / "table_routes.csv", [dict(zip(hdr, r)) for r in rows], hdr)
    md += ["## Certification routes (overall)", md_table(hdr, rows)]

    # ---- per suite (CSV only)
    fields = sorted({k for g in P.values() for k in g})
    write_csv(out / "per_suite.csv", [dict(suite=s, **g) for s, g in P.items()], ["suite"] + fields)
    (out / "summary.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print("\n\n".join(md))
    print(f"\nTables written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
