from __future__ import annotations

import argparse
import faulthandler
import json
import os
import time

from .cocomot_adapter import CocomotBackend, DependencyError
from .symbolic_search import SymbolicIncrementalBackend
from .monitor import OnlineMonitor
from .ocel import extract_schema, load_ocel1, simulate_stream
from .report import result_dict, schema_to_json


def build_parser():
    p = argparse.ArgumentParser(
        prog="odacc",
        description="Online data-aware object-centric conformance checking prototype",
    )
    p.add_argument("--model", required=False, help="DOPID PNML model")
    p.add_argument("--log", required=True, help="OCEL 1.0 JSON log used to simulate the stream")
    p.add_argument("--cocomot-root", help="Path to the cocomot-main checkout")
    p.add_argument("--mode", choices=["online", "offline", "both", "stream"], default="both")
    p.add_argument("--backend", choices=["smt", "symbolic"], default="smt",
                   help="Online backend: smt whole-prefix baseline or symbolic incremental A* + JODAP")
    p.add_argument("--offline-backend", choices=["smt", "symbolic", "same"], default="smt",
                   help="Backend for final offline alignment (default: smt). Use 'same' to reuse --backend.")
    p.add_argument("--model-depth-margin", type=int, default=4,
                   help="Extra model-move exploration depth for the symbolic backend")
    p.add_argument("--jodap-contexts", type=int, default=24,
                   help="Maximum reusable incremental JODAP contexts (symbolic backend)")
    p.add_argument("--local-repair-cost-budget", type=float, default=4.0,
                   help="Maximum additional alignment cost explored by the bounded compositional local repair search")
    p.add_argument("--local-repair-max-candidates", type=int, default=48,
                   help="Maximum local repair candidates before exact A* fallback")
    p.add_argument("--local-repair-transition-cap", type=int, default=12,
                   help="Maximum structurally relevant model transitions considered by local repair search")
    p.add_argument("--fixed-objects", action="store_true",
                   help="Use all observed component objects in the model initial marking")
    p.add_argument("--solve-on-updates", action="store_true",
                   help="Recompute even for update-only observable units")
    p.add_argument("--output", help="Write the complete JSON result to this file")
    p.add_argument("--print-stream", action="store_true")
    p.add_argument("--quiet-solver", action="store_true",
                   help="Suppress CoCoMoT distance/move debug matrices")
    p.add_argument("--print-full-results", action="store_true",
                   help="Print the complete JSON result even when --output is used")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress schema and result summaries (errors are still shown)")
    p.add_argument("--progress-updates", action="store_true",
                   help="Emit machine-readable start/done lines for every online update")
    p.add_argument("--diagnostic-heartbeat", type=float, default=0.0, metavar="SECONDS",
                   help="While an online solve is running, print a heartbeat every N seconds; 0 disables it")
    p.add_argument("--diagnostic-stack", type=float, default=0.0, metavar="SECONDS",
                   help="Dump all Python thread stacks to stderr every N seconds; 0 disables it")
    p.add_argument("--symbolic-query-diagnostics", default=None, metavar="JSONL",
                   help="Write detailed lazy/eager JODAP phase diagnostics as JSONL")
    p.add_argument("--symbolic-query-diagnostics-print", action="store_true",
                   help="Echo detailed JODAP query diagnostics to stdout in real time")
    return p


def _cost_text(result):
    if result is None:
        return "-"
    if not result.get("feasible", False):
        return "INFEASIBLE"
    cb = result.get("cost_breakdown") or {}
    total = cb.get("total", result.get("cost"))
    cf = cb.get("control_flow", "?")
    data = cb.get("data", "?")
    obj = cb.get("object", "?")
    return f"total={total} (cf={cf}, data={data}, obj={obj})"


def _print_summary(args, stream, results):
    if args.quiet:
        return
    print(f"\nlog: {args.log}")
    print(f"observable units: {len(stream)}")

    prefix = results.get("prefix", [])
    if prefix:
        print(f"prefix alignments: {len(prefix)}")
        shown = prefix if len(prefix) <= 10 else prefix[:3] + [None] + prefix[-3:]
        for r in shown:
            if r is None:
                print("  ...")
                continue
            print(
                f"  component={r['component']} prefix={r['prefix_position']}: "
                f"{_cost_text(r)} "
                f"[encode={r.get('encode_seconds', 0):.3f}s, solve={r.get('solve_seconds', 0):.3f}s]"
            )

    offline = results.get("offline", [])
    if offline:
        print("offline alignment(s):")
        for r in offline:
            print(
                f"  component={r['component']}: {_cost_text(r)} "
                f"[encode={r.get('encode_seconds', 0):.3f}s, solve={r.get('solve_seconds', 0):.3f}s]"
            )

    timing = results.get("timing") or {}
    if timing:
        print(
            "timing: "
            f"online_total={timing.get('online_total_seconds', 0):.3f}s, "
            f"online_avg={timing.get('online_avg_seconds', 0):.3f}s, "
            f"online_max={timing.get('online_max_seconds', 0):.3f}s, "
            f"offline={timing.get('offline_total_seconds', 0):.3f}s"
        )

    progress = results.get("progress") or {}
    if progress:
        print(
            "progress: "
            f"{progress.get('completed_stream_units', 0)}/{progress.get('total_stream_units', '?')} units, "
            f"{progress.get('completed_online_updates', 0)} solved prefixes, "
            f"complete={progress.get('complete', False)}"
        )

    if args.output:
        print(f"full result: {os.path.abspath(args.output)}")


def _replace_with_retry(src: str, dst: str, *, attempts: int = 10) -> None:
    """Replace a checkpoint robustly on Windows.

    Antivirus/indexer processes can briefly hold either the old JSON or the
    freshly closed temporary file, causing ``os.replace`` to raise WinError 5.
    Retry with bounded exponential backoff; genuine persistent failures are
    still surfaced after the final attempt.
    """
    delay = 0.025
    last = None
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.8, 0.75)
    if last is not None:
        raise last


def _atomic_write_json(path: str, payload: dict) -> None:
    """Atomically replace ``path`` with ``payload`` with Windows-safe retries."""
    target = os.path.abspath(path)
    outdir = os.path.dirname(target)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    # A per-process temporary name avoids collisions with a stale/locked .part
    # file from an interrupted prior run.
    tmp = f"{target}.part.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    try:
        _replace_with_retry(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _unit_meta(unit, component, merged):
    event = unit.event
    try:
        comp_events = len(component.execution.event_ids) if component is not None else 0
    except Exception:
        comp_events = sum(
            1 for u in getattr(component, "units", ()) if getattr(u, "event", None) is not None
        ) if component is not None else 0
    return {
        "stream_position": unit.position,
        "component": component.component_id if component is not None else None,
        "merged": bool(merged),
        "event_id": event.event_id if event is not None else None,
        "activity": event.activity if event is not None else None,
        "event_object_count": len(event.objects) if event is not None else 0,
        "component_event_count": comp_events,
        "component_object_count": len(component.objects) if component is not None else 0,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.diagnostic_stack > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.diagnostic_stack, repeat=True)

    data = load_ocel1(args.log)
    schema = extract_schema(data)
    stream = simulate_stream(data)
    schema_json = json.loads(schema_to_json(schema))

    if not args.quiet and not args.output:
        print("OBJECT-CENTRIC SCHEMA")
        print(schema_to_json(schema))
        print(f"\nobservable units: {len(stream)}")

    if args.print_stream and not args.quiet:
        for u in stream:
            print(u)
    if args.mode == "stream":
        return 0
    if not args.model or not args.cocomot_root:
        raise SystemExit("--model and --cocomot-root are required for conformance checking")

    def make_backend(name):
        if name == "symbolic":
            return SymbolicIncrementalBackend(
                args.cocomot_root, args.model, args.fixed_objects,
                model_depth_margin=args.model_depth_margin,
                max_prepared_contexts=args.jodap_contexts,
                query_diagnostics_path=args.symbolic_query_diagnostics,
                query_diagnostics_print=args.symbolic_query_diagnostics_print,
                local_repair_cost_budget=args.local_repair_cost_budget,
                local_repair_max_candidates=args.local_repair_max_candidates,
                local_repair_transition_cap=args.local_repair_transition_cap,
            )
        return CocomotBackend(
            args.cocomot_root, args.model, args.fixed_objects,
            quiet_solver=args.quiet_solver,
        )

    try:
        backend = make_backend(args.backend)
        offline_name = args.backend if args.offline_backend == "same" else args.offline_backend
        offline_backend = backend if offline_name == args.backend else make_backend(offline_name)
    except DependencyError as exc:
        raise SystemExit(str(exc))

    monitor = OnlineMonitor(
        backend,
        offline_backend=offline_backend,
        diagnostic_heartbeat_seconds=args.diagnostic_heartbeat,
    )
    online = args.mode in ("online", "both")
    offline = args.mode in ("offline", "both")
    results = {"schema": schema_json, "prefix": [], "offline": [], "timing": {}}

    def result_snapshot(run, *, progress: dict, complete: bool, offline_complete: bool = False):
        p_times = list(run.prefix_wall_seconds)
        o_times = list(run.offline_wall_seconds)
        snap = {
            "schema": schema_json,
            "prefix": (
                [result_dict(r) for r in run.prefix_results]
                if complete else [dict(x) for x in run.certified_prefix_snapshots]
            ),
            "offline": [result_dict(r) for r in run.offline_results] if complete else [],
            "timing": {
                "online_updates": len(p_times),
                "online_wall_seconds": p_times,
                "prefix_updates": list(run.prefix_timing_records),
                "online_total_seconds": sum(p_times),
                "online_avg_seconds": (sum(p_times) / len(p_times)) if p_times else 0.0,
                "online_max_seconds": max(p_times) if p_times else 0.0,
                "offline_components": len(o_times),
                "offline_wall_seconds": o_times,
                "offline_total_seconds": sum(o_times),
                "conformance_total_seconds": sum(p_times) + sum(o_times),
            },
            "progress": {
                "complete": bool(complete),
                "online_complete": bool(complete),
                "offline_complete": bool(offline_complete),
                "total_stream_units": len(stream),
                "completed_stream_units": run.completed_stream_units,
                "completed_online_updates": len(run.certified_prefix_snapshots),
                "last_completed_stream_position": run.last_completed_stream_position,
                **progress,
            },
            "backends": {
                "online": args.backend,
                "offline": (args.backend if args.offline_backend == "same" else args.offline_backend),
            },
        }
        if args.backend == "symbolic":
            snap["symbolic_stats"] = dict(backend.jodap.stats)
        return snap

    if online:
        progress_seq = {"n": 0}

        def progress_callback(phase, unit, component, merged, result, elapsed):
            if not args.progress_updates and phase != "heartbeat":
                return
            if phase == "heartbeat" and args.diagnostic_heartbeat <= 0:
                return
            if phase == "start":
                progress_seq["n"] += 1
            event = unit.event
            activity = event.activity if event is not None else "update"
            event_id = event.event_id if event is not None else "-"
            comp_id = component.component_id if component is not None else "-"
            event_objs = len(event.objects) if event is not None else 0
            comp_objs = len(component.objects) if component is not None else 0
            try:
                comp_events = len(component.execution.event_ids) if component is not None else 0
            except Exception:
                comp_events = sum(
                    1 for u in getattr(component, "units", ()) if getattr(u, "event", None) is not None
                ) if component is not None else 0
            fields = [
                "ODACC_UPDATE", phase, str(progress_seq["n"]),
                str(unit.position), str(comp_id), "1" if merged else "0",
                str(event_id), str(activity).replace("\t", " "),
                str(event_objs), str(comp_events), str(comp_objs),
            ]
            if phase == "done":
                cb = (result.cost_breakdown or {}) if result is not None else {}
                fields += [
                    f"{elapsed:.6f}",
                    str(cb.get("total", result.cost if result is not None else "-")),
                ]
            elif phase == "heartbeat":
                fields += [f"{elapsed:.6f}"]
            print("\t".join(fields), flush=True)

        def checkpoint_callback(phase, run, unit, component, merged, result, elapsed):
            if not args.output or phase not in ("start", "heartbeat", "done"):
                return
            meta = _unit_meta(unit, component, merged)
            if phase in ("start", "heartbeat"):
                progress = {
                    "status": "RUNNING",
                    "current_unit": {
                        **meta,
                        **({"wall_seconds": elapsed} if phase == "heartbeat" and elapsed is not None else {}),
                    },
                    "last_completed_unit": None,
                }
            else:
                progress = {
                    "status": "RUNNING",
                    "current_unit": None,
                    "last_completed_unit": {**meta, "wall_seconds": elapsed},
                }
            _atomic_write_json(
                args.output,
                result_snapshot(run, progress=progress, complete=False, offline_complete=False),
            )

        run = monitor.run(
            stream,
            offline_at_end=offline,
            solve_on_updates=args.solve_on_updates,
            progress_callback=progress_callback,
            checkpoint_callback=checkpoint_callback,
        )
        results = result_snapshot(
            run,
            progress={"status": "COMPLETE", "current_unit": None, "last_completed_unit": None},
            complete=True,
            offline_complete=offline,
        )
    else:
        # Consume stream without prefix solves, then solve each final component.
        for u in stream:
            monitor.components.apply(u)
        o_times = []
        for comp in sorted(monitor.components.components.values(), key=lambda c: c.component_id):
            t0 = time.perf_counter()
            results["offline"].append(result_dict(offline_backend.solve(comp, offline=True)))
            o_times.append(time.perf_counter() - t0)
        results["timing"] = {
            "online_updates": 0,
            "online_wall_seconds": [],
            "prefix_updates": [],
            "online_total_seconds": 0.0,
            "online_avg_seconds": 0.0,
            "online_max_seconds": 0.0,
            "offline_components": len(o_times),
            "offline_wall_seconds": o_times,
            "offline_total_seconds": sum(o_times),
            "conformance_total_seconds": sum(o_times),
        }
        results["progress"] = {
            "status": "COMPLETE",
            "complete": True,
            "online_complete": False,
            "offline_complete": True,
            "total_stream_units": len(stream),
            "completed_stream_units": len(stream),
            "completed_online_updates": 0,
            "last_completed_stream_position": stream[-1].position if stream else None,
            "current_unit": None,
        }
        if args.backend == "symbolic":
            results["symbolic_stats"] = dict(backend.jodap.stats)
        results["backends"] = {
            "online": args.backend,
            "offline": (args.backend if args.offline_backend == "same" else args.offline_backend),
        }

    if args.output:
        _atomic_write_json(args.output, results)

    if args.diagnostic_stack > 0:
        faulthandler.cancel_dump_traceback_later()

    text = json.dumps(results, indent=2)
    if args.print_full_results or not args.output:
        if not args.quiet:
            print(text)
    else:
        _print_summary(args, stream, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
