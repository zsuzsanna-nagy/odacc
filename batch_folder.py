#!/usr/bin/env python3
"""Batch-run ODACC on every OCEL JSON file in a folder.

Each log receives a complete JSON result file.  The terminal prints only a
one-line cost summary, and the output folder also gets summary.json and
summary.csv for later analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import signal


def _replace_with_retry(src, dst, attempts=10):
    """Windows-safe atomic replace for result/checkpoint files."""
    delay = 0.025
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.8, 0.75)


def parser():
    p = argparse.ArgumentParser(description="Run ODACC on all *.jsonocel files in a folder")
    p.add_argument("--folder", required=True, help="Folder containing the OCEL files")
    p.add_argument("--model", help="PNML model; defaults to <folder>/net.pnml")
    p.add_argument("--cocomot-root", required=True)
    p.add_argument("--out", default="batch-results", help="Output directory")
    p.add_argument("--mode", choices=["online", "offline", "both"], default="both")
    p.add_argument("--backend", choices=["symbolic", "smt"], default="symbolic",
                   help="Backend used for online prefix computation")
    p.add_argument("--offline-backend", choices=["smt", "symbolic", "same"], default="smt",
                   help="Backend used for final offline alignment (default: smt)")
    p.add_argument("--online-timeout", type=float, default=120.0,
                   help="Timeout in seconds for each file's complete online phase; <=0 disables the timeout")
    p.add_argument("--observation-timeout", type=float, default=0.0, metavar="SECONDS",
                   help=("Maximum processing time for one online observation/update. "
                         "<=0 disables the per-observation timeout. When it fires, "
                         "the ODACC process is terminated, the latest certified checkpoint "
                         "is preserved, and the file is marked OBSERVATION_TIMEOUT."))
    p.add_argument("--offline-timeout", type=float, default=120.0,
                   help="Timeout in seconds for each file's offline phase; <=0 disables the timeout")
    p.add_argument("--model-depth-margin", type=int, default=4,
                   help="Extra model-move depth for the symbolic backend")
    p.add_argument("--jodap-contexts", type=int, default=24,
                   help="Maximum reusable incremental JODAP contexts per run")
    p.add_argument("--local-repair-cost-budget", type=float, default=4.0,
                   help="Additional alignment-cost budget for compositional local repair")
    p.add_argument("--local-repair-max-candidates", type=int, default=48,
                   help="Maximum local-repair candidates before exact A* fallback")
    p.add_argument("--local-repair-transition-cap", type=int, default=12,
                   help="Maximum structurally relevant model transitions considered locally")
    p.add_argument("--pattern", default="*.jsonocel")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--fixed-objects", action="store_true")
    p.add_argument("--solve-on-updates", action="store_true")
    p.add_argument("--odacc", default=str(Path(__file__).with_name("odacc.py")))
    p.add_argument("--show-errors", action="store_true",
                   help="Print full stderr for failed logs")
    p.add_argument("--progress-updates", action="store_true",
                   help="Print each online update as it starts/completes; useful for merge/deviation stress tests")
    p.add_argument("--diagnostic-heartbeat", type=float, default=0.0, metavar="SECONDS",
                   help="Print the currently running online observation every N seconds; 0 disables it")
    p.add_argument("--diagnostic-stack", type=float, default=0.0, metavar="SECONDS",
                   help="Ask ODACC to dump Python stacks to stderr every N seconds; 0 disables it")
    p.add_argument("--symbolic-query-diagnostics", action="store_true",
                   help="Save per-file JODAP lazy/eager phase diagnostics and print them live")
    p.add_argument("--native-crash-retries", type=int, default=1, metavar="N",
                   help=("Retry an online run in a fresh isolated process after a native crash "
                         "(e.g. Z3 access violation/SIGSEGV). Default: 1; 0 disables retry."))
    return p


def cost_fields(result, prefix):
    if result is None:
        return {
            f"{prefix}_feasible": None,
            f"{prefix}_total": None,
            f"{prefix}_control_flow": None,
            f"{prefix}_data": None,
            f"{prefix}_object": None,
            f"{prefix}_encode_seconds": None,
            f"{prefix}_solve_seconds": None,
        }
    cb = result.get("cost_breakdown") or {}
    return {
        f"{prefix}_feasible": result.get("feasible"),
        f"{prefix}_total": cb.get("total", result.get("cost")),
        f"{prefix}_control_flow": cb.get("control_flow"),
        f"{prefix}_data": cb.get("data"),
        f"{prefix}_object": cb.get("object"),
        f"{prefix}_encode_seconds": result.get("encode_seconds"),
        f"{prefix}_solve_seconds": result.get("solve_seconds"),
    }


def fmt(rec, prefix):
    feasible = rec.get(f"{prefix}_feasible")
    if feasible is None:
        return "-"
    if not feasible:
        return "INFEASIBLE"
    return (
        f"{rec.get(f'{prefix}_total')} "
        f"[cf={rec.get(f'{prefix}_control_flow')}, "
        f"d={rec.get(f'{prefix}_data')}, o={rec.get(f'{prefix}_object')}]"
    )



def _load_json_if_exists(path: Path):
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def _preserve_partial_online(
    rec,
    data,
    outfile: Path,
    rel,
    args,
    offline_backend,
    prefix_timing_rows,
    *,
    termination_reason="online_timeout",
):
    """Keep the last atomic CLI checkpoint when an online subprocess is killed."""
    if not isinstance(data, dict):
        return False
    prefix = data.get("prefix") or []
    prefix_result = prefix[-1] if prefix else None
    rec.update(cost_fields(prefix_result, "prefix"))
    rec["prefix_count"] = len(prefix)
    timing = data.get("timing") or {}
    rec["online_updates"] = timing.get("online_updates")
    rec["online_total_seconds"] = timing.get("online_total_seconds")
    rec["online_avg_seconds"] = timing.get("online_avg_seconds")
    rec["online_max_seconds"] = timing.get("online_max_seconds")
    rec["conformance_total_seconds"] = timing.get("conformance_total_seconds")
    progress = data.get("progress") or {}
    rec["partial_result_saved"] = True
    rec["completed_stream_units"] = progress.get("completed_stream_units")
    rec["total_stream_units"] = progress.get("total_stream_units")
    rec["completed_online_updates"] = progress.get("completed_online_updates")
    rec["last_completed_stream_position"] = progress.get("last_completed_stream_position")
    current = progress.get("current_unit") or {}
    rec["timeout_stream_position"] = current.get("stream_position")
    rec["timeout_event_id"] = current.get("event_id")
    rec["timeout_activity"] = current.get("activity")
    rec["timeout_component"] = current.get("component")
    rec["timeout_component_events"] = current.get("component_event_count")
    rec["timeout_component_objects"] = current.get("component_object_count")

    timing_records = timing.get("prefix_updates") or []
    for tr in timing_records:
        row = {
            "file": str(rel),
            "online_backend": args.backend,
            "offline_backend": offline_backend,
            "partial": True,
        }
        row.update(tr)
        prefix_timing_rows.append(row)
    rec["prefix_timing_count"] = len(timing_records)

    stats = data.get("symbolic_stats") or {}
    if isinstance(stats, dict):
        for key, value in stats.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                rec[f"symbolic_{key}"] = value

    # Keep the final per-log filename stable for downstream scripts even though
    # the run was interrupted.  The progress.complete flag remains false.
    payload = dict(data)
    payload["backends"] = {"online": args.backend, "offline": offline_backend}
    progress_payload = payload.setdefault("progress", {})
    progress_payload["terminated_by_batch_timeout"] = termination_reason in ("online_timeout", "observation_timeout")
    progress_payload["termination_reason"] = termination_reason
    if termination_reason == "native_crash":
        progress_payload["terminated_by_native_crash"] = True
    if termination_reason == "observation_timeout":
        progress_payload["terminated_by_observation_timeout"] = True
        progress_payload["observation_timeout_seconds"] = args.observation_timeout
        progress_payload["observation_elapsed_seconds"] = rec.get("observation_elapsed_seconds")
    tmp = outfile.with_suffix(outfile.suffix + f".part.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    try:
        _replace_with_retry(tmp, outfile)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return True


def _popen_isolation_kwargs():
    """Return platform-specific flags that isolate the child process tree."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def _kill_process_tree(proc):
    """Best-effort hard termination of an isolated ODACC process tree."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            # /T also terminates descendants.  Suppress taskkill chatter.
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _native_crash_reason(returncode, output=""):
    """Classify common native-process failures without relying on Python exceptions."""
    if returncode is None or returncode == 0:
        return None
    rc = int(returncode)
    unsigned = rc & 0xFFFFFFFF
    windows = {
        0xC0000005: "ACCESS_VIOLATION",
        0xC0000409: "STACK_BUFFER_OVERRUN",
        0xC000001D: "ILLEGAL_INSTRUCTION",
        0xC0000094: "INTEGER_DIVIDE_BY_ZERO",
    }
    if unsigned in windows:
        return windows[unsigned]
    if os.name != "nt" and rc < 0:
        sig = -rc
        if sig == getattr(signal, "SIGSEGV", 11):
            return "SIGSEGV"
        if sig == getattr(signal, "SIGABRT", 6):
            return "SIGABRT"
        if sig == getattr(signal, "SIGBUS", 7):
            return "SIGBUS"
        if sig == getattr(signal, "SIGILL", 4):
            return "SIGILL"
    low = (output or "").lower()
    if "access violation" in low or "segmentation fault" in low or "fatal python error" in low:
        return "NATIVE_CRASH_TEXT"
    return None


def _partial_score(data):
    if not isinstance(data, dict):
        return (-1, -1)
    progress = data.get("progress") or {}
    return (
        len(data.get("prefix") or []),
        int(progress.get("completed_stream_units") or 0),
    )


def _better_partial(a, b):
    return b if _partial_score(b) > _partial_score(a) else a


def _run_isolated_online(cmd, args, checkpoint_path: Path):
    """Run one online ODACC phase in a fresh process with crash containment.

    Native crashes are retried from scratch in a brand-new process.  No Z3
    objects cross the boundary.  The best atomic certified checkpoint observed
    across failed attempts is returned to the caller for final preservation.
    """
    max_retries = max(0, int(getattr(args, "native_crash_retries", 0)))
    best_partial = None
    native_failures = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Do not let a stale checkpoint from the previous crashed process be
            # mistaken for progress made by the retry.
            try:
                checkpoint_path.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"       native crash retry {attempt}/{max_retries}: fresh ODACC/Z3 process", flush=True)

        popen_kw = _popen_isolation_kwargs()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, **popen_kw,
        )
        online_timed_out = False
        observation_timed_out = False
        observation_timeout_meta = {}
        observation_state = {"started_at": None, "meta": None}
        observation_lock = threading.Lock()

        def kill_online():
            nonlocal online_timed_out
            if proc.poll() is None:
                online_timed_out = True
                _kill_process_tree(proc)

        def observation_watchdog():
            nonlocal observation_timed_out, observation_timeout_meta
            if args.observation_timeout <= 0:
                return
            sleep_for = min(0.10, max(0.02, args.observation_timeout / 20.0))
            while proc.poll() is None:
                with observation_lock:
                    started_at = observation_state["started_at"]
                    meta = observation_state["meta"]
                if started_at is not None:
                    elapsed = time.perf_counter() - started_at
                    if elapsed >= args.observation_timeout:
                        observation_timed_out = True
                        observation_timeout_meta = dict(meta or {})
                        observation_timeout_meta["elapsed_seconds"] = elapsed
                        _kill_process_tree(proc)
                        return
                time.sleep(sleep_for)

        online_timer = None
        if args.online_timeout > 0:
            online_timer = threading.Timer(args.online_timeout, kill_online)
            online_timer.daemon = True
            online_timer.start()

        observation_thread = None
        if args.observation_timeout > 0:
            observation_thread = threading.Thread(
                target=observation_watchdog, name="odacc-observation-watchdog", daemon=True
            )
            observation_thread.start()

        lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            text = line.rstrip("\r\n")
            if not text.startswith("ODACC_UPDATE\t"):
                continue
            parts = text.split("\t")
            if len(parts) < 11:
                continue
            (_, phase, upd, stream_pos, comp, merged_flag, event_id, activity,
             event_objs, comp_events, comp_objs, *tail) = parts
            merge_tag = " [MERGE]" if merged_flag == "1" else ""
            size_tag = f" event_objs={event_objs} comp_events={comp_events} comp_objs={comp_objs}"
            if phase == "start":
                with observation_lock:
                    observation_state["started_at"] = time.perf_counter()
                    observation_state["meta"] = {
                        "update": int(upd) if str(upd).isdigit() else upd,
                        "stream_position": int(stream_pos) if str(stream_pos).isdigit() else stream_pos,
                        "component": int(comp) if str(comp).isdigit() else comp,
                        "event_id": event_id, "activity": activity,
                        "event_objects": int(event_objs) if str(event_objs).isdigit() else event_objs,
                        "component_events": int(comp_events) if str(comp_events).isdigit() else comp_events,
                        "component_objects": int(comp_objs) if str(comp_objs).isdigit() else comp_objs,
                    }
                print(f"       update {upd} stream={stream_pos} comp={comp}{merge_tag}: "
                      f"{activity} ({event_id}) running...{size_tag}", flush=True)
            elif phase == "heartbeat":
                elapsed = tail[0] if tail else "?"
                print(f"       update {upd} stream={stream_pos} comp={comp}{merge_tag}: "
                      f"still running after {elapsed}s -- {activity} ({event_id}){size_tag}", flush=True)
            else:
                with observation_lock:
                    observation_state["started_at"] = None
                    observation_state["meta"] = None
                elapsed = tail[0] if tail else "?"
                cost = tail[1] if len(tail) > 1 else "?"
                print(f"       update {upd} stream={stream_pos} comp={comp}{merge_tag}: "
                      f"done in {elapsed}s cost={cost}{size_tag}", flush=True)

        rc = proc.wait()
        if online_timer is not None:
            online_timer.cancel()
        if observation_thread is not None:
            observation_thread.join(timeout=0.25)
        output = "".join(lines)
        partial = _load_json_if_exists(checkpoint_path)
        best_partial = _better_partial(best_partial, partial)

        if observation_timed_out:
            return {"kind": "observation_timeout", "returncode": 124, "stdout": output,
                    "partial": best_partial, "observation_meta": observation_timeout_meta,
                    "native_failures": native_failures, "retry_count": attempt}
        if online_timed_out:
            return {"kind": "online_timeout", "returncode": 124, "stdout": output,
                    "partial": best_partial, "native_failures": native_failures,
                    "retry_count": attempt}
        native = _native_crash_reason(rc, output)
        if native is not None:
            native_failures.append({"attempt": attempt + 1, "returncode": rc, "reason": native})
            if attempt < max_retries:
                continue
            return {"kind": "native_crash", "returncode": rc, "stdout": output,
                    "partial": best_partial, "native_reason": native,
                    "native_failures": native_failures, "retry_count": attempt}
        return {"kind": "ok" if rc == 0 else "error", "returncode": rc,
                "stdout": output, "partial": partial,
                "native_failures": native_failures, "retry_count": attempt}

    raise AssertionError("unreachable")

def main():
    args = parser().parse_args()
    folder = Path(args.folder).resolve()
    model = Path(args.model).resolve() if args.model else folder / "net.pnml"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    logs = sorted(folder.rglob(args.pattern) if args.recursive else folder.glob(args.pattern))
    if not logs:
        raise SystemExit(f"No files matching {args.pattern!r} found in {folder}")

    print(f"Model: {model}")
    print(f"Logs: {len(logs)}")
    print(f"Results: {out}")
    offline_backend = args.backend if args.offline_backend == "same" else args.offline_backend
    print(f"Online backend: {args.backend}")
    print(f"Offline backend: {offline_backend}")
    if args.observation_timeout > 0:
        print(f"Per-observation online timeout: {args.observation_timeout:.1f}s")
    else:
        print("Per-observation online timeout: disabled")
    print()
    print(
        f"{'#':>3}  {'OCEL':<38} {'PREFIX':<22} {'OFFLINE':<22} "
        f"{'ONLINE':<14} {'OFFLINE ST':<16} "
        f"{'ONL TOT':>8} {'ONL AVG':>8} {'ONL MAX':>8} {'OFF':>8} {'TOTAL':>8}"
    )
    print("-" * 177)

    summary = []
    # Long-format per-prefix measurements across all logs.  This is intentionally
    # separate from summary.csv so each online update remains one plot-ready row.
    prefix_timing_rows = []
    for idx, log in enumerate(logs, 1):
        rel = log.relative_to(folder)
        safe_stem = "__".join(rel.with_suffix("").parts)
        outfile = out / f"{safe_stem}.json"
        base_cmd = [
            sys.executable, args.odacc,
            "--cocomot-root", args.cocomot_root,
            "--model", str(model),
            "--log", str(log),
            "--output", str(outfile),
            "--quiet-solver",
            "--quiet",
        ]
        if args.fixed_objects:
            base_cmd.append("--fixed-objects")
        if args.solve_on_updates:
            base_cmd.append("--solve-on-updates")
        if args.diagnostic_heartbeat > 0:
            base_cmd += ["--diagnostic-heartbeat", str(args.diagnostic_heartbeat)]
        if args.diagnostic_stack > 0:
            base_cmd += ["--diagnostic-stack", str(args.diagnostic_stack)]
        if args.backend == "symbolic":
            base_cmd += [
                "--local-repair-cost-budget", str(args.local_repair_cost_budget),
                "--local-repair-max-candidates", str(args.local_repair_max_candidates),
                "--local-repair-transition-cap", str(args.local_repair_transition_cap),
            ]

        rec = {
            "file": str(rel),
            "result_file": str(outfile),
            "returncode": 0,
            "online_status": "NOT_RUN",
            "offline_status": "NOT_RUN",
            "offline_semantics": "fresh_offline_alignment",
        }
        t_all = time.perf_counter()
        full_error = ""

        try:
            if args.mode == "both":
                # Run the online and offline phases separately.  Besides allowing
                # different backends, this lets the batch runner expose online
                # results immediately and put a timeout around only the optional
                # offline phase.
                online_file = out / f"{safe_stem}.online.tmp.json"
                online_cmd = base_cmd.copy()
                oi = online_cmd.index("--output") + 1
                online_cmd[oi] = str(online_file)
                online_cmd += ["--mode", "online", "--backend", args.backend,
                               "--offline-backend", "same"]
                if args.backend == "symbolic":
                    online_cmd += ["--model-depth-margin", str(args.model_depth_margin),
                                   "--jodap-contexts", str(args.jodap_contexts)]
                    if args.symbolic_query_diagnostics:
                        diag_file = out / f"{safe_stem}.jodap_diagnostics.jsonl"
                        online_cmd += ["--symbolic-query-diagnostics", str(diag_file),
                                       "--symbolic-query-diagnostics-print"]
                if not online_cmd or online_cmd[-1] != "--progress-updates":
                    online_cmd.append("--progress-updates")
                run_on = _run_isolated_online(online_cmd, args, online_file)
                rec["native_crash_retry_count"] = run_on.get("retry_count", 0)
                rec["native_crash_attempts"] = len(run_on.get("native_failures") or [])
                if run_on.get("native_failures"):
                    rec["native_crash_history"] = json.dumps(run_on["native_failures"], sort_keys=True)
                kind = run_on.get("kind")
                if kind == "observation_timeout":
                    meta = run_on.get("observation_meta") or {}
                    rec["observation_timeout"] = True
                    rec["online_status"] = "OBSERVATION_TIMEOUT"
                    rec["returncode"] = 124
                    rec["observation_timeout_seconds"] = args.observation_timeout
                    rec["observation_elapsed_seconds"] = meta.get("elapsed_seconds")
                    rec["timeout_update"] = meta.get("update")
                    rec["timeout_stream_position"] = meta.get("stream_position")
                    rec["timeout_component"] = meta.get("component")
                    rec["timeout_event_id"] = meta.get("event_id")
                    rec["timeout_activity"] = meta.get("activity")
                    rec["timeout_event_objects"] = meta.get("event_objects")
                    rec["timeout_component_events"] = meta.get("component_events")
                    rec["timeout_component_objects"] = meta.get("component_objects")
                    _preserve_partial_online(
                        rec, run_on.get("partial"), outfile, rel, args, offline_backend, prefix_timing_rows,
                        termination_reason="observation_timeout",
                    )
                    raise RuntimeError(
                        "Online observation timed out after "
                        f"{args.observation_timeout}s at update "
                        f"{rec.get('timeout_update')} / {rec.get('timeout_activity')} "
                        f"({rec.get('timeout_event_id')})"
                    )
                if kind == "online_timeout":
                    rec["online_timeout"] = True
                    rec["online_status"] = "ONLINE_TIMEOUT"
                    rec["returncode"] = 124
                    _preserve_partial_online(
                        rec, run_on.get("partial"), outfile, rel, args, offline_backend, prefix_timing_rows,
                        termination_reason="online_timeout",
                    )
                    raise RuntimeError(f"Online phase timed out after {args.online_timeout}s")
                if kind == "native_crash":
                    rec["native_crash"] = True
                    rec["native_crash_reason"] = run_on.get("native_reason")
                    rec["native_crash_returncode"] = run_on.get("returncode")
                    rec["online_status"] = "ONLINE_NATIVE_CRASH"
                    rec["returncode"] = run_on.get("returncode")
                    _preserve_partial_online(
                        rec, run_on.get("partial"), outfile, rel, args, offline_backend, prefix_timing_rows,
                        termination_reason="native_crash",
                    )
                    raise RuntimeError(
                        f"Online native crash after {rec.get('native_crash_attempts')} attempt(s): "
                        f"{rec.get('native_crash_reason')} rc={rec.get('native_crash_returncode')}"
                    )
                class _CP:
                    pass
                cp_on = _CP()
                cp_on.returncode = run_on.get("returncode", 1)
                cp_on.stdout = run_on.get("stdout", "")
                cp_on.stderr = ""
                if cp_on.returncode != 0:
                    raise RuntimeError(cp_on.stderr or cp_on.stdout or "Online phase failed")
                with open(online_file, encoding="utf-8") as fh:
                    online_data = json.load(fh)
                online_file.unlink(missing_ok=True)

                prefix_result = online_data.get("prefix", [])[-1] if online_data.get("prefix") else None
                rec.update(cost_fields(prefix_result, "prefix"))
                rec["prefix_count"] = len(online_data.get("prefix", []))
                timing_on = online_data.get("timing") or {}
                rec["online_updates"] = timing_on.get("online_updates")
                rec["online_total_seconds"] = timing_on.get("online_total_seconds")
                rec["online_avg_seconds"] = timing_on.get("online_avg_seconds")
                rec["online_max_seconds"] = timing_on.get("online_max_seconds")
                rec["online_status"] = "ONLINE_OK"

                # Print the online result before starting potentially expensive
                # offline completion, so a long offline solve no longer looks like
                # the online monitor itself is stuck.
                print(
                    f"     online done: {str(rel):<42.42} {fmt(rec, 'prefix'):<24.24} "
                    f"avg={rec.get('online_avg_seconds', 0) or 0:.2f}s "
                    f"max={rec.get('online_max_seconds', 0) or 0:.2f}s; offline...",
                    flush=True,
                )

                offline_file = out / f"{safe_stem}.offline.tmp.json"
                offline_cmd = base_cmd.copy()
                oi = offline_cmd.index("--output") + 1
                offline_cmd[oi] = str(offline_file)
                offline_cmd += ["--mode", "offline", "--backend", offline_backend,
                                "--offline-backend", offline_backend]
                if offline_backend == "symbolic":
                    offline_cmd += ["--model-depth-margin", str(args.model_depth_margin),
                                    "--jodap-contexts", str(args.jodap_contexts)]
                timeout = None if args.offline_timeout <= 0 else args.offline_timeout
                try:
                    cp_off = subprocess.run(offline_cmd, capture_output=True, text=True, timeout=timeout, **_popen_isolation_kwargs())
                except subprocess.TimeoutExpired as exc:
                    rec["offline_timeout"] = True
                    rec["offline_status"] = "OFFLINE_TIMEOUT"
                    rec["returncode"] = 124
                    cp_off = None
                    full_error = f"Offline phase timed out after {args.offline_timeout}s"

                if cp_off is not None and cp_off.returncode == 0:
                    with open(offline_file, encoding="utf-8") as fh:
                        offline_data = json.load(fh)
                    offline_file.unlink(missing_ok=True)
                    offline_results = offline_data.get("offline", [])
                    timing_off = offline_data.get("timing") or {}
                    rec["offline_status"] = "OFFLINE_OK"
                elif cp_off is not None:
                    native = _native_crash_reason(
                        cp_off.returncode, (cp_off.stderr or "") + (cp_off.stdout or "")
                    )
                    if native is not None:
                        rec["offline_status"] = "OFFLINE_NATIVE_CRASH"
                        rec["offline_native_crash"] = True
                        rec["offline_native_crash_reason"] = native
                        rec["offline_native_crash_returncode"] = cp_off.returncode
                    raise RuntimeError(cp_off.stderr or cp_off.stdout or "Offline phase failed")
                else:
                    offline_results = []
                    timing_off = {}

                if len(offline_results) == 1:
                    offline_result = offline_results[0]
                elif offline_results:
                    feasible = all(r.get("feasible", False) for r in offline_results)
                    sums = {"total": 0, "control_flow": 0, "data": 0, "object": 0}
                    for r in offline_results:
                        cb = r.get("cost_breakdown") or {}
                        for key in sums:
                            value = cb.get(key)
                            if value is not None:
                                sums[key] += value
                    offline_result = {
                        "feasible": feasible, "cost": sums["total"], "cost_breakdown": sums,
                        "encode_seconds": sum(r.get("encode_seconds", 0) for r in offline_results),
                        "solve_seconds": sum(r.get("solve_seconds", 0) for r in offline_results),
                    }
                else:
                    offline_result = None
                rec.update(cost_fields(offline_result, "offline"))
                rec["offline_component_count"] = len(offline_results)
                rec["offline_total_seconds"] = timing_off.get("offline_total_seconds")
                rec["conformance_total_seconds"] = (rec.get("online_total_seconds") or 0) + (rec.get("offline_total_seconds") or 0)

                # Store one combined result file for downstream analysis.
                combined = dict(online_data)
                combined["offline"] = offline_results
                combined["backends"] = {"online": args.backend, "offline": offline_backend}
                combined["timing"] = {
                    **timing_on,
                    "offline_components": timing_off.get("offline_components", 0),
                    "offline_wall_seconds": timing_off.get("offline_wall_seconds", []),
                    "offline_total_seconds": timing_off.get("offline_total_seconds", 0.0),
                    "conformance_total_seconds": rec["conformance_total_seconds"],
                }
                with open(outfile, "w", encoding="utf-8") as fh:
                    json.dump(combined, fh, indent=2)
                data = combined

            else:
                cmd = base_cmd.copy()
                phase_backend = args.backend if args.mode == "online" else offline_backend
                cmd += ["--mode", args.mode, "--backend", phase_backend,
                        "--offline-backend", phase_backend]
                if phase_backend == "symbolic":
                    cmd += ["--model-depth-margin", str(args.model_depth_margin),
                            "--jodap-contexts", str(args.jodap_contexts)]
                    if args.mode == "online" and args.symbolic_query_diagnostics:
                        diag_file = out / f"{safe_stem}.jodap_diagnostics.jsonl"
                        cmd += ["--symbolic-query-diagnostics", str(diag_file),
                                "--symbolic-query-diagnostics-print"]
                timeout = None
                if args.mode == "offline" and args.offline_timeout > 0:
                    timeout = args.offline_timeout
                elif args.mode == "online" and args.online_timeout > 0:
                    timeout = args.online_timeout
                if args.mode == "online":
                    if not cmd or cmd[-1] != "--progress-updates":
                        cmd.append("--progress-updates")
                    run_on = _run_isolated_online(cmd, args, outfile)
                    rec["native_crash_retry_count"] = run_on.get("retry_count", 0)
                    rec["native_crash_attempts"] = len(run_on.get("native_failures") or [])
                    if run_on.get("native_failures"):
                        rec["native_crash_history"] = json.dumps(run_on["native_failures"], sort_keys=True)
                    kind = run_on.get("kind")
                    if kind == "observation_timeout":
                        meta = run_on.get("observation_meta") or {}
                        rec["observation_timeout"] = True
                        rec["online_status"] = "OBSERVATION_TIMEOUT"
                        rec["returncode"] = 124
                        rec["observation_timeout_seconds"] = args.observation_timeout
                        rec["observation_elapsed_seconds"] = meta.get("elapsed_seconds")
                        rec["timeout_update"] = meta.get("update")
                        rec["timeout_stream_position"] = meta.get("stream_position")
                        rec["timeout_component"] = meta.get("component")
                        rec["timeout_event_id"] = meta.get("event_id")
                        rec["timeout_activity"] = meta.get("activity")
                        rec["timeout_event_objects"] = meta.get("event_objects")
                        rec["timeout_component_events"] = meta.get("component_events")
                        rec["timeout_component_objects"] = meta.get("component_objects")
                        _preserve_partial_online(
                            rec, run_on.get("partial"), outfile, rel, args, offline_backend, prefix_timing_rows,
                            termination_reason="observation_timeout",
                        )
                        raise RuntimeError(
                            "Online observation timed out after "
                            f"{args.observation_timeout}s at update "
                            f"{rec.get('timeout_update')} / {rec.get('timeout_activity')} "
                            f"({rec.get('timeout_event_id')})"
                        )
                    if kind == "online_timeout":
                        rec["online_timeout"] = True
                        rec["online_status"] = "ONLINE_TIMEOUT"
                        rec["returncode"] = 124
                        _preserve_partial_online(
                            rec, run_on.get("partial"), outfile, rel, args, offline_backend, prefix_timing_rows,
                            termination_reason="online_timeout",
                        )
                        raise RuntimeError(f"Online phase timed out after {args.online_timeout}s")
                    if kind == "native_crash":
                        rec["native_crash"] = True
                        rec["native_crash_reason"] = run_on.get("native_reason")
                        rec["native_crash_returncode"] = run_on.get("returncode")
                        rec["online_status"] = "ONLINE_NATIVE_CRASH"
                        rec["returncode"] = run_on.get("returncode")
                        _preserve_partial_online(
                            rec, run_on.get("partial"), outfile, rel, args, offline_backend, prefix_timing_rows,
                            termination_reason="native_crash",
                        )
                        raise RuntimeError(
                            f"Online native crash after {rec.get('native_crash_attempts')} attempt(s): "
                            f"{rec.get('native_crash_reason')} rc={rec.get('native_crash_returncode')}"
                        )
                    class _CP:
                        pass
                    cp = _CP()
                    cp.returncode = run_on.get("returncode", 1)
                    cp.stdout = run_on.get("stdout", "")
                    cp.stderr = ""
                else:
                    try:
                        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **_popen_isolation_kwargs())
                    except subprocess.TimeoutExpired:
                        raise
                if cp.returncode != 0:
                    if args.mode == "offline":
                        native = _native_crash_reason(
                            cp.returncode, (cp.stderr or "") + (cp.stdout or "")
                        )
                        if native is not None:
                            rec["offline_status"] = "OFFLINE_NATIVE_CRASH"
                            rec["offline_native_crash"] = True
                            rec["offline_native_crash_reason"] = native
                            rec["offline_native_crash_returncode"] = cp.returncode
                    raise RuntimeError(cp.stderr or cp.stdout or "ODACC phase failed")
                with open(outfile, encoding="utf-8") as fh:
                    data = json.load(fh)
                prefix_result = data.get("prefix", [])[-1] if data.get("prefix") else None
                offline_results = data.get("offline", [])
                offline_result = offline_results[0] if len(offline_results) == 1 else None
                rec.update(cost_fields(prefix_result, "prefix"))
                rec.update(cost_fields(offline_result, "offline"))
                timing = data.get("timing") or {}
                rec["online_updates"] = timing.get("online_updates")
                rec["online_total_seconds"] = timing.get("online_total_seconds")
                rec["online_avg_seconds"] = timing.get("online_avg_seconds")
                rec["online_max_seconds"] = timing.get("online_max_seconds")
                rec["offline_total_seconds"] = timing.get("offline_total_seconds")
                rec["conformance_total_seconds"] = timing.get("conformance_total_seconds")
                if args.mode == "online":
                    rec["online_status"] = "ONLINE_OK"
                else:
                    rec["offline_status"] = "OFFLINE_OK"

            timing_records = ((data.get("timing") or {}).get("prefix_updates") or [])
            for tr in timing_records:
                row = {
                    "file": str(rel),
                    "online_backend": args.backend,
                    "offline_backend": offline_backend,
                }
                row.update(tr)
                prefix_timing_rows.append(row)
            rec["prefix_timing_count"] = len(timing_records)

            stats = data.get("symbolic_stats") or {}
            for key in (
                "queries", "prepared_builds", "prepared_hits",
                "filtered_transitions", "por_pruned", "signature_pruned",
                "state_dominated", "upper_bound_pruned",
                "upper_bound_seeded", "offline_seeded",
                "object_attribute_soft_terms",
                "query_cache_hits", "lower_bound_hits",
                "merge_warm_starts", "merge_checkpoint_reused",
                "zero_cost_sync_attempts", "zero_cost_sync_hits",
                "zero_cost_delta_checks", "zero_cost_delta_hits",
                "canonical_creation_restrictions",
                "silent_macro_attempts", "silent_macro_hits",
                "silent_macro_moves",
                "merge_frontier_compositions", "merge_frontier_hits",
                "symmetry_canonicalizations",
                "focused_model_queries", "focused_model_hits",
                "focused_model_fallbacks",
                "component_context_builds", "component_context_reuses",
                "component_context_extensions", "context_capacity",
                "context_build_seconds_ms",
                "context_variable_seconds_ms", "context_initial_seconds_ms",
                "context_transition_seconds_ms", "context_object_type_seconds_ms",
                "context_freshness_seconds_ms", "context_moving_seconds_ms",
                "context_remaining_seconds_ms", "context_data_seconds_ms",
                "context_cache_seconds_ms", "context_require_seconds_ms",
                "lazy_path_attempts", "lazy_path_hits", "lazy_path_fallbacks",
                "lazy_path_build_seconds_ms", "lazy_path_solve_seconds_ms",
                "eager_context_avoided", "eager_fallback_queries",
                "lazy_certification_failures", "lazy_nonzero_fallbacks",
                "delta_local_guard_checks", "delta_local_guard_hits",
                "delta_dependency_fallbacks",
                "delta_historical_reads", "delta_token_data_hits",
                "delta_provenance_hits",
                "delta_token_lookup_attempts", "delta_token_lookup_misses",
                "delta_global_lookup_attempts", "delta_global_lookup_misses",
                "exact_extension_attempts", "exact_extension_hits",
                "exact_extension_fallbacks",
                "nonzero_pruned_before_fallback",
                "positive_lower_bound_proofs",
                "positive_lower_bound_terminations",
                "positive_fixed_path_incumbents",
                "zero_deviation_guard_checks",
                "zero_deviation_guard_false",
                "zero_deviation_guard_true",
                "zero_deviation_guard_unknown",
                "zero_cost_early_terminations",
                "analytic_all_log_seeds",
                "merge_direct_compositions", "merge_direct_hits",
                "merge_structural_log_proofs", "merge_lower_bound_terminations",
                "merge_frontier_heap_pops", "merge_frontier_incompatible",
                "merge_incompatible_object_binding",
                "merge_incompatible_marking",
                "merge_incompatible_process_data",
                "merge_incompatible_token_data",
                "merge_incompatible_partial_order",
                "merge_incompatible_variable_version",
                "merge_namespaced_data_conflicts",
                "merge_bridge_checks", "merge_bridge_hits",
                "merge_bridge_log_hits",
                "merge_one_parent_repairs", "merge_one_parent_repair_hits",
                "merge_full_search_fallbacks",
                "local_repair_search_attempts", "local_repair_search_candidates",
                "local_repair_search_solver_calls", "local_repair_search_hits",
                "local_repair_search_proven", "local_repair_search_incumbents",
                "local_repair_incumbents_preserved", "local_repair_incumbents_restored",
                "local_repair_search_fallbacks", "local_repair_search_pruned_budget",
                "local_repair_search_max_depth", "local_repair_event_binding_probes",
                "local_repair_missing_token_producers",
                "local_repair_missing_token_targeted_enqueues",
                "local_repair_relation_candidates_pruned",
                "local_repair_invariant_failures_pruned",
                "checkpoint_bound_binding_attempts",
                "checkpoint_bound_binding_candidates",
                "checkpoint_bound_binding_hits",
                "checkpoint_bound_binding_proven",
                "candidate_structural_lb_raised",
                "pre_jodap_upper_bound_pruned",
                "pre_jodap_log_cost_pruned",
                "pre_jodap_model_cost_pruned",
                "strict_improvement_mode_entries",
                "strict_improvement_pre_solve_pruned",
                "strict_improvement_lazy_unsat_pruned",
                "strict_improvement_eager_queries",
                "strict_improvement_eager_unsat",
                "strict_improvement_lower_bound_closed",
                "strict_improvement_generation_pruned",
                "strict_improvement_generation_log_pruned",
                "strict_improvement_generation_model_pruned",
                "strict_improvement_generation_node_closed",
                "guard_directed_list_attempts",
                "guard_directed_list_supported",
                "guard_directed_list_candidates_pruned",
                "guard_directed_list_candidates_kept",
                "guard_directed_list_fallbacks",
                "guard_directed_binding_generation_attempts",
                "guard_directed_binding_generation_supported",
                "guard_directed_binding_generation_candidates",
                "guard_directed_binding_generation_subsets_avoided",
                "guard_directed_binding_generation_fallbacks",
                "marking_domain_reduction_attempts",
                "marking_domain_reduction_supported",
                "marking_domain_objects_pruned",
                "marking_domain_candidates_pruned",
                "marking_domain_reduction_fallbacks",
                "checkpoint_context_attempts",
                "checkpoint_context_builds",
                "checkpoint_context_hits",
                "checkpoint_context_fallbacks",
                "checkpoint_context_objects_full",
                "checkpoint_context_objects_kept",
                "checkpoint_context_objects_pruned",
                "checkpoint_context_subset_space_full",
                "checkpoint_context_subset_space_kept",
                "checkpoint_context_build_seconds_ms",
                "checkpoint_context_moving_seconds_ms",
                "persistent_checkpoint_lineage_seeded",
                "persistent_checkpoint_lineage_propagated",
                "persistent_checkpoint_lineage_reused",
                "persistent_checkpoint_lineage_assignment_recovered",
                "persistent_checkpoint_lineage_missing_on_parent",
                "persistent_checkpoint_lineage_missing_on_candidate",
                "persistent_checkpoint_lineage_dropped_on_canonical_reuse",
                "persistent_checkpoint_lineage_attached_on_reuse",
                "persistent_checkpoint_context_attempts",
                "persistent_checkpoint_context_proven",
                "persistent_checkpoint_context_upper_bounds",
                "persistent_checkpoint_context_fallbacks",
                "context_absolute_depth_escalations",
            ):
                if key in stats:
                    rec[f"symbolic_{key}"] = stats[key]

        except subprocess.TimeoutExpired:
            if rec.get("online_status") != "ONLINE_OK" and args.mode == "online":
                rec["online_status"] = "ONLINE_TIMEOUT"
                rec["online_timeout"] = True
                full_error = f"Online phase timed out after {args.online_timeout}s"
            else:
                rec["offline_status"] = "OFFLINE_TIMEOUT"
                rec["offline_timeout"] = True
                full_error = f"Offline phase timed out after {args.offline_timeout}s"
            rec["returncode"] = 124
        except Exception as exc:
            full_error = str(exc)
            status = rec.get("online_status")
            timed_status = status in ("ONLINE_TIMEOUT", "OBSERVATION_TIMEOUT")
            native_status = status == "ONLINE_NATIVE_CRASH"
            offline_native_status = rec.get("offline_status") == "OFFLINE_NATIVE_CRASH"
            if timed_status:
                rec["returncode"] = 124
            elif native_status:
                rec["returncode"] = rec.get("native_crash_returncode", rec.get("returncode", 1))
            elif offline_native_status:
                rec["returncode"] = rec.get("offline_native_crash_returncode", rec.get("returncode", 1))
            else:
                rec["returncode"] = 1
            if timed_status or native_status or offline_native_status:
                pass
            elif rec.get("online_status") == "ONLINE_OK":
                rec["offline_status"] = "OFFLINE_ERROR"
            elif args.mode == "offline":
                rec["offline_status"] = "OFFLINE_ERROR"
            else:
                rec["online_status"] = "ONLINE_ERROR"

        rec["wall_seconds"] = time.perf_counter() - t_all
        if full_error:
            rec["error"] = full_error[-4000:]
            error_file = out / f"{safe_stem}.error.txt"
            error_file.write_text(full_error, encoding="utf-8")
            rec["error_file"] = str(error_file)
            if args.show_errors:
                print(full_error)

        summary.append(rec)
        if rec.get("partial_result_saved"):
            done = rec.get("completed_stream_units")
            total = rec.get("total_stream_units")
            stuck_activity = rec.get("timeout_activity") or "update"
            stuck_event = rec.get("timeout_event_id") or "-"
            timeout_kind = (
                "observation timeout" if rec.get("online_status") == "OBSERVATION_TIMEOUT"
                else "native crash" if rec.get("online_status") == "ONLINE_NATIVE_CRASH"
                else "online timeout"
            )
            print(
                f"     partial checkpoint ({timeout_kind}): completed {done}/{total} stream units; "
                f"stuck at {stuck_activity} ({stuck_event}); "
                f"last certified prefix cost={rec.get('prefix_total')}",
                flush=True,
            )
        on_tot = rec.get("online_total_seconds")
        on_avg = rec.get("online_avg_seconds")
        on_max = rec.get("online_max_seconds")
        off_t = rec.get("offline_total_seconds")
        conf_t = rec.get("conformance_total_seconds")
        def tfmt(v):
            return "-" if v is None else f"{float(v):.2f}s"
        print(
            f"{idx:>3}  {str(rel):<38.38} "
            f"{fmt(rec, 'prefix'):<22.22} "
            f"{fmt(rec, 'offline'):<22.22} "
            f"{rec.get('online_status','NOT_RUN'):<14.14} "
            f"{rec.get('offline_status','NOT_RUN'):<16.16} "
            f"{tfmt(on_tot):>8} {tfmt(on_avg):>8} {tfmt(on_max):>8} "
            f"{tfmt(off_t):>8} {tfmt(conf_t):>8}"
        )

    summary_json = out / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary_csv = out / "summary.csv"
    fieldnames = []
    for rec in summary:
        for key in rec:
            if key != "error" and key not in fieldnames:
                fieldnames.append(key)
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    prefix_json = out / "prefix_timings.json"
    with open(prefix_json, "w", encoding="utf-8") as fh:
        json.dump(prefix_timing_rows, fh, indent=2)

    prefix_csv = out / "prefix_timings.csv"
    prefix_fields = []
    for row in prefix_timing_rows:
        for key in row:
            if key not in prefix_fields:
                prefix_fields.append(key)
    with open(prefix_csv, "w", newline="", encoding="utf-8") as fh:
        if prefix_fields:
            writer = csv.DictWriter(fh, fieldnames=prefix_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(prefix_timing_rows)

    ok = sum(1 for r in summary if r["returncode"] == 0)
    print("-" * 177)
    online_ok = sum(r.get("online_status") in ("ONLINE_OK", "NOT_RUN") for r in summary)
    offline_ok = sum(r.get("offline_status") in ("OFFLINE_OK", "NOT_RUN") for r in summary)
    print(f"Online phase acceptable:  {online_ok}/{len(summary)}")
    print(f"Offline phase acceptable: {offline_ok}/{len(summary)}")
    print(f"Fully successful:         {ok}/{len(summary)}")
    print(f"Summary JSON:       {summary_json}")
    print(f"Summary CSV:        {summary_csv}")
    print(f"Per-prefix JSON:    {prefix_json}")
    print(f"Per-prefix CSV:     {prefix_csv}")
    return 0 if ok == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
