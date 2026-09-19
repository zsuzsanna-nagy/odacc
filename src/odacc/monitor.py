from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List
import time
import threading

from .components import ComponentManager
from .domain import AlignmentResult, ObservableUnit
from .report import certified_result_dict


@dataclass
class OnlineRun:
    prefix_results: List[AlignmentResult] = field(default_factory=list)
    offline_results: List[AlignmentResult] = field(default_factory=list)
    # End-to-end wall time of each online alignment update.  These values
    # include A*/JODAP orchestration in addition to the solver's own timing.
    prefix_wall_seconds: List[float] = field(default_factory=list)
    # End-to-end wall time of each final offline alignment computation.
    offline_wall_seconds: List[float] = field(default_factory=list)
    # One record per solved online prefix.  Unlike prefix_wall_seconds, this
    # preserves the stream/event/component metadata required for per-prefix
    # latency plots and for locating expensive updates.
    prefix_timing_records: List[dict] = field(default_factory=list)
    # Compact, JSON-safe snapshots copied immediately after each certified
    # update.  Heartbeat/checkpoint writers use these immutable payloads instead
    # of traversing live AlignmentResult/search/solver structures.
    certified_prefix_snapshots: List[dict] = field(default_factory=list)
    # Number of observable units that returned successfully, including update-
    # only units for which no behavioral alignment query was requested.
    completed_stream_units: int = 0
    last_completed_stream_position: int | None = None


class OnlineMonitor:
    """Component-incremental online monitor.

    Only the component touched by an incoming observable unit is recomputed.
    If the unit merges components, their observation formulas and partial-order
    executions are merged and the affected joint component is recomputed from
    the conservative safe checkpoint (the beginning of the merged histories).
    """

    def __init__(self, backend, offline_backend=None, diagnostic_heartbeat_seconds: float = 0.0):
        self.backend = backend
        self.offline_backend = offline_backend or backend
        self.components = ComponentManager()
        self.diagnostic_heartbeat_seconds = max(0.0, float(diagnostic_heartbeat_seconds or 0.0))

    @staticmethod
    def _symbolic_stats_snapshot(backend):
        """Return a compact copy of symbolic counters, if available.

        The snapshot is taken immediately before/after one online solve so the
        final certification route can be classified from *successful proof*
        counters rather than from cumulative run-level diagnostics.  This is
        reporting-only metadata and does not affect search or certification.
        """
        jodap = getattr(backend, "jodap", None)
        stats = getattr(jodap, "stats", None)
        return dict(stats) if isinstance(stats, dict) else None

    @staticmethod
    def _certification_route(before, after, result):
        """Classify the exclusive final route used for one certified prefix.

        Only counters denoting a successful/proven route are considered.
        Attempts and unsuccessful fallbacks are deliberately ignored.  The
        precedence mirrors the method hierarchy: an exact proof dominates a
        local repair, which dominates direct continuation.  If the symbolic
        solver returns a feasible prefix without one of the specialised proof
        counters firing, the result came from the unrestricted exact A* path.
        """
        if before is None or after is None or result is None or not result.feasible:
            return None

        def increased(name):
            return int(after.get(name, 0) or 0) > int(before.get(name, 0) or 0)

        checkpoint_exact = (
            "persistent_checkpoint_context_proven",
            "checkpoint_slice_proven",
            "checkpoint_bound_binding_proven",
            "provenance_slice_proven",
        )
        local_repair = (
            "one_step_repair_proven",
            "extra_event_fast_proven",
            "guard_data_repair_proven",
            "object_relation_repair_proven",
            "local_repair_search_proven",
            "fresh_object_sync_successes",
            "first_event_latent_relation_hits",
        )
        direct = (
            "zero_cost_sync_hits",
            "zero_cost_delta_hits",
            "silent_macro_hits",
            "merge_direct_hits",
            "merge_bridge_hits",
            "merge_fast_successes",
        )

        if any(increased(k) for k in checkpoint_exact):
            return "checkpoint_exact"
        if any(increased(k) for k in local_repair):
            return "local_repair"
        if any(increased(k) for k in direct):
            return "direct"
        return "full_exact"

    def process(self, unit: ObservableUnit, solve_on_updates: bool = False,
                progress_callback=None, checkpoint_callback=None):
        component, merged = self.components.apply(unit)
        if progress_callback is not None:
            progress_callback("start", unit, component, merged, None, None)
        if checkpoint_callback is not None:
            checkpoint_callback("start", unit, component, merged, None, None)

        # Attribute/O2O-only units update Phi_obs. By default they do not cause
        # a behavioral prefix alignment query, matching the paper formulation.
        if unit.event is None and not solve_on_updates:
            return None, merged

        # Optional heartbeat while an expensive backend call is in progress.
        stop_heartbeat = threading.Event()
        heartbeat_thread = None
        solve_started = time.perf_counter()
        if self.diagnostic_heartbeat_seconds > 0 and progress_callback is not None:
            interval = self.diagnostic_heartbeat_seconds

            def heartbeat():
                while not stop_heartbeat.wait(interval):
                    elapsed = time.perf_counter() - solve_started
                    progress_callback(
                        "heartbeat", unit, component, merged, None,
                        elapsed,
                    )
                    # Persist a live diagnostic snapshot as well.  The prefix
                    # result itself remains the last certified checkpoint, but
                    # backend counters now reflect work performed inside the
                    # currently running observation if the process is killed by
                    # the per-observation watchdog.
                    if checkpoint_callback is not None:
                        checkpoint_callback(
                            "heartbeat", unit, component, merged, None, elapsed
                        )

            heartbeat_thread = threading.Thread(
                target=heartbeat, name="odacc-progress-heartbeat", daemon=True
            )
            heartbeat_thread.start()

        stats_before = self._symbolic_stats_snapshot(self.backend)
        try:
            result = self.backend.solve(component, offline=False)
        finally:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)
        stats_after = self._symbolic_stats_snapshot(self.backend)
        certification_route = self._certification_route(stats_before, stats_after, result)
        if result is not None:
            # Dynamic reporting metadata: AlignmentResult remains unchanged, so
            # existing JSON/result consumers stay backwards compatible.
            result.certification_route = certification_route
        component.current_alignment = result
        return result, merged

    def run(self, stream: Iterable[ObservableUnit], *, offline_at_end: bool = False,
            solve_on_updates: bool = False, progress_callback=None,
            checkpoint_callback=None) -> OnlineRun:
        """Consume the stream and optionally checkpoint after every unit.

        ``checkpoint_callback`` is invoked around an online unit:
        ``start`` after the unit has been incorporated into the component but
        before the potentially expensive solve, and ``done`` after the unit has
        completed and the successful prefix result/timing has been appended.
        When diagnostic heartbeats are enabled it also receives ``heartbeat``
        snapshots while the solve is in progress.

        This is intentionally separate from ``progress_callback``.  A caller can
        therefore persist the last certified prefix atomically without writing a
        file for every diagnostic heartbeat.
        """
        out = OnlineRun()
        update_index = 0

        def process_checkpoint(phase, unit, component, merged, result, elapsed):
            if checkpoint_callback is not None:
                checkpoint_callback(phase, out, unit, component, merged, result, elapsed)

        for unit in stream:
            t0 = time.perf_counter()
            result, merged = self.process(
                unit,
                solve_on_updates=solve_on_updates,
                progress_callback=progress_callback,
                checkpoint_callback=process_checkpoint,
            )
            elapsed = time.perf_counter() - t0

            if result is not None:
                update_index += 1
                out.prefix_results.append(result)
                out.prefix_wall_seconds.append(elapsed)
                out.certified_prefix_snapshots.append(certified_result_dict(result))
                component = self.components.components.get(result.component_id)
                event = unit.event
                cb = result.cost_breakdown or {}
                out.prefix_timing_records.append({
                    "update_index": update_index,
                    "stream_position": unit.position,
                    "component": result.component_id,
                    "prefix_position": result.prefix_position,
                    "event_id": event.event_id if event is not None else None,
                    "activity": event.activity if event is not None else None,
                    "event_object_count": len(event.objects) if event is not None else 0,
                    "component_object_count": (
                        len(component.objects) if component is not None else None
                    ),
                    "component_event_count": (
                        len(component.execution.event_ids) if component is not None else None
                    ),
                    "merged": bool(merged),
                    "wall_seconds": elapsed,
                    "feasible": result.feasible,
                    "cost": cb.get("total", result.cost),
                    "control_flow_cost": cb.get("control_flow"),
                    "data_cost": cb.get("data"),
                    "object_cost": cb.get("object"),
                    "encode_seconds": result.encode_seconds,
                    "solve_seconds": result.solve_seconds,
                    "certification_route": getattr(result, "certification_route", None),
                })

            out.completed_stream_units += 1
            out.last_completed_stream_position = unit.position
            component = self.components.components.get(result.component_id) if result is not None else None
            if component is None:
                # For update-only units, find the component touched by the just-
                # applied unit if possible; purely diagnostic/checkpoint metadata.
                try:
                    component = next(
                        c for c in self.components.components.values()
                        if unit in getattr(c, "units", ())
                    )
                except Exception:
                    component = None

            # Persist only after the result/timing has been appended, so a killed
            # process always leaves a self-consistent certified prefix on disk.
            if checkpoint_callback is not None:
                checkpoint_callback("done", out, unit, component, merged, result, elapsed)

            if progress_callback is not None:
                progress_callback("done", unit, component, merged, result, elapsed)

        if offline_at_end:
            for component in sorted(self.components.components.values(), key=lambda c: c.component_id):
                t0 = time.perf_counter()
                out.offline_results.append(self.offline_backend.solve(component, offline=True))
                out.offline_wall_seconds.append(time.perf_counter() - t0)
        return out
