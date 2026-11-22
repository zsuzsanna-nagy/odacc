"""Regression tests for correctness issues found during development.

Run from the repository root:

    ODACC_COCOMOT_ROOT=/path/to/cocomot python -m unittest tests.test_regressions -v

The end-to-end tests call ``odacc.py`` in a subprocess (as the experiment
runner does) and inspect the JSON output.  The selected cases are small; the
whole file runs in roughly two minutes.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COCOMOT = os.environ.get("ODACC_COCOMOT_ROOT", str(ROOT.parent / "cocomot"))
OM_MODEL = ROOT / "examples/order_management/order_management_dopid.pnml"
CM = ROOT / "examples/order_management/controlled_mutations"
COMPREHENSIVE = ROOT / "examples/comprehensive"


def run_online(model: Path, log: Path, backend: str, timeout: int = 240) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        cmd = [sys.executable, str(ROOT / "odacc.py"), "--cocomot-root", COCOMOT,
               "--model", str(model), "--log", str(log), "--mode", "online",
               "--backend", backend, "--quiet-solver", "--quiet", "--output", str(out)]
        subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(out.read_text(encoding="utf-8"))["prefix"]


def explained_events(result: dict) -> list:
    return [m["event_id"] for m in result["moves"] if m["kind"] in ("log", "sync")]


@unittest.skipUnless(Path(COCOMOT, "src").is_dir(), "set ODACC_COCOMOT_ROOT to a CoCoMoT checkout")
class EndToEnd(unittest.TestCase):

    def test_no_stale_alignment_at_fresh_object_update(self):
        """Missing 'pick item': 'create package' (fresh PACKAGE) costs 1, not a stale 0."""
        log = CM / "cf_missing_pick/component_032_events_0007_objects_0003.jsonocel"
        prefixes = run_online(OM_MODEL, log, "symbolic")
        seen = set()
        for r in prefixes:
            events = explained_events(r)
            self.assertEqual(len(events), len(set(events)))
            self.assertTrue(seen <= set(events), f"prefix {r['prefix_position']} dropped events")
            seen = set(events)
        self.assertEqual([r["cost"] for r in prefixes], [0, 0, 1, 1, 1, 1])
        self.assertEqual(prefixes[2]["certification_route"], "local_repair")

    def test_smt_step_bound_is_certified(self):
        """CoCoMoT's default bound forced cost 3; the certified bound finds cost 1."""
        log = CM / "cf_missing_pick/component_032_events_0007_objects_0003.jsonocel"
        prefixes = run_online(OM_MODEL, log, "smt")
        self.assertEqual(prefixes[-1]["cost"], 1)
        self.assertTrue(all(r.get("step_bound_certified") for r in prefixes))

    def test_anchored_exact_synchronization_both_backends(self):
        """'ship' lists fewer products than the order holds: exact sync is violated."""
        log = COMPREHENSIVE / "04_object_missing_product_at_ship.jsonocel"
        model = COMPREHENSIVE / "net.pnml"
        sym = run_online(model, log, "symbolic")
        smt = run_online(model, log, "smt")
        self.assertEqual([r["cost"] for r in sym], [r["cost"] for r in smt])
        self.assertEqual(sym[-1]["cost"], 2)

    def test_sync_moves_bind_exactly_the_observed_objects(self):
        """No object-relation repair moves: every sync move has object_match True."""
        log = COMPREHENSIVE / "05_object_extra_product_at_ship.jsonocel"
        model = COMPREHENSIVE / "net.pnml"
        for r in run_online(model, log, "symbolic"):
            self.assertEqual(r["cost_breakdown"].get("object", 0), 0)
            for m in r["moves"]:
                if m["kind"] == "sync":
                    self.assertTrue(m.get("object_match", True), m)

    def test_log_reexplanation_and_goal_anchoring(self):
        """Missing ITEM relation: prefix 6 needs a co-optimal re-explanation.

        Before 1.1.1 the B&B-returned alignment of the merge prefix was not
        anchored as certified boundary and Levels 1-2 were skipped; the
        full-prefix search then exceeded several minutes.
        """
        log = CM / "object_missing_item_relation/component_032_events_0007_objects_0003.jsonocel"
        prefixes = run_online(OM_MODEL, log, "symbolic", timeout=120)
        self.assertEqual([r["cost"] for r in prefixes], [0, 0, 0, 1, 3, 3, 3])
        self.assertEqual(prefixes[5]["certification_route_detail"],
                         "local_repair:log_reexplanation")

    def test_smt_level3_fallback(self):
        """Level 3 decided by the SMT strict-improvement check (unsat and sat)."""
        log = CM / "object_missing_item_relation/component_038_events_0008_objects_0004.jsonocel"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            cmd = [sys.executable, str(ROOT / "odacc.py"), "--cocomot-root", COCOMOT,
                   "--model", str(OM_MODEL), "--log", str(log), "--mode", "online",
                   "--backend", "symbolic", "--quiet-solver", "--quiet", "--output", str(out),
                   "--symbolic-smt-fallback-after", "0"]
            subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=240,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            prefixes = json.loads(out.read_text(encoding="utf-8"))["prefix"]
        self.assertEqual([r["cost"] for r in prefixes], [0, 0, 0, 0, 1, 4, 5, 5])
        details = [r["certification_route_detail"] for r in prefixes]
        self.assertIn("smt_exact:unsat", details)
        self.assertEqual(details[-1], "direct:zero_cost_sync")  # incremental state kept

    def test_anytime_bounds_sandwich_the_optimum(self):
        """Deadline 0: every result is feasible and lower_bound <= optimum <= cost."""
        log = CM / "cf_missing_pick/component_032_events_0007_objects_0003.jsonocel"
        optima = [0, 0, 1, 1, 1, 1]  # SMT reference (certified run bound)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            cmd = [sys.executable, str(ROOT / "odacc.py"), "--cocomot-root", COCOMOT,
                   "--model", str(OM_MODEL), "--log", str(log), "--mode", "online",
                   "--backend", "symbolic", "--quiet-solver", "--quiet", "--output", str(out),
                   "--anytime-deadline", "0"]
            subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=240,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            prefixes = json.loads(out.read_text(encoding="utf-8"))["prefix"]
        self.assertEqual(len(prefixes), len(optima))
        for r, opt in zip(prefixes, optima):
            self.assertTrue(r["feasible"])
            self.assertLessEqual(r["lower_bound"], opt)
            self.assertGreaterEqual(r["cost"], opt)
            if r["certified"]:
                self.assertEqual(r["cost"], opt)
            events = explained_events(r)
            self.assertEqual(len(events), len(set(events)))

    def test_candidate_heuristics_keep_bounds_sound(self):
        """All four candidate heuristics: certified prefixes equal SMT, lower bounds
        never exceed the SMT optimum, and the beam does not worsen the result."""
        log = CM / "object_extra_item_relation/component_021_events_0013_objects_0005.jsonocel"
        smt = [0, 0, 0, 0, 0, 2, 2, 4, 5, 5, 5, 5, 5]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            cmd = [sys.executable, str(ROOT / "odacc.py"), "--cocomot-root", COCOMOT,
                   "--model", str(OM_MODEL), "--log", str(log), "--mode", "online",
                   "--backend", "symbolic", "--quiet-solver", "--quiet", "--output", str(out),
                   "--anytime-deadline", "5", "--symbolic-smt-fallback-after", "2",
                   "--token-path-repair", "on", "--incumbent-beam", "4",
                   "--deviation-priority", "on", "--replay-reexplanation", "on"]
            subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=200,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data = json.loads(out.read_text(encoding="utf-8"))
        prefixes = data["prefix"]
        self.assertEqual(len(prefixes), len(smt))
        for r, opt in zip(prefixes, smt):
            if r.get("certified"):
                self.assertEqual(r["cost"], opt)
            else:
                self.assertLessEqual(r["lower_bound"], opt)
                self.assertGreaterEqual(r["cost"], opt)
        self.assertEqual(data["symbolic_stats"].get("validity_failures", 0), 0)

    def test_joint_lower_bound_sound_and_tight(self):
        """Joint LP bound never exceeds the SMT optimum; exact on known hard prefixes."""
        import contextlib
        import io
        from odacc.symbolic_search import SymbolicIncrementalBackend
        from odacc.components import ComponentManager
        from odacc.ocel import load_ocel1, simulate_stream
        cases = [
            (CM / "object_missing_item_relation/component_038_events_0008_objects_0004.jsonocel",
             [0, 0, 0, 0, 1, 4, 5, 5], True),
            (ROOT / "examples/order_management/multi_deviation_benchmark/three_mixed/"
                    "component_032_events_0007_objects_0003__three_mixed_047.jsonocel",
             [0, 0, 0, 1, 1, 4, 4, 5], True),
            (CM / "object_extra_item_relation/component_021_events_0013_objects_0005.jsonocel",
             [0, 0, 0, 0, 0, 2, 2, 4, 5, 5], False),
        ]
        for log, optima, exact in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                be = SymbolicIncrementalBackend(COCOMOT, str(OM_MODEL))
            be.use_joint_lb = True
            cm = ComponentManager()
            lbs = []
            for unit in simulate_stream(load_ocel1(str(log))):
                comp, _ = cm.apply(unit)
                if unit.event is not None:
                    lbs.append(be._joint_lb_value(comp))
                if len(lbs) == len(optima):
                    break
            for lb, opt in zip(lbs, optima):
                self.assertIsNotNone(lb)
                self.assertLessEqual(lb, opt, f"{log.name}: {lbs} vs {optima}")
            if exact:
                self.assertEqual(lbs, optima)

    def test_incremental_smt_check_agrees(self):
        """Incremental SMT check gives the same answers as the plain one."""
        import contextlib
        import io
        from odacc.cocomot_adapter import CocomotBackend
        from odacc.components import ComponentManager
        from odacc.ocel import load_ocel1, simulate_stream
        log = CM / "object_missing_item_relation/component_038_events_0008_objects_0004.jsonocel"
        plain = CocomotBackend(COCOMOT, str(OM_MODEL), quiet_solver=True)
        inc = CocomotBackend(COCOMOT, str(OM_MODEL), quiet_solver=True)
        inc.incremental_smt = True
        cm = ComponentManager()
        n = 0
        for unit in simulate_stream(load_ocel1(str(log))):
            comp, _ = cm.apply(unit)
            if unit.event is None:
                continue
            n += 1
            for pos, ub in ((5, 2), (6, 4), (6, 5)):
                if pos != n:
                    continue
                with contextlib.redirect_stdout(io.StringIO()):
                    a = plain.strict_improvement(comp, ub, timeout_s=60)
                    b = inc.strict_improvement(comp, ub, timeout_s=60)
                self.assertEqual(a[0], b[0])
                self.assertEqual(None if a[1] is None else a[1].cost,
                                 None if b[1] is None else b[1].cost)
        self.assertGreaterEqual(inc.inc_stats["base_reuses"], 1)

    def test_v2_lp_candidates_and_heuristic(self):
        """v2.0 switches: optimum found and certified where v1.9 timed out."""
        log = CM / "object_missing_item_relation/component_038_events_0008_objects_0004.jsonocel"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            cmd = [sys.executable, str(ROOT / "odacc.py"), "--cocomot-root", COCOMOT,
                   "--model", str(OM_MODEL), "--log", str(log), "--mode", "online",
                   "--backend", "symbolic", "--quiet-solver", "--quiet", "--output", str(out),
                   "--joint-lb", "on", "--lp-guided-candidates", "on", "--lp-heuristic", "on"]
            subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            prefixes = json.loads(out.read_text(encoding="utf-8"))["prefix"]
        self.assertEqual([r["cost"] for r in prefixes], [0, 0, 0, 0, 1, 4, 5, 5])
        self.assertTrue(all(r.get("certified") for r in prefixes))

    def test_smt_component_timeout_gives_up_only_that_component(self):
        """Per-component give-up: other components continue; merges inherit."""
        from odacc.cocomot_adapter import SMTUpdateTimeout
        from odacc.domain import AlignmentResult
        from odacc.monitor import OnlineMonitor
        from odacc.ocel import load_ocel1, simulate_stream

        class FakeBackend:
            """Times out on the first component at its second update."""
            def __init__(self):
                self.seen = {}
                self.victim = None

            def solve(self, component, offline=False):
                cid = component.component_id
                self.seen[cid] = self.seen.get(cid, 0) + 1
                if self.victim is None:
                    self.victim = cid
                if cid == self.victim and self.seen[cid] == 2:
                    raise SMTUpdateTimeout()
                pos = max(u.position for u in component.units)
                return AlignmentResult(cid, pos, "prefix", 0, True)

        log = ROOT / "examples/comprehensive/11_two_independent_components.jsonocel"
        backend = FakeBackend()
        run = OnlineMonitor(backend).run(simulate_stream(load_ocel1(str(log))))
        recs = run.prefix_timing_records
        victim = backend.victim
        after = [r for r in recs if r["component"] == victim and r["update_index"] > 2]
        others = [r for r in recs if r["component"] != victim]
        self.assertIn(victim, run.given_up_components)
        self.assertTrue(any(r.get("skipped") == "timeout" for r in recs))
        self.assertTrue(all(r.get("skipped") == "after_timeout" for r in after
                            if r.get("skipped") is not None))
        self.assertTrue(others and all(r.get("skipped") is None for r in others))

        # a component that merges with a given-up one is given up as well
        mon = OnlineMonitor(backend)
        mon._given_up_components = {7}

        class Comp:
            component_id = 9
            merged_from = (7, 8)
            units = []
        self.assertTrue(set(Comp.merged_from) & mon._given_up_components)

    def test_two_orders_in_one_trace(self):
        """Upstream exact sync made 'pay bank transfer' impossible with two orders."""
        model = ROOT / "examples/scalability/models/net_guard_medium.pnml"
        log = ROOT / "examples/scalability/merge_scale/merge_c02_p01.jsonocel"
        sym = run_online(model, log, "symbolic")
        self.assertEqual(sym[-1]["cost"], 4)


@unittest.skipUnless(Path(COCOMOT, "src").is_dir(), "set ODACC_COCOMOT_ROOT to a CoCoMoT checkout")
class Units(unittest.TestCase):

    def test_depth_bound_condition_holds_for_benchmark_models(self):
        from odacc.cocomot_adapter import load_cocomot, model_depth_bound_condition
        mods = load_cocomot(COCOMOT)
        for model in sorted(glob.glob(str(ROOT / "examples/**/*.pnml"), recursive=True)):
            net = mods.OPI(mods.read_pnml_input(model))
            ok, reasons = model_depth_bound_condition(net)
            self.assertTrue(ok, f"{model}: {reasons}")

    def test_validity_guard_recovers_from_a_malformed_result(self):
        """A result that drops an observed event is detected and recomputed."""
        from odacc.symbolic_search import SymbolicIncrementalBackend
        from odacc.components import ComponentManager
        from odacc.ocel import load_ocel1, simulate_stream
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            backend = SymbolicIncrementalBackend(COCOMOT, str(COMPREHENSIVE / "net.pnml"))
        original = backend._solve_impl
        calls = {"n": 0}

        def corrupting(component, offline=False):
            res = original(component, offline=offline)
            calls["n"] += 1
            if calls["n"] == 2 and res.moves:  # corrupt the second online update once
                res.moves = [m for m in res.moves if m.kind == "model"]
            return res

        backend._solve_impl = corrupting
        cm = ComponentManager()
        last = None
        with contextlib.redirect_stdout(io.StringIO()):
            for unit in simulate_stream(load_ocel1(str(COMPREHENSIVE / "00_fit.jsonocel"))):
                comp, _ = cm.apply(unit)
                if unit.event is not None:
                    last = backend.solve(comp)
                    events = [m.event_id for m in last.moves if m.kind in ("log", "sync")]
                    self.assertEqual(set(events), set(comp.execution.event_ids))
        self.assertEqual(backend.jodap.stats["validity_failures"], 1)
        self.assertEqual(backend.jodap.stats["validity_unrecovered"], 0)
        self.assertEqual(last.cost, 0)

    def test_object_projection_bound_is_admissible(self):
        """Per-object projection LB never exceeds known optima (SMT, per prefix)."""
        from odacc.symbolic_search import ObjectProjectionBound
        from odacc.cocomot_adapter import load_cocomot
        from odacc.components import ComponentManager
        from odacc.ocel import load_ocel1, simulate_stream
        import contextlib
        import io
        mods = load_cocomot(COCOMOT)
        cases = [
            ("examples/object_attributes/net_object_attrs.pnml",
             "examples/object_attributes/00_fit_all_object_guards.jsonocel", [0, 0, 0, 0, 0]),
            (str(OM_MODEL.relative_to(ROOT)),
             "examples/order_management/controlled_mutations/object_missing_item_relation/"
             "component_038_events_0008_objects_0004.jsonocel", [0, 0, 0, 0, 1, 4, 5, 5]),
            (str(OM_MODEL.relative_to(ROOT)),
             "examples/order_management/controlled_mutations/cf_missing_pick/"
             "component_061_events_0007_objects_0003.jsonocel", [0, 0, 1, 1, 1, 1]),
        ]
        for model, log, optima in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                bound = ObjectProjectionBound(mods.OPI(mods.read_pnml_input(str(ROOT / model))))
            self.assertTrue(bound.ok)
            cm = ComponentManager()
            lbs = []
            for unit in simulate_stream(load_ocel1(str(ROOT / log))):
                comp, _ = cm.apply(unit)
                if unit.event is not None:
                    lbs.append(bound.prefix_lower_bound(comp))
            self.assertEqual(len(lbs), len(optima))
            for lb, opt in zip(lbs, optima):
                self.assertLessEqual(lb, opt, f"{log}: {lbs} vs {optima}")

    def test_nonvacuous_firing_rule(self):
        """Compact non-vacuous encoding: only transitions whose input arcs all
        contain LIST variables get a constraint, stated over the LIST slots."""
        import contextlib
        import io
        from odacc.cocomot_adapter import CocomotBackend
        from odacc.components import ComponentManager
        from odacc.ocel import load_ocel1, simulate_stream
        log = CM / "object_extra_item_relation/component_021_events_0013_objects_0005.jsonocel"
        with contextlib.redirect_stdout(io.StringIO()):
            be = CocomotBackend(COCOMOT, str(OM_MODEL), quiet_solver=True)
            cm = ComponentManager()
            for unit in simulate_stream(load_ocel1(str(log))):
                comp, _ = cm.apply(unit)
                if unit.position == 8:
                    break
            enc = be._build_encoding(comp, step_bound=12)[1]
        by_label = {t.get("label"): t for t in enc._net._transitions}
        # scalar-only input arc: never vacuous, no constraint
        self.assertIsNone(enc._nonvacuous_condition(by_label["pick item"], 0))
        # only (p, I LIST) input: constraint over the LIST slots
        self.assertIsNotNone(enc._nonvacuous_condition(by_label["failed delivery"], 0))

    def test_structural_replay_rejects_vacuous_and_accepts_valid(self):
        from odacc.structural_replay import ordered_marking_errors
        from odacc.domain import AlignmentMove

        class Net:
            _data_types = set()
            _places = [{"id": "q"}, {"id": "r"}]
            _transitions = [{"id": "t", "label": "fd"}]
            _arcs = [{"source": "q", "target": "t", "inscription": [("p", "PACKAGE"), ("I", "ITEM LIST")]},
                     {"source": "t", "target": "r", "inscription": [("p", "PACKAGE")]}]

            @staticmethod
            def is_exact_sync_arc(p, t):
                return False

        def bind(component, net, t, objects, fresh_object=None):
            return {"p": "p1", "I": [o for o in objects if o.startswith("i")]}

        vacuous = [AlignmentMove(kind="model", cost=1, activity="fd", transition="fd",
                                 objects=("p1",), model_objects=("p1",))]
        errors, unknown = ordered_marking_errors(Net, None, vacuous, bind)
        self.assertIsNone(unknown)
        self.assertTrue(errors and "vacuous" in errors[0])

    def test_two_mixed_037_cost_two(self):
        """Case that motivated the non-vacuous rule: final cost 2, all certified."""
        log = ROOT / ("examples/order_management/multi_deviation_benchmark/two_mixed/"
                      "component_037_events_0012_objects_0006__two_mixed_016.jsonocel")
        prefixes = run_online(OM_MODEL, log, "symbolic", timeout=120)
        self.assertEqual(prefixes[-1]["cost"], 2)
        self.assertTrue(all(r.get("certified") for r in prefixes))

    def test_exact_sync_unconsumed_is_anchored(self):
        from odacc.symbolic_search import JODAPSolver

        class Net:
            _data_types = set()

            @staticmethod
            def is_exact_sync_arc(p, t):
                return True

        place = {"id": 5}
        arc = {"source": 5, "target": 9, "inscription": [("o", "ORDER"), ("P", "PRODUCT LIST")]}
        marked = [("o1", "p1"), ("o1", "p2"), ("o2", "p3")]
        f = JODAPSolver._exact_sync_unconsumed
        # Binding o1 with both of its products: other order's token is irrelevant.
        self.assertIsNone(f(Net, place, arc, {"o": "o1", "P": ["p1", "p2"]}, marked,
                            [("o1", "p1"), ("o1", "p2")]))
        # Binding o1 with only one product violates exactness.
        self.assertEqual(f(Net, place, arc, {"o": "o1", "P": ["p1"]}, marked, [("o1", "p1")]),
                         ("o1", "p2"))


if __name__ == "__main__":
    unittest.main()
