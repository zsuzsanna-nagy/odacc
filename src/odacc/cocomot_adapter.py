from __future__ import annotations

import io
import itertools
import os
import sys
import time
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .domain import AlignmentMove, AlignmentResult
from .components import ComponentState
from .soft_observations import install_soft_object_property_semantics, observed_object_attribute_terms


class DependencyError(RuntimeError):
    pass


@dataclass
class CocomotModules:
    OPI: Any
    read_pnml_input: Any
    Encoding: Any
    Trace: Any
    Event: Any
    Z3Solver: Any


def load_cocomot(cocomot_root: str) -> CocomotModules:
    src = os.path.abspath(os.path.join(cocomot_root, "src"))
    if not os.path.isdir(src):
        raise DependencyError(f"CoCoMoT src directory not found: {src}")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from objectcentric.opi import OPI
        from objectcentric.encoding import Encoding
        from objectcentric.input import Trace, Event
        from dpn.read import read_pnml_input
        from smt.z3solver import Z3Solver
    except Exception as exc:
        raise DependencyError(
            "Could not load CoCoMoT/Z3. Install CoCoMoT's dependencies "
            "(in particular z3-solver) and pass --cocomot-root. Original error: "
            + repr(exc)
        ) from exc
    return CocomotModules(OPI, read_pnml_input, Encoding, Trace, Event, Z3Solver)


def make_observation_encoding_classes(base_encoding):
    class SoftObservationEncoding(base_encoding):
        """CoCoMoT encoding with selected-transition-local data semantics.

        CoCoMoT's object-centric encoder enumerates every transition at every
        model step.  Its token-data cache originally identifies a token firing
        only from the concrete object parameters; it does *not* include the
        transition variable ``T_i``.  Consequently data-transfer constraints of
        an unselected transition can constrain the data variables of the
        transition that actually fires.  This becomes observable once log-side
        and model-side values are separated (e.g. the ``late_ship_data`` case).

        ODACC therefore makes both token firing and guards conditional on the
        transition selected at the current, active model step.  This preserves
        the original marking/data-store machinery while preventing unselected
        transitions from leaking data constraints into the chosen run.
        """

        def _selected_at(self, t, i):
            s = self._solver
            return s.land([
                s.eq(self._transition_vars[i], s.num(t["id"])),
                s.lt(s.num(i), self._run_length_var),
            ])

        def is_fired_token(self, p, t, tok, j, incoming):
            raw = super().is_fired_token(p, t, tok, j, incoming)
            return self._solver.land([self._selected_at(t, j), raw])

        def transition_constraint(self, t, i):
            raw = super().transition_constraint(t, i)
            # Be defensive around legacy/custom CoCoMoT encoders: an
            # object-dependent guard with no admissible concrete binding used
            # to be represented by None.  Semantically that guard cannot hold
            # for the selected transition, so encode it as false rather than
            # passing None into Z3.
            if raw is None:
                raw = self._solver.false()
            return self._solver.implies(self._selected_at(t, i), raw)

        def odacc_boundary_initial_state(self, marking_signature, token_store_values=()):
            """Fix instant 0 to a certified ODACC checkpoint witness.

            ``token_store_values`` contains already validated positional store
            values ``(place_id, token, values)``.  ODACC constructs it only when
            the retained witness and the fixed suffix inscriptions identify every
            data slot unambiguously; otherwise it does not use the specialized
            eager context.
            """
            s = self._solver
            marked = set()
            for item in marking_signature or ():
                try:
                    marked.add((int(item[0]), tuple(item[1])))
                except Exception:
                    continue

            constraints = []
            m0 = self._marking_vars[0]
            for pid, by_tok in m0.items():
                for tok, var in by_tok.items():
                    constraints.append(var if (int(pid), tuple(tok)) in marked else s.neg(var))

            stores = {}
            for item in token_store_values or ():
                try:
                    stores[(int(item[0]), tuple(item[1]))] = tuple(item[2])
                except Exception:
                    continue
            if getattr(self, "_data_store_vars", None):
                for pid, by_tok in self._data_store_vars[0].items():
                    for tok, vars_ in by_tok.items():
                        vals = stores.get((int(pid), tuple(tok)))
                        if vals is None:
                            continue
                        for var, value in zip(vars_, vals):
                            try:
                                val = s.real(value) if isinstance(value, float) else s.num(int(value))
                            except Exception:
                                continue
                            constraints.append(s.eq(var, val))
            return s.land(constraints)

    class PrefixEncoding(SoftObservationEncoding):
        """Soft-observation encoding with a reachable-prefix terminal condition."""

        def prefix_state(self):
            s = self._solver
            rl = self._run_length_var
            return s.land([s.le(s.num(0), rl), s.le(rl, s.num(self._step_bound))])

    return SoftObservationEncoding, PrefixEncoding


class CocomotBackend:
    def __init__(self, cocomot_root: str, model_path: str, fixed_objects: bool = False,
                 quiet_solver: bool = False):
        self.mods = load_cocomot(cocomot_root)
        self.model_path = model_path
        self.fixed_objects = fixed_objects
        self.quiet_solver = quiet_solver
        # Use the same observation/model-state separation as JODAP.  Event
        # values were already soft in CoCoMoT; this extends the semantics to
        # object-carried attributes as well.
        install_soft_object_property_semantics()
        self.OfflineEncoding, self.PrefixEncoding = make_observation_encoding_classes(self.mods.Encoding)

    def _new_net(self):
        return self.mods.OPI(self.mods.read_pnml_input(self.model_path))

    @staticmethod
    def _coerce_data(value: Any) -> Optional[Any]:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        return None

    def _trace_for_component(self, component: ComponentState):
        net = self._new_net()
        declared = dict(net.get_data_variables())
        events = []
        # CoCoMoT orders Trace events by integer id. Re-number according to
        # stream order while retaining original IDs in the outer result.
        ordered_units = [u for u in component.units if u.event is not None]
        ordered_units.sort(key=lambda u: u.position)
        for seq, unit in enumerate(ordered_units):
            e = unit.event
            vals: Dict[str, Any] = {}
            for name, value in e.attributes.items():
                if name not in declared:
                    continue
                coerced = self._coerce_data(value)
                if coerced is not None:
                    vals[name] = coerced
            events.append(self.mods.Event(seq, e.activity, e.timestamp, list(e.objects), vals))

        object_attrs = component.observation_formula.current_object_attributes()
        objs = {}
        for obj in sorted(component.objects):
            typ = component.observation_formula.object_types[obj]
            objs[obj] = {"type": typ, "ovmap": dict(object_attrs.get(obj, {}))}
        # CoCoMoT's Trace constructor intentionally stores only a set of object
        # identifiers. The object-centric Encoding, however, expects the trace
        # objects to have been enriched with their type/attribute dictionaries
        # via Trace.add_object_types(), as done by CoCoMoT's Log.split_into_traces().
        trace = self.mods.Trace(events, ["timestamp"], objs.keys())
        trace.add_object_types(objs)
        return net, trace, ordered_units

    @staticmethod
    def _eval_data_value(model, var, value_type: str):
        if value_type in ("Rational", "Real", "java.lang.Double"):
            return model.eval_real(var)
        return model.eval_int(var)

    @staticmethod
    def _guard_variable_names(constraint) -> set[str]:
        """Return variable names occurring anywhere in a DOPID guard AST.

        This is intentionally syntax-driven and conservative.  It is used only
        to decide whether ODACC must add its semantic guard overlay; unsupported
        nodes simply contribute no names and therefore leave the original
        CoCoMoT constraint untouched.
        """
        if constraint is None:
            return set()
        out: set[str] = set()
        seen: set[int] = set()
        stack = [constraint]
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if node.__class__.__name__ == "Var":
                name = getattr(node, "name", None)
                if isinstance(name, str):
                    out.add(name)
            for field in ("left", "right", "expr", "_left", "_right", "_expr"):
                child = getattr(node, field, None)
                if child is not None:
                    stack.append(child)
            for field in ("_args", "args"):
                children = getattr(node, field, None)
                if isinstance(children, (list, tuple)):
                    stack.extend(children)
            for value in getattr(node, "__dict__", {}).values():
                if isinstance(value, (list, tuple)):
                    stack.extend(x for x in value if x is not None)
                elif value is not None and value is not node:
                    mod = getattr(value.__class__, "__module__", "")
                    if isinstance(mod, str) and mod.startswith("dpn."):
                        stack.append(value)
        return out

    @staticmethod
    def _transition_data_flow(net, transition) -> tuple[set[str], set[str]]:
        """Return data variables carried into and out of ``transition``."""
        reads: set[str] = set()
        writes: set[str] = set()
        tid = transition["id"]
        data_types = set(getattr(net, "_data_types", ()))
        for arc in getattr(net, "_arcs", ()):
            incoming = arc.get("target") == tid
            outgoing = arc.get("source") == tid
            if not incoming and not outgoing:
                continue
            for name, typ in arc.get("inscription", ()):
                if typ not in data_types:
                    continue
                if incoming:
                    reads.add(str(name))
                if outgoing:
                    writes.add(str(name))
        return reads, writes

    def _semantic_output_guard_constraints(self, encoding, net, component: ComponentState):
        """Align CoCoMoT's exact guard semantics with JODAP's model-side values.

        JODAP interprets a data variable written by the selected transition as
        the model-side value of that transition.  An observed event value is a
        separate immutable log fact and equality between both sides is soft.

        The original object-centric CoCoMoT encoding can, for a variable that is
        *written but not carried on an input arc*, evaluate a guard against a
        different/pre-state symbolic value while the edit-distance observation
        is compared with ``_data_vars[i][name]``.  This admits zero-cost models
        such as ``d=3,m=1`` for a guard requiring ``m=0``.

        We add a selected-transition-local copy of precisely those guards over
        the same per-step model-side variables that JODAP and the observation
        cost use.  Existing CoCoMoT constraints remain in force.  Thus this is a
        semantic strengthening only where read/write versions were previously
        disconnected; it is not a search/pruning optimization.

        Object parameters referenced by the guard are concretely enumerated and
        guarded by the corresponding object-variable equality.  If a guard
        depends on a LIST-valued object parameter, we conservatively skip the
        overlay for that transition rather than risk imposing an unsound binding;
        CoCoMoT's original exact constraint then remains authoritative.
        """
        s = encoding._solver
        constraints = []
        object_types = component.observation_formula.object_types
        object_attrs = component.observation_formula.current_object_attributes()
        objects_for_replacer = {
            obj: {"type": object_types.get(obj), "ovmap": dict(object_attrs.get(obj, {}))}
            for obj in component.objects
        }

        try:
            from dpn.expr_utils import VarReplacer, ListExpander
            import objectcentric.encoding as oc_encoding
            ObjectPropertyReplacer = oc_encoding.ObjectPropertyReplacer
        except Exception:
            return s.true()

        for t in getattr(net, "_transitions", ()):
            guard = t.get("constraint")
            if guard is None:
                continue
            reads, writes = self._transition_data_flow(net, t)
            guard_names = self._guard_variable_names(guard)
            # The problematic semantic gap concerns a model value introduced by
            # this transition and used by its guard.  Variables already carried
            # into the transition retain CoCoMoT's normal pre-state semantics.
            if not ((writes - reads) & guard_names):
                continue

            try:
                params = list(net.object_params_of_transition(t, encoding._objects))
            except Exception:
                params = []
            by_name = {str(p.get("name")): p for p in params}
            relevant_param_names = [n for n in guard_names if n in by_name]
            if any("LIST" in str(by_name[n].get("type", "")) for n in relevant_param_names):
                continue

            # Candidate concrete bindings only for object parameters actually
            # referenced by this guard.  Unreferenced PRODUCT LIST parameters,
            # for example, do not multiply the overlay.
            candidates = []
            supported = True
            for name in relevant_param_names:
                p = by_name[name]
                typ = str(p.get("type", ""))
                base = typ[:typ.rfind(" LIST")] if " LIST" in typ else typ
                objs = sorted(o for o in component.objects if object_types.get(o) == base)
                if not objs:
                    supported = False
                    break
                candidates.append((name, p, objs))
            if not supported:
                continue

            binding_products = itertools.product(*(x[2] for x in candidates)) if candidates else [()]
            for values in binding_products:
                binding = {candidates[k][0]: values[k] for k in range(len(candidates))}
                # Object parameters denote concrete participating objects; do
                # not admit the same concrete object in two distinct scalar slots.
                if len(set(binding.values())) != len(binding):
                    continue
                try:
                    g = deepcopy(guard)
                    if binding:
                        g.accept(VarReplacer(dict(binding)))
                    exp = ListExpander()
                    g.accept(exp)
                    while exp._change:
                        exp._change = False
                        g.accept(exp)
                    g.accept(ObjectPropertyReplacer(objects_for_replacer))
                except Exception:
                    continue

                for i in range(encoding.get_step_bound()):
                    try:
                        guard_formula = g.toSMT(s, encoding._data_vars[i])
                    except Exception:
                        continue
                    conds = [encoding._selected_at(t, i)]
                    valid_binding = True
                    for name, obj in binding.items():
                        p = by_name[name]
                        try:
                            idx = int(p.get("index"))
                            oid = encoding._id_by_object_name[obj]
                            conds.append(s.eq(encoding._object_vars[i][idx], s.num(oid)))
                        except Exception:
                            valid_binding = False
                            break
                    if not valid_binding:
                        continue
                    constraints.append(s.implies(s.land(conds), guard_formula))

        return s.land(constraints) if constraints else s.true()

    @staticmethod
    def _is_object_creation_transition(net, trans) -> bool:
        if not trans.get("invisible", False):
            return False
        tid = trans["id"]
        for arc in net._arcs:
            if arc.get("source") != tid:
                continue
            for name, _typ in arc.get("inscription", []):
                if "nu" in str(name):
                    return True
        return False

    def _decode(self, encoding, model, component: ComponentState, position: int, mode: str,
                ordered_units, encode_s: float, solve_s: float,
                objective_expr=None, object_attr_meta=None) -> AlignmentResult:
        component_id = component.component_id
        run_length = model.eval_int(encoding._run_length_var)
        tvars = encoding._transition_vars
        ovars = encoding._object_vars
        declared = dict(encoding._net.get_data_variables())

        model_run: List[Dict[str, Any]] = []
        for i in range(run_length):
            tid = model.eval_int(tvars[i])
            trans = next(t for t in encoding._net._transitions if t["id"] == tid)
            objs = []
            for k in range(encoding._max_objs_per_trans):
                oid = model.eval_int(ovars[i][k])
                if oid in encoding._object_name_by_id:
                    objs.append(encoding._object_name_by_id[oid])
            data_values: Dict[str, Any] = {}
            if getattr(encoding, "_data_vars", None):
                for name, vtype in declared.items():
                    if name in encoding._data_vars[i]:
                        try:
                            data_values[name] = self._eval_data_value(
                                model, encoding._data_vars[i][name], vtype
                            )
                        except Exception:
                            # Diagnostic extraction should not invalidate an alignment.
                            pass
            model_run.append({
                "step": i,
                "transition": trans.get("label"),
                "transition_id": trans.get("id"),
                "silent": bool(trans.get("invisible", False)),
                "object_creation": self._is_object_creation_transition(encoding._net, trans),
                "objects": objs,
                "model_state_data": data_values,
            })

        # Recover the optimal edit-distance path and annotate every move with the
        # decomposition used by CoCoMoT: synchronous data mismatch cost versus
        # structural/model-log cost. Exact object equality is a hard condition
        # for synchronous moves in the reference backend, hence object_cost=0.
        moves: List[AlignmentMove] = []
        assignments: List[Dict[str, Any]] = []
        i, j = run_length, len(ordered_units)
        while i > 0 or j > 0:
            is_model = i > 0 and model.eval_bool(encoding._vs_mod_move[i][j])
            is_sync = i > 0 and j > 0 and model.eval_bool(encoding._vs_sync_move[i][j])
            if is_model:
                prev_i, prev_j, kind = i - 1, j, "model"
            elif is_sync:
                prev_i, prev_j, kind = i - 1, j - 1, "sync"
            else:
                prev_i, prev_j, kind = i, j - 1, "log"
            event = ordered_units[j - 1].event if kind in ("log", "sync") else None
            trans_info = model_run[i - 1] if kind in ("model", "sync") else None
            observed_objects = tuple(sorted(event.objects)) if event else ()
            model_objects = tuple(sorted(trans_info["objects"])) if trans_info else ()

            observed_data: Dict[str, Any] = {}
            model_data: Dict[str, Any] = {}
            mismatches: List[str] = []
            data_cost = 0
            if kind == "sync" and event is not None and trans_info is not None:
                observed_data = {
                    name: value for name, value in event.attributes.items() if name in declared
                }
                # Report only model values that are directly comparable to an
                # observed event attribute. This avoids exposing arbitrary values
                # of unconstrained variables in the diagnostic output.
                for name, obs_value in observed_data.items():
                    if name in trans_info["model_state_data"]:
                        model_value = trans_info["model_state_data"][name]
                        model_data[name] = model_value
                        if model_value != obs_value:
                            mismatches.append(name)
                data_cost = len(mismatches)

            # Activity/object equality is a hard synchronous condition.  A
            # structural deviation is therefore represented by a log/model move,
            # while event/object-attribute value disagreement is soft.
            object_cost = 0
            if kind == "log":
                control_flow_cost = len(observed_objects)
            elif kind == "model":
                control_flow_cost = 0 if (trans_info and trans_info.get("silent")) else len(model_objects)
            else:
                control_flow_cost = 0
            step_cost = control_flow_cost + data_cost

            move = AlignmentMove(
                kind=kind,
                cost=step_cost,
                activity=event.activity if event else (trans_info["transition"] if trans_info else None),
                event_id=event.event_id if event else None,
                transition=trans_info["transition"] if trans_info else None,
                objects=observed_objects if event else model_objects,
                silent=bool(trans_info and trans_info.get("silent")),
                object_creation=bool(trans_info and trans_info.get("object_creation")),
                observed_objects=observed_objects,
                model_objects=model_objects,
                object_match=(observed_objects == model_objects) if kind == "sync" else None,
                observed_data=observed_data,
                model_data=model_data,
                data_mismatches=tuple(sorted(mismatches)),
                control_flow_cost=control_flow_cost,
                data_cost=data_cost,
                object_cost=object_cost,
            )
            moves.append(move)
            if kind in ("model", "sync"):
                assignments.append({
                    "model_step": i - 1,
                    "move_kind": kind,
                    "event_id": event.event_id if event else None,
                    "transition": trans_info["transition"] if trans_info else None,
                    "objects": list(model_objects),
                    "observed_objects": list(observed_objects),
                    "move_written_data": dict(model_data),
                    "observed_data": dict(observed_data),
                    "data_mismatches": sorted(mismatches),
                })
            i, j = prev_i, prev_j

        moves.reverse()
        assignments.reverse()

        object_attr_assignments = []
        object_attr_cost = 0
        for obj, attr, observed, _encoded, var in (object_attr_meta or []):
            try:
                mv_num = model.eval_real(var)
                if isinstance(observed, bool):
                    model_value = bool(round(mv_num))
                elif isinstance(observed, int) and not isinstance(observed, bool):
                    model_value = int(round(mv_num))
                elif isinstance(observed, float):
                    model_value = float(mv_num)
                elif isinstance(observed, str):
                    try:
                        from dpn.expr import Expr
                        model_value = Expr.strval(int(round(mv_num)))
                    except Exception:
                        model_value = mv_num
                else:
                    model_value = mv_num
                mismatch = model_value != observed
                object_attr_cost += int(mismatch)
                object_attr_assignments.append({
                    "object": obj, "attribute": attr,
                    "observed_value": observed, "model_value": model_value,
                    "mismatch": mismatch, "cost": int(mismatch),
                })
            except Exception:
                pass

        cf_cost = sum(m.control_flow_cost for m in moves)
        event_data_cost = sum(m.data_cost for m in moves)
        data_cost = event_data_cost + object_attr_cost
        object_cost = sum(m.object_cost for m in moves)
        # Evaluate the actual optimized objective when available.  The explicit
        # decomposition is constructed from move semantics and soft observations
        # and should coincide with it.
        try:
            total_cost = model.eval_int(objective_expr) if objective_expr is not None else cf_cost + data_cost + object_cost
        except Exception:
            total_cost = cf_cost + data_cost + object_cost
        cost_breakdown = {
            "total": int(total_cost),
            "control_flow": int(cf_cost),
            "data": int(data_cost),
            "object": int(object_cost),
        }

        # CoCoMoT's decode() prints its distance matrices as a side effect. Keep
        # raw diagnostics, but optionally suppress those console prints.
        if self.quiet_solver:
            buf = io.StringIO()
            with redirect_stdout(buf):
                raw = encoding.decode(model)
        else:
            raw = encoding.decode(model)

        return AlignmentResult(
            component_id=component_id,
            prefix_position=position,
            mode=mode,
            cost=int(total_cost),
            feasible=True,
            moves=moves,
            model_run=model_run,
            assignments=assignments,
            cost_breakdown=cost_breakdown,
            raw=raw,
            encode_seconds=encode_s,
            solve_seconds=solve_s,
            joint_assignment={"object_attribute_assignments": object_attr_assignments},
        )

    def solve(self, component: ComponentState, *, offline: bool = False) -> AlignmentResult:
        net, trace, ordered_units = self._trace_for_component(component)
        solver = self.mods.Z3Solver(incremental=False)
        EncodingClass = self.OfflineEncoding if offline else self.PrefixEncoding
        encoding = EncodingClass(solver, net, trace)
        t0 = time.perf_counter()
        encoding.create_variables()
        formulas = [
            encoding.initial_state(self.fixed_objects),
            encoding.transition_range(),
            encoding.object_types(),
            encoding.freshness(),
            encoding.moving_tokens(),
            encoding.remaining_tokens(),
            encoding.data_constraints(),
            # Semantic alignment with JODAP: when a selected transition guard
            # refers to a value introduced/written by that transition, constrain
            # the very same model-side step variable that is compared softly to
            # the observation.  Search-only JODAP optimizations (LB/UB, caching,
            # fixed-path incumbents) deliberately do not appear here.
            self._semantic_output_guard_constraints(encoding, net, component),
            encoding.final_state() if offline else encoding.prefix_state(),
        ]
        solver.require(formulas)
        # Token movement expressions are cached while the formulas above are built.
        solver.require([encoding.cache_constraints()])
        solver.require([encoding.edit_distance()])
        # Observations and model-side values are represented separately.  The
        # original edit-distance objective already handles event-data mismatch;
        # add the analogous soft object-attribute equalities.
        object_attr_terms, object_attr_meta = observed_object_attribute_terms(component, solver)
        dist = encoding.optimization_expression()
        objective = dist
        for term in object_attr_terms:
            objective = solver.plus(objective, term)
        encode_s = time.perf_counter() - t0
        optbound = (encoding.get_step_bound() * max(1, encoding.get_max_objs_per_trans())
                    + len(trace) + len(object_attr_terms) + 5)
        model = solver.minimize(objective, max=optbound)
        solve_s = solver.t_solve
        position = max((u.position for u in component.units), default=0)
        if model is None:
            return AlignmentResult(component.component_id, position, "offline" if offline else "prefix",
                                   None, False, encode_seconds=encode_s, solve_seconds=solve_s)
        result = self._decode(encoding, model, component, position,
                              "offline" if offline else "prefix", ordered_units, encode_s, solve_s,
                              objective_expr=objective, object_attr_meta=object_attr_meta)
        model.destroy()
        return result


def semantic_output_guard_constraints(encoding, net, component: ComponentState):
    """Public semantic overlay shared by the monolithic SMT and eager JODAP encoders."""
    helper = object.__new__(CocomotBackend)
    return helper._semantic_output_guard_constraints(encoding, net, component)
