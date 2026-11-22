"""Conservative ordered token-flow validation of a decoded ODACC alignment.

This deliberately does not validate data guards or prove optimality.  It checks
concrete object bindings, token availability and actual input consumption, using
CoCoMoT's already parsed DOPID metadata but not the optimizer/SMT model.
Ambiguous bindings or unsupported initial markings are reported as *unverified*.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, Sequence


def _nonvacuous_enabled() -> bool:
    try:
        from . import cocomot_adapter
        return bool(cocomot_adapter.NONVACUOUS_FIRING)
    except Exception:
        return True


def _arc_tokens(net: Any, inscription: Sequence[tuple[str, str]], binding: dict):
    cols = []
    for name, typ in inscription:
        if typ in net._data_types:
            continue
        if name not in binding:
            raise ValueError(f"unbound arc variable {name}")
        v = binding[name]
        if "LIST" in typ:
            cols.append(list(v))
        else:
            cols.append([v])
    return [tuple(tok) for tok in product(*cols)] if cols else [tuple()]


def ordered_marking_errors(net: Any, component: Any, moves: Sequence[Any],
                           bind_function: Any) -> tuple[list[str], str | None]:
    """Return (definite errors, unknown reason).

    Unknown is *not* equivalent to valid. Callers must not promote unknown to an
    independent feasibility certificate.  Objects are immutable token colors.
    """
    try:
        places = {p["id"]: p for p in net._places}
        trans = {t["id"]: t for t in net._transitions}
        by_label = defaultdict(list)
        for t in trans.values():
            by_label[t.get("label")].append(t)
        for p in net._places:
            initial = p.get("initialMarking", p.get("initial_marking", 0))
            if initial and str(initial) not in ("0", "0.0"):
                return [], "nonempty initial marking needs token-level initialization"
        marking = {pid: set() for pid in places}
        arcs = list(net._arcs)
        for step, move in enumerate(moves):
            if move.kind == "log":
                continue
            if move.kind not in ("model", "sync"):
                return [], f"unsupported move kind at {step}: {move.kind}"
            matches = [trans[move.transition_id]] if hasattr(move, "transition_id") and move.transition_id in trans else by_label.get(move.transition, [])
            if len(matches) != 1:
                return [], f"ambiguous transition at {step}: {getattr(move, 'transition', None)}"
            t = matches[0]
            objects = tuple(move.model_objects or move.objects or ())
            nu = {name for a in arcs if a["source"] == t["id"]
                  for name, typ in a.get("inscription", ()) if "nu" in str(name) and typ not in net._data_types}
            fresh = objects[0] if nu and len(objects) == 1 else None
            binding = bind_function(component, net, t, objects, fresh_object=fresh)
            if binding is None:
                return [], f"ambiguous binding at step {step}, {t.get('label')}"
            pre = [a for a in arcs if a["target"] == t["id"] and a["source"] in places]
            post = [a for a in arcs if a["source"] == t["id"] and a["target"] in places]
            consumed = []
            for arc in pre:
                p = places[arc["source"]]
                requested = _arc_tokens(net, arc.get("inscription", ()), binding)
                for tok in requested:
                    if tok not in marking[p["id"]]:
                        return [f"step {step}: {t.get('label')} requires absent token "
                                f"{p['id']}:{tok}"], None
                    consumed.append((p["id"], tok))
                try:
                    exact = net.is_exact_sync_arc(p["id"], t["id"])
                except (AttributeError, KeyError):
                    exact = False
                if exact:
                    scalars = [(i, binding[name]) for i, (name, typ) in enumerate(
                        (x for x in arc.get("inscription", ()) if x[1] not in net._data_types))
                        if "LIST" not in typ]
                    needed = set(requested)
                    for tok in marking[p["id"]]:
                        if all(i < len(tok) and tok[i] == v for i, v in scalars) and tok not in needed:
                            return [f"step {step}: {t.get('label')} leaves exact-sync token "
                                    f"{p['id']}:{tok}"], None
            if pre and not consumed and _nonvacuous_enabled():
                return [f"step {step}: {t.get('label')} fires vacuously "
                        f"(input LIST binding empty; no input token consumed)"], None
            for pid, tok in consumed:
                marking[pid].remove(tok)
            for arc in post:
                for tok in _arc_tokens(net, arc.get("inscription", ()), binding):
                    marking[arc["target"]].add(tok)
        return [], None
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return [], f"structural replay unavailable: {exc}"
