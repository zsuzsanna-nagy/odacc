from __future__ import annotations

import json
from typing import Any, Iterable

from .domain import AlignmentMove, AlignmentResult, ObjectCentricSchema


def schema_to_json(schema: ObjectCentricSchema) -> str:
    payload = {
        "activities": sorted(schema.activities),
        "object_types": sorted(schema.object_types),
        "qualifiers": sorted(schema.qualifiers),
        "attributes": {k: sorted(v) for k, v in sorted(schema.attributes.items())},
        "attribute_types": dict(sorted(schema.attribute_types.items())),
    }
    return json.dumps(payload, indent=2)


def _primitive(value: Any) -> Any:
    """Return a JSON scalar without invoking arbitrary object protocols.

    Unknown/native values are represented only by their Python type name.  In
    particular, we deliberately do *not* call str(value), iterate value, inspect
    dataclasses, copy it, or test Mapping/Iterable protocols.  A checkpoint
    writer must never touch retained solver/native internals merely to report a
    result.
    """
    if value is None or type(value) in (str, int, float, bool):
        return value
    return f"<{type(value).__name__}>"


def _plain_builtin(value: Any, *, depth: int = 0, max_depth: int = 6) -> Any:
    """Convert only exact builtin containers to JSON-safe values.

    This is intentionally *not* a generic serializer.  It walks exact dict,
    list, tuple, set and frozenset instances only.  Subclasses, dataclasses,
    solver wrappers, Mapping implementations and arbitrary iterables are never
    traversed.
    """
    if value is None or type(value) in (str, int, float, bool):
        return value
    if depth >= max_depth:
        return f"<{type(value).__name__}>"

    if type(value) is dict:
        out: dict[str, Any] = {}
        for key, item in value.items():
            # Keys in report payloads should be primitive.  Do not call str on
            # arbitrary/native keys.
            if key is None or type(key) in (str, int, float, bool):
                safe_key = str(key)
            else:
                safe_key = f"<{type(key).__name__}>"
            out[safe_key] = _plain_builtin(item, depth=depth + 1, max_depth=max_depth)
        return out

    if type(value) in (list, tuple):
        return [
            _plain_builtin(item, depth=depth + 1, max_depth=max_depth)
            for item in value
        ]

    if type(value) in (set, frozenset):
        # Never compare or stringify arbitrary members while sorting.  Convert
        # each member first, then sort only the resulting primitive JSON forms
        # by repr, which cannot call back into the original native object.
        safe_items = [
            _plain_builtin(item, depth=depth + 1, max_depth=max_depth)
            for item in value
        ]
        return sorted(safe_items, key=repr)

    return f"<{type(value).__name__}>"


def _string_sequence(value: Any) -> list[str]:
    """Serialize only an exact list/tuple/set of primitive identifiers."""
    if type(value) not in (list, tuple, set, frozenset):
        return []
    result: list[str] = []
    for item in value:
        if item is None:
            result.append("None")
        elif type(item) in (str, int, float, bool):
            result.append(str(item))
        else:
            result.append(f"<{type(item).__name__}>")
    return result


def _plain_dict(value: Any) -> dict[str, Any]:
    """Serialize an exact builtin dict; reject all mapping-like objects."""
    if type(value) is not dict:
        return {}
    converted = _plain_builtin(value)
    return converted if type(converted) is dict else {}


def _plain_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    """Serialize only an exact list/tuple whose entries are exact dicts."""
    if type(value) not in (list, tuple):
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        if type(entry) is dict:
            out.append(_plain_dict(entry))
        else:
            # Preserve list shape without inspecting the unexpected object.
            out.append({"_unserialized_type": type(entry).__name__})
    return out


def move_dict(move: AlignmentMove) -> dict[str, Any]:
    """Serialize a move using a strict public-field whitelist only.

    No ``__dict__`` traversal, dataclass conversion, deepcopy, generic Mapping
    traversal or arbitrary-object stringification is performed.
    """
    return {
        "kind": _primitive(move.kind),
        "cost": _primitive(move.cost),
        "activity": _primitive(move.activity),
        "event_id": _primitive(move.event_id),
        "transition": _primitive(move.transition),
        "objects": _string_sequence(move.objects),
        "silent": bool(move.silent) if type(move.silent) is bool else False,
        "object_creation": (
            bool(move.object_creation) if type(move.object_creation) is bool else False
        ),
        "observed_objects": _string_sequence(move.observed_objects),
        "model_objects": _string_sequence(move.model_objects),
        "object_match": (
            move.object_match
            if move.object_match is None or type(move.object_match) is bool
            else None
        ),
        "observed_data": _plain_dict(move.observed_data),
        "model_data": _plain_dict(move.model_data),
        "data_mismatches": _string_sequence(move.data_mismatches),
        "control_flow_cost": _primitive(move.control_flow_cost),
        "data_cost": _primitive(move.data_cost),
        "object_cost": _primitive(move.object_cost),
    }


def result_dict(r: AlignmentResult) -> dict[str, Any]:
    """Serialize only the public report fields of an alignment result.

    The checkpoint/report layer is deliberately isolated from internal search,
    continuation, provenance and solver state.  Any such state attached to a
    result or move is ignored rather than recursively inspected.
    """
    if type(r.moves) in (list, tuple):
        moves = [move_dict(m) for m in r.moves if isinstance(m, AlignmentMove)]
    else:
        moves = []

    return {
        "component": _primitive(r.component_id),
        "prefix_position": _primitive(r.prefix_position),
        "mode": _primitive(r.mode),
        "feasible": bool(r.feasible) if type(r.feasible) is bool else False,
        "cost": _primitive(r.cost),
        "cost_breakdown": _plain_dict(r.cost_breakdown),
        "encode_seconds": (
            float(r.encode_seconds)
            if type(r.encode_seconds) in (int, float)
            else 0.0
        ),
        "solve_seconds": (
            float(r.solve_seconds)
            if type(r.solve_seconds) in (int, float)
            else 0.0
        ),
        "moves": moves,
        "model_run": _plain_list_of_dicts(r.model_run),
        "assignments": _plain_list_of_dicts(r.assignments),
        "joint_assignment": _plain_dict(r.joint_assignment),
    }



def certified_result_dict(r: AlignmentResult) -> dict[str, Any]:
    """Return a compact immutable checkpoint payload for one certified result.

    Checkpoint/heartbeat persistence must never traverse model runs, decoded
    assignments, joint-assignment dictionaries, solver objects, or search state.
    Only scalar report fields and a compact move summary are copied while the
    result is known to be certified.  The normal final report still uses
    :func:`result_dict` after a successful run completes.
    """
    moves = []
    if type(r.moves) in (list, tuple):
        for move in r.moves:
            if not isinstance(move, AlignmentMove):
                continue
            moves.append({
                "kind": _primitive(move.kind),
                "cost": _primitive(move.cost),
                "activity": _primitive(move.activity),
                "event_id": _primitive(move.event_id),
                "transition": _primitive(move.transition),
                "objects": _string_sequence(move.objects),
                "control_flow_cost": _primitive(move.control_flow_cost),
                "data_cost": _primitive(move.data_cost),
                "object_cost": _primitive(move.object_cost),
            })
    cb = r.cost_breakdown if type(r.cost_breakdown) is dict else {}
    safe_cb = {}
    for key in ("total", "control_flow", "data", "object"):
        value = cb.get(key)
        if value is None or type(value) in (str, int, float, bool):
            safe_cb[key] = value
    return {
        "component": _primitive(r.component_id),
        "prefix_position": _primitive(r.prefix_position),
        "mode": _primitive(r.mode),
        "feasible": bool(r.feasible) if type(r.feasible) is bool else False,
        "cost": _primitive(r.cost),
        "cost_breakdown": safe_cb,
        "encode_seconds": float(r.encode_seconds) if type(r.encode_seconds) in (int, float) else 0.0,
        "solve_seconds": float(r.solve_seconds) if type(r.solve_seconds) in (int, float) else 0.0,
        "moves": moves,
        "checkpoint_compact": True,
    }

def results_to_json(results: Iterable[AlignmentResult]) -> str:
    return json.dumps([result_dict(r) for r in results], indent=2)
