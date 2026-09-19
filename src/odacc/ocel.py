from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .domain import (
    ObjectAttributeObservation,
    ObjectCentricSchema,
    ObjectCreation,
    ObservableUnit,
    O2ORelation,
    StreamEvent,
)


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "Integer"
    if isinstance(value, float):
        return "Real"
    if isinstance(value, str):
        return "String"
    if value is None:
        return "Null"
    return type(value).__name__


def _register_type(schema: ObjectCentricSchema, attr: str, value: Any) -> None:
    typ = _type_name(value)
    old = schema.attribute_types.get(attr)
    if old is None or old == "Null":
        schema.attribute_types[attr] = typ
    elif typ != "Null" and old != typ:
        # Keep the schema deterministic while flagging inconsistent inputs.
        raise ValueError(f"attribute {attr!r} has inconsistent types: {old} and {typ}")


def load_ocel1(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "ocel:events" not in data or "ocel:objects" not in data:
        raise ValueError("Only OCEL 1.0 JSON is currently supported by the CoCoMoT adapter")
    return data


def extract_schema(data: Mapping[str, Any]) -> ObjectCentricSchema:
    schema = ObjectCentricSchema()
    objects = data["ocel:objects"]
    events = data["ocel:events"]

    for _, obj in objects.items():
        ot = obj["ocel:type"]
        schema.object_types.add(ot)
        attrs = obj.get("ocel:ovmap") or {}
        schema.attributes.setdefault(ot, set()).update(attrs.keys())
        for a, v in attrs.items():
            _register_type(schema, a, v)

    for _, ev in events.items():
        act = ev["ocel:activity"]
        schema.activities.add(act)
        attrs = ev.get("ocel:vmap") or {}
        schema.attributes.setdefault(act, set()).update(attrs.keys())
        for a, v in attrs.items():
            _register_type(schema, a, v)

    # OCEL 1.0 omap entries are unqualified. Keep the qualifier set empty.
    return schema


def _event_sort_key(item: Tuple[str, Mapping[str, Any]]) -> Tuple[str, str]:
    event_id, ev = item
    # ISO timestamps are lexicographically sortable in the supplied examples.
    return str(ev.get("ocel:timestamp", "")), str(event_id)


def simulate_stream(data: Mapping[str, Any]) -> List[ObservableUnit]:
    """Turn an OCEL 1.0 log into the observable-unit stream used by ODACC.

    An object is created in the first unit that references it. Static OCEL 1.0
    ovmap values are emitted as initial object-attribute observations in that
    same unit. OCEL 1.0 contains no explicit O2O updates, so that component is
    empty. Event omap entries become unqualified E2O relations.
    """

    objects = data["ocel:objects"]
    events = sorted(data["ocel:events"].items(), key=_event_sort_key)
    observed_objects = set()
    result: List[ObservableUnit] = []

    for pos, (eid, raw) in enumerate(events, start=1):
        timestamp = str(raw.get("ocel:timestamp", pos))
        omap = tuple(raw.get("ocel:omap") or ())
        new_objects = [o for o in omap if o not in observed_objects]
        creations = tuple(ObjectCreation(o, objects[o]["ocel:type"]) for o in new_objects)
        attrs = []
        for o in new_objects:
            for name, value in (objects[o].get("ocel:ovmap") or {}).items():
                attrs.append(ObjectAttributeObservation(o, name, timestamp, value))
        observed_objects.update(new_objects)
        event = StreamEvent(
            event_id=str(eid),
            activity=raw["ocel:activity"],
            timestamp=timestamp,
            attributes=dict(raw.get("ocel:vmap") or {}),
            relations=tuple(("", o) for o in omap),
        )
        result.append(
            ObservableUnit(
                position=pos,
                event=event,
                object_creations=creations,
                o2o_updates=(),
                attribute_updates=tuple(attrs),
            )
        )

    # Objects never referenced by an event still become observable. They are
    # appended as update-only units so they are not silently lost.
    for o, raw_obj in objects.items():
        if o in observed_objects:
            continue
        pos = len(result) + 1
        timestamp = str(pos)
        attrs = tuple(
            ObjectAttributeObservation(o, a, timestamp, v)
            for a, v in (raw_obj.get("ocel:ovmap") or {}).items()
        )
        result.append(
            ObservableUnit(
                position=pos,
                object_creations=(ObjectCreation(o, raw_obj["ocel:type"]),),
                attribute_updates=attrs,
            )
        )
    return result
