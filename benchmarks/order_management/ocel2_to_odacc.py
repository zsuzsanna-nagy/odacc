#!/usr/bin/env python3
"""Convert OCEL 2.0 XML to ODACC-ready data.

Two output modes are implemented.

``components``
    Extract connected components using selected *core object types* and write
    one OCEL-1-compatible ``.jsonocel`` per component.  This is immediately
    consumable by the current ODACC CLI.  Event relations to contextual/global
    object types can be filtered away, which is important for order management:
    shared products/employees would otherwise connect almost the complete log.

``stream``
    Write an explicit JSONL observable stream that preserves OCEL 2.0 object
    creations, O2O relations, time-varying object attributes and qualified E2O
    relations.  This is the lossless representation to use when the ODACC XML
    loader is extended (and is already useful for inspection/reproducibility).

The order-management profile uses ORDER/ITEM/PACKAGE as the conformance
perspective.  Packages may contain items from several orders, so the connected
component extraction naturally retains genuine component merges.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class Obj:
    oid: str
    typ: str
    attrs: List[Tuple[str, str, Any]] = field(default_factory=list)
    rels: List[Tuple[str, str]] = field(default_factory=list)  # target, qualifier


@dataclass
class Ev:
    eid: str
    activity: str
    time: str
    attrs: Dict[str, Any]
    rels: List[Tuple[str, str]]  # object, qualifier


def cast_value(text: Optional[str], typ: Optional[str]) -> Any:
    s = (text or "").strip()
    t = (typ or "string").lower()
    if t in {"integer", "int", "long"}:
        try: return int(s)
        except ValueError: return s
    if t in {"float", "double", "real", "number"}:
        try: return float(s)
        except ValueError: return s
    if t in {"boolean", "bool"}:
        return s.lower() in {"1", "true", "yes"}
    return s


def load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ocel2(path: Path) -> Tuple[Dict[str, Obj], List[Ev], dict]:
    # The supplied public OCEL files are modest enough to parse with iterparse;
    # this avoids keeping the entire XML DOM in memory for the larger hinge log.
    object_attr_types: Dict[Tuple[str, str], str] = {}
    event_attr_types: Dict[Tuple[str, str], str] = {}
    objects: Dict[str, Obj] = {}
    events: List[Ev] = []

    # First obtain declarations.  iterparse is repeated deliberately to keep the
    # implementation dependency-free and robust to element ordering.
    current_ot = current_et = None
    for phase, elem in ET.iterparse(path, events=("start", "end")):
        if phase == "start" and elem.tag == "object-type":
            current_ot = elem.get("name")
        elif phase == "end" and elem.tag == "object-type":
            current_ot = None; elem.clear()
        elif phase == "start" and elem.tag == "event-type":
            current_et = elem.get("name")
        elif phase == "end" and elem.tag == "event-type":
            current_et = None; elem.clear()
        elif phase == "end" and elem.tag == "attribute":
            # Declarations have a type attribute; observations do not.
            if elem.get("type") is not None and current_ot:
                object_attr_types[(current_ot, elem.get("name") or "")] = elem.get("type") or "string"
            if elem.get("type") is not None and current_et:
                event_attr_types[(current_et, elem.get("name") or "")] = elem.get("type") or "string"
            elem.clear()

    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "object" and elem.get("id") is not None:
            oid, typ = elem.get("id") or "", elem.get("type") or ""
            attrs = []
            for a in elem.findall("./attributes/attribute"):
                name = a.get("name") or ""
                attrs.append((name, a.get("time") or "", cast_value(a.text, object_attr_types.get((typ, name)))))
            rels = [(r.get("object-id") or "", r.get("qualifier") or "") for r in elem.findall("./objects/relationship")]
            objects[oid] = Obj(oid, typ, attrs, rels)
            elem.clear()
        elif elem.tag == "event" and elem.get("id") is not None:
            eid, act, tm = elem.get("id") or "", elem.get("type") or "", elem.get("time") or ""
            attrs: Dict[str, Any] = {}
            for a in elem.findall("./attributes/attribute"):
                name = a.get("name") or ""
                attrs[name] = cast_value(a.text, event_attr_types.get((act, name)))
            rels = [(r.get("object-id") or "", r.get("qualifier") or "") for r in elem.findall("./objects/relationship")]
            events.append(Ev(eid, act, tm, attrs, rels))
            elem.clear()

    events.sort(key=lambda e: (e.time, e.eid))
    meta = {"object_attribute_types": object_attr_types, "event_attribute_types": event_attr_types}
    return objects, events, meta


def mapped_type(profile: dict, typ: str) -> Optional[str]:
    return profile.get("object_type_map", {}).get(typ)


def mapped_activity(profile: dict, act: str) -> Optional[str]:
    return profile.get("activity_map", {}).get(act)


def core_components(objects: Dict[str, Obj], profile: dict) -> List[Set[str]]:
    core_types = set(profile["core_object_types"])
    graph: Dict[str, Set[str]] = defaultdict(set)
    for oid, obj in objects.items():
        if obj.typ in core_types:
            graph[oid]  # create singleton
    for oid, obj in objects.items():
        if obj.typ not in core_types:
            continue
        for target, _q in obj.rels:
            other = objects.get(target)
            if other is not None and other.typ in core_types:
                graph[oid].add(target); graph[target].add(oid)
    seen: Set[str] = set(); out = []
    for root in sorted(graph):
        if root in seen: continue
        comp: Set[str] = set(); stack = [root]; seen.add(root)
        while stack:
            x = stack.pop(); comp.add(x)
            for y in graph[x]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        out.append(comp)
    return out


def latest_attr_value(obj: Obj, name: str) -> Any:
    vals = [(tm, v) for n, tm, v in obj.attrs if n == name]
    if not vals: return None
    vals.sort(key=lambda x: x[0])
    return vals[-1][1]


def component_jsonocel(objects: Dict[str, Obj], events: List[Ev], comp: Set[str], profile: dict) -> dict:
    core_types = set(profile["core_object_types"])
    act_map = profile.get("activity_map", {})
    attr_map = profile.get("object_attribute_map", {})

    # An event belongs to the component iff it touches one of its core objects.
    # Relations to core objects outside the component are not expected for a
    # connected component; if present they are dropped conservatively.
    selected_events: List[Ev] = []
    for e in events:
        if e.activity not in act_map: continue
        if any(o in comp for o, _q in e.rels):
            selected_events.append(e)

    # Keep only core objects actually relevant to this component.
    out_objects = {}
    for oid in sorted(comp):
        obj = objects[oid]
        mapped = mapped_type(profile, obj.typ)
        if mapped is None: continue
        ovmap = {}
        for src, dst in attr_map.get(obj.typ, {}).items():
            v = latest_attr_value(obj, src)
            if v is not None: ovmap[dst] = v
        out_objects[oid] = {"ocel:type": mapped, "ocel:ovmap": ovmap}

    out_events = {}
    for e in selected_events:
        omap = []
        for oid, _qual in e.rels:
            if oid in comp and oid in out_objects:
                omap.append(oid)
        # Drop events that become objectless after applying the perspective.
        if not omap: continue
        out_events[e.eid] = {
            "ocel:activity": mapped_activity(profile, e.activity),
            "ocel:timestamp": e.time,
            "ocel:omap": omap,
            "ocel:vmap": dict(e.attrs),
        }
    return {
        "ocel:global-log": {"ocel:version": "1.0", "ocel:ordering": "timestamp"},
        "ocel:global-event": {},
        "ocel:global-object": {},
        "ocel:objects": out_objects,
        "ocel:events": out_events,
    }


def write_components(objects: Dict[str, Obj], events: List[Ev], profile: dict, outdir: Path,
                     min_events: int, max_events: int, limit: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    comps = core_components(objects, profile)
    rows = []
    emitted = 0
    for idx, comp in enumerate(comps, start=1):
        data = component_jsonocel(objects, events, comp, profile)
        ne = len(data["ocel:events"]); no = len(data["ocel:objects"])
        counts = defaultdict(int)
        for o in comp: counts[objects[o].typ] += 1
        if ne < min_events or (max_events and ne > max_events):
            status = "filtered"
            filename = ""
        elif limit and emitted >= limit:
            status = "limit"
            filename = ""
        else:
            filename = f"component_{idx:03d}_events_{ne:04d}_objects_{no:04d}.jsonocel"
            (outdir / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
            emitted += 1; status = "written"
        rows.append({
            "component": idx, "events": ne, "objects": no,
            "orders": counts.get("orders",0), "items": counts.get("items",0), "packages": counts.get("packages",0),
            "status": status, "file": filename,
        })
    with (outdir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["component"])
        w.writeheader(); w.writerows(rows)
    (outdir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Components discovered: {len(comps)}; written: {emitted}")
    print(f"Manifest: {outdir / 'manifest.csv'}")


def write_stream(objects: Dict[str, Obj], events: List[Ev], profile: dict, outfile: Path) -> None:
    """Write the selected OCEL2 perspective as observable-stream JSONL.

    Attribute handling is incremental: attributes up to an object's first event
    are emitted at creation, and only that object's future changes are inserted
    into a global heap.  Complexity is O(events + attributes log attributes),
    rather than scanning every observed object at every event.
    """
    import heapq

    core_types = set(profile.get("core_object_types", [])) or {o.typ for o in objects.values()}
    act_map = profile.get("activity_map", {})
    type_map = profile.get("object_type_map", {})
    attr_map = profile.get("object_attribute_map", {})

    filtered_events = []
    for e in events:
        if act_map and e.activity not in act_map:
            continue
        rr = [(oid, q) for oid, q in e.rels if oid in objects and objects[oid].typ in core_types]
        if rr:
            filtered_events.append((e, rr))

    first_seen: Set[str] = set()
    emitted_rel_sources: Set[str] = set()
    sorted_attrs = {
        oid: sorted(obj.attrs, key=lambda x: x[1])
        for oid, obj in objects.items() if obj.typ in core_types
    }
    # (timestamp, sequence, object_id, attr_name, value)
    future_attrs = []
    heap_seq = 0

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as fh:
        pos = 0
        for e, rr in filtered_events:
            pos += 1
            involved = [oid for oid, _ in rr]
            creations = []
            updates = []

            # First emit already-scheduled changes that became observable before
            # this event.
            while future_attrs and future_attrs[0][0] <= e.time:
                tm, _seq, oid, name, val = heapq.heappop(future_attrs)
                dst = attr_map.get(objects[oid].typ, {}).get(name, name)
                updates.append({"object_id": oid, "attribute": dst, "timestamp": tm, "value": val})

            for oid in involved:
                if oid in first_seen:
                    continue
                first_seen.add(oid)
                creations.append({
                    "object_id": oid,
                    "object_type": type_map.get(objects[oid].typ, objects[oid].typ),
                })
                for name, tm, val in sorted_attrs.get(oid, []):
                    dst = attr_map.get(objects[oid].typ, {}).get(name, name)
                    if not tm or tm <= e.time:
                        updates.append({"object_id": oid, "attribute": dst,
                                        "timestamp": tm or e.time, "value": val})
                    else:
                        heap_seq += 1
                        heapq.heappush(future_attrs, (tm, heap_seq, oid, name, val))

            o2o = []
            for oid in involved:
                if oid in emitted_rel_sources:
                    continue
                emitted_rel_sources.add(oid)
                for target, q in objects[oid].rels:
                    if target in objects and objects[target].typ in core_types:
                        o2o.append({"source": oid, "qualifier": q, "target": target})

            row = {
                "position": pos,
                "event": {
                    "event_id": e.eid,
                    "activity": act_map.get(e.activity, e.activity),
                    "timestamp": e.time,
                    "attributes": e.attrs,
                    "relations": [{"qualifier": q, "object_id": oid} for oid, q in rr],
                },
                "object_creations": creations,
                "o2o_updates": o2o,
                "attribute_updates": updates,
            }
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Stream: {outfile} ({pos} event units)")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="OCEL 2.0 XML")
    ap.add_argument("--profile-json", type=Path, default=Path(__file__).with_name("order_management_profile.json"))
    ap.add_argument("--mode", choices=["components", "stream"], default="components")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-events", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=0, help="0 = no upper bound")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    profile = load_profile(args.profile_json)
    objects, events, _meta = parse_ocel2(args.input)
    print(f"Parsed {len(objects)} objects and {len(events)} events")
    if args.mode == "components":
        write_components(objects, events, profile, args.out, args.min_events, args.max_events, args.limit)
    else:
        write_stream(objects, events, profile, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
