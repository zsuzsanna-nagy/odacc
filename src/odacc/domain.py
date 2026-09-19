from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple


@dataclass(frozen=True)
class ObjectCreation:
    object_id: str
    object_type: str


@dataclass(frozen=True)
class O2ORelation:
    source: str
    qualifier: str
    target: str


@dataclass(frozen=True)
class ObjectAttributeObservation:
    object_id: str
    attribute: str
    timestamp: str
    value: Any


@dataclass(frozen=True)
class StreamEvent:
    event_id: str
    activity: str
    timestamp: str
    attributes: Mapping[str, Any]
    # qualified E2O relations. OCEL 1.0 has no qualifiers; "" is used there.
    relations: Tuple[Tuple[str, str], ...]

    @property
    def objects(self) -> FrozenSet[str]:
        return frozenset(o for _, o in self.relations)


@dataclass(frozen=True)
class ObservableUnit:
    position: int
    event: Optional[StreamEvent] = None
    object_creations: Tuple[ObjectCreation, ...] = ()
    o2o_updates: Tuple[O2ORelation, ...] = ()
    attribute_updates: Tuple[ObjectAttributeObservation, ...] = ()

    @property
    def referenced_objects(self) -> FrozenSet[str]:
        result: Set[str] = {c.object_id for c in self.object_creations}
        result.update(x.object_id for x in self.attribute_updates)
        for r in self.o2o_updates:
            result.add(r.source)
            result.add(r.target)
        if self.event:
            result.update(self.event.objects)
        return frozenset(result)


@dataclass
class ObjectCentricSchema:
    activities: Set[str] = field(default_factory=set)
    object_types: Set[str] = field(default_factory=set)
    qualifiers: Set[str] = field(default_factory=set)
    # allowed attributes per activity/object type
    attributes: Dict[str, Set[str]] = field(default_factory=dict)
    # one global value type per attribute
    attribute_types: Dict[str, str] = field(default_factory=dict)


@dataclass
class PrefixExecution:
    """Partially ordered observed execution for one connected component."""

    event_ids: Set[str] = field(default_factory=set)
    # directly-follows edges induced by individual objects
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    last_event_by_object: Dict[str, str] = field(default_factory=dict)

    def add_event(self, event: StreamEvent) -> None:
        if event.event_id in self.event_ids:
            return
        for obj in event.objects:
            prev = self.last_event_by_object.get(obj)
            if prev is not None and prev != event.event_id:
                self.edges.add((prev, event.event_id))
            self.last_event_by_object[obj] = event.event_id
        self.event_ids.add(event.event_id)


@dataclass
class ObservationFormula:
    """Structured representation of Phi_obs^C.

    The implementation keeps facts structured and translates them to the
    backend representation only when a conformance query is made. This avoids
    coupling stream processing to a particular SMT/OMT library.
    """

    object_types: Dict[str, str] = field(default_factory=dict)
    events: Dict[str, StreamEvent] = field(default_factory=dict)
    o2o_relations: Set[O2ORelation] = field(default_factory=set)
    attribute_history: Dict[Tuple[str, str], List[Tuple[str, Any]]] = field(default_factory=dict)
    first_seen_position: Dict[str, int] = field(default_factory=dict)

    def add_unit(self, unit: ObservableUnit) -> None:
        for creation in unit.object_creations:
            self.object_types[creation.object_id] = creation.object_type
            self.first_seen_position.setdefault(creation.object_id, unit.position)
        for rel in unit.o2o_updates:
            self.o2o_relations.add(rel)
        for upd in unit.attribute_updates:
            self.attribute_history.setdefault((upd.object_id, upd.attribute), []).append(
                (upd.timestamp, upd.value)
            )
        if unit.event is not None:
            self.events[unit.event.event_id] = unit.event

    def merge(self, other: "ObservationFormula") -> None:
        self.object_types.update(other.object_types)
        self.events.update(other.events)
        self.o2o_relations.update(other.o2o_relations)
        for key, vals in other.attribute_history.items():
            current = self.attribute_history.setdefault(key, [])
            for val in vals:
                if val not in current:
                    current.append(val)
        for obj, pos in other.first_seen_position.items():
            self.first_seen_position[obj] = min(self.first_seen_position.get(obj, pos), pos)

    def current_object_attributes(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for (obj, attr), values in self.attribute_history.items():
            if not values:
                continue
            # Input is assumed ordered; timestamps are retained for reporting.
            result.setdefault(obj, {})[attr] = values[-1][1]
        return result


@dataclass
class AlignmentMove:
    kind: str  # log | model | sync
    cost: int
    activity: Optional[str] = None
    event_id: Optional[str] = None
    transition: Optional[str] = None
    objects: Tuple[str, ...] = ()
    # Additional diagnostics for validating the multi-perspective alignment.
    silent: bool = False
    object_creation: bool = False
    observed_objects: Tuple[str, ...] = ()
    model_objects: Tuple[str, ...] = ()
    object_match: Optional[bool] = None
    observed_data: Dict[str, Any] = field(default_factory=dict)
    model_data: Dict[str, Any] = field(default_factory=dict)
    data_mismatches: Tuple[str, ...] = ()
    control_flow_cost: int = 0
    data_cost: int = 0
    object_cost: int = 0


@dataclass
class AlignmentResult:
    component_id: int
    prefix_position: int
    mode: str
    cost: Optional[int]
    feasible: bool
    moves: List[AlignmentMove] = field(default_factory=list)
    model_run: List[Dict[str, Any]] = field(default_factory=list)
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    cost_breakdown: Dict[str, int] = field(default_factory=dict)
    raw: str = ""
    encode_seconds: float = 0.0
    solve_seconds: float = 0.0
    joint_assignment: Dict[str, Any] = field(default_factory=dict)
