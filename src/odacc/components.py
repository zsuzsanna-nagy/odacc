from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .domain import AlignmentResult, ObservableUnit, ObservationFormula, PrefixExecution


@dataclass
class ComponentState:
    component_id: int
    objects: Set[str] = field(default_factory=set)
    execution: PrefixExecution = field(default_factory=PrefixExecution)
    observation_formula: ObservationFormula = field(default_factory=ObservationFormula)
    units: List[ObservableUnit] = field(default_factory=list)
    current_alignment: Optional[AlignmentResult] = None
    # Conservative safe checkpoint. A future A* backend can store OPEN/CLOSED
    # snapshots here; the SMT backend needs only the stream position.
    checkpoint_position: int = 0
    # Component ids that were joined to create this component. Kept so the
    # symbolic backend can derive a feasible incumbent from their last prefix
    # alignments when a merge occurs.
    merged_from: Tuple[int, ...] = ()
    # Position of the observable unit that connected previously independent
    # components.  The prefix immediately before this position is a
    # compositional checkpoint: each parent component has already been solved
    # independently up to that point.
    merge_position: int = 0
    merge_event_id: Optional[str] = None
    parent_checkpoint_positions: Tuple[Tuple[int, int], ...] = ()

    @property
    def first_position(self) -> int:
        return min((u.position for u in self.units), default=0)

    def add_unit(self, unit: ObservableUnit) -> None:
        if unit not in self.units:
            self.units.append(unit)
            self.units.sort(key=lambda u: u.position)
        self.objects.update(unit.referenced_objects)
        self.observation_formula.add_unit(unit)
        if unit.event is not None:
            self.execution.add_event(unit.event)


class ComponentManager:
    """Maintains connected components induced by E2O co-participation and O2O."""

    def __init__(self) -> None:
        self.components: Dict[int, ComponentState] = {}
        self.object_to_component: Dict[str, int] = {}
        self._next_id = 1

    def affected_component_ids(self, unit: ObservableUnit) -> Set[int]:
        return {
            self.object_to_component[o]
            for o in unit.referenced_objects
            if o in self.object_to_component
        }

    def _new_component(self) -> ComponentState:
        cid = self._next_id
        self._next_id += 1
        comp = ComponentState(cid)
        self.components[cid] = comp
        return comp

    def _merge(self, ids: Set[int], unit: ObservableUnit) -> ComponentState:
        states = [self.components[i] for i in sorted(ids)]
        merged = self._new_component()
        merged.merged_from = tuple(sorted(ids))
        merged.merge_position = unit.position
        merged.merge_event_id = unit.event.event_id if unit.event is not None else None
        merged.parent_checkpoint_positions = tuple(
            (s.component_id, max((u.position for u in s.units), default=0))
            for s in states
        )
        for state in states:
            merged.objects.update(state.objects)
            merged.observation_formula.merge(state.observation_formula)
            for old_unit in state.units:
                merged.add_unit(old_unit)
        # Compositional safe checkpoint: immediately before the observation
        # that first connected the parent components.  The parent prefixes have
        # already been monitored independently up to this point.  The symbolic
        # backend may warm-start from their retained optimal/frontier states; a
        # root state is still kept as an exact fallback, so correctness does not
        # depend on the warm start.
        merged.checkpoint_position = max(0, unit.position - 1)
        for state in states:
            del self.components[state.component_id]
        merged.add_unit(unit)
        for obj in merged.objects:
            self.object_to_component[obj] = merged.component_id
        return merged

    def apply(self, unit: ObservableUnit) -> Tuple[ComponentState, bool]:
        ids = self.affected_component_ids(unit)
        if not ids:
            comp = self._new_component()
            merged = False
        elif len(ids) == 1:
            comp = self.components[next(iter(ids))]
            merged = False
        else:
            comp = self._merge(ids, unit)
            return comp, True

        comp.add_unit(unit)
        for obj in comp.objects:
            self.object_to_component[obj] = comp.component_id
        return comp, merged
