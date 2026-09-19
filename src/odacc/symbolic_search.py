from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import re
from copy import deepcopy
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .components import ComponentState
from .domain import AlignmentMove, AlignmentResult, StreamEvent
from .cocomot_adapter import (DependencyError, load_cocomot, make_observation_encoding_classes,
                              semantic_output_guard_constraints)


@dataclass(frozen=True)
class SymbolicMove:
    """A control/object-flow move chosen by A*.

    Concrete model-only object bindings are intentionally not stored here.
    They remain symbolic and are selected by JODAP.  For synchronous moves,
    JODAP constrains the symbolic binding to the event's observed object set.
    """

    kind: str  # log | model | sync
    event_id: Optional[str] = None
    transition_id: Optional[int] = None
    transition_label: Optional[str] = None


@dataclass
class SearchNode:
    node_id: int
    consumed: FrozenSet[str]
    # A linear extension induced by the sequence of log/sync moves on this
    # particular A* path. The observed execution itself remains partially ordered.
    event_order: Tuple[str, ...]
    model_depth: int
    g: float
    h: float = 0.0
    assignment_cost: float = 0.0
    # Path signature is used only for conservative duplicate detection. It does
    # not contain object/data assignments.
    model_signature: Tuple[int, ...] = ()
    move_signature: Tuple[Tuple[str, Optional[str], Optional[int]], ...] = ()
    # Persistent certified-boundary lineage.  This is deliberately independent
    # of ``SearchState.predecessor``: canonical node reuse may keep a predecessor
    # chain from a co-optimal route and thereby hide the certified online
    # checkpoint from later eager queries.  Nodes extended from the last
    # certified prefix therefore carry the boundary witness plus the exact
    # post-boundary move suffix explicitly.
    checkpoint_boundary_assignment: Any = None
    checkpoint_boundary_snapshot: Any = None
    checkpoint_boundary_model_depth: int = 0
    checkpoint_boundary_prefix_moves: Tuple[SymbolicMove, ...] = ()
    checkpoint_suffix_moves: Tuple[SymbolicMove, ...] = ()

    @property
    def f(self) -> float:
        return self.g + self.h


def _clone_search_node(node: SearchNode, *, node_id: Optional[int] = None) -> SearchNode:
    """Create a lightweight clone of a search node.

    SearchNode contains only immutable Python values (frozenset/tuple/numbers).
    Reconstructing it explicitly avoids ``copy.deepcopy`` traversing unrelated
    retained solver/search objects through future extensions or wrappers.
    """
    return SearchNode(
        node_id=node.node_id if node_id is None else int(node_id),
        consumed=node.consumed,
        event_order=node.event_order,
        model_depth=int(node.model_depth),
        g=float(node.g),
        h=float(node.h),
        assignment_cost=float(node.assignment_cost),
        model_signature=node.model_signature,
        move_signature=node.move_signature,
        checkpoint_boundary_assignment=node.checkpoint_boundary_assignment,
        checkpoint_boundary_snapshot=node.checkpoint_boundary_snapshot,
        checkpoint_boundary_model_depth=int(node.checkpoint_boundary_model_depth),
        checkpoint_boundary_prefix_moves=node.checkpoint_boundary_prefix_moves,
        checkpoint_suffix_moves=node.checkpoint_suffix_moves,
    )


@dataclass
class JointAssignment:
    total_cost: float
    object_bindings: List[Dict[str, Any]] = field(default_factory=list)
    data_assignments: List[Dict[str, Any]] = field(default_factory=list)
    # Soft model-side object-attribute assignments.  CoCoMoT normally replaces
    # object properties in guards by their observed constants; JODAP keeps them
    # symbolic and penalizes disagreement with the observation instead.
    object_attribute_assignments: List[Dict[str, Any]] = field(default_factory=list)
    # Concrete operational state selected by the optimum.  These signatures
    # support conservative state-dominance checks between alternative paths.
    marking_signature: Tuple[Any, ...] = ()
    data_state_signature: Tuple[Any, ...] = ()
    # Data carried by the current object-aware marking. Each entry is
    # (place_id, object-token, ((data_name, value), ...)). This is retained
    # as part of the certified witness so later transitions can read values
    # written many moves earlier without reconstructing the complete prefix.
    token_data_signature: Tuple[Any, ...] = ()
    # Provenance of currently known process/data values. Entries are
    # (name, value, source_model_step, source_kind). The provenance is
    # diagnostic as well as operational: a local extension may safely reuse a
    # historical value if its source is still represented in the witness.
    data_provenance_signature: Tuple[Any, ...] = ()
    solve_seconds: float = 0.0
    encode_seconds: float = 0.0




@dataclass
class PreparedJODAP:
    """Reusable CoCoMoT encoding for one observed linearization.

    The expensive model/marking/data constraints are asserted once. Candidate
    paths sharing the same event order and object domain only add their
    transition/binding constraints and objective temporarily via push/pop.
    """

    key: Tuple[Any, ...]
    solver: Any
    encoding: Any
    net: Any
    trans_by_id: Dict[int, Dict[str, Any]]
    data_types: Dict[str, str]
    base_encode_seconds: float = 0.0
    final_formula: Any = None
    # Maximum model depth represented by this persistent component context.
    capacity: int = 0
    # Object-domain version. Event prefixes and observed attribute values do not
    # affect the structural base encoding and therefore do not force rebuilds.
    domain_key: Tuple[Any, ...] = ()
    # Metadata for an exact context whose semantic instant 0 is a certified
    # online checkpoint rather than the DOPID initial marking.
    checkpoint_boundary: Optional[JointAssignment] = None
    checkpoint_model_depth: int = 0
    checkpoint_object_domain: FrozenSet[str] = frozenset()
    checkpoint_static_marking: Tuple[Any, ...] = ()
    checkpoint_static_token_data: Tuple[Any, ...] = ()

@dataclass
class SearchState:
    component_id: int
    nodes: Dict[int, SearchNode] = field(default_factory=dict)
    predecessor: Dict[int, Tuple[int, SymbolicMove]] = field(default_factory=dict)
    open_heap: List[Tuple[float, int, int]] = field(default_factory=list)
    open_ids: Set[int] = field(default_factory=set)
    closed_ids: Set[int] = field(default_factory=set)
    signatures: Dict[Tuple[Any, ...], int] = field(default_factory=dict)
    current_goal: Optional[int] = None
    current_assignment: Optional[JointAssignment] = None
    current_event_ids: FrozenSet[str] = frozenset()
    current_objects: FrozenSet[str] = frozenset()
    current_attribute_observations: int = 0
    current_object_attribute_snapshot: Tuple[Any, ...] = ()
    model_bound: int = 0
    # Best complete prefix explanation known so far.  It is used as an upper
    # bound; nodes whose admissible f-value cannot improve it are not expanded.
    upper_bound: float = float("inf")
    incumbent_moves: Tuple[SymbolicMove, ...] = ()
    incumbent_assignment: Optional[JointAssignment] = None
    incumbent_offline: bool = False
    # Explicit handoff for a complete feasible witness discovered by a local
    # repair while ``_sync_increment`` is extending the observed prefix.  This
    # must not be reconstructed later from ``incumbent_moves`` because local
    # materialization may hit an already-existing canonical search node whose
    # predecessor chain represents a different (co-optimal) path.
    pending_increment_incumbent_moves: Tuple[SymbolicMove, ...] = ()
    pending_increment_incumbent_assignment: Optional[JointAssignment] = None
    pending_increment_incumbent_cost: float = float("inf")
    pending_increment_incumbent_event_ids: FrozenSet[str] = frozenset()
    search_offline: bool = False
    root_expanded: bool = False
    # Proven lower bound for the complete current prefix. Normally zero. Merge
    # processing can raise this when the connecting observation is structurally
    # incapable of synchronization and therefore has an unavoidable log cost.
    proven_prefix_lower_bound: float = 0.0
    # Exact operational-state dominance table.  The key contains the consumed
    # log state plus a canonical object-aware model/data state.  The value also
    # stores model depth so a deeper equal-cost path cannot dominate a shallower
    # one when a finite exploration bound is used.
    state_dominance: Dict[Tuple[Any, ...], Tuple[float, int, int]] = field(default_factory=dict)
    # JODAP optimum retained for each evaluated search node.  Besides avoiding
    # repeated decoding this provides the concrete marking used by focused
    # model-successor queries.
    assignments_by_node: Dict[int, JointAssignment] = field(default_factory=dict)
    # Optional certified boundary used by provenance-focused repair.  When set,
    # the temporary focused search treats this node as its semantic initial
    # state: JODAP evaluates only moves after the boundary and carries the
    # concrete marking/data witness forward as fixed facts.
    boundary_assignment: Optional[JointAssignment] = None
    boundary_prefix_moves: Tuple[SymbolicMove, ...] = ()
    boundary_model_depth: int = 0
    boundary_node_original: Optional[int] = None
    # Co-optimal complete-prefix witnesses retained specifically for future
    # continuation/merge composition.  A shortest reported alignment (for
    # example a one-unit log move for an orphan event) may be optimal yet leave
    # no useful model-side state.  We therefore keep equally cheap concrete
    # witnesses that reach a different operational state without changing the
    # reported optimum.
    cooptimal_continuation_boundaries: List[Tuple[float, Tuple[SymbolicMove, ...], JointAssignment]] = field(default_factory=list)
    next_id: int = 0
    tie: int = 0

    def new_id(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid

    def push(self, node: SearchNode) -> None:
        self.nodes[node.node_id] = node
        self.open_ids.add(node.node_id)
        self.closed_ids.discard(node.node_id)
        self.tie += 1
        heapq.heappush(self.open_heap, (node.f, self.tie, node.node_id))

    def pop(self) -> Optional[SearchNode]:
        while self.open_heap:
            f, _, nid = heapq.heappop(self.open_heap)
            if nid not in self.open_ids:
                continue
            node = self.nodes[nid]
            # stale heap entry after a cost update
            if abs(node.f - f) > 1e-9:
                continue
            self.open_ids.remove(nid)
            return node
        return None


class JODAPSolver:
    @staticmethod
    def _objects_in_value(value: Any, known: Set[str]) -> Set[str]:
        """Return object ids from a concrete token/binding value.

        JODAP's zero-cost diagnostics run before SymbolicIncrementalBackend
        gets control, so they cannot use the backend helper with the same
        purpose. Keep this small extractor local to JODAP as well.
        """
        out: Set[str] = set()
        if isinstance(value, str):
            if value in known:
                out.add(value)
            return out
        if isinstance(value, dict):
            for k, v in value.items():
                out.update(JODAPSolver._objects_in_value(k, known))
                out.update(JODAPSolver._objects_in_value(v, known))
            return out
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                out.update(JODAPSolver._objects_in_value(item, known))
            return out
        return out

    """Incremental Joint Object and Data Assignment Problem.

    CoCoMoT's object-centric encoding is expensive to construct.  The first
    prototype rebuilt it for every A* successor and, later, for almost every
    newly observed prefix because event order and the current search bound were
    part of the cache key.  The current implementation keeps one persistent
    structural Z3 context per component/object-domain version.  Event values,
    synchronization choices, observed/model mismatches and path constraints are
    query-local, so they do not force a base rebuild.

    Each context is created with spare model-depth capacity and is reused while
    the component grows.  If the capacity is exceeded it is rebuilt only with a
    geometrically larger bound.  A newly observed object (or a newly introduced
    object-attribute name used by the structural guard encoding) creates a new
    object-domain version.  The static DOPID semantics are therefore asserted
    once per domain version, while candidate queries use push/pop.
    """

    @staticmethod
    def _base_var(name: str) -> str:
        """Return the unversioned/base name of a data variable.

        CoCoMoT denotes post-state/versioned variables by trailing apostrophes
        (for example ``d'``).  Provenance slicing compares dependencies across
        token data, guards and retained boundary assignments, so all of those
        references must use the same base name.
        """
        return str(name).rstrip("'")

    def __init__(self, cocomot_root: str, model_path: str, fixed_objects: bool = False,
                 max_prepared_contexts: int = 24, provenance_slicing: str = "off",
                 query_diagnostics_path: Optional[str] = None, query_diagnostics_print: bool = False):

        self.mods = load_cocomot(cocomot_root)
        self.model_path = model_path
        self.fixed_objects = fixed_objects
        self.max_prepared_contexts = max(2, int(max_prepared_contexts))
        self.provenance_slicing = str(provenance_slicing or "off").lower()
        self.query_diagnostics_path = os.path.abspath(query_diagnostics_path) if query_diagnostics_path else None
        self.query_diagnostics_print = bool(query_diagnostics_print or query_diagnostics_path)
        self._diag_seq = 0
        if self.query_diagnostics_path:
            os.makedirs(os.path.dirname(self.query_diagnostics_path) or ".", exist_ok=True)
            try:
                with open(self.query_diagnostics_path, "w", encoding="utf-8") as _fh:
                    _fh.write("")
            except OSError:
                self.query_diagnostics_path = None
        self._static_net = None
        self._prepared: "OrderedDict[Tuple[Any, ...], PreparedJODAP]" = OrderedDict()
        # One grow-only structural encoding per component/object domain.  Unlike
        # _prepared in the earlier prototype, this context is independent of the
        # current event linearization and observed attribute values.
        self._component_contexts: Dict[Tuple[Any, ...], PreparedJODAP] = {}
        self._query_cache: "OrderedDict[Tuple[Any, ...], Optional[JointAssignment]]" = OrderedDict()
        self.max_query_cache = max(256, self.max_prepared_contexts * 256)
        self._install_soft_object_property_semantics()
        # Exact/eager JODAP must use the same selected-transition-local token
        # data semantics as the monolithic SMT backend.  The optimized lazy
        # path already reasons only about the chosen transition; using the raw
        # CoCoMoT Encoding here allowed constraints from unselected transitions
        # to leak into fallback/offline queries.
        self.ExactEncoding, _ = make_observation_encoding_classes(self.mods.Encoding)
        self.stats: Dict[str, int] = {
            "prepared_builds": 0,
            "prepared_hits": 0,
            "queries": 0,
            "filtered_transitions": 0,
            "por_pruned": 0,
            "signature_pruned": 0,
            "state_dominated": 0,
            "upper_bound_pruned": 0,
            "pre_jodap_upper_bound_pruned": 0,
            "pre_jodap_log_cost_pruned": 0,
            "pre_jodap_model_cost_pruned": 0,
            "strict_improvement_mode_entries": 0,
            "strict_improvement_pre_solve_pruned": 0,
            "strict_improvement_lazy_unsat_pruned": 0,
            "strict_improvement_eager_queries": 0,
            "strict_improvement_eager_unsat": 0,
            "strict_improvement_lower_bound_closed": 0,
            "strict_improvement_generation_pruned": 0,
            "strict_improvement_generation_log_pruned": 0,
            "strict_improvement_generation_model_pruned": 0,
            "strict_improvement_generation_node_closed": 0,
            "guard_directed_list_attempts": 0,
            "guard_directed_list_supported": 0,
            "guard_directed_list_candidates_pruned": 0,
            "guard_directed_list_candidates_kept": 0,
            "guard_directed_list_fallbacks": 0,
            "guard_directed_binding_generation_attempts": 0,
            "guard_directed_binding_generation_supported": 0,
            "guard_directed_binding_generation_candidates": 0,
            "guard_directed_binding_generation_subsets_avoided": 0,
            "guard_directed_binding_generation_fallbacks": 0,
            "marking_domain_reduction_attempts": 0,
            "marking_domain_reduction_supported": 0,
            "marking_domain_objects_pruned": 0,
            "marking_domain_candidates_pruned": 0,
            "marking_domain_reduction_fallbacks": 0,
            # Certified-checkpoint-specialized eager suffix contexts.
            "checkpoint_context_attempts": 0,
            "checkpoint_context_builds": 0,
            "checkpoint_context_hits": 0,
            "checkpoint_context_fallbacks": 0,
            "checkpoint_context_objects_full": 0,
            "checkpoint_context_objects_kept": 0,
            "checkpoint_context_objects_pruned": 0,
            "checkpoint_context_subset_space_full": 0,
            "checkpoint_context_subset_space_kept": 0,
            "checkpoint_context_build_seconds_ms": 0,
            "checkpoint_context_moving_seconds_ms": 0,
            "persistent_checkpoint_lineage_seeded": 0,
            "persistent_checkpoint_lineage_propagated": 0,
            "persistent_checkpoint_lineage_reused": 0,
            "persistent_checkpoint_lineage_assignment_recovered": 0,
            "persistent_checkpoint_lineage_missing_on_parent": 0,
            "persistent_checkpoint_lineage_missing_on_candidate": 0,
            "persistent_checkpoint_lineage_dropped_on_canonical_reuse": 0,
            "persistent_checkpoint_lineage_attached_on_reuse": 0,
            "persistent_checkpoint_context_attempts": 0,
            "persistent_checkpoint_context_proven": 0,
            "persistent_checkpoint_context_upper_bounds": 0,
            "persistent_checkpoint_context_fallbacks": 0,
            "upper_bound_seeded": 0,
            "offline_seeded": 0,
            "object_attribute_soft_terms": 0,
            "query_cache_hits": 0,
            "lower_bound_hits": 0,
            "merge_warm_starts": 0,
            "merge_checkpoint_reused": 0,
            "zero_cost_sync_attempts": 0,
            "zero_cost_sync_hits": 0,
            "symmetry_canonicalizations": 0,
            "focused_model_queries": 0,
            "focused_model_hits": 0,
            "focused_model_fallbacks": 0,
            "zero_cost_delta_checks": 0,
            "zero_cost_delta_hits": 0,
            "canonical_creation_restrictions": 0,
            "silent_macro_attempts": 0,
            "silent_macro_hits": 0,
            "silent_macro_moves": 0,
            "merge_frontier_compositions": 0,
            "merge_frontier_hits": 0,
            "merge_direct_compositions": 0,
            "merge_direct_hits": 0,
            "merge_structural_log_proofs": 0,
            "merge_lower_bound_terminations": 0,
            "merge_frontier_heap_pops": 0,
            "merge_frontier_incompatible": 0,
            "merge_incompatible_object_binding": 0,
            "merge_incompatible_marking": 0,
            "merge_incompatible_process_data": 0,
            "merge_incompatible_token_data": 0,
            "merge_incompatible_partial_order": 0,
            "merge_incompatible_variable_version": 0,
            "merge_namespaced_data_conflicts": 0,
            "merge_bridge_checks": 0,
            "merge_bridge_hits": 0,
            "merge_bridge_log_hits": 0,
            "merge_one_parent_repairs": 0,
            "merge_one_parent_repair_hits": 0,
            "merge_full_search_fallbacks": 0,
            "component_context_builds": 0,
            "component_context_reuses": 0,
            "component_context_extensions": 0,
            "context_adaptive_builds": 0,
            "context_adaptive_extensions": 0,
            "context_requested_depth_max": 0,
            "context_geometric_growths": 0,
            "checkpoint_slice_attempts": 0,
            "checkpoint_slice_hits": 0,
            "checkpoint_slice_proven": 0,
            "checkpoint_slice_upper_bounds": 0,
            "checkpoint_slice_fallbacks": 0,
            "checkpoint_slice_model_steps": 0,
            "checkpoint_bound_binding_attempts": 0,
            "checkpoint_bound_binding_candidates": 0,
            "checkpoint_bound_binding_hits": 0,
            "checkpoint_bound_binding_proven": 0,
            "candidate_structural_lb_raised": 0,
            "merge_bridge_fresh_creations": 0,
            "merge_bridge_guarded_zero_cost_proofs": 0,
            "merge_fast_attempts": 0,
            "merge_fast_no_compositions": 0,
            "merge_fast_structural_sync": 0,
            "merge_fast_bridge_prepare_failures": 0,
            "merge_fast_materialize_failures": 0,
            "merge_fast_sync_attempts": 0,
            "merge_fast_sync_failures": 0,
            "merge_fast_successes": 0,
            "merge_cooptimal_parent_rewrite_attempts": 0,
            "merge_cooptimal_parent_rewrite_candidates": 0,
            "merge_cooptimal_parent_rewrite_hits": 0,
            "merge_cooptimal_parent_rewrite_failures": 0,
            "exact_zero_cost_goal_hits": 0,
            "exact_zero_cost_expansion_stops": 0,
            "zero_cost_sync_declines": 0,
            "zero_cost_sync_binding_failures": 0,
            "zero_cost_sync_marking_failures": 0,
            "zero_cost_sync_token_data_failures": 0,
            "zero_cost_sync_data_mismatches": 0,
            "zero_cost_sync_guard_unknown": 0,
            "zero_cost_sync_guard_false": 0,
            "zero_cost_sync_output_failures": 0,
            "merge_bridge_creation_declines": 0,
            "fresh_object_sync_attempts": 0,
            "fresh_object_sync_preparations": 0,
            "fresh_object_sync_created_objects": 0,
            "fresh_object_sync_successes": 0,
            "fresh_object_sync_failures": 0,
            "fresh_object_sync_prepare_failures": 0,
            "one_step_repair_attempts": 0,
            "one_step_repair_candidates": 0,
            "one_step_repair_hits": 0,
            "one_step_repair_proven": 0,
            "one_step_repair_incumbents": 0,
            "one_step_repair_fallbacks": 0,
            "one_step_repair_lb_proofs": 0,
            "extra_event_fast_attempts": 0,
            "extra_event_fast_recognized": 0,
            "extra_event_fast_lb_proofs": 0,
            "extra_event_fast_hits": 0,
            "extra_event_fast_proven": 0,
            "extra_event_fast_incumbents": 0,
            "extra_event_fast_fallbacks": 0,
            "guard_data_repair_attempts": 0,
            "guard_data_repair_candidates": 0,
            "guard_data_repair_hits": 0,
            "guard_data_repair_proven": 0,
            "guard_data_repair_fallbacks": 0,
            "guard_data_repair_lb_proofs": 0,
            "object_relation_repair_attempts": 0,
            "object_relation_repair_candidates": 0,
            "object_relation_repair_hits": 0,
            "object_relation_repair_proven": 0,
            "object_relation_repair_incumbents": 0,
            "object_relation_repair_fallbacks": 0,
            "object_relation_repair_lb_proofs": 0,
            "local_repair_search_attempts": 0,
            "local_repair_search_candidates": 0,
            "local_repair_search_solver_calls": 0,
            "local_repair_search_hits": 0,
            "local_repair_search_proven": 0,
            "local_repair_search_incumbents": 0,
            "local_repair_incumbents_preserved": 0,
            "local_repair_incumbents_restored": 0,
            "local_repair_search_fallbacks": 0,
            "local_repair_search_pruned_budget": 0,
            "local_repair_search_max_depth": 0,
            "local_repair_event_binding_probes": 0,
            "local_repair_missing_token_producers": 0,
            "local_repair_missing_token_targeted_enqueues": 0,
            "local_repair_relation_candidates_pruned": 0,
            "local_repair_invariant_failures_pruned": 0,
            "object_relation_extra_hits": 0,
            "object_relation_missing_hits": 0,
            "object_relation_carry_attempts": 0,
            "object_relation_carry_hits": 0,
            "object_relation_carry_misses": 0,
            "object_relation_carry_free_extensions": 0,
            "object_relation_provenance_candidates": 0,
            "object_relation_cross_component_hits": 0,
            "object_relation_virtual_dependencies": 0,
            "object_relation_virtual_dependency_reuses": 0,
            "object_relation_latent_completion_attempts": 0,
            "object_relation_latent_completion_hits": 0,
            "first_event_latent_relation_attempts": 0,
            "first_event_latent_relation_hits": 0,
            "cooptimal_parent_boundary_reuses": 0,
            "cooptimal_continuation_boundaries": 0,
            "cooptimal_continuation_reuses": 0,
            "first_event_latent_after_macro_attempts": 0,
            "first_event_latent_after_macro_hits": 0,
            "parent_boundary_fallbacks": 0,
            "object_relation_multirelation_candidates": 0,
            "object_relation_multirelation_hits": 0,
            "object_relation_max_repair_cardinality": 0,
            "context_absolute_depth_escalations": 0,
            "context_checkpoint_relative_depth_max": 0,
            "context_capacity": 0,
            "context_build_seconds_ms": 0,
            "context_variable_seconds_ms": 0,
            "context_initial_seconds_ms": 0,
            "context_transition_seconds_ms": 0,
            "context_object_type_seconds_ms": 0,
            "context_freshness_seconds_ms": 0,
            "context_moving_seconds_ms": 0,
            "context_remaining_seconds_ms": 0,
            "context_data_seconds_ms": 0,
            "context_cache_seconds_ms": 0,
            "context_require_seconds_ms": 0,
            "lazy_path_attempts": 0,
            "lazy_path_hits": 0,
            "lazy_path_fallbacks": 0,
            "lazy_path_build_seconds_ms": 0,
            "lazy_path_solve_seconds_ms": 0,
            "eager_context_avoided": 0,
            "eager_fallback_queries": 0,
            "lazy_certification_failures": 0,
            "lazy_nonzero_fallbacks": 0,
            "delta_local_guard_checks": 0,
            "delta_local_guard_hits": 0,
            "delta_dependency_fallbacks": 0,
            "delta_historical_reads": 0,
            "delta_token_data_hits": 0,
            "delta_provenance_hits": 0,
            "delta_token_lookup_attempts": 0,
            "delta_token_lookup_misses": 0,
            "delta_global_lookup_attempts": 0,
            "delta_global_lookup_misses": 0,
            "exact_extension_attempts": 0,
            "exact_extension_hits": 0,
            "exact_extension_fallbacks": 0,
            "nonzero_pruned_before_fallback": 0,
            "positive_lower_bound_proofs": 0,
            "positive_lower_bound_terminations": 0,
            "positive_fixed_path_incumbents": 0,
            "zero_deviation_guard_checks": 0,
            "zero_deviation_guard_false": 0,
            "zero_deviation_guard_true": 0,
            "zero_deviation_guard_unknown": 0,
            "zero_cost_early_terminations": 0,
            "analytic_all_log_seeds": 0,
            "provenance_slice_attempts": 0,
            "provenance_slice_hits": 0,
            "provenance_slice_proven": 0,
            "provenance_slice_incumbent_improvements": 0,
            "provenance_slice_fallbacks": 0,
            "provenance_slice_guard_vars": 0,
            "provenance_slice_transitions_total": 0,
            "provenance_slice_transitions_kept": 0,
            "provenance_slice_model_steps_reopened": 0,
            "provenance_slice_nodes_expanded": 0,
            "provenance_slice_seconds_ms": 0,
            "provenance_boundary_builds": 0,
            "provenance_boundary_marking_facts": 0,
            "provenance_boundary_token_data_facts": 0,
            "provenance_boundary_data_facts": 0,
            "provenance_boundary_object_attr_facts": 0,
            "provenance_sliced_jodap_queries": 0,
            "provenance_sliced_jodap_hits": 0,
            "provenance_sliced_jodap_fallbacks": 0,
            "provenance_sliced_jodap_build_seconds_ms": 0,
            "provenance_sliced_jodap_solve_seconds_ms": 0,
            "provenance_query_slice_attempts": 0,
            "provenance_query_slice_hits": 0,
            "provenance_query_slice_proven": 0,
            "provenance_query_slice_upper_bounds": 0,
            "provenance_query_slice_fallbacks": 0,
            # Exact symbolic LIST-aggregate guard encoding.  These counters are
            # copied from the CoCoMoT Encoding after persistent-context build.
            "symbolic_list_guard_fast_paths": 0,
            "symbolic_list_guard_fallbacks": 0,
            "symbolic_list_guard_subsets_avoided": 0,
        }
        # Set by the local delta checker when it declines only because a
        # historical dependency could not be certified from the retained
        # witness.  The backend then tries one fixed-path exact extension before
        # reopening the full A* search.
        self._last_delta_unknown = False
        self._last_zero_cost_sync_decline_reason = None
        self._last_zero_cost_sync_decline_detail = None
        # Set when the lazy fixed-path encoder has already produced a lower
        # bound strictly above a feasible incumbent for the *current* prefix.
        # In that case an eager fallback cannot improve the incumbent and must
        # not be constructed merely to refine an already dominated candidate.
        self._last_lazy_pruned = False

    def _diag(self, phase: str, **fields) -> None:
        """Emit one structured JODAP diagnostic record.

        Diagnostics are deliberately side-effect free with respect to solving. They
        are written as JSONL (when a path is configured) and echoed to stdout so
        batch_folder can show the currently active symbolic phase in real time.
        """
        if not (self.query_diagnostics_print or self.query_diagnostics_path):
            return
        self._diag_seq += 1
        rec = {
            "seq": self._diag_seq,
            "time": time.time(),
            "phase": phase,
        }
        rec.update(fields)
        if self.query_diagnostics_path:
            try:
                with open(self.query_diagnostics_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                    fh.flush()
            except OSError:
                pass
        if self.query_diagnostics_print:
            compact = " ".join(f"{k}={v}" for k, v in rec.items() if k not in {"time"})
            print("JODAP_DIAG\t" + compact, flush=True)

    def _binding_complexity(self, net, component: ComponentState, transition=None, objects=None):
        """Return logical scalar/LIST binding complexity before CoCoMoT expansion.

        ``object_params_of_transition`` expands one ``ITEM LIST`` inscription into
        one optional scalar slot per concrete ITEM.  Inspecting those expanded
        slots made diagnostics misleading.  Read the raw arc inscriptions here so
        one logical LIST variable is reported once.
        """
        object_types = component.observation_formula.object_types
        by_type: Dict[str, int] = {}
        for obj in component.objects:
            typ = str(object_types.get(obj, "?"))
            by_type[typ] = by_type.get(typ, 0) + 1
        out = {"component_objects_by_type": by_type}
        if transition is None:
            return out

        logical: Dict[str, str] = {}
        tid = transition.get("id")
        for arc in getattr(net, "_arcs", ()):
            if arc.get("source") != tid and arc.get("target") != tid:
                continue
            for name, typ in arc.get("inscription", ()):
                if typ in getattr(net, "_data_types", ()):
                    continue
                name, typ = str(name), str(typ)
                if name not in logical or "LIST" in typ:
                    logical[name] = typ

        scalar, lists = [], []
        for name, typ in sorted(logical.items()):
            item = {"name": name, "type": typ}
            if "LIST" in typ:
                base = typ[:typ.rfind(" LIST")]
                n = by_type.get(base, 0)
                item["candidate_objects"] = n
                item["generic_subset_count"] = (1 << n) if n <= 62 else f"2^{n}"
                lists.append(item)
            else:
                item["candidate_objects"] = by_type.get(typ, 0)
                scalar.append(item)
        out.update({"scalar_params": scalar, "list_params": lists})
        return out

    @staticmethod
    def _object_attr_var_name(obj: str, attr: str) -> str:
        # Z3 accepts rich symbol names, but an ASCII-only encoding keeps dumps
        # readable and avoids accidental separator collisions.
        import base64
        enc = lambda x: base64.urlsafe_b64encode(str(x).encode("utf-8")).decode("ascii").rstrip("=")
        return f"__odacc_oa__{enc(obj)}__{enc(attr)}"

    def _install_soft_object_property_semantics(self) -> None:
        """Keep observed object properties symbolic inside CoCoMoT guards.

        The original CoCoMoT visitor replaces e.g. ``vip(o)`` by the concrete
        value stored in the OCEL object.  That makes a conflicting observation
        a hard guard violation.  For JODAP we instead replace an observed
        property by a model-side SMT variable.  The corresponding observed
        value is added later as a *soft* equality in the optimization objective.

        This patch is local to the symbolic-backend process.  The standalone
        SMT baseline, executed in a different CLI process by batch_folder.py,
        retains the original CoCoMoT semantics.
        """
        try:
            import objectcentric.encoding as oc_encoding
            from dpn.expr_utils import ObjectPropertyReplacer as BaseReplacer
            from dpn.expr import Var
        except Exception:
            return

        name_fun = self._object_attr_var_name

        class SoftObjectPropertyReplacer(BaseReplacer):
            def replace_arg(self, arg):
                unary_fun = lambda t: hasattr(t, "_args") and hasattr(t, "_name") and len(t._args) == 1
                if unary_fun(arg) and isinstance(arg._args[0], Var):
                    obj = arg._args[0].name
                    # List-valued variables are expanded by CoCoMoT before this
                    # visitor, so an observed property should have a concrete id.
                    if isinstance(obj, str) and obj in self._objects:
                        ovmap = self._objects[obj].get("ovmap", {})
                        if arg._name in ovmap:
                            return Var(name_fun(obj, arg._name), None)
                return arg

        # encoding.py imported the class symbol directly, therefore patch that
        # module binding (patching only dpn.expr_utils would not be sufficient).
        oc_encoding.ObjectPropertyReplacer = SoftObjectPropertyReplacer

    @staticmethod
    def _guard_object_attribute_names(constraint, available: Set[str]) -> Set[str]:
        """Collect object-property function names occurring in one guard.

        CoCoMoT represents an object property such as ``budget(o)`` as a
        function expression. Aggregate functions (e.g. ``sum(cost(P))``) may
        nest such expressions, so we traverse the expression tree recursively.
        Only names that are actually observed as object attributes are returned.
        """
        if constraint is None or not available:
            return set()
        found: Set[str] = set()
        seen: Set[int] = set()
        stack = [constraint]
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            name = getattr(node, "_name", None)
            if isinstance(name, str) and name in available:
                found.add(name)

            # Explicitly walk every expression edge used by CoCoMoT.  The old
            # implementation relied primarily on ``__dict__``.  That happened
            # to work for aggregate expressions, but was brittle for simple
            # scalar object properties such as ``vip(o)`` and ``priority(o)``
            # under different parser/class versions.
            for field in ("left", "right", "expr", "_left", "_right", "_expr"):
                child = getattr(node, field, None)
                if child is not None:
                    stack.append(child)
            for field in ("_args", "args"):
                children = getattr(node, field, None)
                if isinstance(children, (list, tuple)):
                    stack.extend(children)

            # Keep a generic fallback for any future expression-node fields.
            for value in getattr(node, "__dict__", {}).values():
                if isinstance(value, (list, tuple)):
                    stack.extend(x for x in value if x is not None)
                elif value is not None and hasattr(value, "__class__"):
                    if value.__class__.__module__.startswith("dpn."):
                        stack.append(value)
        return found

    def _observed_object_attribute_terms(self, component: ComponentState, solver,
                                         relevant_attributes: Set[str],
                                         allowed_objects: Optional[Set[str]] = None):
        """Return path-local soft equalities for observed object attributes.

        An observed object attribute is penalized only if an object-property
        expression with the same attribute name occurs in a transition guard on
        the *selected candidate path*. This prevents guards of unchosen model
        branches from creating artificial data deviations.
        """
        try:
            from dpn.expr import Expr
        except Exception:
            Expr = None
        terms = []
        metadata = []
        attrs = component.observation_formula.current_object_attributes()
        for obj in sorted(attrs):
            if allowed_objects is not None and str(obj) not in allowed_objects:
                continue
            for attr, observed in sorted(attrs[obj].items()):
                if attr not in relevant_attributes:
                    continue
                name = self._object_attr_var_name(obj, attr)
                var = solver.realvar(name)
                if isinstance(observed, bool):
                    encoded = int(observed)
                elif isinstance(observed, (int, float)):
                    encoded = observed
                elif isinstance(observed, str) and Expr is not None:
                    encoded = Expr.numval(observed)
                else:
                    continue
                valexpr = solver.real(encoded)
                terms.append(solver.ite(solver.eq(var, valexpr), solver.num(0), solver.num(1)))
                metadata.append((obj, attr, observed, encoded, var))
        return terms, metadata

    def new_net(self):
        return self.mods.OPI(self.mods.read_pnml_input(self.model_path))

    def close(self) -> None:
        for prepared in self._prepared.values():
            try:
                prepared.solver.destroy()
            except Exception:
                pass
        self._prepared.clear()
        self._component_contexts.clear()
        self._query_cache.clear()

    def invalidate_component(self, component_id: int, *, object_domain_changed: bool = True) -> None:
        """Invalidate persistent contexts only when the encoded object domain changes.

        Event arrivals and object-attribute observations are query-local in JODAP:
        they alter synchronization/objective constraints but not the structural
        DOPID encoding.  A newly observed object is different because CoCoMoT's
        token universe is object-instantiated; that domain version must be rebuilt.
        """
        if object_domain_changed:
            for key in list(self._component_contexts.keys()):
                if key and key[0] == component_id:
                    prepared = self._component_contexts.pop(key)
                    self._prepared.pop(prepared.key, None)
                    try:
                        prepared.solver.destroy()
                    except Exception:
                        pass
        # Query results contain observations/path choices and are always cheap to
        # invalidate component-locally.
        for key in list(self._query_cache.keys()):
            if key and key[0] == component_id:
                self._query_cache.pop(key, None)

    @staticmethod
    def _coerce(value: Any) -> Optional[Any]:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        return None

    def _trace(self, component: ComponentState, event_order: Sequence[str], net,
               object_domain: Optional[Set[str]] = None):
        declared = dict(net.get_data_variables())
        events = []
        for seq, eid in enumerate(event_order):
            e = component.observation_formula.events[eid]
            vals = {}
            for name, value in e.attributes.items():
                if name in declared:
                    cv = self._coerce(value)
                    if cv is not None:
                        vals[name] = cv
            unit = next((u for u in component.units if u.event and u.event.event_id == eid), None)
            if unit is not None:
                for upd in unit.attribute_updates:
                    if upd.attribute in declared and upd.attribute not in vals:
                        cv = self._coerce(upd.value)
                        if cv is not None:
                            vals[upd.attribute] = cv
            events.append(self.mods.Event(seq, e.activity, e.timestamp, list(e.objects), vals))

        objattrs = component.observation_formula.current_object_attributes()
        objs = {}
        domain = set(component.objects) if object_domain is None else set(map(str, object_domain))
        for obj in sorted(domain):
            if obj not in component.observation_formula.object_types:
                continue
            objs[obj] = {
                "type": component.observation_formula.object_types[obj],
                "ovmap": dict(objattrs.get(obj, {})),
            }
        trace = self.mods.Trace(events, ["timestamp"], objs.keys())
        trace.add_object_types(objs)
        return trace, events

    @staticmethod
    def _sum(solver, terms):
        acc = solver.num(0)
        for t in terms:
            acc = solver.plus(acc, t)
        return acc

    def _path(self, state: SearchState, node_id: int) -> List[SymbolicMove]:
        moves: List[SymbolicMove] = []
        cur = node_id
        while cur in state.predecessor:
            prev, move = state.predecessor[cur]
            moves.append(move)
            cur = prev
        moves.reverse()
        return moves

    @staticmethod
    def _freeze_value(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((k, JODAPSolver._freeze_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(JODAPSolver._freeze_value(v) for v in value)
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    def _object_domain_key(self, component: ComponentState,
                           object_domain: Optional[Set[str]] = None,
                           boundary: Optional[JointAssignment] = None) -> Tuple[Any, ...]:
        # Only identifiers/types shape CoCoMoT's structural token universe.
        # Checkpoint contexts additionally key on the exact certified boundary
        # marking/token-data witness because instant 0 is no longer the model
        # initial marking.
        attrs = component.observation_formula.current_object_attributes()
        domain = set(component.objects) if object_domain is None else set(map(str, object_domain))
        key = (
            component.component_id,
            bool(self.fixed_objects),
            tuple((o, component.observation_formula.object_types.get(o),
                   tuple(sorted(attrs.get(o, {}).keys())))
                  for o in sorted(domain)),
        )
        if boundary is not None:
            key += (
                "checkpoint",
                self._freeze_value(boundary.marking_signature),
                self._freeze_value(boundary.token_data_signature),
                round(float(boundary.total_cost), 9),
            )
        return key

    def _checkpoint_context_domain(self, component: ComponentState, state: SearchState,
                                   node: SearchNode, net,
                                   canonical_fresh_bindings: Optional[Dict[int, str]] = None) \
            -> Tuple[Set[str], Tuple[str, ...], bool, Dict[str, Any]]:
        """Return a conservative object slice for an exact checkpoint suffix.

        The suffix transition sequence is fixed by A*.  An existing object can
        participate only if it occurs in a token at an input place of one of
        those transitions; observed suffix objects are always retained.  Entire
        composite tokens are retained atomically.  Output-only object variables
        are potentially fresh/unbounded and force a full-domain fallback unless
        the concrete object is fixed by synchronization/canonical creation.
        """
        full = {str(o) for o in component.objects}
        moves = self._path(state, node.node_id)
        tids = [int(m.transition_id) for m in moves
                if m.kind in ("model", "sync") and m.transition_id is not None]
        suffix_events = tuple(m.event_id for m in moves
                              if m.kind in ("log", "sync") and m.event_id is not None)
        keep: Set[str] = set()
        for eid in suffix_events:
            ev = component.observation_formula.events.get(eid)
            if ev is not None:
                keep.update(map(str, ev.objects))

        pre_places = {int(a.get("source")) for a in getattr(net, "_arcs", ())
                      if a.get("target") in set(tids) and a.get("source") is not None}
        boundary = state.boundary_assignment
        static_marking = []
        if boundary is not None:
            for item in boundary.marking_signature or ():
                try:
                    pid, tok = int(item[0]), tuple(map(str, item[1]))
                except Exception:
                    continue
                if pid in pre_places:
                    keep.update(tok)
                else:
                    static_marking.append(item)

        # Detect object parameters that can appear without being carried by an
        # input token.  Such a parameter may choose any compatible object and a
        # global slice would be unsound unless its identity is fixed.
        unsafe_output_only = False
        fixed_fresh = set(map(str, (canonical_fresh_bindings or {}).values()))
        keep.update(fixed_fresh)
        trans_by_id = {int(t["id"]): t for t in getattr(net, "_transitions", ())}
        data_types = set(getattr(net, "_data_types", ()))
        for step, move in enumerate([m for m in moves if m.kind in ("model", "sync")]):
            if move.transition_id is None:
                continue
            tid = int(move.transition_id)
            incoming_names, outgoing_names, nu_names = set(), set(), set()
            for arc in getattr(net, "_arcs", ()):
                incoming = arc.get("target") == tid
                outgoing = arc.get("source") == tid
                if not incoming and not outgoing:
                    continue
                for name, typ in arc.get("inscription", ()):
                    if typ in data_types:
                        continue
                    base = str(name).replace("nu ", "")
                    if incoming:
                        incoming_names.add(base)
                    if outgoing:
                        outgoing_names.add(base)
                    if outgoing and "nu" in str(name):
                        nu_names.add(base)
            output_only = outgoing_names - incoming_names
            if output_only:
                if move.kind == "sync" and move.event_id is not None:
                    # Synchronization fixes the complete object set to the OCEL
                    # event, so output-only identities cannot escape the slice.
                    continue
                global_step = int(state.boundary_model_depth) + step
                if output_only <= nu_names and global_step in (canonical_fresh_bindings or {}):
                    continue
                unsafe_output_only = True
                break

        safe = not unsafe_output_only
        if not keep:
            safe = False
        if not safe:
            keep = set(full)
            static_marking = []

        # If a retained token is composite, include every object in it.  Repeat
        # to closure because the same object can participate in another token.
        if boundary is not None and keep != full:
            changed = True
            while changed:
                changed = False
                for item in boundary.marking_signature or ():
                    try:
                        tok = set(map(str, item[1]))
                    except Exception:
                        continue
                    if tok & keep and not tok <= keep:
                        keep.update(tok)
                        changed = True

        keep &= full
        info = {
            "full_objects": len(full), "kept_objects": len(keep),
            "pruned_objects": max(0, len(full) - len(keep)),
            "suffix_model_depth": len(tids), "suffix_events": len(suffix_events),
            "transition_ids": tuple(tids), "pre_places": tuple(sorted(pre_places)),
            "safe": bool(safe), "unsafe_output_only": bool(unsafe_output_only),
        }
        return keep, suffix_events, safe, info

    def _checkpoint_boundary_store_values(self, net, boundary: JointAssignment,
                                          info: Dict[str, Any], object_domain: Set[str]) \
            -> Tuple[bool, Tuple[Any, ...]]:
        """Translate retained token data into positional CoCoMoT store slots.

        The mapping is accepted only when an input inscription of the *fixed
        suffix* names every data slot of a relevant boundary token and the
        retained witness contains all those names.  Ambiguous/missing mappings
        make the specialized context ineligible; the caller falls back to the
        ordinary full-prefix eager query rather than weakening data semantics.
        """
        if not boundary.token_data_signature:
            return True, ()
        tids = set(map(int, info.get("transition_ids", ())))
        pre_places = set(map(int, info.get("pre_places", ())))
        data_types = set(getattr(net, "_data_types", ()))
        places = {int(p["id"]): p for p in getattr(net, "_places", ())}
        fields_by_token = {}
        for item in boundary.token_data_signature or ():
            try:
                fields_by_token[(int(item[0]), tuple(map(str, item[1])))] = dict(item[2])
            except Exception:
                continue

        out = []
        for (pid, tok), fields in fields_by_token.items():
            if pid not in pre_places:
                continue
            if not set(tok) <= set(object_domain):
                continue
            place = places.get(pid)
            if place is None:
                return False, ()
            slot_count = sum(1 for typ in place.get("color", ()) if typ in data_types)
            if slot_count == 0:
                continue
            candidate_names = []
            for arc in getattr(net, "_arcs", ()):
                if arc.get("source") != pid or int(arc.get("target", -1)) not in tids:
                    continue
                names = [str(name) for name, typ in arc.get("inscription", ()) if typ in data_types]
                if len(names) == slot_count:
                    candidate_names.append(tuple(names))
            # Different fixed suffix transitions may rename the same positional
            # token field.  Any inscription whose complete names are available
            # in the retained witness gives the same physical slot values.
            names = next((ns for ns in candidate_names if all(n in fields for n in ns)), None)
            if names is None:
                return False, ()
            vals = tuple(fields[n] for n in names)
            out.append((pid, tok, vals))
        return True, tuple(out)

    def _context_key(self, component: ComponentState, state: SearchState,
                     event_order: Sequence[str]) -> Tuple[Any, ...]:
        # Query caches distinguish certified-boundary suffix queries from
        # initial-state/full-prefix queries.  The concrete candidate move
        # signature is added by the caller, so the exact per-candidate object
        # slice need not be duplicated here.
        if state.boundary_assignment is not None:
            return self._object_domain_key(
                component, boundary=state.boundary_assignment
            ) + ("boundary_depth", int(state.boundary_model_depth), tuple(event_order))
        return self._object_domain_key(component) + (tuple(event_order),)

    def _evict_if_needed(self) -> None:
        # Persistent component contexts are intentionally not LRU-evicted during
        # an active run; there is normally only one per current component/domain.
        # Retain the legacy map bound for non-persistent/offline contexts.
        while len(self._prepared) > max(self.max_prepared_contexts, len(self._component_contexts) + 2):
            key, prepared = self._prepared.popitem(last=False)
            if prepared in self._component_contexts.values():
                self._prepared[key] = prepared
                break
            try:
                prepared.solver.destroy()
            except Exception:
                pass

    @staticmethod
    def _geometric_capacity(required: int, limit: Optional[int] = None, minimum: int = 4) -> int:
        """Return a power-of-two-ish capacity without exceeding a known hard limit.

        Persistent CoCoMoT contexts are expensive to rebuild because every model
        layer materializes marking and token-movement constraints.  Growing by
        two layers at a time therefore caused the same component to be rebuilt at
        depths 3, 6, 9, 12, ... .  Geometric growth amortizes that construction
        while the optional limit prevents a 45-step request from rounding to 64
        when the complete search bound is only 51.
        """
        req = max(1, int(required))
        cap = max(int(minimum), 1)
        while cap < req:
            cap *= 2
        if limit is not None:
            cap = min(cap, max(req, int(limit)))
        return max(req, cap)

    def _initial_context_capacity(self, component: ComponentState, required: int,
                                  limit: Optional[int] = None) -> int:
        return self._geometric_capacity(required, limit=limit)

    def _build_persistent_context(self, component: ComponentState, state: SearchState,
                                  event_order: Sequence[str], capacity: int,
                                  domain_key: Tuple[Any, ...],
                                  *, object_domain: Optional[Set[str]] = None,
                                  boundary: Optional[JointAssignment] = None,
                                  boundary_store_values: Sequence[Any] = ()) -> PreparedJODAP:
        net = self.new_net()
        trace, _ = self._trace(component, event_order, net, object_domain=object_domain)
        # Force a capacity independent of the current trace length.  JODAP does
        # not use CoCoMoT's edit-distance/move matrix; event observations are
        # supplied by candidate-specific constraints.  Thus the structural
        # marking/object/data layers are safe to reuse for later event prefixes.
        net._step_bound = max(int(capacity), 1)
        import z3
        solver = self.mods.Z3Solver(incremental=False)
        solver.ctx = z3.Solver()
        solver.ctx.set("timeout", 600000)
        solver._incremental = True
        encoding = self.ExactEncoding(solver, net, trace)

        t0 = time.perf_counter()
        guarded_profiles = []
        for _t in getattr(net, "_transitions", ()):
            if _t.get("constraint") is not None:
                _p = self._binding_complexity(net, component, _t, getattr(encoding, "_objects", None))
                guarded_profiles.append({
                    "transition": _t.get("label"), "transition_id": _t.get("id"),
                    "list_params": _p.get("list_params", []),
                    "scalar_params": _p.get("scalar_params", []),
                })
        self._diag(
            "context_build_start", component=component.component_id, capacity=int(capacity),
            component_events=len(event_order), component_objects=len(component.objects),
            encoded_objects=len(getattr(encoding, "_objects", {}) or {}),
            checkpoint_specialized=(boundary is not None),
            guarded_transitions=guarded_profiles,
        )
        def timed(stat_name, fun):
            ts = time.perf_counter()
            self._diag("context_phase_start", component=component.component_id, phase_name=stat_name)
            value = fun()
            elapsed = time.perf_counter() - ts
            self.stats[stat_name] += int(round(1000.0 * elapsed))
            if boundary is not None and stat_name == "context_moving_seconds_ms":
                self.stats["checkpoint_context_moving_seconds_ms"] += int(round(1000.0 * elapsed))
            self._diag("context_phase_done", component=component.component_id, phase_name=stat_name,
                       elapsed_seconds=elapsed, checkpoint_specialized=(boundary is not None))
            return value

        timed("context_variable_seconds_ms", encoding.create_variables)
        base = [
            timed("context_initial_seconds_ms",
                  (lambda: encoding.odacc_boundary_initial_state(
                      boundary.marking_signature, boundary_store_values
                  )) if boundary is not None else
                  (lambda: encoding.initial_state(self.fixed_objects))),
            timed("context_transition_seconds_ms", encoding.transition_range),
            timed("context_object_type_seconds_ms", encoding.object_types),
            timed("context_freshness_seconds_ms", encoding.freshness),
            timed("context_moving_seconds_ms", encoding.moving_tokens),
            timed("context_remaining_seconds_ms", encoding.remaining_tokens),
            timed("context_data_seconds_ms", encoding.data_constraints),
            # Use the same model-side read/write guard semantics as the full SMT
            # reference.  This matters only for the exact/eager fallback; the
            # lazy path already evaluates selected guards directly.
            semantic_output_guard_constraints(encoding, net, component),
        ]
        # ``encoding.data_constraints`` may use the O(n) symbolic LIST aggregate
        # encoder instead of CoCoMoT's generic powerset substitution.  Expose the
        # counts in the normal symbolic statistics and diagnostics.
        fast_list = int(getattr(encoding, "_odacc_list_guard_fast_paths", 0) or 0)
        fallback_list = int(getattr(encoding, "_odacc_list_guard_fallbacks", 0) or 0)
        avoided_list = int(getattr(encoding, "_odacc_list_guard_subsets_avoided", 0) or 0)
        self.stats["symbolic_list_guard_fast_paths"] += fast_list
        self.stats["symbolic_list_guard_fallbacks"] += fallback_list
        self.stats["symbolic_list_guard_subsets_avoided"] += avoided_list
        self._diag(
            "context_list_guard_encoding", component=component.component_id,
            fast_paths=fast_list, fallbacks=fallback_list,
            generic_subsets_avoided=avoided_list,
        )
        base.append(timed("context_cache_seconds_ms", encoding.cache_constraints))
        timed("context_require_seconds_ms", lambda: solver.require(base))
        encode_seconds = time.perf_counter() - t0

        key = domain_key + (int(capacity),)
        prepared = PreparedJODAP(
            key=key, solver=solver, encoding=encoding, net=net,
            trans_by_id={t["id"]: t for t in net._transitions},
            data_types=dict(net.get_data_variables()),
            base_encode_seconds=encode_seconds, capacity=int(capacity),
            domain_key=domain_key,
            checkpoint_boundary=boundary,
            checkpoint_model_depth=int(state.boundary_model_depth if boundary is not None else 0),
            checkpoint_object_domain=frozenset(object_domain or ()),
            checkpoint_static_marking=tuple(
                item for item in ((boundary.marking_signature if boundary is not None else ()) or ())
                if not any(str(o) in set(object_domain or ()) for o in (item[1] if len(item) > 1 else ()))
            ),
            checkpoint_static_token_data=tuple(
                item for item in ((boundary.token_data_signature if boundary is not None else ()) or ())
                if not any(str(o) in set(object_domain or ()) for o in (item[1] if len(item) > 1 else ()))
            ),
        )
        self._prepared[key] = prepared
        self._component_contexts[domain_key] = prepared
        self.stats["prepared_builds"] += 1
        self.stats["component_context_builds"] += 1
        self.stats["context_capacity"] = max(self.stats["context_capacity"], int(capacity))
        self.stats["context_build_seconds_ms"] += int(round(1000.0 * encode_seconds))
        if boundary is not None:
            self.stats["checkpoint_context_builds"] += 1
            self.stats["checkpoint_context_build_seconds_ms"] += int(round(1000.0 * encode_seconds))
        self._diag("context_build_done", component=component.component_id, capacity=int(capacity),
                   elapsed_seconds=encode_seconds, checkpoint_specialized=(boundary is not None))
        return prepared

    def _prepare(self, component: ComponentState, state: SearchState,
                 event_order: Sequence[str], *, required_depth: Optional[int] = None,
                 node: Optional[SearchNode] = None,
                 canonical_fresh_bindings: Optional[Dict[int, str]] = None) -> PreparedJODAP:
        """Return the smallest reusable eager context that can encode this query.

        For ordinary searches this is the persistent initial-state context used
        by earlier versions.  For a temporary state rooted at a certified online
        checkpoint, instant 0 is instead fixed to the retained concrete boundary
        marking/token-data witness and only the suffix depth is encoded.  The
        checkpoint context also receives a conservative object-domain slice; if
        that slice cannot be justified, the full component object domain is used
        while retaining the suffix-depth reduction.
        """
        boundary = state.boundary_assignment
        if boundary is not None and node is not None:
            self.stats["checkpoint_context_attempts"] += 1
            net = self._query_slice_static_net()
            object_domain, suffix_events, slice_safe, info = self._checkpoint_context_domain(
                component, state, node, net,
                canonical_fresh_bindings=canonical_fresh_bindings,
            )
            store_safe, boundary_store_values = self._checkpoint_boundary_store_values(
                net, boundary, info, object_domain
            )
            if not store_safe:
                self.stats["checkpoint_context_fallbacks"] += 1
                self._diag(
                    "checkpoint_context_data_fallback", component=component.component_id,
                    boundary_model_depth=int(state.boundary_model_depth),
                    reason="boundary_token_data_mapping_ambiguous",
                )
                return None
            full_n = int(info.get("full_objects", len(component.objects)))
            kept_n = int(info.get("kept_objects", len(object_domain)))
            self.stats["checkpoint_context_objects_full"] += full_n
            self.stats["checkpoint_context_objects_kept"] += kept_n
            self.stats["checkpoint_context_objects_pruned"] += max(0, full_n - kept_n)
            # Diagnostic subset-space proxy: this is intentionally just the
            # powerset of the complete/kept object universes; transition-specific
            # LIST profiles remain available in context_build_start.
            if full_n <= 30:
                self.stats["checkpoint_context_subset_space_full"] += (1 << full_n)
            if kept_n <= 30:
                self.stats["checkpoint_context_subset_space_kept"] += (1 << kept_n)

            moves = self._path(state, node.node_id)
            suffix_depth = sum(1 for m in moves if m.kind in ("model", "sync"))
            required = max(1, int(suffix_depth if required_depth is None else required_depth))
            # A fixed suffix query benefits more from a tight context than from
            # geometric spare capacity; sibling suffixes are cached by the exact
            # boundary/domain key and rebuilt only when they need greater depth.
            attrs = component.observation_formula.current_object_attributes()
            domain_key = self._object_domain_key(component, object_domain, boundary) + (
                "suffix_events", tuple(suffix_events),
            )
            prepared = self._component_contexts.get(domain_key)
            if prepared is not None and prepared.capacity >= required:
                self.stats["prepared_hits"] += 1
                self.stats["component_context_reuses"] += 1
                self.stats["checkpoint_context_hits"] += 1
                self._diag(
                    "checkpoint_context_reuse", component=component.component_id,
                    requested_depth=required, capacity=prepared.capacity,
                    full_objects=full_n, kept_objects=kept_n,
                    slice_safe=bool(slice_safe),
                )
                return prepared
            if prepared is not None:
                try:
                    prepared.solver.destroy()
                except Exception:
                    pass
                self._prepared.pop(prepared.key, None)
                self._component_contexts.pop(domain_key, None)

            self._diag(
                "checkpoint_context_slice", component=component.component_id,
                requested_depth=required, full_objects=full_n,
                kept_objects=kept_n, pruned_objects=max(0, full_n-kept_n),
                slice_safe=bool(slice_safe), suffix_events=len(suffix_events),
                boundary_model_depth=int(state.boundary_model_depth),
            )
            return self._build_persistent_context(
                component, state, suffix_events, required, domain_key,
                object_domain=object_domain, boundary=boundary,
                boundary_store_values=boundary_store_values,
            )

        domain_key = self._object_domain_key(component)
        requested = int(state.model_bound if required_depth is None else required_depth)
        required = max(requested, 1)
        self.stats["context_requested_depth_max"] = max(
            self.stats["context_requested_depth_max"], required
        )
        relative_required = required
        self.stats["context_checkpoint_relative_depth_max"] = max(
            self.stats["context_checkpoint_relative_depth_max"], relative_required
        )
        prepared = self._component_contexts.get(domain_key)
        if prepared is not None and prepared.capacity >= required:
            self.stats["prepared_hits"] += 1
            self.stats["component_context_reuses"] += 1
            return prepared

        hard_limit = max(required, int(state.model_bound))
        if required >= max(8, int(0.75 * max(1, int(state.model_bound)))):
            self.stats["context_absolute_depth_escalations"] += 1
            self._diag(
                "context_absolute_depth_escalation", component=component.component_id,
                requested_depth=required, global_model_bound=int(state.model_bound),
                boundary_model_depth=0, checkpoint_relative_depth=relative_required,
            )
        if prepared is None:
            capacity = self._initial_context_capacity(component, required, limit=hard_limit)
            self.stats["context_adaptive_builds"] += 1
        else:
            capacity = max(prepared.capacity, 1)
            while capacity < required:
                capacity = min(hard_limit, max(capacity * 2, required))
                self.stats["context_geometric_growths"] += 1
                if capacity >= hard_limit:
                    break
            capacity = max(required, capacity)
            self.stats["component_context_extensions"] += 1
            self.stats["context_adaptive_extensions"] += 1
            try:
                prepared.solver.destroy()
            except Exception:
                pass
            self._prepared.pop(prepared.key, None)
            self._component_contexts.pop(domain_key, None)

        self._diag(
            "context_adaptive_capacity", component=component.component_id,
            requested_depth=required, capacity=capacity,
            global_model_bound=int(state.model_bound),
        )
        return self._build_persistent_context(
            component, state, event_order, capacity, domain_key
        )

    def structurally_reachable_transition_ids(self, component: ComponentState,
                                              state: SearchState,
                                              node: SearchNode) -> Set[int]:
        """Return a safe structural transition over-approximation *without*
        forcing construction of the eager CoCoMoT JODAP context.

        The previous implementation called :meth:`_prepare` merely to obtain
        CoCoMoT's reachability table.  On paths fully handled by the lazy JODAP
        encoder this defeated laziness: a 100+ second eager context could be
        built even though it was never queried.

        If an eager context already exists (because a genuine fallback query was
        required), its reachability table is reused.  Otherwise all model
        transitions are returned.  This weakens only an optimization; it cannot
        remove a valid successor and, crucially, keeps eager construction fully
        on-demand.
        """
        domain_key = self._object_domain_key(component)
        prepared = self._component_contexts.get(domain_key)
        if prepared is not None:
            try:
                reach = prepared.net.reachable(node.model_depth)
                if reach:
                    return {t["id"] for t in reach}
            except Exception:
                pass
        else:
            self.stats["eager_context_avoided"] += 1
        # Structural reachability is model-static. Reuse one parsed PNML/net
        # instead of reparsing the model on every zero-cost update. This is
        # especially important on large retained components, where repeated
        # pyparsing/new_net() calls add substantial overhead and native-object
        # churn without changing the answer.
        net = self._query_slice_static_net()
        return {t["id"] for t in net._transitions}

    @staticmethod
    def _eval_int(model, expr) -> int:
        if isinstance(expr, int):
            return expr
        value = model.eval(expr, model_completion=True)
        return value.as_long()

    @staticmethod
    def _eval_real(model, expr) -> float:
        if isinstance(expr, (int, float)):
            return float(expr)
        value = model.eval(expr, model_completion=True)
        try:
            return float(value.as_long())
        except Exception:
            return float(value.as_fraction())

    def _minimize_incremental(self, prepared: PreparedJODAP, objective: Any,
                              max_cost: int, lower_bound: int = 0,
                              upper_bound_hint: Optional[int] = None) \
            -> Tuple[Optional[Any], Optional[int], float]:
        """Minimize an integer objective using the reusable SAT context.

        Native Optimize objectives persist across push/pop.  A plain incremental
        Solver plus monotone ``objective <= k`` checks avoids that problem and
        allows the expensive base constraints to remain asserted. Binary search
        needs O(log max_cost) satisfiability calls instead of the linear scan in
        CoCoMoT's incremental helper.
        """
        import z3

        ctx = prepared.solver.ctx
        t0 = time.perf_counter()

        if upper_bound_hint is not None:
            # ``upper_bound_hint`` is an inclusive feasibility cap.  In exact
            # incumbent-certification mode the caller deliberately passes
            # UB-1, because another solution at the incumbent cost cannot
            # improve the known witness.  If the independently admissible
            # lower bound already exceeds that cap, this candidate is proved
            # unable to improve the incumbent without issuing a solver query.
            if int(upper_bound_hint) < int(lower_bound):
                self.stats["strict_improvement_pre_solve_pruned"] += 1
                return None, None, time.perf_counter() - t0
            max_cost = min(int(max_cost), int(upper_bound_hint))
        lb = max(0, min(int(lower_bound), int(max_cost)))
        # Appending an alignment move only adds constraints and non-negative
        # objective terms, hence the predecessor optimum is a valid lower bound.
        # A satisfiable query at that bound proves the child optimum immediately.
        ctx.push()
        ctx.add(objective <= lb)
        status = ctx.check()
        if status == z3.sat:
            ctx.add(objective == lb)
            status2 = ctx.check()
            if status2 == z3.sat:
                model = ctx.model()
                ctx.pop()
                self.stats["lower_bound_hits"] += 1
                return model, lb, time.perf_counter() - t0
        ctx.pop()

        ctx.push()
        ctx.add(objective <= int(max_cost))
        status = ctx.check()
        if status != z3.sat:
            ctx.pop()
            return None, None, time.perf_counter() - t0
        ctx.pop()

        lo, hi = lb + 1, int(max_cost)
        while lo < hi:
            mid = (lo + hi) // 2
            ctx.push()
            ctx.add(objective <= mid)
            status = ctx.check()
            ctx.pop()
            if status == z3.sat:
                hi = mid
            elif status == z3.unsat:
                lo = mid + 1
            else:
                return None, None, time.perf_counter() - t0

        ctx.push()
        ctx.add(objective == lo)
        status = ctx.check()
        if status != z3.sat:
            ctx.pop()
            return None, None, time.perf_counter() - t0
        model = ctx.model()
        # ModelRef remains usable after pop, and extracting the assignments is
        # done by the caller before another query mutates the context.
        ctx.pop()
        return model, lo, time.perf_counter() - t0


    # ------------------------------------------------------------------
    # Lazy path JODAP
    # ------------------------------------------------------------------
    # The persistent CoCoMoT context is retained as an exact fallback for paths
    # with unresolved model-only bindings.  When every model step has a concrete
    # binding (synchronous event or explicitly fixed fresh-object binding), the
    # following encoder works directly on the selected transition path.  It
    # creates no marking variables for unused objects/transitions and therefore
    # scales with |gamma| rather than capacity x all possible object tokens.

    @staticmethod
    def _lazy_unique_object_decls(net, transition) -> Dict[str, str]:
        decls: Dict[str, str] = {}
        for arc in net._arcs:
            if arc.get("source") != transition["id"] and arc.get("target") != transition["id"]:
                continue
            for name, typ in arc.get("inscription", []):
                if typ in net._data_types:
                    continue
                decls.setdefault(name, typ)
        return decls

    def _lazy_binding(self, component: ComponentState, net, transition,
                      objects: Sequence[str], fresh_object: Optional[str] = None) \
            -> Optional[Dict[str, Any]]:
        decls = self._lazy_unique_object_decls(net, transition)
        by_type: Dict[str, List[str]] = {}
        for obj in objects:
            typ = component.observation_formula.object_types.get(obj)
            if typ is not None:
                by_type.setdefault(typ, []).append(obj)
        for vals in by_type.values():
            vals.sort()

        binding: Dict[str, Any] = {}
        used: Set[str] = set()
        # Fresh variable is explicit and must never be guessed.
        for name, typ in decls.items():
            base = typ[:typ.rfind(" LIST")] if "LIST" in typ else typ
            if "nu" in name:
                if fresh_object is None or component.observation_formula.object_types.get(fresh_object) != base:
                    return None
                binding[name] = fresh_object
                used.add(fresh_object)

        # Bind scalar parameters first.  If two indistinguishable scalar
        # parameters of one type exist, the path is genuinely ambiguous and we
        # conservatively fall back to the full JODAP encoding.
        scalar_by_type: Dict[str, List[str]] = {}
        for name, typ in decls.items():
            if "LIST" not in typ and "nu" not in name:
                scalar_by_type.setdefault(typ, []).append(name)
        for typ, names in scalar_by_type.items():
            candidates = [o for o in by_type.get(typ, []) if o not in used]
            if len(names) != 1 or len(candidates) != 1:
                return None
            binding[names[0]] = candidates[0]
            used.add(candidates[0])

        # A list parameter denotes all remaining participating objects of its
        # base type. Multiple list variables of the same type would again be
        # ambiguous and are delegated to the exact fallback.
        list_by_type: Dict[str, List[str]] = {}
        for name, typ in decls.items():
            if "LIST" in typ:
                base = typ[:typ.rfind(" LIST")]
                list_by_type.setdefault(base, []).append(name)
        for typ, names in list_by_type.items():
            if len(names) != 1:
                return None
            binding[names[0]] = [o for o in by_type.get(typ, []) if o not in used]
            used.update(binding[names[0]])

        # A synchronous binding must explain exactly the observed object set.
        if set(objects) != used:
            return None
        return binding

    @staticmethod
    def _lazy_arc_tokens(net, place, inscription, binding: Dict[str, Any]) -> List[Tuple[str, ...]]:
        obj_entries = [(n, t) for (n, t) in inscription if t not in net._data_types]
        if not obj_entries:
            return [tuple()]
        columns: List[List[str]] = []
        for name, typ in obj_entries:
            value = binding.get(name)
            if value is None:
                return []
            columns.append(list(value) if "LIST" in typ else [value])
        return [tuple(x) for x in itertools.product(*columns)]

    def _lazy_guard(self, component: ComponentState, net, transition,
                    binding: Dict[str, Any], solver, data_vars: Dict[str, Any]):
        if "constraint" not in transition:
            return solver.true(), set()
        try:
            from dpn.expr_utils import VarReplacer, ListExpander
            import objectcentric.encoding as oc_encoding
            ObjectPropertyReplacer = oc_encoding.ObjectPropertyReplacer
        except Exception:
            return None, set()
        guard = deepcopy(transition["constraint"])
        guard.accept(VarReplacer(dict(binding)))
        exp = ListExpander()
        guard.accept(exp)
        while exp._change:
            exp._change = False
            guard.accept(exp)
        objects = {
            o: {"type": component.observation_formula.object_types.get(o),
                "ovmap": dict(component.observation_formula.current_object_attributes().get(o, {}))}
            for o in component.objects
        }
        # This module binding was patched by _install_soft_object_property_semantics,
        # so observed object properties become model-side variables rather than
        # hard constants.
        guard.accept(ObjectPropertyReplacer(objects))
        available = {a for vals in component.observation_formula.current_object_attributes().values() for a in vals}
        relevant = self._guard_object_attribute_names(transition.get("constraint"), available)
        return guard.toSMT(solver, data_vars), relevant

    @staticmethod
    def _lazy_num_value(node):
        raw = getattr(node, "num", None)
        if raw is None:
            return None
        try:
            if isinstance(raw, (int, float)):
                return raw
            text = str(raw)
            return float(text) if any(c in text for c in ".eE/") else int(text)
        except Exception:
            try:
                from fractions import Fraction
                return float(Fraction(str(raw)))
            except Exception:
                return None

    def _lazy_eval_expr(self, node, data_values: Dict[str, Any], binding: Dict[str, Any],
                        object_values: Dict[Tuple[str, str], Any]):
        """Evaluate the subset of CoCoMoT expressions used by DOPID guards.

        This evaluator is deliberately used only as an independent certificate
        for a lazy solver result.  Unsupported constructs return ``None`` and
        force the exact eager fallback rather than risking an unsound fast path.
        """
        if node is None:
            return None
        cls = node.__class__.__name__
        if cls == "Num":
            return self._lazy_num_value(node)
        if cls == "Var":
            name = getattr(node, "name", None)
            key = str(node)
            if key in data_values:
                return data_values[key]
            if isinstance(name, str) and name in data_values:
                return data_values[name]
            if isinstance(name, str) and name in binding:
                return binding[name]
            if isinstance(name, (list, tuple)):
                return list(name)
            return None
        if cls == "Fun":
            name = getattr(node, "_name", None)
            args = [self._lazy_eval_expr(a, data_values, binding, object_values)
                    for a in getattr(node, "_args", [])]
            if any(a is None for a in args):
                return None
            if name == "sum":
                flat = []
                for a in args:
                    flat.extend(a if isinstance(a, list) else [a])
                try:
                    return sum(flat)
                except Exception:
                    return None
            if len(args) == 1:
                objs = args[0] if isinstance(args[0], list) else [args[0]]
                vals = []
                for obj in objs:
                    key = (str(obj), str(name))
                    if key not in object_values:
                        return None
                    vals.append(object_values[key])
                return vals if isinstance(args[0], list) else vals[0]
            return None
        left = getattr(node, "left", getattr(node, "_left", None))
        right = getattr(node, "right", getattr(node, "_right", None))
        op = getattr(node, "op", None)
        if left is not None and right is not None and op is not None:
            l = self._lazy_eval_expr(left, data_values, binding, object_values)
            r = self._lazy_eval_expr(right, data_values, binding, object_values)
            if l is None or r is None:
                return None
            try:
                def numeric_equal(a, b):
                    # OCEL numeric values reach this lightweight certificate as
                    # Python floats, while the exact SMT encoding uses Real
                    # arithmetic.  Expressions such as 2200 + 199.99 + ... + 5
                    # can therefore differ from the observed decimal by a few
                    # machine ulps.  Do not reject an otherwise exact zero-cost
                    # witness because of binary floating-point representation.
                    if (isinstance(a, (int, float)) and not isinstance(a, bool)
                            and isinstance(b, (int, float)) and not isinstance(b, bool)):
                        import math
                        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-9)
                    return a == b

                return {
                    "==": lambda: numeric_equal(l, r),
                    "!=": lambda: not numeric_equal(l, r),
                    "<": lambda: l < r, "<=": lambda: l <= r,
                    ">": lambda: l > r, ">=": lambda: l >= r,
                    "&&": lambda: bool(l) and bool(r),
                    "||": lambda: bool(l) or bool(r),
                    "+": lambda: l + r, "-": lambda: l - r,
                    "*": lambda: l * r, "/": lambda: l / r,
                }.get(op, lambda: None)()
            except Exception:
                return None
        # Unary Boolean negation in the CoCoMoT expression tree.
        child = getattr(node, "expr", getattr(node, "_expr", None))
        if child is not None:
            v = self._lazy_eval_expr(child, data_values, binding, object_values)
            return None if v is None else (not bool(v))
        return None

    def _certify_lazy_assignment(self, component: ComponentState, moves: Sequence[SymbolicMove],
                                 trans_by_id: Dict[int, Dict[str, Any]],
                                 assignment: JointAssignment) -> bool:
        """Independently certify guard satisfaction and the complete lazy cost.

        Lazy JODAP is an optimization, never a change in semantics.  A result is
        accepted only when we can reconstruct every hard selected-transition
        guard and every soft event/object-attribute deviation from the decoded
        witness.  Anything unsupported falls back to the exact CoCoMoT JODAP.
        """
        bind_by = {int(x.get("step", -1)): x for x in assignment.object_bindings}
        data_by = {int(x.get("step", -1)): dict(x.get("values", {}))
                   for x in assignment.data_assignments}
        observed_oa = component.observation_formula.current_object_attributes()
        model_oa = {(o, a): v for o, vals in observed_oa.items() for a, v in vals.items()}
        for x in assignment.object_attribute_assignments:
            model_oa[(str(x.get("object")), str(x.get("attribute")))] = x.get("model_value")

        expected = 0
        model_i = 0
        relevant_pairs: Set[Tuple[str, str]] = set()
        available = {a for vals in observed_oa.values() for a in vals}
        for move in moves:
            if move.kind == "log":
                e = component.observation_formula.events.get(move.event_id)
                if e is None:
                    return False
                expected += len(e.objects)
                continue
            t = trans_by_id.get(move.transition_id)
            if t is None:
                return False
            bitem = bind_by.get(model_i)
            if bitem is None:
                return False
            objects = tuple(bitem.get("objects", ()))
            binding = self._lazy_binding(component, self.new_net(), t, objects)
            # Fresh-object transitions use a nu variable and cannot be rebuilt by
            # the generic call without the explicit fresh id.  Their guards are
            # absent in our supported models, so retain only hard move cost here.
            if binding is None and t.get("invisible", False):
                binding = {}
            if binding is None:
                return False
            vals = data_by.get(model_i, {})
            if "constraint" in t:
                attrs = self._guard_object_attribute_names(t.get("constraint"), available)
                for obj in objects:
                    for attr in attrs:
                        if attr in observed_oa.get(obj, {}):
                            relevant_pairs.add((obj, attr))
                ok = self._lazy_eval_expr(t["constraint"], vals, binding, model_oa)
                if ok is not True:
                    return False
            if move.kind == "model" and not t.get("invisible", False):
                expected += len(objects)
            elif move.kind == "sync":
                e = component.observation_formula.events.get(move.event_id)
                if e is None or t.get("label") != e.activity or set(objects) != set(e.objects):
                    return False
                for name, obs in e.attributes.items():
                    # If a recorded event attribute cannot be related to a
                    # model-side value, the lazy encoder is not complete enough.
                    if name not in vals:
                        return False
                    if vals[name] != obs:
                        expected += 1
            model_i += 1

        # Every observed object attribute referenced by a selected guard is a
        # soft equality exactly once for the candidate alignment.
        for obj, attr in relevant_pairs:
            if (obj, attr) not in model_oa or attr not in observed_oa.get(obj, {}):
                return False
            if model_oa[(obj, attr)] != observed_oa[obj][attr]:
                expected += 1
        return abs(float(assignment.total_cost) - float(expected)) <= 1e-9

    def _current_prefix_upper_bound(self, component: ComponentState, state: SearchState) -> float:
        """Return a usable incumbent only when it explains the current prefix.

        During ``_sync_increment`` the state may temporarily still contain the
        incumbent of the previous prefix.  Using that value for branch-and-bound
        would be unsound because it does not yet consume the newly observed
        event.  We therefore accept the bound only if the incumbent move sequence
        consumes exactly the events currently present in the component.
        """
        if state.incumbent_assignment is None or state.incumbent_offline:
            return float("inf")
        consumed = {
            m.event_id for m in state.incumbent_moves
            if m.kind in ("log", "sync") and m.event_id is not None
        }
        if consumed != set(component.execution.event_ids):
            return float("inf")
        return float(state.upper_bound)

    def _lazy_solve_fixed_path(self, component: ComponentState, state: SearchState,
                               node: SearchNode, *, lower_bound: int = 0,
                               canonical_fresh_bindings: Optional[Dict[int, str]] = None,
                               fixed_binding_by_step: Optional[Dict[int, Sequence[str]]] = None,
                               accept_certified_nonzero: bool = False,
                               upper_bound_hint: Optional[int] = None) \
            -> Optional[JointAssignment]:
        """Solve a fully bound candidate without constructing CoCoMoT Encoding.

        Returns ``None`` both for infeasibility and for unsupported/ambiguous
        paths. The caller distinguishes neither case because the exact eager
        backend is always used as a fallback. Consequently this optimization
        cannot remove a feasible alignment.
        """
        import z3
        self.stats["lazy_path_attempts"] += 1
        self._last_lazy_pruned = False
        build_start = time.perf_counter()
        moves = self._path(state, node.node_id)
        last_move = moves[-1] if moves else None
        last_event = None
        if last_move is not None and last_move.event_id is not None:
            last_event = component.observation_formula.events.get(last_move.event_id)
        self._diag(
            "lazy_query_start", component=component.component_id, node=node.node_id,
            model_depth=node.model_depth, move_count=len(moves), lower_bound=lower_bound,
            last_move=(last_move.kind if last_move else None),
            activity=(last_event.activity if last_event is not None else getattr(last_move, "transition_label", None)),
            event_id=(last_event.event_id if last_event is not None else None),
            event_objects=(len(last_event.objects) if last_event is not None else 0),
            component_objects=len(component.objects), component_events=len(component.execution.event_ids),
        )
        net = self.new_net()
        trans_by_id = {t["id"]: t for t in net._transitions}
        data_types = dict(net.get_data_variables())
        zsolver = self.mods.Z3Solver(incremental=False)
        zsolver.ctx = z3.Solver()
        zsolver._incremental = True

        # Concrete object marking, symbolic data attached only to tokens that
        # actually occur on this path.  A provenance-focused temporary search
        # can start from a certified boundary witness rather than reconstructing
        # the complete history.  Boundary values are constants; only the suffix
        # variables remain symbolic.
        marking: Dict[int, Set[Tuple[str, ...]]] = {p["id"]: set() for p in net._places}
        token_data: Dict[Tuple[int, Tuple[str, ...]], Dict[str, Any]] = {}
        data_provenance: Dict[str, Tuple[Any, int, str]] = {}
        boundary = state.boundary_assignment
        base_cost = 0.0
        boundary_object_attrs: Dict[Tuple[str, str], Any] = {}
        boundary_global_data: Dict[str, Any] = {}

        def const_data(name: str, value: Any):
            typ = data_types.get(name, "Integer")
            cv = self._coerce(value)
            if cv is None:
                return None
            return zsolver.real(cv) if typ in ("Real", "Rational") else zsolver.num(cv)

        if boundary is not None:
            base_cost = float(boundary.total_cost)
            self.stats["provenance_boundary_builds"] += 1
            for item in boundary.marking_signature or ():
                try:
                    pid, tok = item[0], tuple(item[1])
                    marking.setdefault(pid, set()).add(tok)
                    self.stats["provenance_boundary_marking_facts"] += 1
                except Exception:
                    continue
            for item in boundary.token_data_signature or ():
                try:
                    pid, tok, fields = item[0], tuple(item[1]), item[2]
                except Exception:
                    continue
                vals = {}
                for name, value in fields:
                    c = const_data(str(name), value)
                    if c is not None:
                        vals[str(name)] = c
                        self.stats["provenance_boundary_token_data_facts"] += 1
                if vals:
                    token_data[(pid, tok)] = vals
            for item in boundary.data_provenance_signature or ():
                if len(item) < 4:
                    continue
                name, value, src_step, src_kind = item[:4]
                c = const_data(str(name), value)
                if c is not None:
                    data_provenance[self._base_var(str(name))] = (c, int(src_step), str(src_kind))
                    self.stats["provenance_boundary_data_facts"] += 1
            for item in boundary.data_state_signature or ():
                try:
                    name, value = self._base_var(str(item[0])), item[1]
                except Exception:
                    continue
                c = const_data(name, value)
                if c is not None:
                    boundary_global_data[name] = c
                    self.stats["provenance_boundary_data_facts"] += 1
            for item in boundary.object_attribute_assignments or ():
                try:
                    boundary_object_attrs[(str(item["object"]), str(item["attribute"]))] = item["model_value"]
                    self.stats["provenance_boundary_object_attr_facts"] += 1
                except Exception:
                    continue
        elif self.fixed_objects:
            for p in net._places:
                if not p.get("initial"):
                    continue
                obj_types = [t for t in p["color"] if t not in net._data_types]
                if len(obj_types) == 1:
                    for o in component.objects:
                        if component.observation_formula.object_types.get(o) == obj_types[0]:
                            marking[p["id"]].add((o,))

        constraints = []
        objective_terms = [zsolver.num(int(round(base_cost)))] if boundary is not None else []
        bindings_out: List[Dict[str, Any]] = []
        data_out: List[Dict[str, Any]] = []
        relevant_attrs: Set[str] = set()
        attr_vars: Dict[Tuple[str, str], Any] = {}
        mi = int(state.boundary_model_depth) if boundary is not None else 0

        def mk_data_var(name: str, step: int):
            typ = data_types.get(name, "Integer")
            return zsolver.realvar(f"__lazy_{name}_{step}") if typ in ("Real", "Rational") else zsolver.intvar(f"__lazy_{name}_{step}")

        for move in moves:
            if move.kind == "log":
                e = component.observation_formula.events[move.event_id]
                objective_terms.append(zsolver.num(len(e.objects)))
                continue
            if move.transition_id not in trans_by_id:
                self.stats["lazy_path_fallbacks"] += 1
                return None
            t = trans_by_id[move.transition_id]
            fixed_objs = None
            if fixed_binding_by_step and mi in fixed_binding_by_step:
                fixed_objs = tuple(fixed_binding_by_step[mi])
            elif move.kind == "sync":
                fixed_objs = tuple(component.observation_formula.events[move.event_id].objects)
            fresh = (canonical_fresh_bindings or {}).get(mi)
            # A nu move with one fixed fresh object is fully determined. Other
            # model-only moves need symbolic binding choices and use the exact
            # fallback backend.
            if fixed_objs is None and fresh is not None:
                fixed_objs = (fresh,)
            if fixed_objs is None:
                self.stats["lazy_path_fallbacks"] += 1
                return None
            _bind_t0 = time.perf_counter()
            profile = self._binding_complexity(net, component, t, list(component.objects))
            self._diag(
                "lazy_binding_start", component=component.component_id, model_step=mi,
                transition=t.get("label"), transition_id=t.get("id"), move_kind=move.kind,
                fixed_object_count=(len(fixed_objs) if fixed_objs is not None else 0),
                list_params=profile.get("list_params", []), scalar_params=profile.get("scalar_params", []),
            )
            binding = self._lazy_binding(component, net, t, fixed_objs, fresh_object=fresh)
            _bind_elapsed = time.perf_counter() - _bind_t0
            if binding is None:
                self._diag("lazy_binding_fallback", component=component.component_id, model_step=mi, transition=t.get("label"), elapsed_seconds=_bind_elapsed)
                self.stats["lazy_path_fallbacks"] += 1
                return None
            self._diag("lazy_binding_done", component=component.component_id, model_step=mi, transition=t.get("label"), elapsed_seconds=_bind_elapsed, binding_keys=sorted(map(str, binding.keys())))

            step_data = {name: mk_data_var(name, mi) for name in data_types}
            _guard_t0 = time.perf_counter()
            self._diag("lazy_guard_start", component=component.component_id, model_step=mi, transition=t.get("label"))
            guard, attrs = self._lazy_guard(component, net, t, binding, zsolver, step_data)
            _guard_elapsed = time.perf_counter() - _guard_t0
            if guard is None:
                self._diag("lazy_guard_fallback", component=component.component_id, model_step=mi, transition=t.get("label"), elapsed_seconds=_guard_elapsed)
                self.stats["lazy_path_fallbacks"] += 1
                return None
            self._diag("lazy_guard_done", component=component.component_id, model_step=mi, transition=t.get("label"), elapsed_seconds=_guard_elapsed, referenced_object_attributes=sorted(attrs))
            constraints.append(guard)
            relevant_attrs.update(attrs)

            pre_arcs = [a for a in net._arcs if a.get("target") == t["id"]]
            post_arcs = [a for a in net._arcs if a.get("source") == t["id"]]
            pre_places = {a["source"] for a in pre_arcs}
            post_places = {a["target"] for a in post_arcs}

            if boundary is not None and boundary_global_data:
                token_read_names = {
                    str(n) for a in pre_arcs for n, typ in a.get("inscription", [])
                    if typ in net._data_types
                }
                explicit_reads = {self._base_var(str(n)) for n in t.get("read", [])}
                for name in explicit_reads - token_read_names:
                    if name in step_data and name in boundary_global_data:
                        constraints.append(zsolver.eq(step_data[name], boundary_global_data[name]))

            # Validate and consume selected input tokens. Data members carried
            # by a token are equated with the transition's corresponding data
            # variables before the token is removed.
            consumed: List[Tuple[int, Tuple[str, ...]]] = []
            for arc in pre_arcs:
                place = next(p for p in net._places if p["id"] == arc["source"])
                toks = self._lazy_arc_tokens(net, place, arc.get("inscription", []), binding)
                if not toks:
                    self.stats["lazy_path_fallbacks"] += 1
                    return None
                data_names = [n for n, typ in arc.get("inscription", []) if typ in net._data_types]
                for tok in toks:
                    if tok not in marking[place["id"]]:
                        self.stats["lazy_path_fallbacks"] += 1
                        return None
                    if data_names:
                        stored = token_data.get((place["id"], tok))
                        if stored is None or any(name not in stored for name in data_names):
                            self.stats["lazy_path_fallbacks"] += 1
                            return None
                        for name in data_names:
                            constraints.append(zsolver.eq(step_data[name], stored[name]))
                    consumed.append((place["id"], tok))
            for pid, tok in consumed:
                marking[pid].discard(tok)
                token_data.pop((pid, tok), None)

            # Produce output tokens and attach only the data fields actually
            # present in their inscriptions.
            for arc in post_arcs:
                place = next(p for p in net._places if p["id"] == arc["target"])
                toks = self._lazy_arc_tokens(net, place, arc.get("inscription", []), binding)
                if not toks:
                    self.stats["lazy_path_fallbacks"] += 1
                    return None
                data_names = [n for n, typ in arc.get("inscription", []) if typ in net._data_types]
                for tok in toks:
                    marking[place["id"]].add(tok)
                    if data_names:
                        token_data[(place["id"], tok)] = {n: step_data[n] for n in data_names}
                        for n in data_names:
                            data_provenance[n] = (step_data[n], mi, "token-write")

            if boundary is not None:
                explicit_writes = {self._base_var(str(n)) for n in t.get("write", [])}
                for arc in post_arcs:
                    explicit_writes.update(
                        str(n) for n, typ in arc.get("inscription", []) if typ in net._data_types
                    )
                for name in explicit_writes:
                    if name in step_data:
                        boundary_global_data[name] = step_data[name]

            if move.kind == "model" and not t.get("invisible", False):
                objective_terms.append(zsolver.num(len(fixed_objs)))
            elif move.kind == "sync":
                e = component.observation_formula.events[move.event_id]
                # Event attributes and object-attribute observations are two
                # different perspectives.  Object updates in the same stream
                # unit must not be folded into the event-value comparison.
                observed = dict(e.attributes)
                for name, value in observed.items():
                    if name not in step_data:
                        continue
                    cv = self._coerce(value)
                    if cv is None:
                        continue
                    val = zsolver.real(cv) if data_types.get(name) in ("Real", "Rational") else zsolver.num(cv)
                    objective_terms.append(zsolver.ite(zsolver.eq(step_data[name], val), zsolver.num(0), zsolver.num(1)))

            bindings_out.append({"step": mi, "transition_id": t["id"], "transition": t.get("label"), "objects": tuple(sorted(fixed_objs))})
            data_out.append({"step": mi, "values": step_data})
            mi += 1

        # Add each relevant observed object-property equality exactly once.
        object_attr_meta = []
        attrs_now = component.observation_formula.current_object_attributes()
        try:
            from dpn.expr import Expr
        except Exception:
            Expr = None
        for obj in sorted(attrs_now):
            for attr, observed in sorted(attrs_now[obj].items()):
                if attr not in relevant_attrs:
                    continue
                name = self._object_attr_var_name(obj, attr)
                var = zsolver.realvar(name)
                attr_vars[(obj, attr)] = var
                if isinstance(observed, bool):
                    encoded = int(observed)
                elif isinstance(observed, (int, float)):
                    encoded = observed
                elif isinstance(observed, str) and Expr is not None:
                    encoded = Expr.numval(observed)
                else:
                    continue
                if (obj, attr) in boundary_object_attrs:
                    # The prefix already paid (or did not pay) for this global
                    # model-side object attribute.  Keep that certified choice
                    # fixed across the sliced suffix and do not charge it twice.
                    bv = boundary_object_attrs[(obj, attr)]
                    if isinstance(bv, bool):
                        bv = int(bv)
                    constraints.append(zsolver.eq(var, zsolver.real(bv)))
                else:
                    objective_terms.append(zsolver.ite(zsolver.eq(var, zsolver.real(encoded)), zsolver.num(0), zsolver.num(1)))
                object_attr_meta.append((obj, attr, observed, var))

        zsolver.ctx.add(*constraints)
        objective = self._sum(zsolver, objective_terms)
        max_cost = max(1, sum(len(component.observation_formula.events[e].objects) for e in node.event_order) + len(moves) * max(1, len(component.objects)) + len(objective_terms) + 3)
        if upper_bound_hint is not None:
            if int(upper_bound_hint) < int(lower_bound):
                # The lazy encoding is a relaxation of the eager fixed-path
                # encoding.  If even its admissible lower bound is already
                # above the strict improvement cap, eager JODAP cannot help.
                self.stats["strict_improvement_pre_solve_pruned"] += 1
                self._last_lazy_pruned = True
                return None
            max_cost = min(int(max_cost), int(upper_bound_hint))
        build_ms = int(round(1000.0 * (time.perf_counter() - build_start)))
        self.stats["lazy_path_build_seconds_ms"] += build_ms
        if boundary is not None:
            self.stats["provenance_sliced_jodap_build_seconds_ms"] += build_ms

        solve_start = time.perf_counter()
        self._diag(
            "lazy_solver_start", component=component.component_id, node=node.node_id,
            constraint_count=len(constraints), objective_terms=len(objective_terms),
            max_cost=max_cost, lower_bound=lower_bound, upper_bound_hint=upper_bound_hint,
            build_seconds=build_ms / 1000.0,
        )
        # Monotone integer minimization, using the predecessor optimum as lower bound.
        lb = max(0, int(lower_bound))
        best_model = None
        best_cost = None
        zsolver.ctx.push(); zsolver.ctx.add(objective <= lb)
        if zsolver.ctx.check() == z3.sat:
            best_model, best_cost = zsolver.ctx.model(), lb
        zsolver.ctx.pop()
        if best_model is None:
            zsolver.ctx.push(); zsolver.ctx.add(objective <= max_cost)
            if zsolver.ctx.check() != z3.sat:
                zsolver.ctx.pop()
                self.stats["lazy_path_solve_seconds_ms"] += int(round(1000.0 * (time.perf_counter() - solve_start)))
                if upper_bound_hint is not None:
                    # UNSAT in a relaxation under the strict improvement cap
                    # proves that the exact eager fixed-path query cannot beat
                    # the current incumbent either.
                    self.stats["strict_improvement_lazy_unsat_pruned"] += 1
                    self._last_lazy_pruned = True
                return None
            zsolver.ctx.pop()
            lo, hi = lb + 1, max_cost
            while lo < hi:
                mid = (lo + hi) // 2
                zsolver.ctx.push(); zsolver.ctx.add(objective <= mid)
                sat = zsolver.ctx.check() == z3.sat
                zsolver.ctx.pop()
                if sat: hi = mid
                else: lo = mid + 1
            zsolver.ctx.push(); zsolver.ctx.add(objective == lo)
            if zsolver.ctx.check() != z3.sat:
                zsolver.ctx.pop(); return None
            best_model, best_cost = zsolver.ctx.model(), lo
            zsolver.ctx.pop()
        solve_ms = int(round(1000.0 * (time.perf_counter() - solve_start)))
        self._diag("lazy_solver_done", component=component.component_id, node=node.node_id, elapsed_seconds=solve_ms / 1000.0, best_cost=best_cost)
        self.stats["lazy_path_solve_seconds_ms"] += solve_ms
        if boundary is not None:
            self.stats["provenance_sliced_jodap_solve_seconds_ms"] += solve_ms

        # Materialize symbolic values before the temporary solver is discarded.
        decoded_data = []
        for item in data_out:
            vals = {}
            for name, var in item["values"].items():
                try:
                    vals[name] = self._eval_real(best_model, var) if data_types.get(name) in ("Real", "Rational") else self._eval_int(best_model, var)
                except Exception:
                    pass
            decoded_data.append({"step": item["step"], "values": vals})
        object_attr_out = []
        for obj, attr, observed, var in object_attr_meta:
            try:
                mv = self._eval_real(best_model, var)
                if isinstance(observed, int) and not isinstance(observed, bool): mv = int(round(mv))
                elif isinstance(observed, bool): mv = bool(round(mv))
                mismatch = mv != observed
                object_attr_out.append({"object": obj, "attribute": attr, "observed_value": observed, "model_value": mv, "mismatch": mismatch, "cost": int(mismatch)})
            except Exception:
                pass
        marking_sig = tuple(sorted((pid, tok) for pid, toks in marking.items() for tok in toks))

        # Materialize the *current* token-carried state, not merely the data
        # variables of the last transition. This is what makes later local
        # checks able to read, e.g., d written by ``place order`` after several
        # unrelated ``pick item`` moves.
        token_data_sig = []
        current_data: Dict[str, Any] = {}
        provenance_sig = []
        for (pid, tok), fields in token_data.items():
            decoded_fields = []
            for name, var in fields.items():
                try:
                    value = self._eval_real(best_model, var) if data_types.get(name) in ("Real", "Rational") else self._eval_int(best_model, var)
                except Exception:
                    continue
                decoded_fields.append((name, self._freeze_value(value)))
                # If several live tokens carry the same global/data name with
                # different values, do not pretend there is a unique global
                # value; token-level provenance remains available.
                if name not in current_data:
                    current_data[name] = value
                elif current_data[name] != value:
                    current_data.pop(name, None)
            token_data_sig.append((pid, tok, tuple(sorted(decoded_fields))))

        for name, (var, src_step, src_kind) in data_provenance.items():
            try:
                value = self._eval_real(best_model, var) if data_types.get(name) in ("Real", "Rational") else self._eval_int(best_model, var)
                provenance_sig.append((name, self._freeze_value(value), int(src_step), str(src_kind)))
                current_data.setdefault(name, value)
            except Exception:
                pass

        # Event-only/global values that are not carried by a live token can
        # still be retained from the last selected model step.
        if decoded_data:
            for name, value in decoded_data[-1]["values"].items():
                current_data.setdefault(name, value)
        data_sig = tuple(sorted((name, self._freeze_value(value)) for name, value in current_data.items()))
        if boundary is not None:
            bindings_out = list(boundary.object_bindings or ()) + bindings_out
            decoded_data = list(boundary.data_assignments or ()) + decoded_data
            merged_oa = {(x.get("object"), x.get("attribute")): dict(x)
                         for x in (boundary.object_attribute_assignments or ())}
            for x in object_attr_out:
                merged_oa[(x.get("object"), x.get("attribute"))] = x
            object_attr_out = list(merged_oa.values())

        result = JointAssignment(total_cost=float(best_cost), object_bindings=bindings_out,
                                 data_assignments=decoded_data,
                                 object_attribute_assignments=object_attr_out,
                                 marking_signature=marking_sig,
                                 data_state_signature=data_sig,
                                 token_data_signature=tuple(sorted(token_data_sig, key=repr)),
                                 data_provenance_signature=tuple(sorted(provenance_sig, key=repr)),
                                 solve_seconds=(time.perf_counter() - solve_start),
                                 encode_seconds=(time.perf_counter() - build_start))
        if boundary is not None:
            # This temporary solver starts from a concrete certified marking and
            # simulates only the selected suffix.  Unsupported symbolic model
            # bindings return earlier; therefore every result reaching here is a
            # concrete feasible sliced witness.  The focused A* caller still
            # accepts it as globally final only when it reaches an independent
            # lower bound.
            self.stats["provenance_sliced_jodap_hits"] += 1
            return result

        # The lazy encoder is used as a proof of *no additional deviation*.
        # If the candidate needs a larger cost than the predecessor lower bound,
        # delegate it to the exact JODAP encoding.  This keeps the common fitting
        # path cheap while all repairing/re-optimising paths retain full semantics.
        if result.total_cost > float(lower_bound) + 1e-9:
            # A fully reconstructed lazy witness is a *feasible upper bound*
            # even when its cost is positive.  This is useful for online repair:
            # when an independent prefix lower-bound proof reaches the same
            # value, we can terminate without constructing the eager CoCoMoT
            # universe.  We still keep the default exact-fallback semantics for
            # ordinary JODAP calls.
            if accept_certified_nonzero:
                if self._certify_lazy_assignment(component, moves, trans_by_id, result):
                    self.stats["lazy_path_hits"] += 1
                    return result
                self.stats["lazy_certification_failures"] += 1

            # The lazy fixed-path model is intentionally a relaxation of the
            # exact JODAP path constraints.  Its optimum is therefore a lower
            # bound for the exact cost of this concrete candidate.  If even this
            # lower bound is already *strictly* worse than a feasible incumbent
            # for the current prefix, constructing the eager CoCoMoT context can
            # never improve the result.
            ub = self._current_prefix_upper_bound(component, state)
            if ub < float("inf") and result.total_cost > ub + 1e-9:
                self.stats["nonzero_pruned_before_fallback"] += 1
                self._last_lazy_pruned = True
                return None
            self.stats["lazy_nonzero_fallbacks"] += 1
            self.stats["lazy_path_fallbacks"] += 1
            return None
        if not self._certify_lazy_assignment(component, moves, trans_by_id, result):
            self.stats["lazy_certification_failures"] += 1
            self.stats["lazy_path_fallbacks"] += 1
            return None
        self.stats["lazy_path_hits"] += 1
        return result

    def _query_slice_static_net(self):
        if self._static_net is None:
            self._static_net = self.new_net()
        return self._static_net

    def _query_slice_boundary(self, component: ComponentState, state: SearchState,
                              node: SearchNode) -> Optional[Tuple[int, JointAssignment]]:
        """Choose a certified ancestor boundary for a query-local provenance slice.

        This hook lives immediately before an eager/full JODAP query.  It uses
        provenance of values read by the latest selected transition to avoid
        fixing a writer that may need to be reconsidered.  If no scalar writer
        is relevant (for example a guard over retained object attributes), the
        nearest certified ancestor is the smallest safe speculative boundary.

        The boundary-fixed result is never assumed exact merely because it is
        feasible; callers accept it without the eager query only when it reaches
        the independent lower bound.  Otherwise it is used only as an upper
        bound for the exact full-path minimization.
        """
        if self.provenance_slicing != "focus" or state.boundary_assignment is not None:
            return None
        if node.node_id not in state.predecessor:
            return None

        # Walk from the candidate towards the root and collect certified
        # ancestors.  The immediate parent is normally already evaluated by A*.
        ancestors: List[int] = []
        cur = node.node_id
        while cur in state.predecessor:
            prev, _mv = state.predecessor[cur]
            if prev in state.assignments_by_node:
                ancestors.append(prev)
            cur = prev
        if not ancestors:
            if state.current_goal is not None and state.current_assignment is not None:
                ancestors.append(state.current_goal)
            else:
                return None

        nearest_id = ancestors[0]
        nearest_assn = state.assignments_by_node.get(nearest_id)
        if nearest_assn is None and state.current_goal == nearest_id:
            nearest_assn = state.current_assignment
        if nearest_assn is None:
            return None

        # Determine model-side values read by the latest selected transition.
        # Input-arc data are reads even when the PNML read list is incomplete.
        moves = self._path(state, node.node_id)
        last_model = next((m for m in reversed(moves)
                           if m.kind in ("model", "sync") and m.transition_id is not None), None)
        read_names: Set[str] = set()
        if last_model is not None:
            try:
                net = self._query_slice_static_net()
                t = next(t for t in net._transitions if int(t["id"]) == int(last_model.transition_id))
                read_names.update(self._base_var(str(x)) for x in t.get("read", []))
                for arc in net._arcs:
                    if arc.get("target") != t["id"]:
                        continue
                    for name, typ in arc.get("inscription", []):
                        if typ in net._data_types:
                            read_names.add(self._base_var(str(name)))
            except Exception:
                read_names.clear()

        provenance = {}
        for item in nearest_assn.data_provenance_signature or ():
            if len(item) >= 4:
                name, _value, src_step, _kind = item[:4]
                try:
                    provenance[self._base_var(str(name))] = int(src_step)
                except Exception:
                    pass
        source_steps = [provenance[n] for n in read_names if n in provenance and provenance[n] >= 0]

        if source_steps:
            # source step k is the k-th selected model move; a boundary at model
            # depth <= k lies before that write and permits the sliced suffix to
            # reconsider it.
            target_depth = min(source_steps)
            candidates = []
            for aid in ancestors:
                an = state.nodes.get(aid)
                aa = state.assignments_by_node.get(aid)
                if aa is None and state.current_goal == aid:
                    aa = state.current_assignment
                if an is not None and aa is not None and an.model_depth <= target_depth:
                    candidates.append((an.model_depth, aid, aa))
            if candidates:
                _depth, aid, aa = max(candidates, key=lambda x: x[0])
                return aid, aa

        # Static object-attribute dependencies and already-certified values can
        # be fixed at the immediate certified parent.  This gives the smallest
        # suffix and is only a speculative candidate/upper bound unless it hits
        # the independent lower bound.
        return nearest_id, nearest_assn

    def _query_slice_state(self, state: SearchState, node: SearchNode,
                           boundary_id: int, boundary_assn: JointAssignment) \
            -> Optional[Tuple[SearchState, SearchNode]]:
        """Clone only the candidate chain after ``boundary_id`` into a suffix state."""
        chain: List[Tuple[int, SymbolicMove]] = []
        cur = node.node_id
        while cur != boundary_id:
            if cur not in state.predecessor:
                return None
            prev, mv = state.predecessor[cur]
            chain.append((cur, mv))
            cur = prev
        chain.reverse()

        bnode = state.nodes.get(boundary_id)
        if bnode is None:
            return None
        focused = SearchState(state.component_id)
        focused.model_bound = state.model_bound
        focused.search_offline = state.search_offline
        focused.current_event_ids = state.current_event_ids
        focused.current_objects = state.current_objects
        focused.current_attribute_observations = state.current_attribute_observations
        focused.current_object_attribute_snapshot = state.current_object_attribute_snapshot
        focused.proven_prefix_lower_bound = state.proven_prefix_lower_bound
        # Preserve only the numeric/complete-prefix incumbent information needed
        # for strict cheaper-than-UB certification.  Its move sequence remains
        # in global coordinates, which is intentional: _current_prefix_upper_bound
        # validates it against the complete observed prefix, not the suffix root.
        focused.upper_bound = state.upper_bound
        focused.incumbent_moves = tuple(state.incumbent_moves)
        focused.incumbent_assignment = state.incumbent_assignment
        focused.incumbent_offline = state.incumbent_offline
        focused.boundary_assignment = boundary_assn
        focused.boundary_prefix_moves = tuple(self._path(state, boundary_id))
        focused.boundary_model_depth = int(bnode.model_depth)
        focused.boundary_node_original = boundary_id

        root = _clone_search_node(bnode, node_id=0)
        root.g = float(boundary_assn.total_cost)
        root.assignment_cost = float(boundary_assn.total_cost)
        focused.nodes[0] = root
        focused.assignments_by_node[0] = boundary_assn
        prev_new = 0
        next_new = 1
        final_node = root
        for old_id, mv in chain:
            old = state.nodes.get(old_id)
            if old is None:
                return None
            nn = _clone_search_node(old, node_id=next_new)
            focused.nodes[next_new] = nn
            focused.predecessor[next_new] = (prev_new, mv)
            prev_new = next_new
            final_node = nn
            next_new += 1
        focused.next_id = next_new
        return focused, final_node

    def _persistent_checkpoint_state(self, state: SearchState, node: SearchNode) \
            -> Optional[Tuple[SearchState, SearchNode]]:
        """Build a suffix state from a node-carried certified checkpoint lineage.

        Unlike ``_query_slice_state`` this does not depend on the canonical
        predecessor chain reaching the original boundary node.  The post-boundary
        move sequence was propagated explicitly when the node was generated.
        """
        boundary = node.checkpoint_boundary_assignment
        snapshot = node.checkpoint_boundary_snapshot
        if boundary is None or snapshot is None:
            return None

        focused = SearchState(state.component_id)
        focused.model_bound = state.model_bound
        focused.search_offline = state.search_offline
        focused.current_event_ids = state.current_event_ids
        focused.current_objects = state.current_objects
        focused.current_attribute_observations = state.current_attribute_observations
        focused.current_object_attribute_snapshot = state.current_object_attribute_snapshot
        focused.proven_prefix_lower_bound = state.proven_prefix_lower_bound
        focused.upper_bound = state.upper_bound
        focused.incumbent_moves = tuple(state.incumbent_moves)
        focused.incumbent_assignment = state.incumbent_assignment
        focused.incumbent_offline = state.incumbent_offline
        focused.boundary_assignment = boundary
        focused.boundary_prefix_moves = tuple(node.checkpoint_boundary_prefix_moves)
        focused.boundary_model_depth = int(node.checkpoint_boundary_model_depth)
        focused.boundary_node_original = None

        root = _clone_search_node(snapshot, node_id=0)
        root.checkpoint_boundary_assignment = None
        root.checkpoint_boundary_snapshot = None
        root.checkpoint_boundary_model_depth = 0
        root.checkpoint_boundary_prefix_moves = ()
        root.checkpoint_suffix_moves = ()
        root.g = float(boundary.total_cost)
        root.assignment_cost = float(boundary.total_cost)
        focused.nodes[0] = root
        focused.assignments_by_node[0] = boundary

        prev = root
        prev_id = 0
        next_id = 1
        for mv in tuple(node.checkpoint_suffix_moves):
            consumed = prev.consumed
            event_order = prev.event_order
            model_depth = int(prev.model_depth)
            model_sig = prev.model_signature
            move_sig = prev.move_signature + ((mv.kind, mv.event_id, mv.transition_id),)
            if mv.kind in ("log", "sync"):
                if mv.event_id is None:
                    return None
                consumed = frozenset(set(consumed) | {mv.event_id})
                event_order = event_order + (mv.event_id,)
            if mv.kind in ("model", "sync"):
                model_depth += 1
                model_sig = model_sig + (mv.transition_id,)
            nn = SearchNode(
                next_id, consumed, event_order, model_depth,
                g=float("inf"), h=0.0, assignment_cost=0.0,
                model_signature=model_sig, move_signature=move_sig,
            )
            focused.nodes[next_id] = nn
            focused.predecessor[next_id] = (prev_id, mv)
            prev = nn
            prev_id = next_id
            next_id += 1
        focused.next_id = next_id
        return focused, prev

    def _try_persistent_checkpoint_context(
            self, component: ComponentState, state: SearchState, node: SearchNode,
            *, lower_bound: int = 0,
            canonical_fresh_bindings: Optional[Dict[int, str]] = None,
            upper_bound_hint: Optional[int] = None
    ) -> Tuple[Optional[JointAssignment], Optional[int]]:
        """Try an eager checkpoint-relative query using node-carried lineage.

        The result is globally final only if it reaches the caller's independent
        lower bound.  Otherwise it is an upper bound and the ordinary full-prefix
        exact query remains the correctness fallback.
        """
        built = self._persistent_checkpoint_state(state, node)
        if built is None:
            return None, None
        focused, final_node = built
        self.stats["persistent_checkpoint_context_attempts"] += 1
        self._diag(
            "persistent_checkpoint_context_attempt",
            component=component.component_id, node=node.node_id,
            absolute_model_depth=int(node.model_depth),
            boundary_model_depth=int(focused.boundary_model_depth),
            checkpoint_relative_depth=max(0, int(node.model_depth) - int(focused.boundary_model_depth)),
            suffix_moves=len(node.checkpoint_suffix_moves),
        )
        result = self.solve(
            component, focused, final_node, require_final=False,
            lower_bound=lower_bound,
            canonical_fresh_bindings=canonical_fresh_bindings,
            _force_checkpoint_eager=True,
        )
        if result is None:
            self.stats["persistent_checkpoint_context_fallbacks"] += 1
            return None, None
        ub = max(int(lower_bound), int(math.ceil(result.total_cost - 1e-9)))
        if result.total_cost <= float(lower_bound) + 1e-9:
            self.stats["persistent_checkpoint_context_proven"] += 1
            self._diag(
                "persistent_checkpoint_context_proven",
                component=component.component_id, node=node.node_id,
                total_cost=float(result.total_cost), lower_bound=int(lower_bound),
            )
            return result, ub
        self.stats["persistent_checkpoint_context_upper_bounds"] += 1
        return None, ub

    def _nearest_certified_checkpoint(self, state: SearchState, node: SearchNode) \
            -> Optional[Tuple[int, JointAssignment]]:
        """Return the nearest certified ancestor of ``node``.

        This is a solver-independent incremental checkpoint.  Unlike provenance
        slicing it does not claim that the prefix before the checkpoint can never
        need revision.  It is used only to solve the concrete suffix cheaply: a
        result is accepted as globally optimal only when it reaches the caller's
        independent lower bound; otherwise it is merely an incumbent for the
        complete eager query.
        """
        if state.boundary_assignment is not None or node.node_id not in state.predecessor:
            return None
        cur = node.node_id
        while cur in state.predecessor:
            prev, _move = state.predecessor[cur]
            assn = state.assignments_by_node.get(prev)
            if assn is None and state.current_goal == prev:
                assn = state.current_assignment
            if assn is not None:
                return prev, assn
            cur = prev
        return None

    def _checkpoint_min_cost_binding_candidates(
            self, component: ComponentState, boundary: JointAssignment,
            transition_id: int, *, max_candidates: int = 32
    ) -> List[Tuple[str, ...]]:
        """Return cheap concrete bindings for one model-only suffix move.

        This is deliberately *not* a complete binding enumerator.  It generates
        only very small bindings (one object for each scalar parameter and, when
        present, one object for each LIST parameter).  Such a witness is sufficient
        to prove the candidate optimum when its JODAP
        cost reaches the independently computed structural lower bound.  If no
        witness reaches that bound, callers fall back to the unrestricted exact
        encoding, so completeness is unchanged.

        Candidate objects are taken only from tokens currently marked at the
        transition's input places.  This makes the common repair move (for
        example ``pick item``) linear in the number of marked objects and avoids
        constructing the absolute-depth eager context merely to discover a
        minimum-cost binding.
        """
        try:
            net = self._query_slice_static_net()
            transition = next(t for t in net._transitions
                              if int(t.get("id")) == int(transition_id))
        except Exception:
            return []
        if transition.get("invisible", False):
            # Fresh/silent creation is already handled by canonical-fresh logic;
            # other zero-cost silent bindings need the ordinary exact fallback.
            return []

        decls = self._lazy_unique_object_decls(net, transition)
        if not decls or any("nu" in str(name) for name in decls):
            return []

        # The current lazy concrete binder is exact only when there is at most
        # one logical parameter of a given base type.  Do not invent a choice in
        # genuinely ambiguous same-type scalar/list signatures.
        by_base: Dict[str, List[Tuple[str, str]]] = {}
        for name, typ in decls.items():
            base = typ[:typ.rfind(" LIST")] if "LIST" in typ else typ
            by_base.setdefault(str(base), []).append((str(name), str(typ)))
        if any(len(v) != 1 for v in by_base.values()):
            return []

        marked_by_place: Dict[Any, Set[str]] = {}
        known = set(component.objects)
        for item in boundary.marking_signature or ():
            try:
                pid, token = item[0], tuple(item[1])
            except Exception:
                continue
            bucket = marked_by_place.setdefault(pid, set())
            for value in token:
                if value in known:
                    bucket.add(str(value))

        input_arcs = [a for a in net._arcs if a.get("target") == transition["id"]]
        pools: List[List[str]] = []
        for name, typ in decls.items():
            base = typ[:typ.rfind(" LIST")] if "LIST" in typ else typ
            relevant_sets: List[Set[str]] = []
            for arc in input_arcs:
                mentions = any(str(n) == str(name) and str(t) == str(typ)
                               for n, t in arc.get("inscription", ()))
                if not mentions:
                    continue
                vals = {
                    obj for obj in marked_by_place.get(arc.get("source"), set())
                    if component.observation_formula.object_types.get(obj) == base
                }
                relevant_sets.append(vals)
            if not relevant_sets:
                return []
            candidates = set.intersection(*relevant_sets) if len(relevant_sets) > 1 else set(relevant_sets[0])
            if not candidates:
                return []
            pools.append(sorted(candidates))

        out: List[Tuple[str, ...]] = []
        seen: Set[Tuple[str, ...]] = set()
        for choice in itertools.product(*pools):
            # A concrete binding uses distinct object identities across logical
            # parameters; LIST parameters contribute exactly one object here.
            if len(set(choice)) != len(choice):
                continue
            candidate = tuple(sorted(str(x) for x in choice))
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= max(1, int(max_candidates)):
                break
        return out

    def _try_checkpoint_suffix(
            self, component: ComponentState, state: SearchState, node: SearchNode,
            *, lower_bound: int = 0,
            canonical_fresh_bindings: Optional[Dict[int, str]] = None,
            upper_bound_hint: Optional[int] = None
    ) -> Tuple[Optional[JointAssignment], Optional[int], bool]:
        """Try a boundary-state fixed-path solve before constructing eager JODAP.

        Returns ``(result, upper_bound_hint, strictly_closed)``.  The third
        value is true when the checkpoint-local relaxation has already proved
        that no suffix cheaper than the current incumbent exists.  That proof
        must terminate the candidate rather than falling through to the
        absolute-depth eager context.

        The retained ancestor assignment supplies the concrete marking, token
        data, provenance and already-paid cost.  Only moves after that checkpoint
        are encoded.  This makes the effective context depth relative to the
        latest certified online prefix instead of the candidate's absolute model
        depth.  Completeness is preserved because any non-proven result falls
        through to the ordinary full eager query.
        """
        chosen = self._nearest_certified_checkpoint(state, node)
        if chosen is None:
            return None, None, False
        boundary_id, boundary_assn = chosen
        built = self._query_slice_state(state, node, boundary_id, boundary_assn)
        if built is None:
            return None, None, False
        focused, final_node = built
        suffix_depth = max(0, int(final_node.model_depth) - int(focused.boundary_model_depth))
        self.stats["checkpoint_slice_attempts"] += 1
        self.stats["checkpoint_slice_model_steps"] += suffix_depth
        self._diag(
            "checkpoint_suffix_start", component=component.component_id, node=node.node_id,
            boundary_node=boundary_id, boundary_model_depth=focused.boundary_model_depth,
            candidate_model_depth=final_node.model_depth, suffix_model_depth=suffix_depth,
            lower_bound=lower_bound,
        )
        result = self._lazy_solve_fixed_path(
            component, focused, final_node, lower_bound=lower_bound,
            canonical_fresh_bindings=canonical_fresh_bindings,
            accept_certified_nonzero=True,
            upper_bound_hint=upper_bound_hint,
        )
        if result is None:
            if self._last_lazy_pruned:
                # The relaxed checkpoint suffix itself is UNSAT below the
                # incumbent.  The eager exact encoding is a restriction of that
                # relaxation, so this candidate cannot improve the known UB.
                self._diag(
                    "checkpoint_suffix_strict_improvement_unsat",
                    component=component.component_id, node=node.node_id,
                    lower_bound=lower_bound, upper_bound_hint=upper_bound_hint,
                )
                return None, None, True
            # The nearest certified parent normally leaves exactly the newly
            # appended move in this suffix.  When that move is a visible
            # model-only repair, the ordinary lazy encoder declines because its
            # object binding is symbolic.  Try minimum-cardinality bindings from
            # the certified parent marking.  Reaching ``lower_bound`` is a full
            # proof: no unrestricted binding can be cheaper than that independent
            # structural bound.  Failure to reach it remains only an incumbent
            # hint and falls through to the complete eager backend.
            suffix_moves = self._path(focused, final_node.node_id)
            best_bound_witness = None
            if len(suffix_moves) == 1 and suffix_moves[0].kind == "model" \
                    and suffix_moves[0].transition_id is not None:
                bindings = self._checkpoint_min_cost_binding_candidates(
                    component, boundary_assn, suffix_moves[0].transition_id
                )
                if bindings:
                    self.stats["checkpoint_bound_binding_attempts"] += 1
                fixed_step = int(focused.boundary_model_depth)
                for objects in bindings:
                    self.stats["checkpoint_bound_binding_candidates"] += 1
                    cand = self._lazy_solve_fixed_path(
                        component, focused, final_node, lower_bound=lower_bound,
                        canonical_fresh_bindings=canonical_fresh_bindings,
                        fixed_binding_by_step={fixed_step: objects},
                        accept_certified_nonzero=True,
                        upper_bound_hint=upper_bound_hint,
                    )
                    if cand is None:
                        continue
                    self.stats["checkpoint_bound_binding_hits"] += 1
                    if best_bound_witness is None or cand.total_cost < best_bound_witness.total_cost:
                        best_bound_witness = cand
                    if cand.total_cost <= float(lower_bound) + 1e-9:
                        self.stats["checkpoint_slice_hits"] += 1
                        self.stats["checkpoint_slice_proven"] += 1
                        self.stats["checkpoint_bound_binding_proven"] += 1
                        self._diag(
                            "checkpoint_bound_binding_proven",
                            component=component.component_id, node=node.node_id,
                            transition_id=suffix_moves[0].transition_id,
                            binding=list(objects), lower_bound=lower_bound,
                            total_cost=cand.total_cost,
                        )
                        return cand, max(int(lower_bound), int(math.ceil(cand.total_cost - 1e-9))), False
            if best_bound_witness is not None:
                self.stats["checkpoint_slice_hits"] += 1
                self.stats["checkpoint_slice_upper_bounds"] += 1
                ub = max(int(lower_bound), int(math.ceil(best_bound_witness.total_cost - 1e-9)))
                self._diag(
                    "checkpoint_bound_binding_upper_bound",
                    component=component.component_id, node=node.node_id,
                    suffix_model_depth=suffix_depth, total_cost=best_bound_witness.total_cost,
                )
                return None, ub, False
            # Last checkpoint-local attempt: exact eager suffix with a
            # checkpoint-specialized object universe.  This is still only an
            # upper bound for the unrestricted full-prefix problem unless it
            # reaches the independent lower bound.
            exact_suffix = self.solve(
                component, focused, final_node, require_final=False,
                lower_bound=lower_bound,
                canonical_fresh_bindings=canonical_fresh_bindings,
                _force_checkpoint_eager=True,
            )
            if exact_suffix is not None:
                self.stats["checkpoint_context_hits"] += 1
                ub = max(int(lower_bound), int(math.ceil(exact_suffix.total_cost - 1e-9)))
                if exact_suffix.total_cost <= float(lower_bound) + 1e-9:
                    self.stats["checkpoint_slice_hits"] += 1
                    self.stats["checkpoint_slice_proven"] += 1
                    self._diag("checkpoint_eager_suffix_proven",
                               component=component.component_id, node=node.node_id,
                               suffix_model_depth=suffix_depth, total_cost=exact_suffix.total_cost)
                    return exact_suffix, ub, False
                self.stats["checkpoint_slice_upper_bounds"] += 1
                self._diag("checkpoint_eager_suffix_upper_bound",
                           component=component.component_id, node=node.node_id,
                           suffix_model_depth=suffix_depth, total_cost=exact_suffix.total_cost)
                return None, ub, False
            self.stats["checkpoint_slice_fallbacks"] += 1
            self._diag("checkpoint_suffix_fallback", component=component.component_id,
                       node=node.node_id, suffix_model_depth=suffix_depth)
            return None, None, False
        self.stats["checkpoint_slice_hits"] += 1
        ub = max(int(lower_bound), int(math.ceil(result.total_cost - 1e-9)))
        if result.total_cost <= float(lower_bound) + 1e-9:
            self.stats["checkpoint_slice_proven"] += 1
            self._diag("checkpoint_suffix_proven", component=component.component_id,
                       node=node.node_id, suffix_model_depth=suffix_depth,
                       total_cost=result.total_cost)
            return result, ub, False
        self.stats["checkpoint_slice_upper_bounds"] += 1
        self._diag("checkpoint_suffix_upper_bound", component=component.component_id,
                   node=node.node_id, suffix_model_depth=suffix_depth,
                   total_cost=result.total_cost)
        # This is the real integration point for the checkpoint-specialized
        # eager context.  Previously a feasible-but-unproven lazy suffix was
        # returned only as an upper-bound hint, after which the caller rebuilt
        # the absolute-depth full-prefix context.  Force the exact eager suffix
        # here first; its instant 0 is the certified checkpoint and _prepare()
        # can therefore apply the reduced object domain safely.
        exact_suffix = self.solve(
            component, focused, final_node, require_final=False,
            lower_bound=lower_bound,
            canonical_fresh_bindings=canonical_fresh_bindings,
            _force_checkpoint_eager=True,
        )
        if exact_suffix is not None:
            exact_ub = max(int(lower_bound), int(math.ceil(exact_suffix.total_cost - 1e-9)))
            if exact_suffix.total_cost <= float(lower_bound) + 1e-9:
                self.stats["checkpoint_slice_proven"] += 1
                self._diag(
                    "checkpoint_eager_suffix_proven",
                    component=component.component_id, node=node.node_id,
                    suffix_model_depth=suffix_depth, total_cost=exact_suffix.total_cost,
                )
                return exact_suffix, exact_ub, False
            ub = min(ub, exact_ub)
            self._diag(
                "checkpoint_eager_suffix_upper_bound",
                component=component.component_id, node=node.node_id,
                suffix_model_depth=suffix_depth, total_cost=exact_suffix.total_cost,
            )
        return None, ub, False

    def _try_query_level_provenance_slice(
            self, component: ComponentState, state: SearchState, node: SearchNode,
            *, lower_bound: int = 0,
            canonical_fresh_bindings: Optional[Dict[int, str]] = None,
            upper_bound_hint: Optional[int] = None
    ) -> Tuple[Optional[JointAssignment], Optional[int]]:
        """Try the reduced boundary-state JODAP exactly where eager JODAP would start.

        Returns ``(proven_result, upper_bound_hint)``.  A result is proven only
        if its cost reaches the caller's independent lower bound.  Otherwise a
        feasible sliced witness supplies a safe upper bound to the subsequent
        full exact query.
        """
        chosen = self._query_slice_boundary(component, state, node)
        if chosen is None:
            return None, None
        boundary_id, boundary_assn = chosen
        built = self._query_slice_state(state, node, boundary_id, boundary_assn)
        if built is None:
            return None, None

        self.stats["provenance_slice_attempts"] += 1
        self.stats["provenance_query_slice_attempts"] += 1
        focused, final_node = built
        self.stats["provenance_slice_model_steps_reopened"] += max(
            0, int(final_node.model_depth) - int(focused.boundary_model_depth)
        )
        self.stats["provenance_sliced_jodap_queries"] += 1
        t0 = time.perf_counter()
        result = self._lazy_solve_fixed_path(
            component, focused, final_node, lower_bound=lower_bound,
            canonical_fresh_bindings=canonical_fresh_bindings,
            accept_certified_nonzero=True,
            upper_bound_hint=upper_bound_hint,
        )
        self.stats["provenance_slice_seconds_ms"] += int((time.perf_counter() - t0) * 1000)
        if result is None:
            self.stats["provenance_sliced_jodap_fallbacks"] += 1
            self.stats["provenance_query_slice_fallbacks"] += 1
            return None, None

        self.stats["provenance_slice_hits"] += 1
        self.stats["provenance_query_slice_hits"] += 1
        ub = max(int(lower_bound), int(math.ceil(result.total_cost - 1e-9)))
        if result.total_cost <= float(lower_bound) + 1e-9:
            self.stats["provenance_slice_proven"] += 1
            self.stats["provenance_query_slice_proven"] += 1
            return result, ub
        self.stats["provenance_query_slice_upper_bounds"] += 1
        return None, ub

    def solve(self, component: ComponentState, state: SearchState, node: SearchNode,
              *, require_final: bool = False, lower_bound: int = 0,
              focus_objects: Optional[Set[str]] = None,
              canonical_fresh_object: Optional[str] = None,
              canonical_fresh_bindings: Optional[Dict[int, str]] = None,
              _force_checkpoint_eager: bool = False) -> Optional[JointAssignment]:
        moves = self._path(state, node.node_id)

        # Once a complete feasible current-prefix incumbent exists, exact search
        # no longer needs another solution at the same cost.  Its only remaining
        # task is the decision problem "does any alignment cost strictly less
        # than UB exist?"  Alignment costs are integral, hence UB-1 is the exact
        # inclusive feasibility cap.  This derives solely from the incumbent and
        # objective; it does not use mutation counts or look ahead.
        strict_improvement_cap: Optional[int] = None
        if not require_final:
            incumbent_ub = self._current_prefix_upper_bound(component, state)
            if incumbent_ub < float("inf"):
                incumbent_ub_int = int(math.ceil(incumbent_ub - 1e-9))
                strict_improvement_cap = incumbent_ub_int - 1
                self.stats["strict_improvement_mode_entries"] += 1
                if int(lower_bound) > strict_improvement_cap:
                    self.stats["strict_improvement_lower_bound_closed"] += 1
                    self._diag(
                        "strict_improvement_candidate_closed",
                        component=component.component_id, node=node.node_id,
                        lower_bound=int(lower_bound), incumbent=float(incumbent_ub),
                        improvement_cap=int(strict_improvement_cap),
                    )
                    return None
        boundary_mode = state.boundary_assignment is not None
        if boundary_mode and not _force_checkpoint_eager:
            self.stats["provenance_sliced_jodap_queries"] += 1
            sliced = self._lazy_solve_fixed_path(
                component, state, node, lower_bound=lower_bound,
                canonical_fresh_bindings=canonical_fresh_bindings,
                accept_certified_nonzero=True,
                upper_bound_hint=strict_improvement_cap,
            )
            if sliced is not None:
                return sliced
            # The concrete lazy suffix cannot represent every symbolic LIST or
            # model-only binding.  Instead of abandoning the checkpoint and
            # rebuilding an absolute-depth context, continue below with an exact
            # eager context whose instant 0 is this certified boundary.
            self.stats["provenance_sliced_jodap_fallbacks"] += 1
            self.stats["checkpoint_context_fallbacks"] += 1
        elif boundary_mode and _force_checkpoint_eager:
            self._diag(
                "checkpoint_context_forced_at_fallback",
                component=component.component_id, node=node.node_id,
                boundary_model_depth=int(state.boundary_model_depth),
            )

        # Prefer the genuinely lazy path encoder whenever every model-side
        # binding is already determined by synchronization or explicit fresh
        # creation. It avoids constructing CoCoMoT's eager object-token universe.
        if (not _force_checkpoint_eager) and not require_final and (canonical_fresh_bindings or all(m.kind in ("log", "sync") for m in moves)):
            lazy = self._lazy_solve_fixed_path(
                component, state, node, lower_bound=lower_bound,
                canonical_fresh_bindings=canonical_fresh_bindings,
                upper_bound_hint=strict_improvement_cap,
            )
            if lazy is not None:
                return lazy
            if self._last_lazy_pruned:
                # Candidate is already dominated by a feasible current-prefix
                # incumbent; exact fallback would only refine a losing cost.
                return None

        # Before constructing the expensive full CoCoMoT context, solve the
        # concrete suffix from the nearest certified checkpoint.  This is the
        # normal incremental path and is independent of the optional provenance
        # ablation.  A checkpoint result is final only when it reaches an
        # independent lower bound; otherwise it is a safe upper-bound hint for
        # the complete exact query.
        provenance_upper_bound = None
        # Prefer the boundary descriptor propagated directly on the A* node.
        # This survives canonical predecessor reuse and is therefore available
        # on the real hard-case eager path where nearest-ancestor discovery may
        # otherwise report boundary depth zero.
        if (not require_final and state.boundary_assignment is None
                and node.checkpoint_boundary_assignment is None and not state.search_offline):
            self.stats["persistent_checkpoint_lineage_missing_on_candidate"] += 1
            self._diag(
                "persistent_checkpoint_lineage_missing_on_candidate",
                component=component.component_id, node=node.node_id,
                model_depth=int(node.model_depth), consumed_events=len(node.consumed),
                move_count=len(moves),
            )

        if (not require_final and state.boundary_assignment is None
                and node.checkpoint_boundary_assignment is not None):
            persistent_exact, persistent_ub = self._try_persistent_checkpoint_context(
                component, state, node, lower_bound=lower_bound,
                canonical_fresh_bindings=canonical_fresh_bindings,
                upper_bound_hint=strict_improvement_cap,
            )
            if persistent_exact is not None:
                return persistent_exact
            if persistent_ub is not None:
                provenance_upper_bound = persistent_ub

        if not require_final and state.boundary_assignment is None:
            checkpoint_exact, checkpoint_upper_bound, checkpoint_closed = self._try_checkpoint_suffix(
                component, state, node, lower_bound=lower_bound,
                canonical_fresh_bindings=canonical_fresh_bindings,
                upper_bound_hint=strict_improvement_cap,
            )
            if checkpoint_exact is not None:
                return checkpoint_exact
            if checkpoint_closed:
                self.stats["strict_improvement_pre_solve_pruned"] += 1
                self._diag(
                    "checkpoint_suffix_closed_candidate",
                    component=component.component_id, node=node.node_id,
                    lower_bound=int(lower_bound), upper_bound_hint=strict_improvement_cap,
                )
                return None
            if checkpoint_upper_bound is not None:
                provenance_upper_bound = (checkpoint_upper_bound if provenance_upper_bound is None
                                          else min(provenance_upper_bound, checkpoint_upper_bound))

        # Optional provenance slicing may reopen an earlier defining write than
        # the nearest checkpoint.  It remains an ablation/repair mechanism and
        # can further tighten the incumbent before the full query.
        if self.provenance_slicing == "focus" and state.boundary_assignment is None and not require_final:
            sliced_exact, provenance_ub = self._try_query_level_provenance_slice(
                component, state, node, lower_bound=lower_bound,
                canonical_fresh_bindings=canonical_fresh_bindings,
                upper_bound_hint=strict_improvement_cap,
            )
            if sliced_exact is not None:
                return sliced_exact
            if provenance_ub is not None:
                provenance_upper_bound = (provenance_ub if provenance_upper_bound is None
                                          else min(provenance_upper_bound, provenance_ub))

        # A one-step repair or inherited merge alignment may already provide a
        # feasible explanation for the *current* prefix. Pass that incumbent to
        # every eager JODAP minimization, not only to A*'s outer pruning logic.
        # The old code used only checkpoint/provenance-derived hints here, so
        # diagnostics could still show upper_bound_hint=None after a proven
        # cost-1 repair and hundreds of dominated exact queries followed.
        incumbent_ub = self._current_prefix_upper_bound(component, state)
        if incumbent_ub < float("inf"):
            incumbent_ub_int = int(math.ceil(incumbent_ub - 1e-9))
            # Strict incumbent certification: another solution of cost UB is
            # irrelevant because a feasible UB witness is already known.
            # Search only <= UB-1.
            improvement_cap = incumbent_ub_int - 1
            provenance_upper_bound = (
                improvement_cap if provenance_upper_bound is None
                else min(int(provenance_upper_bound), improvement_cap)
            )
            self.stats["strict_improvement_eager_queries"] += 1

        context_key = self._context_key(component, state, node.event_order)
        focus_key = tuple(sorted(focus_objects)) if focus_objects else ()
        fresh_key = tuple(sorted((canonical_fresh_bindings or {}).items()))
        qkey = (component.component_id, context_key, tuple(node.move_signature),
                bool(require_final), focus_key, canonical_fresh_object, fresh_key,
                provenance_upper_bound)
        if qkey in self._query_cache:
            self.stats["query_cache_hits"] += 1
            cached = self._query_cache.pop(qkey)
            self._query_cache[qkey] = cached
            return cached
        self.stats["queries"] += 1
        self.stats["eager_fallback_queries"] += 1
        last_move = moves[-1] if moves else None
        last_event = None
        if last_move is not None and last_move.event_id is not None:
            last_event = component.observation_formula.events.get(last_move.event_id)
        self._diag(
            "eager_query_start", component=component.component_id, node=node.node_id,
            model_depth=node.model_depth, move_count=len(moves), lower_bound=lower_bound,
            last_move=(last_move.kind if last_move else None),
            activity=(last_event.activity if last_event is not None else getattr(last_move, "transition_label", None)),
            event_id=(last_event.event_id if last_event is not None else None),
            event_objects=(len(last_event.objects) if last_event is not None else 0),
            component_objects=len(component.objects), component_events=len(component.execution.event_ids),
        )
        _prepare_t0 = time.perf_counter()
        boundary_offset = int(state.boundary_model_depth) if boundary_mode else 0
        desired_depth = max(0, int(node.model_depth) - boundary_offset)
        prepared = self._prepare(
            component, state, node.event_order, required_depth=max(1, desired_depth),
            node=node, canonical_fresh_bindings=canonical_fresh_bindings,
        )
        if prepared is None:
            return None
        self._diag("eager_context_ready", component=component.component_id, node=node.node_id, elapsed_seconds=time.perf_counter()-_prepare_t0, capacity=prepared.capacity, checkpoint_specialized=bool(prepared.checkpoint_boundary is not None))
        solver = prepared.solver
        encoding = prepared.encoding
        net = prepared.net
        try:
            nu_transition_ids = {t["id"] for t in net.nu_transitions()}
        except Exception:
            nu_transition_ids = set()

        if desired_depth > encoding._step_bound:
            return None

        objective_terms = ([solver.num(int(round(state.boundary_assignment.total_cost)))]
                           if boundary_mode and state.boundary_assignment is not None else [])
        path_constraints = []
        model_steps: List[Tuple[int, SymbolicMove]] = []
        relevant_object_attrs: Set[str] = set()
        available_object_attrs = {
            attr
            for values in component.observation_formula.current_object_attributes().values()
            for attr in values
        }
        mi = 0

        def num_objects_used(step: int):
            return self._sum(solver, [
                solver.ite(solver.eq(v, solver.num(-1)), solver.num(0), solver.num(1))
                for v in encoding._object_vars[step]
            ])

        for move in moves:
            if move.kind == "log":
                e = component.observation_formula.events[move.event_id]
                objective_terms.append(solver.num(len(e.objects)))
                continue

            assert move.transition_id is not None
            if mi >= len(encoding._transition_vars):
                return None
            tid = move.transition_id
            path_constraints.append(
                solver.eq(encoding._transition_vars[mi], solver.num(tid))
            )
            t = prepared.trans_by_id[tid]
            model_steps.append((mi, move))

            # Symmetry breaking for independent fresh-object creation.  When the
            # caller supplies the canonical object for the *newly appended* nu
            # transition, bind only the nu parameter to that object.  Other
            # transition parameters (if any) remain symbolic.  This prevents
            # factorial permutations of equivalent creation orders before they
            # ever reach the expensive optimization query.
            fresh_obj = None
            global_mi = mi + boundary_offset
            if canonical_fresh_bindings and global_mi in canonical_fresh_bindings:
                fresh_obj = canonical_fresh_bindings[global_mi]
            elif canonical_fresh_object is not None and mi == desired_depth - 1:
                fresh_obj = canonical_fresh_object
            if fresh_obj is not None and tid in nu_transition_ids:
                oid = encoding._id_by_object_name.get(fresh_obj)
                if oid is None:
                    return None
                try:
                    params = net.object_params_of_transition(t, encoding._objects)
                    nu_params = [p for p in params if "nu" in str(p.get("name", ""))]
                except Exception:
                    nu_params = []
                if len(nu_params) == 1:
                    k = int(nu_params[0]["index"])
                    path_constraints.append(
                        solver.eq(encoding._object_vars[mi][k], solver.num(oid))
                    )

            relevant_object_attrs.update(
                self._guard_object_attribute_names(t.get("constraint"), available_object_attrs)
            )

            if move.kind == "model":
                if not t.get("invisible", False):
                    objective_terms.append(num_objects_used(mi))
                # Focus only the newly appended model move. Fresh-object (nu)
                # transitions are exempt because their purpose is precisely to
                # introduce an object that may not yet occur in the immediate
                # observed frontier. The caller falls back to the unrestricted
                # query unless this focused query reaches the predecessor lower
                # bound, so this is an optimization rather than a semantic cut.
                if focus_objects and mi == desired_depth - 1 and tid not in nu_transition_ids:
                    allowed_ids = [encoding._id_by_object_name[o]
                                   for o in sorted(focus_objects)
                                   if o in encoding._id_by_object_name]
                    if allowed_ids:
                        for ov in encoding._object_vars[mi]:
                            path_constraints.append(solver.lor(
                                [solver.eq(ov, solver.num(-1))] +
                                [solver.eq(ov, solver.num(oid)) for oid in allowed_ids]
                            ))
            elif move.kind == "sync":
                e = component.observation_formula.events[move.event_id]
                ids = []
                for obj in e.objects:
                    if obj not in encoding._id_by_object_name:
                        return None
                    ids.append(encoding._id_by_object_name[obj])
                path_constraints.append(
                    solver.eq(num_objects_used(mi), solver.num(len(ids)))
                )
                for oid in ids:
                    path_constraints.append(solver.lor([
                        solver.eq(v, solver.num(oid)) for v in encoding._object_vars[mi]
                    ]))

                observed = dict(e.attributes)
                unit = next((u for u in component.units
                             if u.event and u.event.event_id == e.event_id), None)
                if unit is not None:
                    for upd in unit.attribute_updates:
                        observed.setdefault(upd.attribute, upd.value)
                for name, value in observed.items():
                    if name not in prepared.data_types or name not in encoding._data_vars[mi]:
                        continue
                    cv = self._coerce(value)
                    if cv is None:
                        continue
                    typ = prepared.data_types[name]
                    valexpr = solver.real(cv) if typ in ("Rational", "Real") else solver.num(cv)
                    objective_terms.append(solver.ite(
                        solver.eq(encoding._data_vars[mi][name], valexpr),
                        solver.num(0), solver.num(1)
                    ))
            mi += 1

        # Object-attribute observations are soft only for properties referenced
        # by guards on transitions actually selected on this candidate path.
        encoded_object_domain = set(map(str, getattr(encoding, "_objects", {}).keys())) \
            if isinstance(getattr(encoding, "_objects", None), dict) else None
        object_attr_terms, object_attr_meta = self._observed_object_attribute_terms(
            component, solver, relevant_object_attrs, allowed_objects=encoded_object_domain
        )
        objective_terms.extend(object_attr_terms)
        self.stats["object_attribute_soft_terms"] += len(object_attr_terms)

        path_constraints.append(
            solver.eq(encoding._run_length_var, solver.num(desired_depth))
        )
        if require_final:
            if prepared.final_formula is None:
                prepared.final_formula = encoding.final_state()
            path_constraints.append(prepared.final_formula)

        objective = self._sum(solver, objective_terms)
        max_cost = (
            sum(len(component.observation_formula.events[e].objects) for e in node.event_order)
            + max(1, encoding.get_max_objs_per_trans()) * max(1, node.model_depth)
            + sum(len(component.observation_formula.events[e].attributes) for e in node.event_order)
            + sum(len(u.attribute_updates) for u in component.units if u.event is not None)
            + len(object_attr_terms)
            + 5
        )

        # Candidate-specific scope. The base encoding remains permanently
        # asserted and is reused by subsequent sibling/descendant candidates.
        ctx = solver.ctx
        ctx.push()
        ctx.add(*path_constraints)
        query_encode_start = time.perf_counter()
        self._diag(
            "eager_solver_start", component=component.component_id, node=node.node_id,
            path_constraints=len(path_constraints), objective_terms=len(objective_terms),
            model_steps=len(model_steps), max_cost=max_cost, lower_bound=lower_bound,
            upper_bound_hint=provenance_upper_bound,
        )
        model, total_cost, solve_seconds = self._minimize_incremental(
            prepared, objective, max_cost, lower_bound=lower_bound,
            upper_bound_hint=provenance_upper_bound,
        )
        query_encode_seconds = time.perf_counter() - query_encode_start
        self._diag(
            "eager_solver_done", component=component.component_id, node=node.node_id,
            elapsed_seconds=query_encode_seconds, solver_seconds=solve_seconds, total_cost=total_cost,
            feasible=(model is not None and total_cost is not None),
        )
        if model is None or total_cost is None:
            ctx.pop()
            if strict_improvement_cap is not None:
                self.stats["strict_improvement_eager_unsat"] += 1
            self._query_cache[qkey] = None
            while len(self._query_cache) > self.max_query_cache:
                self._query_cache.popitem(last=False)
            return None

        bindings: List[Dict[str, Any]] = []
        data_vals: List[Dict[str, Any]] = []
        for step, move in model_steps:
            t = prepared.trans_by_id[move.transition_id]
            obj_names = []
            for v in encoding._object_vars[step]:
                oid = self._eval_int(model, v)
                if oid in encoding._object_name_by_id:
                    obj_names.append(encoding._object_name_by_id[oid])
            bindings.append({
                "step": step + boundary_offset,
                "transition_id": move.transition_id,
                "transition": t.get("label"),
                "objects": tuple(obj_names),
            })
            vals = {}
            if net.has_data():
                for name, var in encoding._data_vars[step].items():
                    typ = prepared.data_types.get(name)
                    try:
                        vals[name] = (self._eval_real(model, var)
                                      if typ in ("Rational", "Real")
                                      else self._eval_int(model, var))
                    except Exception:
                        pass
            data_vals.append({"step": step + boundary_offset, "values": vals})

        object_attr_vals: List[Dict[str, Any]] = []
        for obj, attr, observed, encoded, var in object_attr_meta:
            try:
                model_value_num = self._eval_real(model, var)
                # Preserve the user-facing observed type where possible.
                if isinstance(observed, bool):
                    model_value = bool(round(model_value_num))
                elif isinstance(observed, int) and not isinstance(observed, bool):
                    model_value = int(round(model_value_num))
                elif isinstance(observed, float):
                    model_value = float(model_value_num)
                elif isinstance(observed, str):
                    try:
                        from dpn.expr import Expr
                        model_value = Expr.strval(int(round(model_value_num)))
                    except Exception:
                        model_value = model_value_num
                else:
                    model_value = model_value_num
                mismatch = (model_value != observed)
                object_attr_vals.append({
                    "object": obj,
                    "attribute": attr,
                    "observed_value": observed,
                    "model_value": model_value,
                    "mismatch": bool(mismatch),
                    "cost": 1 if mismatch else 0,
                })
            except Exception:
                pass

        # Decode the concrete operational state selected by this optimum.
        # Markings are available at instants 0..step_bound; desired_depth is the
        # marking after the last model/synchronous move of this candidate.
        marking_signature: List[Any] = []
        try:
            import z3
            mvars = encoding._marking_vars[desired_depth]
            for pid in sorted(mvars):
                for tok, var in mvars[pid].items():
                    if z3.is_true(model.eval(var, model_completion=True)):
                        marking_signature.append((pid, self._freeze_value(tok)))
        except Exception:
            marking_signature = []
        if boundary_mode and prepared.checkpoint_static_marking:
            # Tokens containing only objects outside the sliced suffix universe
            # cannot be consumed/produced by the fixed suffix (otherwise their
            # objects would have been retained by the domain closure). Preserve
            # them verbatim in the certified successor state.
            marking_signature.extend(prepared.checkpoint_static_marking)
            marking_signature = list(dict.fromkeys(marking_signature))

        # CoCoMoT stores data variables per model step.  The last model-step
        # valuation is the current process-execution data state for the state
        # dominance optimization.
        data_state_signature: List[Any] = []
        if data_vals:
            last_vals = data_vals[-1].get("values", {})
            data_state_signature = sorted(
                (name, self._freeze_value(value)) for name, value in last_vals.items()
            )

        ctx.pop()
        token_data_signature = tuple(prepared.checkpoint_static_token_data or ())
        data_provenance_signature = ()
        if boundary_mode and state.boundary_assignment is not None:
            boundary = state.boundary_assignment
            bindings = list(boundary.object_bindings or ()) + bindings
            data_vals = list(boundary.data_assignments or ()) + data_vals
            merged_oa = {(x.get("object"), x.get("attribute")): dict(x)
                         for x in (boundary.object_attribute_assignments or ())}
            for x in object_attr_vals:
                merged_oa[(x.get("object"), x.get("attribute"))] = x
            object_attr_vals = list(merged_oa.values())
            # Relevant live-token data inside the sliced universe are
            # represented in the exact SMT store but stock CoCoMoT does not
            # expose a name-stable decoder for every positional token field.
            # Do not carry potentially stale prefix provenance across suffix
            # writes: an empty provenance signature makes later local delta
            # checks fall back conservatively to exact solving.
            data_provenance_signature = ()

        result = JointAssignment(
            total_cost=float(total_cost),
            object_bindings=bindings,
            data_assignments=data_vals,
            object_attribute_assignments=object_attr_vals,
            marking_signature=tuple(sorted(marking_signature, key=repr)),
            data_state_signature=tuple(data_state_signature),
            token_data_signature=token_data_signature,
            data_provenance_signature=data_provenance_signature,
            solve_seconds=solve_seconds,
            # The expensive base encoding cost is charged only when the context
            # was created. This field mainly measures per-query construction.
            encode_seconds=query_encode_seconds,
        )
        self._query_cache[qkey] = result
        while len(self._query_cache) > self.max_query_cache:
            self._query_cache.popitem(last=False)
        return result


    @staticmethod
    def aggregate_list_guard_spec(transition: Dict[str, Any]) -> Optional[Tuple[str, str, str, float]]:
        """Recognize a scalar-object attribute equality against a LIST aggregate.

        Returns ``(attribute, scalar_var, list_var, offset)`` for guards of the
        exact forms used by the order-management DOPID, e.g.::

            weight(p) == sum(weight(I))
            price(o) == sum(price(I)) + 5.0

        Unsupported guards return ``None``.  This helper is syntax recognition
        only; callers must still validate concrete values with the independent
        guard evaluator before pruning anything.
        """
        constraint = transition.get("constraint")
        if constraint is None:
            return None
        text = re.sub(r"\s+", "", str(constraint))
        patterns = (
            re.compile(r"^\(([A-Za-z_]\w*)\(([A-Za-z_]\w*)\)==sum\(\1\(([A-Za-z_]\w*)\)\)\)$"),
            re.compile(r"^\(([A-Za-z_]\w*)\(([A-Za-z_]\w*)\)==\(sum\(\1\(([A-Za-z_]\w*)\)\)([+-]\d+(?:\.\d+)?)\)\)$"),
        )
        for idx, pattern in enumerate(patterns):
            match = pattern.match(text)
            if not match:
                continue
            offset = 0.0
            if idx == 1:
                try:
                    offset = float(match.group(4))
                except Exception:
                    return None
            return match.group(1), match.group(2), match.group(3), offset
        return None

    def evaluate_guard_for_concrete_object_binding(
            self, component: ComponentState, transition: Dict[str, Any],
            binding: Dict[str, Any]) -> Optional[bool]:
        """Evaluate an object-attribute-only guard for one concrete binding.

        ``True``/``False`` are returned only when the existing independent lazy
        evaluator can decide the complete guard from immutable observed object
        attributes.  Guards depending on process data, unsupported functions, or
        unresolved values return ``None`` and therefore cannot be used for
        pruning.  This makes the helper a safe filter for specialized relation
        repair enumeration while exact JODAP remains authoritative.
        """
        constraint = transition.get("constraint")
        if constraint is None or not isinstance(binding, dict):
            return None
        object_values = {
            (str(o), str(a)): v
            for o, vals in component.observation_formula.current_object_attributes().items()
            for a, v in vals.items()
        }
        try:
            value = self._lazy_eval_expr(constraint, {}, binding, object_values)
        except Exception:
            return None
        return value if isinstance(value, bool) else None

    def infer_single_object_attribute_guard_repair(
            self, component: ComponentState, transition: Dict[str, Any],
            binding: Dict[str, Any]
    ) -> Optional[Dict[Tuple[str, str], Any]]:
        """Infer one concrete object-attribute repair for a simple aggregate guard.

        The controlled order-management data mutations use guards of the form
        ``weight(p) == sum(weight(I))`` (and, more generally,
        ``attr(x) == sum(attr(L)) +/- c``).  When the observed object binding is
        already fixed and all list-member attributes are concrete, the unique
        model-side value for the scalar object's attribute can be computed
        without solving the historical prefix again.

        This helper is deliberately narrow.  Unsupported/multi-variable guards
        return ``None`` and the exact JODAP fallback remains responsible for
        them.
        """
        constraint = transition.get("constraint")
        if constraint is None or not isinstance(binding, dict):
            return None
        spec = self.aggregate_list_guard_spec(transition)
        if spec is None:
            return None
        attr, scalar_var, list_var, offset = spec
        scalar_obj = binding.get(scalar_var)
        list_objs = binding.get(list_var)
        if isinstance(scalar_obj, (list, tuple, set)) or scalar_obj is None:
            return None
        if not isinstance(list_objs, (list, tuple, set)) or not list_objs:
            return None

        observed_oa = component.observation_formula.current_object_attributes()
        values = []
        for obj in list_objs:
            value = observed_oa.get(str(obj), {}).get(attr)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append(float(value))
        desired = float(sum(values) + offset)
        scalar_obj = str(scalar_obj)
        observed_scalar = observed_oa.get(scalar_obj, {}).get(attr)
        if not isinstance(observed_scalar, (int, float)) or isinstance(observed_scalar, bool):
            return None
        # If it already matches, this is not a data-deviation repair.
        if math.isclose(float(observed_scalar), desired, rel_tol=1e-9, abs_tol=1e-9):
            return None

        # Re-evaluate with exactly one model-side override.  This guards the
        # algebraic shortcut against parser/guard-shape mistakes.
        model_oa = {
            (str(o), str(a)): v
            for o, vals in observed_oa.items() for a, v in vals.items()
        }
        model_oa[(scalar_obj, attr)] = desired
        guard_ok = self._lazy_eval_expr(constraint, {}, binding, model_oa)
        if guard_ok is not True:
            return None
        return {(scalar_obj, attr): desired}


    def marking_feasible_list_domain(
            self, component: ComponentState, assignment: JointAssignment,
            transition: Dict[str, Any], observed_objects: Sequence[str],
            candidate_list_objects: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Return the zero-cost marking-feasible domain of one LIST parameter.

        This is a conservative pre-binding reduction for synchronous/relation
        fast paths.  It is enabled only when the transition has exactly one
        logical LIST parameter and every input arc using that LIST can be
        evaluated with the concrete scalar binding induced by ``observed_objects``.
        Each LIST member expands independently in ``_lazy_arc_tokens``; therefore
        an object is feasible iff all input tokens induced by binding the LIST to
        that singleton are present in the certified parent marking.

        ``None`` means unsupported/unknown and callers must retain the old path.
        The helper never considers repair effects; it describes *zero-cost*
        synchronization only.
        """
        self.stats["marking_domain_reduction_attempts"] += 1
        try:
            net = self._query_slice_static_net()
            nt = next(t for t in net._transitions if t.get("id") == transition.get("id"))
            decls = self._lazy_unique_object_decls(net, nt)
            list_vars = [(n, typ) for n, typ in decls.items() if "LIST" in str(typ)]
            if len(list_vars) != 1:
                self.stats["marking_domain_reduction_fallbacks"] += 1
                return None
            list_var, list_type = list_vars[0]
            base_type = str(list_type)[:str(list_type).rfind(" LIST")]
            base_binding = self._lazy_binding(component, net, nt, tuple(sorted(map(str, observed_objects))))
            if base_binding is None or not isinstance(base_binding.get(list_var), list):
                self.stats["marking_domain_reduction_fallbacks"] += 1
                return None

            relevant_arcs = []
            for arc in net._arcs:
                if arc.get("target") != nt.get("id"):
                    continue
                obj_entries = [(n, typ) for n, typ in arc.get("inscription", ()) if typ not in net._data_types]
                if list_var not in {n for n, _typ in obj_entries}:
                    continue
                # More than one LIST variable on an input arc would couple the
                # per-object domains; leave that case to the exact fallback.
                if sum(1 for _n, typ in obj_entries if "LIST" in str(typ)) != 1:
                    self.stats["marking_domain_reduction_fallbacks"] += 1
                    return None
                try:
                    place = next(pl for pl in net._places if pl.get("id") == arc.get("source"))
                except StopIteration:
                    self.stats["marking_domain_reduction_fallbacks"] += 1
                    return None
                relevant_arcs.append((place, arc))
            if not relevant_arcs:
                # No input marking constrains the LIST variable.
                self.stats["marking_domain_reduction_supported"] += 1
                cand = {str(o) for o in candidate_list_objects
                        if component.observation_formula.object_types.get(str(o)) == base_type}
                return {"list_var": list_var, "base_type": base_type,
                        "feasible": cand, "infeasible": set()}

            marked = set(assignment.marking_signature or ())
            if not marked:
                self.stats["marking_domain_reduction_fallbacks"] += 1
                return None

            bound_list = {str(o) for o in base_binding.get(list_var, ())}
            candidates = {str(o) for o in candidate_list_objects
                          if str(o) in bound_list
                          and component.observation_formula.object_types.get(str(o)) == base_type}
            feasible, infeasible = set(), set()
            for obj in sorted(candidates):
                binding = dict(base_binding)
                binding[list_var] = [obj]
                ok = True
                for place, arc in relevant_arcs:
                    req = self._lazy_arc_tokens(net, place, arc.get("inscription", ()), binding)
                    if not req:
                        self.stats["marking_domain_reduction_fallbacks"] += 1
                        return None
                    for tok in req:
                        key = (place["id"], self._freeze_value(tok))
                        if key not in marked:
                            ok = False
                            break
                    if not ok:
                        break
                (feasible if ok else infeasible).add(obj)

            self.stats["marking_domain_reduction_supported"] += 1
            self.stats["marking_domain_objects_pruned"] += len(infeasible)
            self._diag(
                "marking_domain_reduction", component=component.component_id,
                transition_id=transition.get("id"), transition=transition.get("label"),
                list_var=str(list_var), base_type=str(base_type),
                candidate_count=len(candidates), feasible_count=len(feasible),
                infeasible_count=len(infeasible),
                infeasible_objects=sorted(infeasible),
            )
            return {"list_var": list_var, "base_type": base_type,
                    "feasible": feasible, "infeasible": infeasible}
        except Exception:
            self.stats["marking_domain_reduction_fallbacks"] += 1
            return None


    def check_zero_cost_sync_extension(
            self, component: ComponentState, state: SearchState,
            parent: SearchNode, child: SearchNode, event: StreamEvent,
            transition_id: int, parent_assignment: JointAssignment, *,
            model_object_attribute_overrides: Optional[Dict[Tuple[str, str], Any]] = None,
            model_objects_override: Optional[Sequence[str]] = None,
            extra_cost: float = 0.0, repair_tag: Optional[str] = None
    ) -> Optional[JointAssignment]:
        """Certify one synchronous extension from the retained optimal witness.

        This is deliberately *not* a solve of the complete alignment prefix.  A
        previous optimal witness already certifies all historical moves.  For a
        candidate ``(e,t)`` we therefore inspect only the new firing:

        * ``t`` must be enabled in the retained object-aware marking;
        * the event objects must form exactly one concrete binding of ``t``;
        * the guard must hold when newly observed event values are used as the
          model-side values (zero new event-data deviation);
        * model-side object properties already chosen by the parent are kept,
          while a property that becomes relevant for the first time is fixed to
          its observed value (zero new object-data deviation).

        If any required value cannot be decided locally we return ``None``.  The
        caller then resumes ordinary A*+JODAP, which is free to re-optimise old
        assignments.  Hence this routine is only a sufficient zero-cost proof;
        it never removes a feasible alignment.
        """
        self.stats["zero_cost_delta_checks"] += 1
        self.stats["delta_local_guard_checks"] += 1
        self._last_delta_unknown = False
        self._last_zero_cost_sync_decline_reason = None
        self._last_zero_cost_sync_decline_detail = None

        transition_label = None
        def decline(reason: str, *, stat: Optional[str] = None, unknown: bool = False, **extra):
            self.stats["zero_cost_sync_declines"] += 1
            self._last_zero_cost_sync_decline_reason = reason
            self._last_zero_cost_sync_decline_detail = {
                "reason": reason, "transition_id": transition_id,
                "transition": transition_label, **extra
            }
            if stat is not None and stat in self.stats:
                self.stats[stat] += 1
            if unknown:
                self._last_delta_unknown = True
            payload = {
                "component": component.component_id,
                "event_id": event.event_id,
                "activity": event.activity,
                "transition_id": transition_id,
                "transition": transition_label,
                "parent_node": parent.node_id,
                "parent_model_depth": parent.model_depth,
                "reason": reason,
            }
            payload.update(extra)
            self._diag("zero_cost_sync_decline", **payload)
            return None

        def safe_binding_repr(binding_value):
            if not isinstance(binding_value, dict):
                return None
            out = {}
            for k, v in binding_value.items():
                out[str(k)] = list(v) if isinstance(v, (list, tuple, set)) else v
            return out

        if child.model_depth != parent.model_depth + 1:
            return decline("model_depth_not_single_extension", expected=parent.model_depth + 1, actual=child.model_depth)
        if child.event_order != parent.event_order + (event.event_id,):
            return decline("event_order_not_single_extension")

        try:
            net = self.new_net()
            transition = next(t for t in net._transitions if t["id"] == transition_id)
        except Exception as exc:
            return decline("transition_lookup_failed", error=repr(exc))
        transition_label = transition.get("label")
        if transition_label != event.activity:
            return decline("activity_mismatch", transition_label=transition_label)

        # Ordinarily a synchronous move uses exactly the observed object set.
        # A certified object-relation repair may deliberately use a model-side
        # set that differs by one relation; the observation itself is retained
        # unchanged and the mismatch is paid via ``extra_cost``.
        binding_objects = tuple(sorted(
            model_objects_override if model_objects_override is not None else event.objects
        ))
        binding = self._lazy_binding(component, net, transition, binding_objects)
        if binding is None:
            try:
                decls = self._lazy_unique_object_decls(net, transition)
            except Exception:
                decls = {}
            return decline(
                "object_binding_failed", stat="zero_cost_sync_binding_failures",
                event_objects=list(event.objects), model_objects=list(binding_objects), object_types={
                    str(o): component.observation_formula.object_types.get(o) for o in set(event.objects) | set(binding_objects)
                }, logical_parameters={str(k): str(v) for k, v in decls.items()},
            )
        self._diag(
            "zero_cost_sync_binding", component=component.component_id, event_id=event.event_id,
            activity=event.activity, transition_id=transition_id, transition=transition_label,
            binding=safe_binding_repr(binding),
        )

        # ------------------------------------------------------------------
        # 1. Local object-flow enablement and successor marking.
        # ------------------------------------------------------------------
        marked = set(parent_assignment.marking_signature)
        if not marked and parent.model_depth > 0:
            # A parent produced by an unsupported/eager witness may not expose a
            # concrete marking.  Do not pretend that the move is enabled.
            self.stats["delta_dependency_fallbacks"] += 1
            return decline("parent_marking_unavailable", stat="zero_cost_sync_marking_failures", unknown=True)

        # Retain token-carried values from the certified parent witness.  This
        # is the crucial historical dependency store: a value written by an
        # earlier transition remains attached to the live token even if several
        # unrelated transitions have fired since then.
        parent_token_data: Dict[Tuple[int, Tuple[str, ...]], Dict[str, Any]] = {}
        for item in parent_assignment.token_data_signature:
            try:
                pid, tok, fields = item
                parent_token_data[(int(pid), self._freeze_value(tok))] = dict(fields)
            except Exception:
                continue

        parent_provenance: Dict[str, Tuple[Any, int, str]] = {}
        for item in parent_assignment.data_provenance_signature:
            try:
                name, value, src_step, src_kind = item
                parent_provenance[str(name)] = (value, int(src_step), str(src_kind))
            except Exception:
                continue

        # Current certified model-side values are a secondary dependency store.
        # They are especially useful for DPN-style execution variables which do
        # not necessarily live on a token.  Token-local values remain preferred
        # whenever an input inscription names the variable.
        parent_current_data: Dict[str, Any] = {}
        for item in parent_assignment.data_state_signature:
            try:
                name, value = item
                parent_current_data[str(name)] = value
            except Exception:
                continue

        consumed: List[Tuple[int, Tuple[str, ...]]] = []
        pre_data: Dict[str, Any] = {}
        input_data_names: Set[str] = set()
        for arc in (a for a in net._arcs if a.get("target") == transition_id):
            try:
                place = next(p for p in net._places if p["id"] == arc["source"])
            except StopIteration:
                return decline("input_place_missing", stat="zero_cost_sync_marking_failures", arc_source=arc.get("source"))
            req = self._lazy_arc_tokens(net, place, arc.get("inscription", []), binding)
            if not req:
                return decline(
                    "input_arc_binding_empty", stat="zero_cost_sync_binding_failures",
                    place_id=place.get("id"), place_name=place.get("name", place.get("label")),
                    inscription=[(str(n), str(tp)) for n, tp in arc.get("inscription", [])],
                    binding=safe_binding_repr(binding),
                )
            data_names = [n for n, typ in arc.get("inscription", []) if typ in net._data_types]
            input_data_names.update(data_names)
            for tok in req:
                key = (place["id"], self._freeze_value(tok))
                if key not in marked:
                    return decline(
                        "required_input_token_not_marked", stat="zero_cost_sync_marking_failures",
                        place_id=place.get("id"), place_name=place.get("name", place.get("label")),
                        required_token=repr(key), marked_token_count=len(marked),
                        required_objects=sorted(self._objects_in_value(key[1], set(component.objects))),
                        binding=safe_binding_repr(binding),
                    )
                if data_names:
                    self.stats["delta_token_lookup_attempts"] += len(data_names)
                    fields = parent_token_data.get(key)
                    if fields is None:
                        self.stats["delta_token_lookup_misses"] += len(data_names)
                        # An eager/legacy witness may know the marking but not
                        # the token-carried values. Exact JODAP remains the safe
                        # fallback in that case.
                        self.stats["delta_dependency_fallbacks"] += 1
                        return decline(
                            "token_data_missing", stat="zero_cost_sync_token_data_failures", unknown=True,
                            place_id=place.get("id"), token=repr(key), data_names=list(data_names),
                        )
                    for name in data_names:
                        if name not in fields:
                            self.stats["delta_token_lookup_misses"] += 1
                            self.stats["delta_dependency_fallbacks"] += 1
                            return decline(
                                "token_data_field_missing", stat="zero_cost_sync_token_data_failures", unknown=True,
                                place_id=place.get("id"), token=repr(key), data_name=name,
                            )
                        value = fields[name]
                        if name in pre_data and pre_data[name] != value:
                            self.stats["delta_dependency_fallbacks"] += 1
                            return decline(
                                "conflicting_token_data", stat="zero_cost_sync_token_data_failures", unknown=True,
                                data_name=name, previous_value=pre_data.get(name), token_value=value,
                            )
                        pre_data[name] = value
                        self.stats["delta_token_data_hits"] += 1
                        self.stats["delta_historical_reads"] += 1
                consumed.append(key)

        successor_marking = set(marked)
        successor_token_data = {k: dict(v) for k, v in parent_token_data.items()}
        for key in consumed:
            successor_marking.discard(key)
            successor_token_data.pop(key, None)

        output_arcs = [a for a in net._arcs if a.get("source") == transition_id]
        output_data_names: Set[str] = {
            n for a in output_arcs for n, typ in a.get("inscription", [])
            if typ in net._data_types
        }

        # ------------------------------------------------------------------
        # 2. Incremental/versioned data witness.
        # ------------------------------------------------------------------
        # Use only values with an explicit provenance as historical global/data
        # state. ``data_state_signature`` may contain arbitrary values for
        # variables irrelevant to the last transition and is therefore not a
        # safe dependency source on its own.
        new_data: Dict[str, Any] = dict(parent_current_data)
        # Explicit provenance and token-carried reads override a merely current
        # scalar snapshot because they identify the actual historical source.
        new_data.update({name: value for name, (value, _, _) in parent_provenance.items()})
        new_data.update(pre_data)
        new_provenance = dict(parent_provenance)

        # Event/model equality is soft in general, but this routine proves only
        # a *zero-cost* extension. A read value carried by an input token is
        # therefore compared with the observation rather than overwritten by it.
        # Attributes that are newly written by this transition can take the
        # observed value directly for a zero-cost candidate.
        step = parent.model_depth
        for name, observed in event.attributes.items():
            if name in input_data_names:
                if name not in pre_data:
                    self.stats["delta_dependency_fallbacks"] += 1
                    return decline(
                        "observed_read_without_token_value", stat="zero_cost_sync_token_data_failures",
                        unknown=True, data_name=name, observed_value=observed,
                    )
                if pre_data[name] != observed:
                    # Structurally valid synchronous move, but it requires a
                    # data deviation. Let exact JODAP compare it against other
                    # possible alignments.
                    return decline(
                        "event_token_data_mismatch", stat="zero_cost_sync_data_mismatches",
                        data_name=name, model_value=pre_data[name], observed_value=observed,
                    )
                new_data[name] = pre_data[name]
                self.stats["delta_provenance_hits"] += 1
            else:
                new_data[name] = observed
                new_provenance[name] = (observed, step, "event-write")

        # Explicit DPN-style write declarations are global process-execution
        # variables. If a written variable has no event value, a zero-cost local
        # proof can retain an already certified value; otherwise the model must
        # choose a new value and exact JODAP is required.
        declared_writes = {self._base_var(v) for v in transition.get("write", [])}
        for name in declared_writes:
            if name in event.attributes:
                new_provenance[name] = (event.attributes[name], step, "process-write")
            elif name not in new_data:
                self.stats["delta_dependency_fallbacks"] += 1
                return decline(
                    "undetermined_process_write", stat="zero_cost_sync_token_data_failures",
                    unknown=True, data_name=name,
                )

        observed_oa = component.observation_formula.current_object_attributes()
        model_oa: Dict[Tuple[str, str], Any] = {
            (str(o), str(a)): v
            for o, vals in observed_oa.items() for a, v in vals.items()
        }
        prior_oa: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in parent_assignment.object_attribute_assignments:
            key = (str(item.get("object")), str(item.get("attribute")))
            prior_oa[key] = dict(item)
            # A previously paid deviation is part of the retained witness and
            # must not silently be reset to the observation.
            model_oa[key] = item.get("model_value")
        # A guarded data-repair fast path may provide one concrete model-side
        # object-attribute value that differs from the observation.  Overrides
        # are applied after retained deviations so the new transition can keep
        # the observed event/object identity while paying exactly one soft data
        # mismatch instead of reopening the complete prefix search.
        for key, value in (model_object_attribute_overrides or {}).items():
            model_oa[(str(key[0]), str(key[1]))] = value

        # Evaluate only the newly selected guard.  This is the dependency test:
        # if the guard references a value not available in the retained witness
        # (or uses an unsupported expression), local certification declines and
        # full JODAP may revise historical values.
        relevant_attrs: Set[str] = set()
        if "constraint" in transition:
            available = {a for vals in observed_oa.values() for a in vals}
            relevant_attrs = self._guard_object_attribute_names(
                transition.get("constraint"), available
            )
            # Pull scalar/process variables required by the newly selected guard
            # from the retained certified state before declaring the dependency
            # unknown. This is the wiring missing in the previous provenance
            # implementation: values were stored but guard reads never asked for
            # them explicitly.
            try:
                guard_vars = {self._base_var(v) for v in transition["constraint"].vars()}
            except Exception:
                guard_vars = set()
            for name in sorted(guard_vars):
                if name in new_data:
                    continue
                self.stats["delta_global_lookup_attempts"] += 1
                if name in parent_provenance:
                    new_data[name] = parent_provenance[name][0]
                    self.stats["delta_provenance_hits"] += 1
                    self.stats["delta_historical_reads"] += 1
                elif name in parent_current_data:
                    new_data[name] = parent_current_data[name]
                    self.stats["delta_provenance_hits"] += 1
                    self.stats["delta_historical_reads"] += 1
                else:
                    self.stats["delta_global_lookup_misses"] += 1
            guard_ok = self._lazy_eval_expr(
                transition["constraint"], new_data, binding, model_oa
            )
            guard_objects = []
            for value in binding.values():
                if isinstance(value, (list, tuple, set)):
                    guard_objects.extend(str(x) for x in value)
                elif value is not None:
                    guard_objects.append(str(value))
            guard_objects = sorted(set(guard_objects))
            attr_snapshot = {
                obj: {attr: model_oa.get((obj, attr)) for attr in sorted(relevant_attrs)
                      if (obj, attr) in model_oa}
                for obj in guard_objects
            }
            # Diagnostic aggregates must respect logical guard bindings.  In
            # particular, weight(p) and sum(weight(I)) use the same attribute
            # name but p must not be included in the aggregate over I.
            aggregate_snapshot = {}
            for var_name, bound in binding.items():
                if not isinstance(bound, (list, tuple, set)):
                    continue
                objs = [str(x) for x in bound]
                for attr in sorted(relevant_attrs):
                    vals = [model_oa[(obj, attr)] for obj in objs if (obj, attr) in model_oa]
                    if not vals:
                        continue
                    numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    aggregate_snapshot[f"{attr}({var_name})"] = {
                        "objects": objs,
                        "values": vals,
                        "numeric_sum": sum(numeric) if len(numeric) == len(vals) else None,
                    }
            guard_diag = {
                "guard": str(transition.get("constraint")),
                "binding": safe_binding_repr(binding),
                "guard_variables": sorted(str(x) for x in guard_vars),
                "data_values": {str(k): v for k, v in new_data.items() if str(k) in guard_vars},
                "relevant_object_attributes": sorted(relevant_attrs),
                "object_attribute_values": attr_snapshot,
                "aggregate_values": aggregate_snapshot,
                "guard_result": guard_ok,
            }
            self._diag(
                "zero_cost_sync_guard", component=component.component_id, event_id=event.event_id,
                activity=event.activity, transition_id=transition_id, transition=transition_label,
                **guard_diag,
            )
            if guard_ok is None:
                self.stats["delta_dependency_fallbacks"] += 1
                return decline(
                    "guard_evaluation_unknown", stat="zero_cost_sync_guard_unknown", unknown=True,
                    **guard_diag,
                )
            if guard_ok is not True:
                # In particular, this catches an event value that would require
                # a model-side repair. Such a synchronous move may still be
                # optimal, but it is not a zero-cost extension.
                return decline(
                    "guard_false", stat="zero_cost_sync_guard_false", **guard_diag,
                )

        # Produce output tokens only after the guard has been certified. Data
        # fields on those tokens inherit the selected model-side value and its
        # provenance, so future transitions can read it locally.
        for arc in output_arcs:
            try:
                place = next(p for p in net._places if p["id"] == arc["target"])
            except StopIteration:
                return decline("output_place_missing", stat="zero_cost_sync_output_failures", arc_target=arc.get("target"))
            produced = self._lazy_arc_tokens(net, place, arc.get("inscription", []), binding)
            if not produced:
                return decline(
                    "output_arc_binding_empty", stat="zero_cost_sync_output_failures",
                    place_id=place.get("id"), place_name=place.get("name", place.get("label")),
                    inscription=[(str(n), str(tp)) for n, tp in arc.get("inscription", [])],
                    binding=safe_binding_repr(binding),
                )
            data_names = [n for n, typ in arc.get("inscription", []) if typ in net._data_types]
            for tok in produced:
                key = (place["id"], self._freeze_value(tok))
                successor_marking.add(key)
                if data_names:
                    fields = {}
                    for name in data_names:
                        if name not in new_data:
                            # A model-side write not determined by the current
                            # event/retained witness requires optimization.
                            self.stats["delta_dependency_fallbacks"] += 1
                            return decline(
                                "output_data_undetermined", stat="zero_cost_sync_output_failures", unknown=True,
                                data_name=name, place_id=place.get("id"),
                            )
                        fields[name] = self._freeze_value(new_data[name])
                        new_provenance[name] = (new_data[name], step, "token-write")
                    successor_token_data[key] = fields

        # If an object property becomes relevant for the first time, equality
        # with its observation is the unique zero-new-cost choice.  Missing
        # observations leave no local value to certify and therefore fall back.
        oa_out = [dict(x) for x in parent_assignment.object_attribute_assignments]
        oa_index = {
            (str(row.get("object")), str(row.get("attribute"))): idx
            for idx, row in enumerate(oa_out)
        }
        for obj in event.objects:
            vals = observed_oa.get(obj, {})
            for attr in sorted(relevant_attrs):
                if attr not in vals:
                    # The selected guard may still be satisfiable using a pure
                    # model value, but deciding that requires JODAP.
                    self.stats["delta_dependency_fallbacks"] += 1
                    return decline(
                        "guard_object_attribute_missing", stat="zero_cost_sync_guard_unknown", unknown=True,
                        object_id=str(obj), attribute=str(attr),
                    )
                key = (str(obj), str(attr))
                model_value = model_oa.get(key, vals[attr])
                observed_value = vals[attr]
                if isinstance(model_value, (int, float)) and isinstance(observed_value, (int, float)) \
                        and not isinstance(model_value, bool) and not isinstance(observed_value, bool):
                    mismatch = not math.isclose(float(model_value), float(observed_value),
                                                rel_tol=1e-9, abs_tol=1e-9)
                else:
                    mismatch = model_value != observed_value
                row = {
                    "object": str(obj),
                    "attribute": str(attr),
                    "observed_value": observed_value,
                    "model_value": model_value,
                    "mismatch": bool(mismatch),
                    "cost": 1 if mismatch else 0,
                }
                if key in oa_index:
                    # Preserve an already paid mismatch unless this transition
                    # deliberately supplies a new concrete model-side override.
                    if key in (model_object_attribute_overrides or {}):
                        oa_out[oa_index[key]] = row
                else:
                    oa_index[key] = len(oa_out)
                    oa_out.append(row)

        result = JointAssignment(
            total_cost=float(parent_assignment.total_cost) + float(extra_cost),
            object_bindings=list(parent_assignment.object_bindings) + [{
                "step": step,
                "transition_id": transition_id,
                "transition": transition.get("label"),
                "objects": tuple(sorted(binding_objects)),
            }],
            data_assignments=list(parent_assignment.data_assignments) + [{
                "step": step,
                "values": dict(new_data),
            }],
            object_attribute_assignments=oa_out,
            marking_signature=tuple(sorted(successor_marking, key=repr)),
            data_state_signature=tuple(sorted(
                (name, self._freeze_value(value)) for name, value in new_data.items()
            )),
            token_data_signature=tuple(sorted(
                ((pid, tok, tuple(sorted(fields.items())))
                 for (pid, tok), fields in successor_token_data.items()),
                key=repr)),
            data_provenance_signature=tuple(sorted(
                ((name, self._freeze_value(value), int(src_step), str(src_kind))
                 for name, (value, src_step, src_kind) in new_provenance.items()),
                key=repr)),
            solve_seconds=0.0,
            encode_seconds=0.0,
        )
        self.stats["delta_local_guard_hits"] += 1
        if float(extra_cost) <= 1e-12:
            self.stats["zero_cost_delta_hits"] += 1
        self._diag(
            "zero_cost_sync_success" if not repair_tag else f"{repair_tag}_sync_success",
            component=component.component_id, event_id=event.event_id,
            activity=event.activity, transition_id=transition_id, transition=transition_label,
            binding=safe_binding_repr(binding), parent_cost=float(parent_assignment.total_cost),
            observed_objects=sorted(event.objects), model_objects=sorted(binding_objects),
            result_cost=float(result.total_cost), successor_marking_size=len(successor_marking),
            repair_tag=repair_tag, extra_cost=float(extra_cost),
            object_attribute_overrides={f"{k[0]}.{k[1]}": v for k, v in (model_object_attribute_overrides or {}).items()},
        )
        return result

    def check_fixed_sync_extension(
            self, component: ComponentState, state: SearchState, child: SearchNode,
            event: StreamEvent, transition_id: int, parent_assignment: JointAssignment
    ) -> Optional[JointAssignment]:
        """Check exactly one proposed synchronous extension without reopening A*.

        All historical model-step bindings are taken from the retained parent
        witness and the new synchronous step is fixed by the observed event. The
        selected path is then solved by the lazy path encoder.  This is the
        intermediate fallback for a local ``UNKNOWN`` result: it can prove that
        the old optimum extends with the same cost while avoiding construction
        of the eager object-instantiated CoCoMoT context.
        """
        self.stats["exact_extension_attempts"] += 1
        fixed: Dict[int, Sequence[str]] = {}
        fresh: Dict[int, str] = {}
        try:
            net = self.new_net()
            by_id = {t["id"]: t for t in net._transitions}
            for b in parent_assignment.object_bindings:
                step = int(b.get("step"))
                tid = b.get("transition_id")
                objs = tuple(b.get("objects", ()))
                t = by_id.get(tid)
                if t is None or not objs:
                    continue
                decls = self._lazy_unique_object_decls(net, t)
                if any("nu" in str(name) for name in decls):
                    if len(objs) != 1:
                        self.stats["exact_extension_fallbacks"] += 1
                        return None
                    fresh[step] = str(objs[0])
                else:
                    fixed[step] = tuple(str(o) for o in objs)
            fixed[child.model_depth - 1] = tuple(event.objects)
        except Exception:
            self.stats["exact_extension_fallbacks"] += 1
            return None

        result = self._lazy_solve_fixed_path(
            component, state, child,
            lower_bound=int(parent_assignment.total_cost),
            canonical_fresh_bindings=fresh or None,
            fixed_binding_by_step=fixed or None,
            accept_certified_nonzero=True,
        )
        if result is None:
            self.stats["exact_extension_fallbacks"] += 1
            return None
        # A certified fixed-path result is useful even when it adds a positive
        # deviation: it is a feasible incumbent.  Global optimality is decided
        # by the backend's independent prefix lower bound.
        self.stats["exact_extension_hits"] += 1
        if result.total_cost > parent_assignment.total_cost + 1e-9:
            self.stats["positive_fixed_path_incumbents"] += 1
        return result


class SymbolicIncrementalBackend:
    """Incremental A* over move structures with JODAP at each candidate node.

    Search state is retained per object component. Object/data assignments are
    not committed by A*: they are optimized jointly by JODAP for the complete
    candidate prefix. The implementation deliberately uses conservative path
    signatures instead of aggressive symbolic-state subsumption; this preserves
    correctness while keeping the prototype auditable.
    """

    def __init__(self, cocomot_root: str, model_path: str, fixed_objects: bool = False,
                 model_depth_margin: int = 4, max_prepared_contexts: int = 24,
                 provenance_slicing: str = "off", provenance_slice_max_expansions: int = 256,
                 query_diagnostics_path: Optional[str] = None, query_diagnostics_print: bool = False,
                 local_repair_cost_budget: float = 4.0, local_repair_max_candidates: int = 48,
                 local_repair_transition_cap: int = 12):
        self.jodap = JODAPSolver(
            cocomot_root, model_path, fixed_objects=fixed_objects,
            max_prepared_contexts=max_prepared_contexts,
            provenance_slicing=provenance_slicing,
            query_diagnostics_path=query_diagnostics_path,
            query_diagnostics_print=query_diagnostics_print,
        )
        self.model_path = model_path
        self.model_depth_margin = model_depth_margin
        self.provenance_slicing = str(provenance_slicing or "off").lower()
        if self.provenance_slicing not in {"off", "focus"}:
            raise ValueError("provenance_slicing must be 'off' or 'focus'")
        self.provenance_slice_max_expansions = max(1, int(provenance_slice_max_expansions))
        self.searches: Dict[int, SearchState] = {}
        # Latest observed ComponentState for every component that is still
        # active in the observation-induced partition.  This registry is used
        # only by conservative cross-component object-relation repair: a
        # missing observed relation may leave the model-side ITEM in another
        # already-observed component even though the DOPID transition relates
        # it to the current PACKAGE.  Future/unseen objects are never added.
        self._active_components: Dict[int, ComponentState] = {}
        self._virtual_component_dependencies: Dict[int, Dict[int, Set[str]]] = {}
        self.max_object_relation_fast_cardinality = 3

        # Before reopening unrestricted A*, explore a small exact neighbourhood
        # around the retained certified prefix.  Candidate repairs are ordered
        # by the same structural cost lower bounds used by the global search,
        # while the lazy fixed-path JODAP objective computes the real combined
        # model/log/data/object cost.  These limits affect performance only:
        # failure to certify locally always falls back to the complete search.
        self.local_repair_cost_budget = max(0.0, float(local_repair_cost_budget))
        self.local_repair_max_candidates = max(1, int(local_repair_max_candidates))
        self.local_repair_transition_cap = max(1, int(local_repair_transition_cap))
        # Use a separate net only for static transition metadata.
        net = self.jodap.new_net()
        # Static net metadata is reused by merge composition checks. Keeping the
        # parsed net avoids constructing a CoCoMoT encoding merely to test hard
        # synchronization compatibility of the connecting event.
        self.net_metadata = net
        self.transitions = list(net._transitions)
        self.transition_by_id = {t["id"]: t for t in self.transitions}
        self.all_transition_ids = set(self.transition_by_id)
        self.visible_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for t in self.transitions:
            if not t.get("invisible", False):
                self.visible_by_label.setdefault(t.get("label"), []).append(t)
        self._independent_pairs = self._compute_independence(net)
        self.nu_transition_types: Dict[int, str] = {}
        try:
            for t in net.nu_transitions():
                params = net.object_params_of_transition(t, [])
                # object_params_of_transition may need a concrete object domain
                # for LIST parameters. Nu parameters themselves are scalar in
                # the DOPID models used here; fall back to arc inscriptions.
                nu = [p for p in params if "nu" in str(p.get("name", ""))]
                if len(nu) == 1:
                    self.nu_transition_types[t["id"]] = str(nu[0]["type"])
        except Exception:
            for t in getattr(net, "_transitions", []):
                tid = t.get("id")
                for a in getattr(net, "_arcs", []):
                    if a.get("source") != tid:
                        continue
                    for name, typ in a.get("inscription", []):
                        if "nu" in str(name):
                            self.nu_transition_types[tid] = str(typ).replace(" LIST", "")
        # Focused model queries are intentionally off: empirical hit rates were
        # very low and each miss duplicated an unrestricted JODAP solve.
        self.enable_focused_model_queries = False
        # Symmetry/dominance bookkeeping was costly and produced no pruning in
        # the scalability runs. Keep it disabled by default; POR and canonical
        # fresh-object generation remain active.
        self.enable_symmetry_dominance = False

    @staticmethod
    def _base_var(name: str) -> str:
        return str(name).rstrip("'")

    def _compute_independence(self, net) -> Set[FrozenSet[int]]:
        """Conservative independence relation used for partial-order reduction.

        Two transitions are considered independent only if their incident place
        sets are disjoint and neither writes a data variable read/written by the
        other.  If guard-variable extraction fails we conservatively treat the
        transition as depending on every data variable.
        """
        footprints: Dict[int, Set[int]] = {}
        reads: Dict[int, Set[str]] = {}
        writes: Dict[int, Set[str]] = {}
        all_data = {str(v.get("name")) for v in net.variables()}
        for t in self.transitions:
            tid = t["id"]
            footprints[tid] = {
                (a["source"] if a["target"] == tid else a["target"])
                for a in net._arcs if a["source"] == tid or a["target"] == tid
            }
            writes[tid] = {self._base_var(v) for v in t.get("write", [])}
            guard = t.get("constraint")
            if guard is None:
                reads[tid] = set()
            else:
                try:
                    reads[tid] = {self._base_var(v) for v in guard.vars()}
                except Exception:
                    reads[tid] = set(all_data)
        independent: Set[FrozenSet[int]] = set()
        for i, a in enumerate(self.transitions):
            for b in self.transitions[i + 1:]:
                ta, tb = a["id"], b["id"]
                if footprints[ta] & footprints[tb]:
                    continue
                if writes[ta] & (writes[tb] | reads[tb]):
                    continue
                if writes[tb] & (writes[ta] | reads[ta]):
                    continue
                independent.add(frozenset((ta, tb)))
        return independent

    def _independent(self, ta: int, tb: int) -> bool:
        return ta != tb and frozenset((ta, tb)) in self._independent_pairs

    def _canonical_move_signature(self, sig: Tuple[Tuple[str, Optional[str], Optional[int]], ...]) \
            -> Tuple[Tuple[str, Optional[str], Optional[int]], ...]:
        """Normalize consecutive model-only runs modulo independent swaps.

        Log and synchronous moves are barriers. Within each model-only run,
        adjacent independent transitions are ordered by transition id. This is
        a compact Mazurkiewicz-trace style normal form and collapses many
        equivalent interleavings without reordering dependent behavior.
        """
        out: List[Tuple[str, Optional[str], Optional[int]]] = []
        run: List[Tuple[str, Optional[str], Optional[int]]] = []

        def flush() -> None:
            nonlocal run
            # Repeatedly swap only adjacent independent inversions.
            changed = True
            while changed:
                changed = False
                for i in range(len(run) - 1):
                    a, b = run[i], run[i + 1]
                    if a[2] is not None and b[2] is not None and a[2] > b[2] \
                            and self._independent(a[2], b[2]):
                        run[i], run[i + 1] = b, a
                        changed = True
            out.extend(run)
            run = []

        for item in sig:
            if item[0] == "model":
                run.append(item)
            else:
                flush()
                out.append(item)
        flush()
        return tuple(out)

    @staticmethod
    def _preds(component: ComponentState) -> Dict[str, Set[str]]:
        p: Dict[str, Set[str]] = {e: set() for e in component.execution.event_ids}
        for a, b in component.execution.edges:
            p.setdefault(b, set()).add(a)
            p.setdefault(a, set())
        return p

    def _bound(self, component: ComponentState) -> int:
        # Online bound grows monotonically with the observed prefix. It is not a
        # semantic finality condition; it only prevents unbounded model-move
        # exploration in the prototype.
        e = len(component.execution.event_ids)
        o = len(component.objects)
        return max(2, e + o + self.model_depth_margin)

    def _heuristic(self, component: ComponentState, consumed: FrozenSet[str]) -> float:
        """Admissible lower bound for the remaining observed behavior.

        If no visible model transition has an event's activity label, that event
        can only be consumed as a log move and its object-count cost is
        unavoidable. All other remaining costs are lower-bounded by zero.
        """
        lb = 0
        for eid in component.execution.event_ids:
            if eid in consumed:
                continue
            e = component.observation_formula.events[eid]
            if not self.visible_by_label.get(e.activity):
                lb += len(e.objects)
        return float(lb)

    def _initialize(self, component: ComponentState) -> SearchState:
        state = SearchState(component.component_id)
        state.model_bound = self._bound(component)
        root = SearchNode(
            node_id=state.new_id(), consumed=frozenset(), event_order=(),
            model_depth=0, g=0.0, h=self._heuristic(component, frozenset()),
            model_signature=(), move_signature=()
        )
        state.signatures[self._signature(root)] = root.node_id
        state.push(root)
        state.current_event_ids = frozenset(component.execution.event_ids)
        state.current_objects = frozenset(component.objects)
        state.current_attribute_observations = sum(
            len(v) for v in component.observation_formula.attribute_history.values())
        attrs_now = component.observation_formula.current_object_attributes()
        state.current_object_attribute_snapshot = tuple(
            sorted((o, a, self.jodap._freeze_value(v))
                   for o, vals in attrs_now.items() for a, v in vals.items())
        )
        self.searches[component.component_id] = state
        # Install an analytical all-log incumbent first. For a merge, compose
        # certified parent boundary witnesses directly before considering any
        # replay-based fallback. This avoids rebuilding the merged history in
        # JODAP merely to obtain a warm start.
        self._seed_incumbent(component, state)
        # Establish a cheap positive lower bound as early as possible.  This is
        # especially important for first-event object-attribute guards such as
        # vip(o): if the guard is false under the unique zero-deviation
        # observation, a zero-cost prefix is impossible even before A* starts.
        if len(component.execution.event_ids) == 1:
            eid0 = next(iter(component.execution.event_ids))
            ev0 = component.observation_formula.events.get(eid0)
            if ev0 is not None:
                state.proven_prefix_lower_bound = max(
                    state.proven_prefix_lower_bound,
                    float(self._event_zero_deviation_increment_lb(component, ev0))
                )
        if getattr(component, "merged_from", ()):
            # Merge-specific work is completely isolated here. Ordinary updates
            # never build frontier-composition metadata.  If direct bridge
            # composition cannot be installed, the untouched root remains the
            # exact fallback; importantly, we no longer replay parent histories
            # through JODAP merely to construct a warm start.
            self._install_direct_merge_compositions(component, state)
        return state

    def _signature(self, node: SearchNode, *, canonical: bool = True) -> Tuple[Any, ...]:
        move_sig = (self._canonical_move_signature(node.move_signature)
                    if canonical else node.move_signature)
        return (node.consumed, node.event_order, move_sig)

    def _path(self, state: SearchState, nid: int) -> List[SymbolicMove]:
        out: List[SymbolicMove] = []
        while nid in state.predecessor:
            prev, move = state.predecessor[nid]
            out.append(move)
            nid = prev
        out.reverse()
        return out

    def _object_observation_classes(self, component: ComponentState) -> Dict[str, str]:
        """Return conservative symmetry classes for currently indistinguishable objects.

        Objects are exchangeable only when all information observed *so far* is
        identical: type, current attributes, E2O participation (including the
        concrete event id/qualifier), and incident O2O relations.  A later event
        may distinguish a class; state-dominance information is therefore cleared
        whenever the observed event set grows.
        """
        attrs = component.observation_formula.current_object_attributes()
        events_by_obj: Dict[str, List[Any]] = {o: [] for o in component.objects}
        for eid, event in component.observation_formula.events.items():
            for qual, obj in event.relations:
                if obj in events_by_obj:
                    events_by_obj[obj].append((eid, event.activity, qual))
        rels_by_obj: Dict[str, List[Any]] = {o: [] for o in component.objects}
        for rel in component.observation_formula.o2o_relations:
            if rel.source in rels_by_obj:
                rels_by_obj[rel.source].append(("out", rel.qualifier, rel.target))
            if rel.target in rels_by_obj:
                rels_by_obj[rel.target].append(("in", rel.qualifier, rel.source))

        groups: Dict[Any, List[str]] = {}
        for obj in sorted(component.objects):
            sig = (
                component.observation_formula.object_types.get(obj),
                tuple(sorted((a, self.jodap._freeze_value(v))
                             for a, v in attrs.get(obj, {}).items())),
                tuple(sorted(events_by_obj.get(obj, ()))),
                tuple(sorted(rels_by_obj.get(obj, ()))),
            )
            groups.setdefault(sig, []).append(obj)

        result: Dict[str, str] = {}
        idx = 0
        for sig, objs in sorted(groups.items(), key=lambda kv: repr(kv[0])):
            if len(objs) < 2:
                continue
            typ = component.observation_formula.object_types.get(objs[0], "obj")
            label = f"@sym:{typ}:{idx}"
            idx += 1
            for obj in objs:
                result[obj] = label
        if result:
            self.jodap.stats["symmetry_canonicalizations"] += 1
        return result

    def _canonical_marking(self, component: ComponentState, marking: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if not self.enable_symmetry_dominance:
            return marking
        classes = self._object_observation_classes(component)
        if not classes:
            return marking

        def repl(value: Any) -> Any:
            if isinstance(value, str):
                return classes.get(value, value)
            if isinstance(value, tuple):
                return tuple(repl(x) for x in value)
            if isinstance(value, list):
                return tuple(repl(x) for x in value)
            return value

        return tuple(sorted((repl(x) for x in marking), key=repr))

    def _state_key(self, component: ComponentState, node: SearchNode,
                   assignment: JointAssignment) -> Tuple[Any, ...]:
        # Future enablement depends on the current object-aware marking and the
        # current data values, not on the order in which independent model moves
        # reached them.  Symmetric objects are canonicalized conservatively.
        obj_attr_state = tuple(sorted(
            (a.get("object"), a.get("attribute"), self.jodap._freeze_value(a.get("model_value")))
            for a in assignment.object_attribute_assignments
        ))
        return (
            node.consumed,
            node.event_order,
            self._canonical_marking(component, assignment.marking_signature),
            assignment.data_state_signature,
            assignment.token_data_signature,
            obj_attr_state,
        )

    @staticmethod
    def _objects_in_value(value: Any, known: Set[str]) -> Set[str]:
        out: Set[str] = set()
        if isinstance(value, str):
            if value in known:
                out.add(value)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for x in value:
                out.update(SymbolicIncrementalBackend._objects_in_value(x, known))
        return out

    def _focused_objects(self, component: ComponentState, parent: SearchNode,
                         assignment: Optional[JointAssignment]) -> Set[str]:
        """Objects most likely to matter before the next observed event.

        This set is used only as a *first-pass* symmetry/focus restriction.  A
        focused result is accepted without fallback only when it reaches the
        predecessor's proven lower bound; otherwise JODAP is solved again without
        the restriction, preserving completeness and optimality.
        """
        focus: Set[str] = set()
        for event in self._enabled_events(component, parent):
            focus.update(event.objects)
        # Normally the next enabled observations are the strongest safe *hint*:
        # model behavior for unrelated objects can usually be postponed.  If no
        # observed event is currently enabled, fall back to the objects in the
        # concrete current marking.  Because a focused solution is accepted only
        # when it attains the predecessor lower bound, an overly narrow hint can
        # never change the optimum; it merely triggers the unrestricted fallback.
        if not focus and assignment is not None:
            known = set(component.objects)
            for item in assignment.marking_signature:
                focus.update(self._objects_in_value(item, known))
        return focus

    def _canonical_fresh_object(self, component: ComponentState, parent: SearchNode,
                                transition_id: int) -> Optional[str]:
        """Return the canonical not-yet-marked object for an invisible nu move.

        Fresh creations of the same type are symmetric until an observed event
        distinguishes their identities.  CoCoMoT's freshness semantics requires
        the nu object not to occur in the current marking.  Choosing the
        lexicographically first known object of the required type that is absent
        from that marking therefore removes only equivalent creation-order
        permutations.  If the required metadata is unavailable, no restriction
        is applied.
        """
        typ = self.nu_transition_types.get(transition_id)
        if not typ:
            return None
        t = self.transition_by_id.get(transition_id)
        if not t or not t.get("invisible", False):
            return None
        assignment = self.searches.get(component.component_id)
        parent_assn = None
        if assignment is not None:
            parent_assn = assignment.assignments_by_node.get(parent.node_id)
        known = set(component.objects)
        marked: Set[str] = set()
        if parent_assn is None:
            # The root marking contains no process objects in the current DOPID
            # benchmarks, so the first fresh creation can already be
            # canonicalized. For non-root states without a retained witness we
            # conservatively skip the restriction.
            if parent.model_depth != 0:
                return None
        else:
            for item in parent_assn.marking_signature:
                marked.update(self._objects_in_value(item, known))
        candidates = sorted(
            o for o in component.objects
            if component.observation_formula.object_types.get(o) == typ and o not in marked
        )
        return candidates[0] if candidates else None

    def _minimum_model_move_cost(self, move: SymbolicMove) -> float:
        """Safe structural lower bound for one visible model-only move.

        The model-move objective charges one unit for every concrete object in
        the selected binding.  Reading CoCoMoT's expanded parameter table with
        an empty object domain can undercount LIST inscriptions (and in some
        versions returns no useful ``needed`` slots at all).  Count logical
        object variables directly from the raw PNML arc inscriptions instead.
        Every scalar variable needs one object slot. LIST variables deliberately
        contribute zero to this lower bound because the generic CoCoMoT binding
        domain includes the empty subset; marking/guard constraints may later
        force a non-empty list, but assuming that here would risk an unsound
        prune. Repeated occurrences of the same scalar variable are counted once.

        This remains a lower bound: a LIST may contain zero or many objects, while
        invisible transitions still cost zero.  Any parsing uncertainty falls
        back to zero rather than risking an unsound prune.
        """
        if move.transition_id is None:
            return 0.0
        t = self.transition_by_id.get(move.transition_id)
        if not t or t.get("invisible", False):
            return 0.0
        try:
            net = self.jodap._static_net or self.jodap.new_net()
            self.jodap._static_net = net
            nt = next(x for x in net._transitions if x.get("id") == move.transition_id)
            logical: Dict[str, str] = {}
            for arc in net._arcs:
                if arc.get("source") != nt["id"] and arc.get("target") != nt["id"]:
                    continue
                for name, typ in arc.get("inscription", ()):
                    if typ in net._data_types:
                        continue
                    name, typ = str(name), str(typ)
                    previous = logical.get(name)
                    if previous is None or "LIST" in typ:
                        logical[name] = typ
            if not logical:
                return 0.0
            required_scalars = {name for name, typ in logical.items() if "LIST" not in typ}
            return float(len(required_scalars))
        except Exception:
            return 0.0

    def _strict_improvement_cap(self, state: SearchState) -> Optional[int]:
        """Return the largest integer cost that can strictly improve the incumbent.

        Alignment costs in ODACC are integral.  Once a feasible current-prefix
        incumbent exists, exact certification needs to consider only costs
        strictly below that incumbent.  Returning ``None`` keeps ordinary search
        semantics for states without a valid incumbent.
        """
        if state.incumbent_assignment is None or state.incumbent_offline != state.search_offline:
            return None
        if not math.isfinite(float(state.upper_bound)):
            return None
        return int(math.floor(float(state.upper_bound) - 1e-9))

    def _strict_generation_lower_bound(
            self, component: ComponentState, parent: SearchNode,
            move: SymbolicMove, consumed_after: FrozenSet[str]) -> float:
        """Cheap admissible lower bound used *before* successor construction.

        This is deliberately the same cost semantics as the later pre-JODAP
        bound, but is evaluated in ``_expand`` so dominated operators never
        allocate a node/signature or enter candidate handling.
        """
        base = float(parent.g) if parent.g != float("inf") else 0.0
        move_lb = 0.0
        if move.kind == "log" and move.event_id is not None:
            ev = component.observation_formula.events.get(move.event_id)
            if ev is not None:
                move_lb = float(len(ev.objects))
        elif move.kind == "model":
            move_lb = self._minimum_model_move_cost(move)
        return base + move_lb + self._heuristic(component, consumed_after)

    def _candidate_pre_jodap_lower_bound(
            self, component: ComponentState, state: SearchState,
            parent: SearchNode, move: SymbolicMove, consumed: FrozenSet[str]) -> tuple[float, str]:
        """Admissible f-bound available before constructing/querying JODAP."""
        base = float(parent.g) if parent.g != float("inf") else 0.0
        move_lb = 0.0
        source = "structural"
        if move.kind == "log" and move.event_id is not None:
            ev = component.observation_formula.events.get(move.event_id)
            if ev is not None:
                # A log move has exactly this control-flow charge in both JODAP
                # and the SMT reference backend.
                move_lb = float(len(ev.objects))
                source = "log"
        elif move.kind == "model":
            move_lb = self._minimum_model_move_cost(move)
            source = "model" if move_lb > 0 else "structural"
        # The ordinary heuristic is independently admissible for all still
        # unconsumed observations, so it can safely be added here.
        return base + move_lb + self._heuristic(component, consumed), source

    def _add_candidate(self, component: ComponentState, state: SearchState,
                       parent: SearchNode, move: SymbolicMove,
                       *, evaluate: bool = True) -> Optional[SearchNode]:
        consumed = parent.consumed
        event_order = parent.event_order
        model_depth = parent.model_depth
        model_sig = parent.model_signature

        # Partial-order reduction: consecutive model-only moves are generated
        # only in the canonical order for independent transitions.
        if (not state.search_offline) and move.kind == "model" and parent.move_signature:
            pkind, _peid, ptid = parent.move_signature[-1]
            if pkind == "model" and ptid is not None and move.transition_id is not None \
                    and ptid > move.transition_id and self._independent(ptid, move.transition_id):
                self.jodap.stats["por_pruned"] += 1
                return None

        move_sig = parent.move_signature + ((move.kind, move.event_id, move.transition_id),)
        if move.kind in ("log", "sync"):
            if move.event_id in consumed:
                return None
            consumed = frozenset(set(consumed) | {move.event_id})
            event_order = event_order + (move.event_id,)
        if move.kind in ("model", "sync"):
            model_depth += 1
            if model_depth > state.model_bound:
                return None
            model_sig = model_sig + (move.transition_id,)

        temp_id = state.new_id()
        inherited_boundary = parent.checkpoint_boundary_assignment
        inherited_snapshot = parent.checkpoint_boundary_snapshot
        inherited_depth = int(parent.checkpoint_boundary_model_depth)
        inherited_prefix = parent.checkpoint_boundary_prefix_moves
        inherited_suffix = parent.checkpoint_suffix_moves
        if inherited_boundary is not None:
            inherited_suffix = tuple(inherited_suffix) + (move,)
        elif not state.search_offline:
            # Diagnostic: after observation-domain revalidation the new event is
            # expanded from every surviving node.  Only nodes genuinely reached
            # from a certified checkpoint may carry a checkpoint suffix; do not
            # fabricate one for unrelated alternatives, but make the bypass
            # visible so hard branches can be classified correctly.
            self.jodap.stats["persistent_checkpoint_lineage_missing_on_parent"] += 1
            self.jodap._diag(
                "persistent_checkpoint_lineage_missing_on_parent",
                component=component.component_id, parent=parent.node_id,
                parent_model_depth=int(parent.model_depth), move_kind=move.kind,
                event_id=move.event_id, transition_id=move.transition_id,
            )

        node = SearchNode(
            temp_id, consumed, event_order, model_depth,
            g=float("inf"), h=self._heuristic(component, consumed),
            model_signature=model_sig, move_signature=move_sig,
            checkpoint_boundary_assignment=inherited_boundary,
            checkpoint_boundary_snapshot=inherited_snapshot,
            checkpoint_boundary_model_depth=inherited_depth,
            checkpoint_boundary_prefix_moves=inherited_prefix,
            checkpoint_suffix_moves=inherited_suffix,
        )
        if inherited_boundary is not None:
            self.jodap.stats["persistent_checkpoint_lineage_propagated"] += 1
        sig = self._signature(node, canonical=not state.search_offline)
        existing_id = state.signatures.get(sig)
        if existing_id is not None:
            self.jodap.stats["signature_pruned"] += 1
            existing = state.nodes.get(existing_id)
            # Preserve a valid certified-boundary route whenever canonical reuse
            # chooses a node whose ordinary predecessor chain came from another
            # route.  The incoming route is independently valid because it was
            # just generated from ``parent``.  Prefer the *nearest* certified
            # boundary (larger boundary model depth), then the shorter suffix.
            # This fixes the hard-case situation where a canonical node already
            # existed with no/older lineage and silently kept that poorer route.
            if existing is not None and inherited_boundary is not None:
                existing_has = existing.checkpoint_boundary_assignment is not None
                same_boundary = (
                    existing.checkpoint_boundary_assignment is inherited_boundary
                    or (existing_has
                        and abs(float(existing.checkpoint_boundary_assignment.total_cost)
                                - float(inherited_boundary.total_cost)) <= 1e-9
                        and int(existing.checkpoint_boundary_model_depth) == inherited_depth
                        and existing.checkpoint_boundary_prefix_moves == inherited_prefix)
                )
                incoming_better = (
                    (not existing_has)
                    or inherited_depth > int(existing.checkpoint_boundary_model_depth)
                    or (same_boundary and (not existing.checkpoint_suffix_moves
                                            or len(inherited_suffix) < len(existing.checkpoint_suffix_moves)))
                )
                if incoming_better:
                    existing.checkpoint_boundary_assignment = inherited_boundary
                    existing.checkpoint_boundary_snapshot = inherited_snapshot
                    existing.checkpoint_boundary_model_depth = inherited_depth
                    existing.checkpoint_boundary_prefix_moves = inherited_prefix
                    existing.checkpoint_suffix_moves = tuple(inherited_suffix)
                    self.jodap.stats["persistent_checkpoint_lineage_reused"] += 1
                    self.jodap.stats["persistent_checkpoint_lineage_attached_on_reuse"] += 1
                    self.jodap._diag(
                        "persistent_checkpoint_lineage_attached_on_reuse",
                        component=component.component_id, node=existing.node_id,
                        boundary_model_depth=inherited_depth,
                        suffix_moves=len(inherited_suffix),
                    )
                else:
                    self.jodap.stats["persistent_checkpoint_lineage_dropped_on_canonical_reuse"] += 1
                    self.jodap._diag(
                        "persistent_checkpoint_lineage_dropped_on_canonical_reuse",
                        component=component.component_id, node=existing.node_id,
                        existing_boundary_depth=int(existing.checkpoint_boundary_model_depth),
                        incoming_boundary_depth=inherited_depth,
                        existing_suffix_moves=len(existing.checkpoint_suffix_moves),
                        incoming_suffix_moves=len(inherited_suffix),
                    )
            elif existing is not None and inherited_boundary is None \
                    and existing.checkpoint_boundary_assignment is None and not state.search_offline:
                self.jodap.stats["persistent_checkpoint_lineage_missing_on_candidate"] += 1
            return existing

        # Prune against the incumbent *before* an exact JODAP query whenever an
        # admissible structural bound already proves that this candidate cannot
        # improve the best prefix alignment.  Previously this check happened
        # only after JODAP had solved the candidate, which is especially costly
        # on realistic object-rich components.
        if evaluate and state.incumbent_assignment is not None \
                and state.incumbent_offline == state.search_offline:
            pre_lb, pre_source = self._candidate_pre_jodap_lower_bound(
                component, state, parent, move, consumed
            )
            if pre_lb >= state.upper_bound - 1e-9:
                self.jodap.stats["upper_bound_pruned"] += 1
                self.jodap.stats["pre_jodap_upper_bound_pruned"] += 1
                if pre_source == "log":
                    self.jodap.stats["pre_jodap_log_cost_pruned"] += 1
                elif pre_source == "model":
                    self.jodap.stats["pre_jodap_model_cost_pruned"] += 1
                self.jodap._diag(
                    "pre_jodap_pruned", component=component.component_id,
                    parent=parent.node_id, move_kind=move.kind,
                    event_id=move.event_id, transition_id=move.transition_id,
                    lower_bound=pre_lb, upper_bound=float(state.upper_bound),
                    source=pre_source,
                )
                return None

        state.predecessor[node.node_id] = (parent.node_id, move)
        assignment: Optional[JointAssignment] = None
        if evaluate:
            parent_lb = int(math.ceil(parent.g - 1e-9)) if parent.g != float("inf") else 0
            candidate_lb = parent_lb
            if move.kind == "model":
                candidate_lb += int(math.ceil(self._minimum_model_move_cost(move) - 1e-9))
            elif move.kind == "log" and move.event_id is not None:
                ev = component.observation_formula.events.get(move.event_id)
                if ev is not None:
                    candidate_lb += int(len(ev.objects))
            if candidate_lb > parent_lb:
                self.jodap.stats["candidate_structural_lb_raised"] += 1
                self.jodap._diag(
                    "candidate_structural_lower_bound",
                    component=component.component_id, parent=parent.node_id,
                    move_kind=move.kind, event_id=move.event_id,
                    transition_id=move.transition_id, parent_lower_bound=parent_lb,
                    candidate_lower_bound=candidate_lb,
                )

            # Model-only successors first try a focused object domain containing
            # the immediate observed frontier (and currently marked objects). If
            # that restricted query attains the predecessor lower bound, it has
            # already proved the unrestricted optimum. Otherwise we fall back to
            # the full JODAP domain, so no valid alignment is removed.
            canonical_fresh = None
            if move.kind == "model" and move.transition_id is not None:
                canonical_fresh = self._canonical_fresh_object(
                    component, parent, move.transition_id
                )
                if canonical_fresh is not None:
                    self.jodap.stats["canonical_creation_restrictions"] += 1

            # Focused model-domain queries were useful only rarely in the
            # benchmark (most queries had to be repeated unrestricted).  They
            # are therefore disabled by default.  The exact incremental JODAP
            # query is issued directly, with only the safe canonical nu-binding
            # restriction above.
            if assignment is None:
                assignment = self.jodap.solve(
                    component, state, node, require_final=False,
                    lower_bound=candidate_lb, canonical_fresh_object=canonical_fresh
                )
            if assignment is None:
                del state.predecessor[node.node_id]
                return None
            node.g = assignment.total_cost
            node.assignment_cost = assignment.total_cost
            state.assignments_by_node[node.node_id] = assignment

            # If an exact query has already produced a complete current-prefix
            # alignment of cost zero, global optimality follows immediately from
            # non-negative alignment costs.  Install it *before* ordinary UB
            # pruning so sibling generation can stop without hundreds of further
            # JODAP calls.
            if not state.search_offline and node.consumed == frozenset(component.execution.event_ids) \
                    and assignment.total_cost <= 1e-9:
                state.upper_bound = 0.0
                state.incumbent_moves = tuple(self._path(state, node.node_id))
                state.incumbent_assignment = assignment
                state.incumbent_offline = False
                state.current_goal = node.node_id
                state.current_assignment = assignment
                self.jodap.stats["exact_zero_cost_goal_hits"] += 1
                self.jodap._diag(
                    "exact_zero_cost_goal", component=component.component_id, node=node.node_id,
                    model_depth=node.model_depth, consumed_events=len(node.consumed),
                )

            # Incumbent/upper-bound pruning. Since h is admissible and all costs
            # are non-negative, a node with f >= incumbent cannot improve it.
            if state.incumbent_assignment is not None \
                    and state.incumbent_offline == state.search_offline \
                    and node.f >= state.upper_bound - 1e-9:
                del state.predecessor[node.node_id]
                self.jodap.stats["upper_bound_pruned"] += 1
                return None

            # Conservative exact-state dominance after JODAP has selected a
            # concrete operational state. Keep the no-more-expensive path.
            if not state.search_offline and self.enable_symmetry_dominance:
                skey = self._state_key(component, node, assignment)
                dominated = state.state_dominance.get(skey)
                # Same current marking/data state means the future behavior is
                # equivalent. Prefer a path that is no more expensive and no
                # deeper (the latter matters only because the prototype uses a
                # finite model-depth exploration bound).
                if dominated is not None and dominated[0] <= node.g + 1e-9 \
                        and dominated[1] <= node.model_depth:
                    del state.predecessor[node.node_id]
                    state.assignments_by_node.pop(node.node_id, None)
                    self.jodap.stats["state_dominated"] += 1
                    return None
                state.state_dominance[skey] = (node.g, node.model_depth, node.node_id)

        state.signatures[sig] = node.node_id
        state.push(node)

        if assignment is not None and not state.search_offline \
                and node.consumed == frozenset(component.execution.event_ids):
            if node.g < state.upper_bound - 1e-9:
                state.upper_bound = node.g
                state.incumbent_moves = tuple(self._path(state, node.node_id))
                state.incumbent_assignment = assignment
                state.incumbent_offline = False
        return node

    def _enabled_events(self, component: ComponentState, node: SearchNode) -> List[StreamEvent]:
        preds = self._preds(component)
        result = []
        for eid in sorted(component.execution.event_ids, key=lambda x: component.observation_formula.events[x].timestamp):
            if eid in node.consumed:
                continue
            if preds.get(eid, set()).issubset(node.consumed):
                result.append(component.observation_formula.events[eid])
        return result

    def _topological_event_order(self, component: ComponentState) -> List[str]:
        preds = self._preds(component)
        remaining = set(component.execution.event_ids)
        order: List[str] = []
        pos = {}
        for u in component.units:
            if u.event is not None:
                pos[u.event.event_id] = u.position
        while remaining:
            enabled = [e for e in remaining if preds.get(e, set()).issubset(set(order))]
            if not enabled:
                # Defensive fallback for malformed/cyclic input.
                enabled = list(remaining)
            enabled.sort(key=lambda e: (pos.get(e, 10**12), e))
            e = enabled[0]
            order.append(e)
            remaining.remove(e)
        return order

    def _evaluate_move_sequence(self, component: ComponentState, state: SearchState,
                                moves: Sequence[SymbolicMove],
                                *, require_final: bool = False) \
            -> Tuple[Optional[JointAssignment], Optional[SearchNode]]:
        """Evaluate a complete candidate without inserting it into OPEN."""
        tmp = SearchState(component.component_id)
        tmp.model_bound = state.model_bound
        root = SearchNode(0, frozenset(), (), 0, 0.0, 0.0, 0.0, (), ())
        tmp.nodes[0] = root
        tmp.next_id = 1
        parent = root
        for move in moves:
            consumed = parent.consumed
            event_order = parent.event_order
            depth = parent.model_depth
            msig = parent.model_signature
            mvsig = parent.move_signature + ((move.kind, move.event_id, move.transition_id),)
            if move.kind in ("log", "sync"):
                if move.event_id in consumed:
                    return None, None
                consumed = frozenset(set(consumed) | {move.event_id})
                event_order = event_order + (move.event_id,)
            if move.kind in ("model", "sync"):
                depth += 1
                if depth > state.model_bound:
                    return None, None
                msig = msig + (move.transition_id,)
            n = SearchNode(tmp.new_id(), consumed, event_order, depth, float("inf"),
                           self._heuristic(component, consumed), 0.0, msig, mvsig)
            tmp.nodes[n.node_id] = n
            tmp.predecessor[n.node_id] = (parent.node_id, move)
            parent = n
        assignment = self.jodap.solve(component, tmp, parent, require_final=require_final)
        if assignment is not None:
            parent.g = assignment.total_cost
            parent.assignment_cost = assignment.total_cost
        return assignment, parent

    @staticmethod
    def _merge_signature_union(parts: Sequence[Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
        """Union tuple signatures while rejecting conflicting keyed entries.

        Marking/token signatures are naturally set-like. Data-state and
        provenance signatures use their first field as a variable key and must
        agree when the same process variable occurs in multiple parent
        components. A conflict means that the parent witnesses cannot be
        composed without re-optimizing them jointly, so the caller falls back to
        ordinary A*+JODAP.
        """
        keyed: Dict[Any, Any] = {}
        unkeyed: Set[Any] = set()
        for part in parts:
            for item in part or ():
                if isinstance(item, tuple) and item:
                    key = item[0]
                    # Process/data signatures use string keys. Place ids in
                    # marking signatures are integers in the imported models and
                    # therefore remain set-like below.
                    if isinstance(key, str) and len(item) >= 2:
                        old = keyed.get(key)
                        if old is not None and old != item:
                            return None
                        keyed[key] = item
                        continue
                unkeyed.add(item)
        return tuple(sorted(list(unkeyed) + list(keyed.values()), key=repr))

    def _compose_parent_assignments(self,
                                    pieces: Sequence[Tuple[JointAssignment, Sequence[SymbolicMove]]]) \
            -> Optional[JointAssignment]:
        """Compose certified parent witnesses without flattening local versions.

        The parents were independent before the connecting observation.  Their
        marking/token state is therefore combined directly.  Process/data names
        are *not* forced to agree merely because two independently solved
        components used the same syntactic variable name: conflicting current
        values are treated as parent-local versions and omitted from the merged
        unqualified current-data map.  The concrete per-token values and the
        original per-step data assignments are retained, so a bridge transition
        can still read the value carried by the object it actually consumes.

        Only a genuine conflict for the same concrete token (same place and
        object tuple but different carried data) makes the witnesses
        incompatible.  This is the operational conflict that cannot be repaired
        by alpha-renaming component-local variables.
        """
        if not pieces:
            return None

        # Namespace process-level values by parent index first.  Equal values
        # can safely be exposed under the original name; conflicting values stay
        # local to their parent and are resolved only if the bridge actually
        # relates them.
        by_name: Dict[str, List[Tuple[int, Any]]] = {}
        prov_by_name: Dict[str, List[Tuple[int, Any]]] = {}
        for parent_idx, (assignment, _moves) in enumerate(pieces):
            for item in assignment.data_state_signature or ():
                if isinstance(item, tuple) and len(item) >= 2:
                    by_name.setdefault(str(item[0]), []).append((parent_idx, item[1]))
            for item in assignment.data_provenance_signature or ():
                if isinstance(item, tuple) and len(item) >= 2:
                    prov_by_name.setdefault(str(item[0]), []).append((parent_idx, item))

        merged_data: List[Any] = []
        merged_prov: List[Any] = []
        for name, vals in by_name.items():
            unique = {repr(v) for _idx, v in vals}
            if len(unique) == 1:
                merged_data.append((name, vals[0][1]))
            else:
                # Alpha-renaming is conceptual here: keep the values in the
                # parent data-assignment histories instead of publishing an
                # ambiguous unqualified value to the bridge witness.
                self.jodap.stats["merge_namespaced_data_conflicts"] += 1
        for name, vals in prov_by_name.items():
            values = {repr(item[1]) for _idx, item in vals}
            if len(values) == 1:
                # Source step/kind may differ across parents; retain one source
                # only when the current value agrees.
                merged_prov.append(vals[0][1])
            else:
                self.jodap.stats["merge_namespaced_data_conflicts"] += 1

        # Some object-relation repairs borrow one already-certified object from
        # another observation component (because the missing E2O relation kept
        # those components separate).  When a later observed event really merges
        # them, the repaired assignment is the authoritative model-side state of
        # that object.  Suppress the stale pre-repair copy from the source parent
        # instead of unioning both places for the same object.
        claimed_owner: Dict[str, int] = {}
        claimed_source: Dict[str, int] = {}
        for parent_idx, (assignment, _moves) in enumerate(pieces):
            for row in assignment.object_bindings or ():
                for obj in row.get("relation_claimed_objects", ()) or ():
                    claimed_owner[str(obj)] = parent_idx
                for pair in row.get("borrowed_object_sources", ()) or ():
                    try:
                        obj, source_cid = pair
                        claimed_owner[str(obj)] = parent_idx
                        claimed_source[str(obj)] = int(source_cid)
                    except Exception:
                        pass

        markings: Set[Any] = set()
        token_by_key: Dict[Tuple[Any, Any], Any] = {}
        obj_attrs: List[Dict[str, Any]] = []
        bindings: List[Dict[str, Any]] = []
        data_assignments: List[Dict[str, Any]] = []
        step_offset = 0
        for parent_idx, (assignment, moves) in enumerate(pieces):
            for mark in assignment.marking_signature or ():
                claimed_here = {obj for obj, owner in claimed_owner.items()
                                if owner != parent_idx and self._objects_in_value(mark, {obj})}
                if claimed_here:
                    continue
                markings.add(mark)
            for token_item in assignment.token_data_signature or ():
                claimed_here = {obj for obj, owner in claimed_owner.items()
                                if owner != parent_idx and self._objects_in_value(token_item, {obj})}
                if claimed_here:
                    continue
                try:
                    pid, tok, fields = token_item
                    key = (pid, self.jodap._freeze_value(tok))
                except Exception:
                    self.jodap.stats["merge_incompatible_token_data"] += 1
                    return None
                old = token_by_key.get(key)
                if old is not None and old != token_item:
                    # Same concrete live token with contradictory carried data
                    # is a genuine state conflict, not a naming collision.
                    self.jodap.stats["merge_incompatible_token_data"] += 1
                    return None
                token_by_key[key] = token_item

            for row in assignment.object_attribute_assignments:
                r = dict(row)
                r.setdefault("parent_component_index", parent_idx)
                obj_attrs.append(r)

            local_model_steps = sum(1 for m in moves if m.kind in ("model", "sync"))
            for row in assignment.object_bindings:
                r = dict(row)
                if isinstance(r.get("step"), int):
                    r["step"] += step_offset
                r.setdefault("parent_component_index", parent_idx)
                bindings.append(r)
            for row in assignment.data_assignments:
                r = dict(row)
                if isinstance(r.get("step"), int):
                    r["step"] += step_offset
                r.setdefault("parent_component_index", parent_idx)
                data_assignments.append(r)
            step_offset += local_model_steps

        return JointAssignment(
            total_cost=float(sum(a.total_cost for a, _ in pieces)),
            object_bindings=bindings,
            data_assignments=data_assignments,
            object_attribute_assignments=obj_attrs,
            marking_signature=tuple(sorted(markings, key=repr)),
            data_state_signature=tuple(sorted(merged_data, key=repr)),
            token_data_signature=tuple(sorted(token_by_key.values(), key=repr)),
            data_provenance_signature=tuple(sorted(merged_prov, key=repr)),
            solve_seconds=0.0,
            encode_seconds=0.0,
        )

    def _parent_boundary_candidates(self, component: ComponentState,
                                    per_parent: int = 8) \
            -> Optional[List[Tuple[int, List[Tuple[float, int, List[SymbolicMove], JointAssignment]]]]]:
        """Return nondominated full-prefix boundary states of each merge parent.

        Fast local proofs sometimes install a certified ``current_assignment``
        before/without leaving a conventional OPEN frontier node (for example a
        latent cross-component object repair).  Such a witness is still a valid
        optimal parent boundary and must remain composable at a later observed
        merge.  We therefore prefer ordinary full-prefix nodes, but retain the
        certified current/incumbent witness as a co-optimal fallback instead of
        forcing the merged component to restart from the global root.
        """
        if not getattr(component, "merged_from", ()):
            return None
        groups = []
        for cid in sorted(component.merged_from):
            pst = self.searches.get(cid)
            if pst is None:
                self.jodap._diag(
                    "parent_boundary_missing", component=component.component_id,
                    parent_component=cid, reason="search_state_missing")
                return None
            full = frozenset(pst.current_event_ids)
            choices: List[Tuple[float, int, List[SymbolicMove], JointAssignment]] = []
            seen: Set[Tuple[Any, ...]] = set()
            # The certified optimum is always considered first.
            ids = []
            if pst.current_goal is not None:
                ids.append(pst.current_goal)
            ids.extend(sorted(pst.nodes))
            for nid in ids:
                node = pst.nodes.get(nid)
                if node is None or node.consumed != full or node.g == float("inf"):
                    continue
                assignment = pst.assignments_by_node.get(nid)
                if assignment is None and nid == pst.current_goal:
                    assignment = pst.current_assignment
                if assignment is None:
                    continue
                sig = self._canonical_move_signature(node.move_signature)
                if sig in seen:
                    continue
                seen.add(sig)
                choices.append((float(assignment.total_cost), nid,
                                self._path(pst, nid), assignment))

            # A fast path may have a certified complete witness whose materialized
            # node was subsequently pruned/rebuilt.  Do not lose that co-optimal
            # boundary: composition only needs its concrete assignment and move
            # sequence.  Total cost, not its decomposition, defines optimality.
            fallback_rows = []
            for cont_cost, cont_moves, cont_assignment in getattr(
                    pst, "cooptimal_continuation_boundaries", ()):
                fallback_rows.append((cont_assignment, tuple(cont_moves),
                                      "cooptimal_continuation"))
            if pst.current_assignment is not None:
                fallback_rows.append((pst.current_assignment,
                                      tuple(pst.incumbent_moves), "current_assignment"))
            if pst.incumbent_assignment is not None and not pst.incumbent_offline:
                fallback_rows.append((pst.incumbent_assignment,
                                      tuple(pst.incumbent_moves), "incumbent_assignment"))
            for assignment, moves_tuple, source in fallback_rows:
                if assignment is None:
                    continue
                moves = list(moves_tuple)
                consumed = {m.event_id for m in moves
                            if m.kind in ("log", "sync") and m.event_id is not None}
                if consumed != set(full):
                    # If current_goal still exists, its path is the authoritative
                    # representation even when incumbent_moves points elsewhere.
                    if pst.current_goal is not None and pst.current_goal in pst.nodes:
                        n = pst.nodes[pst.current_goal]
                        if n.consumed == full:
                            moves = self._path(pst, pst.current_goal)
                            consumed = set(full)
                if consumed != set(full):
                    continue
                sig = self._canonical_move_signature(tuple(
                    (m.kind, m.event_id, m.transition_id) for m in moves))
                if sig in seen:
                    continue
                seen.add(sig)
                synthetic_id = -(int(cid) * 1000 + len(choices) + 1)
                choices.append((float(assignment.total_cost), synthetic_id,
                                moves, assignment))
                self.jodap.stats["cooptimal_parent_boundary_reuses"] += 1
                if source == "cooptimal_continuation":
                    self.jodap.stats["cooptimal_continuation_reuses"] += 1
                self.jodap._diag(
                    "parent_boundary_cooptimal_reuse",
                    component=component.component_id, parent_component=cid,
                    source=source, total_cost=float(assignment.total_cost),
                    move_count=len(moves))

            def _continuation_rank(choice):
                _cost, _nid, _moves, _assignment = choice
                repaired = any(
                    isinstance(row, dict) and (
                        row.get("relation_kind") is not None or
                        row.get("borrowed_object_sources") or
                        row.get("relation_claimed_objects"))
                    for row in (_assignment.object_bindings or ()))
                return (_cost, 0 if repaired else 1, len(_moves), _nid)
            choices.sort(key=_continuation_rank)
            if not choices:
                self.jodap.stats["parent_boundary_fallbacks"] += 1
                self.jodap._diag(
                    "parent_boundary_missing", component=component.component_id,
                    parent_component=cid, reason="no_complete_certified_boundary",
                    current_goal=pst.current_goal,
                    current_event_count=len(pst.current_event_ids),
                    node_count=len(pst.nodes),
                    has_current_assignment=pst.current_assignment is not None,
                    has_incumbent_assignment=pst.incumbent_assignment is not None)
                return None
            groups.append((cid, choices[:max(1, per_parent)]))
        return groups

    def _best_first_parent_compositions(self, component: ComponentState,
                                        max_compositions: int = 16) \
            -> List[Tuple[List[SymbolicMove], JointAssignment]]:
        """Compose the optimal boundary first, then reopen one parent at a time.

        Enumerating a Cartesian product of parent frontiers is exactly the work
        a merge optimization is supposed to avoid.  The first candidate joins
        all certified parent optima.  If the bridge cannot be explained from
        that state, alternatives are generated by changing *one* parent
        boundary state while keeping all other parents at their optimum.  Only
        after these bridge-local repairs fail does the ordinary merged root A*
        remain as the complete fallback.
        """
        groups = self._parent_boundary_candidates(component)
        if not groups:
            return []
        pools = [g[1] for g in groups]

        candidates: List[Tuple[float, Tuple[int, ...], bool]] = []
        optimum = tuple(0 for _ in pools)
        candidates.append((float(sum(pool[0][0] for pool in pools)), optimum, False))
        # Selectively reopen exactly one parent.  Order all such repairs by the
        # resulting parent-prefix lower bound.
        for dim, pool in enumerate(pools):
            for alt in range(1, len(pool)):
                idx = list(optimum)
                idx[dim] = alt
                idx = tuple(idx)
                score = float(sum(pools[i][idx[i]][0] for i in range(len(pools))))
                candidates.append((score, idx, True))
        candidates.sort(key=lambda x: (x[0], x[1]))

        out: List[Tuple[List[SymbolicMove], JointAssignment]] = []
        for _score, idx, is_repair in candidates[:max_compositions]:
            self.jodap.stats["merge_frontier_heap_pops"] += 1
            pieces = []
            moves: List[SymbolicMove] = []
            seen_events: Set[str] = set()
            partial_order_ok = True
            parent_object_sets: List[Set[str]] = []
            for i, choice_idx in enumerate(idx):
                _g, _nid, pmoves, assignment = pools[i][choice_idx]
                event_ids = {m.event_id for m in pmoves
                             if m.kind in ("log", "sync") and m.event_id is not None}
                if seen_events.intersection(event_ids):
                    partial_order_ok = False
                    break
                seen_events.update(event_ids)
                objs: Set[str] = set()
                for row in assignment.object_bindings:
                    objs.update(str(o) for o in row.get("objects", ()) if o is not None)
                parent_object_sets.append(objs)
                pieces.append((assignment, pmoves))
                moves.extend(pmoves)
            if not partial_order_ok:
                self.jodap.stats["merge_incompatible_partial_order"] += 1
                self.jodap.stats["merge_frontier_incompatible"] += 1
                continue
            # Independent parent components must not already share a concrete
            # object. If they do, the checkpoint partition is inconsistent.
            overlap = False
            seen_objs: Set[str] = set()
            claimed_by_parent: List[Set[str]] = []
            for assignment, _pmoves in pieces:
                claimed: Set[str] = set()
                for row in assignment.object_bindings or ():
                    claimed.update(str(o) for o in row.get("relation_claimed_objects", ()) or ())
                    for pair in row.get("borrowed_object_sources", ()) or ():
                        try:
                            claimed.add(str(pair[0]))
                        except Exception:
                            pass
                claimed_by_parent.append(claimed)
            for idx, objs in enumerate(parent_object_sets):
                shared = seen_objs.intersection(objs)
                if shared:
                    prior_claimed = set().union(*claimed_by_parent[:idx]) if idx else set()
                    if not shared.issubset(claimed_by_parent[idx] | prior_claimed):
                        overlap = True
                        break
                    self.jodap.stats["object_relation_virtual_dependency_reuses"] += len(shared)
                seen_objs.update(objs)
            if overlap:
                self.jodap.stats["merge_incompatible_object_binding"] += 1
                self.jodap.stats["merge_frontier_incompatible"] += 1
                continue

            self.jodap.stats["merge_direct_compositions"] += 1
            if is_repair:
                self.jodap.stats["merge_one_parent_repairs"] += 1
            composed = self._compose_parent_assignments(pieces)
            if composed is None:
                self.jodap.stats["merge_frontier_incompatible"] += 1
                continue
            out.append((moves, composed))

        self.jodap.stats["merge_frontier_compositions"] += len(out)
        return out

    def _event_has_structural_sync_candidate(self, component: ComponentState,
                                             event: StreamEvent) -> bool:
        """Return whether any visible transition can *structurally* sync.

        This is deliberately stronger than label matching and deliberately
        independent of the current marking/data valuation.  Scalar object
        parameters require exactly one participating object, while a list
        parameter absorbs the remaining objects of its base type.  Hence a
        ``ship`` transition with one scalar ORDER and one PRODUCT LIST cannot
        synchronize with an event containing two ORDER objects.  Whenever the
        declaration is ambiguous (e.g. several list parameters of one type), we
        conservatively report that a candidate may exist rather than proving an
        unavoidable log move incorrectly.
        """
        observed_by_type: Dict[str, int] = {}
        for obj in event.objects:
            typ = component.observation_formula.object_types.get(obj)
            if typ is None:
                return True  # unknown type: cannot prove impossibility
            observed_by_type[typ] = observed_by_type.get(typ, 0) + 1

        for transition in self.visible_by_label.get(event.activity, ()):
            try:
                decls = self.jodap._lazy_unique_object_decls(self.net_metadata, transition)
            except Exception:
                # The normal helper needs the imported CoCoMoT net.  Fall back
                # to the already available structural metadata when possible.
                try:
                    decls = self.jodap._lazy_unique_object_decls(self.net_metadata, transition)
                except Exception:
                    # If declaration extraction is unavailable, do not claim
                    # structural impossibility.
                    return True

            scalars: Dict[str, int] = {}
            lists: Dict[str, int] = {}
            declared_types: Set[str] = set()
            ambiguous = False
            for name, typ in decls.items():
                base = typ[:typ.rfind(" LIST")] if "LIST" in typ else typ
                declared_types.add(base)
                if "LIST" in typ:
                    lists[base] = lists.get(base, 0) + 1
                    if lists[base] > 1:
                        ambiguous = True
                else:
                    scalars[base] = scalars.get(base, 0) + 1
            if ambiguous:
                return True
            if any(typ not in declared_types for typ in observed_by_type):
                continue

            compatible = True
            for typ in declared_types:
                obs = observed_by_type.get(typ, 0)
                fixed = scalars.get(typ, 0)
                if lists.get(typ, 0):
                    if obs < fixed:
                        compatible = False
                        break
                elif obs != fixed:
                    compatible = False
                    break
            if compatible:
                return True
        return False

    @staticmethod
    def _log_extended_assignment(parent: JointAssignment, log_cost: int) -> JointAssignment:
        return JointAssignment(
            total_cost=float(parent.total_cost + log_cost),
            object_bindings=list(parent.object_bindings),
            data_assignments=list(parent.data_assignments),
            object_attribute_assignments=list(parent.object_attribute_assignments),
            marking_signature=parent.marking_signature,
            data_state_signature=parent.data_state_signature,
            token_data_signature=parent.token_data_signature,
            data_provenance_signature=parent.data_provenance_signature,
            solve_seconds=0.0, encode_seconds=0.0,
        )

    def _apply_zero_cost_fresh_creation(
            self, component: ComponentState, assignment: JointAssignment,
            object_id: str, transition: Dict[str, Any], model_step: int
    ) -> Optional[JointAssignment]:
        """Apply a simple invisible ``nu`` creation directly to a certified state.

        Real OCEL merge events often introduce the bridge object (for example a
        PACKAGE) at the same observation that connects previously independent
        parents.  Replaying all parent histories through JODAP just to create
        that known fresh object defeats incremental merging.  This helper
        handles only the conservative DOPID creation shape: invisible, no input
        arcs, no guard/data dependency, and output arcs whose sole object
        inscription is a ``nu`` variable of the observed object's type.
        Unsupported creation transitions simply return ``None`` and retain the
        exact fallback.
        """
        if not transition.get("invisible", False) or transition.get("constraint") is not None:
            return None
        tid = transition.get("id")
        try:
            net = self.jodap._static_net or self.jodap.new_net()
            self.jodap._static_net = net
            nt = next(t for t in net._transitions if t.get("id") == tid)
        except Exception:
            return None
        if any(a.get("target") == tid for a in getattr(net, "_arcs", ())):
            return None
        out_arcs = [a for a in getattr(net, "_arcs", ()) if a.get("source") == tid]
        if not out_arcs:
            return None
        obj_type = component.observation_formula.object_types.get(object_id)
        if obj_type is None:
            return None

        produced = []
        binding = None
        for arc in out_arcs:
            entries = [(str(n), str(tp)) for n, tp in arc.get("inscription", ())
                       if tp not in getattr(net, "_data_types", ()) ]
            data_entries = [(n, tp) for n, tp in arc.get("inscription", ())
                            if tp in getattr(net, "_data_types", ()) ]
            if data_entries or len(entries) != 1:
                return None
            name, typ = entries[0]
            if "nu" not in name or "LIST" in typ or typ != obj_type:
                return None
            binding = {name: object_id}
            try:
                place = next(p for p in net._places if p.get("id") == arc.get("target"))
                toks = self.jodap._lazy_arc_tokens(net, place, arc.get("inscription", ()), binding)
            except Exception:
                return None
            if not toks:
                return None
            produced.extend((place["id"], self.jodap._freeze_value(tok)) for tok in toks)

        marking = set(assignment.marking_signature or ())
        for item in produced:
            if item in marking:
                return None
            marking.add(item)
        bindings = list(assignment.object_bindings or ())
        bindings.append({
            "step": int(model_step), "transition_id": tid,
            "transition": transition.get("label"), "objects": (object_id,),
            "fresh_object": True,
        })
        return JointAssignment(
            total_cost=float(assignment.total_cost),
            object_bindings=bindings,
            data_assignments=list(assignment.data_assignments or ()),
            object_attribute_assignments=list(assignment.object_attribute_assignments or ()),
            marking_signature=tuple(sorted(marking, key=repr)),
            data_state_signature=tuple(assignment.data_state_signature or ()),
            token_data_signature=tuple(assignment.token_data_signature or ()),
            data_provenance_signature=tuple(assignment.data_provenance_signature or ()),
            solve_seconds=0.0, encode_seconds=0.0,
        )

    def _prepare_direct_merge_bridge(
            self, component: ComponentState, moves: Sequence[SymbolicMove],
            assignment: JointAssignment, event: StreamEvent
    ) -> Optional[Tuple[List[SymbolicMove], JointAssignment]]:
        """Create observed bridge objects locally before a direct merge sync."""
        marked = self._marked_objects_from_assignment(component, assignment)
        missing = sorted(
            (o for o in event.objects if o not in marked),
            key=lambda o: (component.observation_formula.object_types.get(o, ""), o),
        )
        if not missing:
            return list(moves), assignment
        out_moves = list(moves)
        current = assignment
        step = sum(1 for m in out_moves if m.kind in ("model", "sync"))
        for obj in missing:
            typ = component.observation_formula.object_types.get(obj)
            t = self._nu_transition_for_type(typ) if typ is not None else None
            if t is None:
                self.jodap.stats["merge_bridge_creation_declines"] += 1
                self.jodap._diag(
                    "merge_bridge_creation_decline", component=component.component_id,
                    event_id=event.event_id, activity=event.activity, object_id=obj,
                    object_type=typ, reason="no_simple_nu_transition_for_type",
                    missing_objects=list(missing),
                )
                return None
            nxt = self._apply_zero_cost_fresh_creation(component, current, obj, t, step)
            if nxt is None:
                self.jodap.stats["merge_bridge_creation_declines"] += 1
                self.jodap._diag(
                    "merge_bridge_creation_decline", component=component.component_id,
                    event_id=event.event_id, activity=event.activity, object_id=obj,
                    object_type=typ, transition_id=t.get("id"), transition=t.get("label"),
                    reason="simple_nu_creation_rejected", missing_objects=list(missing),
                    current_marking_size=len(current.marking_signature or ()),
                )
                return None
            current = nxt
            out_moves.append(SymbolicMove("model", transition_id=t["id"],
                                          transition_label=t.get("label")))
            step += 1
            self.jodap.stats["merge_bridge_fresh_creations"] += 1
        return out_moves, current

    def _try_cooptimal_orphan_parent_rewrite(
            self, component: ComponentState, state: SearchState,
            bridge_moves: Sequence[SymbolicMove], bridge_assignment: JointAssignment,
            warm: SearchNode, merge_event: StreamEvent
    ) -> Optional[Tuple[List[SymbolicMove], JointAssignment, SearchNode]]:
        """Rewrite a co-optimal orphan log parent into an extendable model state.

        A missing relation can make an earlier event such as ``create package``
        appear as an isolated one-object component.  Its cost-1 log explanation
        is a perfectly valid optimal prefix alignment, but it leaves no package
        token for a later observed ``send package`` merge.  At the later merge
        the observation finally supplies the ITEM relation(s) needed to build an
        equally costly model-side history.  This helper performs that rewrite
        *at the current prefix*, using only information now observed:

        ``LOG create-package(p)``  (cost k)
            -> ``nu-package(p); SYNC create-package(p, I)`` (same total cost k)

        The old log cost is reused as the already-paid object-relation mismatch;
        no cost is refunded or added.  The rewrite is accepted only when the
        symmetric-difference cardinality equals the original log-move cost and
        the concrete historical transition is locally enabled with the current
        merge event's object set.  Thus total-cost optimality is preserved while
        producing a continuation state that the current bridge can extend.
        """
        self.jodap.stats["merge_cooptimal_parent_rewrite_attempts"] += 1
        merge_objects = {str(o) for o in merge_event.objects}
        if not merge_objects:
            return None

        # Search backwards for an earlier LOG event sharing a stable non-ITEM
        # anchor with the current bridge and having a LIST-valued model firing.
        candidates = []
        for idx in range(len(bridge_moves) - 1, -1, -1):
            move = bridge_moves[idx]
            if move.kind != "log" or move.event_id is None:
                continue
            old_event = component.observation_formula.events.get(move.event_id)
            if old_event is None:
                continue
            old_objects = {str(o) for o in old_event.objects}
            common = old_objects & merge_objects
            anchors = {
                o for o in common
                if component.observation_formula.object_types.get(o) != "ITEM"
            }
            if not anchors:
                continue
            for t in self.visible_by_label.get(old_event.activity, ()):
                try:
                    net = self.jodap._query_slice_static_net()
                    nt = next(x for x in net._transitions if x.get("id") == t.get("id"))
                    decls = self.jodap._lazy_unique_object_decls(net, nt)
                except Exception:
                    continue
                if not any("LIST" in str(tp) for tp in decls.values()):
                    continue
                # The current bridge may contain additional non-ITEM objects;
                # retain only types expected by the historical transition.
                scalar_types = {str(tp).replace(" LIST", "") for tp in decls.values()}
                model_objects = {
                    o for o in merge_objects
                    if component.observation_formula.object_types.get(o) in scalar_types
                }
                model_objects |= anchors
                if not old_objects.issubset(model_objects):
                    continue
                relation_cost = len(old_objects ^ model_objects)
                old_log_cost = len(old_event.objects)
                # Reusing the old log cost is sound only when the replacement
                # relation mismatch has exactly the same unit-weighted cost.
                if relation_cost <= 0 or relation_cost != old_log_cost:
                    continue
                candidates.append((idx, old_event, t, tuple(sorted(model_objects)), relation_cost))

        if not candidates:
            self.jodap.stats["merge_cooptimal_parent_rewrite_failures"] += 1
            self.jodap._diag(
                "merge_cooptimal_parent_rewrite_decline",
                component=component.component_id, event_id=merge_event.event_id,
                activity=merge_event.activity, reason="no_matching_orphan_log_parent")
            return None

        self.jodap.stats["merge_cooptimal_parent_rewrite_candidates"] += len(candidates)
        for idx, old_event, transition, model_objects, relation_cost in candidates:
            # ``bridge_assignment`` is the already-composed model state after
            # any simple fresh bridge-object creation.  The log event did not
            # alter this state, so it is exactly the predecessor state needed
            # for the historical synchronous firing.
            synthetic_parent = SearchNode(
                node_id=-1000001,
                consumed=frozenset(set(warm.consumed) - {old_event.event_id}),
                event_order=tuple(e for e in warm.event_order if e != old_event.event_id),
                model_depth=warm.model_depth,
                g=float(bridge_assignment.total_cost), h=0.0,
                assignment_cost=float(bridge_assignment.total_cost),
                model_signature=warm.model_signature,
                move_signature=tuple(
                    sig for sig in warm.move_signature
                    if not (len(sig) >= 2 and sig[0] == "log" and sig[1] == old_event.event_id)
                ),
            )
            synthetic_child = SearchNode(
                node_id=-1000002,
                consumed=frozenset(set(synthetic_parent.consumed) | {old_event.event_id}),
                event_order=synthetic_parent.event_order + (old_event.event_id,),
                model_depth=synthetic_parent.model_depth + 1,
                g=float(bridge_assignment.total_cost), h=0.0,
                assignment_cost=float(bridge_assignment.total_cost),
                model_signature=synthetic_parent.model_signature + (transition["id"],),
                move_signature=synthetic_parent.move_signature +
                    (("sync", old_event.event_id, transition["id"]),),
            )
            rewritten = self.jodap.check_zero_cost_sync_extension(
                component, state, synthetic_parent, synthetic_child,
                old_event, transition["id"], bridge_assignment,
                model_objects_override=model_objects,
                extra_cost=0.0, repair_tag="cooptimal_parent_rewrite")
            if rewritten is None:
                self.jodap._diag(
                    "merge_cooptimal_parent_rewrite_candidate_decline",
                    component=component.component_id, event_id=merge_event.event_id,
                    orphan_event_id=old_event.event_id,
                    orphan_activity=old_event.activity,
                    transition_id=transition.get("id"),
                    model_objects=list(model_objects),
                    detail_reason=self.jodap._last_zero_cost_sync_decline_reason)
                continue

            # Mark the rewritten historical binding as the already-paid object
            # deviation so downstream lifecycle events can carry it forward.
            if rewritten.object_bindings:
                meta = self._relation_repair_metadata(
                    component, old_event.objects, model_objects)
                last = rewritten.object_bindings[-1]
                meta.update({
                    "step": last.get("step"),
                    "transition_id": last.get("transition_id"),
                    "transition": last.get("transition"),
                    "objects": tuple(sorted(model_objects)),
                    "object_cost": relation_cost,
                    "cooptimal_parent_rewrite": True,
                })
                rewritten.object_bindings[-1] = meta

            # Replace the historical log move by the synchronous model firing.
            # Any silent fresh creation needed for the package is already part
            # of ``bridge_moves``/``bridge_assignment`` from bridge preparation.
            new_moves = list(bridge_moves)
            new_moves.pop(idx)
            new_moves.append(SymbolicMove(
                "sync", event_id=old_event.event_id,
                transition_id=transition["id"],
                transition_label=transition.get("label")))

            # Materialize the rewritten parent boundary and retry the *current*
            # bridge from that concrete state.
            rewritten_warm = self._materialize_warm_path(
                component, state, new_moves, rewritten)
            if rewritten_warm is None:
                continue
            state.assignments_by_node[rewritten_warm.node_id] = rewritten
            self.jodap.stats["merge_cooptimal_parent_rewrite_hits"] += 1
            self.jodap._diag(
                "merge_cooptimal_parent_rewrite_success",
                component=component.component_id, event_id=merge_event.event_id,
                orphan_event_id=old_event.event_id,
                orphan_activity=old_event.activity,
                model_objects=list(model_objects), relation_cost=relation_cost,
                total_cost=float(rewritten.total_cost),
                warm_node=rewritten_warm.node_id)
            return new_moves, rewritten, rewritten_warm

        self.jodap.stats["merge_cooptimal_parent_rewrite_failures"] += 1
        return None

    def _install_direct_merge_compositions(self, component: ComponentState,
                                           state: SearchState) -> bool:
        """Compose certified parent boundaries and solve only the new bridge.

        Parent histories are treated as certified checkpoints.  We first join
        their operational witnesses, then inspect only the observation that
        caused the merge.  The optimal-parent join is tried first, followed by
        one-parent repairs.  The ordinary merged root remains in OPEN as the
        complete fallback if the bridge cannot be resolved locally.
        """
        merge_eid = getattr(component, "merge_event_id", None)
        if not merge_eid or merge_eid not in component.observation_formula.events:
            return False
        event = component.observation_formula.events[merge_eid]
        self.jodap.stats["merge_fast_attempts"] += 1
        self.jodap._diag(
            "merge_fast_attempt", component=component.component_id, event_id=merge_eid,
            activity=event.activity, event_objects=len(event.objects),
            merged_from=list(getattr(component, "merged_from", ()) or ()),
        )
        compositions = self._best_first_parent_compositions(component)
        if not compositions:
            self.jodap.stats["merge_fast_no_compositions"] += 1
            self.jodap._diag(
                "merge_fast_decline", component=component.component_id, event_id=merge_eid,
                reason="no_parent_compositions",
            )
            return False

        installed = False
        structural_sync = self._event_has_structural_sync_candidate(component, event)
        if structural_sync:
            self.jodap.stats["merge_fast_structural_sync"] += 1
        self.jodap._diag(
            "merge_fast_structure", component=component.component_id, event_id=merge_eid,
            structural_sync=bool(structural_sync), compositions=len(compositions),
        )
        # The first composition is the independently optimal parent boundary.
        parent_lb = float(compositions[0][1].total_cost)
        # The parent components are independently certified prefixes. Any
        # explanation of the merged prefix must pay at least their already
        # optimal accumulated cost; the bridge can add cost but cannot refund
        # past deviations. Carry this lower bound forward so a synchronous
        # bridge that adds *zero incremental cost* (e.g. parent cost 1 -> child
        # cost 1) is recognized as globally optimal instead of being treated as
        # a failed absolute-zero merge.
        state.proven_prefix_lower_bound = max(
            state.proven_prefix_lower_bound, parent_lb
        )
        self.jodap._diag(
            "merge_parent_lower_bound", component=component.component_id,
            event_id=merge_eid, parent_lower_bound=parent_lb,
            prefix_lower_bound=float(state.proven_prefix_lower_bound),
        )
        if not structural_sync:
            # No marking/data assignment can make the bridge synchronous because
            # its hard activity/type/cardinality signature is incompatible.  The
            # independent parent optima plus the bridge log cost are therefore a
            # valid lower bound for the merged prefix.
            state.proven_prefix_lower_bound = max(
                state.proven_prefix_lower_bound,
                parent_lb + float(len(event.objects)))
            self.jodap.stats["merge_structural_log_proofs"] += 1

        for comp_index, (moves, assignment) in enumerate(compositions):
            bridge_moves = list(moves)
            bridge_assignment = assignment
            if structural_sync:
                prepared_bridge = self._prepare_direct_merge_bridge(
                    component, bridge_moves, bridge_assignment, event
                )
                if prepared_bridge is not None:
                    bridge_moves, bridge_assignment = prepared_bridge
                else:
                    self.jodap.stats["merge_fast_bridge_prepare_failures"] += 1
                    self.jodap._diag(
                        "merge_fast_decline", component=component.component_id, event_id=merge_eid,
                        composition_index=comp_index, reason="bridge_prepare_failed",
                    )

            warm = self._materialize_warm_path(component, state, bridge_moves, bridge_assignment)
            if warm is None:
                self.jodap.stats["merge_incompatible_marking"] += 1
                self.jodap.stats["merge_fast_materialize_failures"] += 1
                self.jodap._diag(
                    "merge_fast_decline", component=component.component_id, event_id=merge_eid,
                    composition_index=comp_index, reason="materialize_warm_path_failed",
                )
                continue
            state.assignments_by_node[warm.node_id] = bridge_assignment
            installed = True
            self.jodap.stats["merge_direct_hits"] += 1
            self.jodap.stats["merge_bridge_checks"] += 1

            if structural_sync:
                self.jodap.stats["merge_fast_sync_attempts"] += 1
                self.jodap._diag(
                    "merge_fast_sync_attempt", component=component.component_id, event_id=merge_eid,
                    composition_index=comp_index, warm_node=warm.node_id,
                    parent_cost=float(bridge_assignment.total_cost),
                )
                # The certified parent markings are composed directly.  Any
                # observed bridge object that is introduced by a simple silent
                # nu transition has already been added analytically above, so
                # the guarded synchronous firing can be checked against exactly
                # the event's concrete object/list binding without replaying the
                # historical parents.
                if self._try_zero_cost_sync_extension(component, state, warm, merge_eid):
                    self.jodap.stats["merge_bridge_hits"] += 1
                    self.jodap.stats["merge_bridge_guarded_zero_cost_proofs"] += 1
                    self.jodap.stats["merge_fast_successes"] += 1
                    self.jodap._diag(
                        "merge_fast_success", component=component.component_id, event_id=merge_eid,
                        composition_index=comp_index, upper_bound=float(state.upper_bound),
                        lower_bound=float(state.proven_prefix_lower_bound),
                    )
                    if comp_index > 0:
                        self.jodap.stats["merge_one_parent_repair_hits"] += 1
                    if state.upper_bound <= state.proven_prefix_lower_bound + 1e-9:
                        break
                    continue

                # If the bridge-local zero-cost checker has already diagnosed
                # a missing required input token, prioritize the dedicated
                # one-step producer repair before relation/data repair.  This
                # is the merge counterpart of the fresh-object fast path: a
                # merge-triggering ``create package`` may need exactly one
                # omitted visible predecessor (e.g. ``pick item``).  The more
                # general relation machinery can otherwise classify/expand the
                # LIST binding first and delay this simple exact repair until
                # the observation timeout.
                merge_decline = getattr(
                    self.jodap, "_last_zero_cost_sync_decline_detail", None
                ) or {}
                prioritized_missing_input = (
                    merge_decline.get("reason") == "required_input_token_not_marked"
                )
                if prioritized_missing_input and self._try_one_step_missing_input_repair(
                        component, state, warm, merge_eid):
                    self.jodap.stats["merge_bridge_hits"] += 1
                    self.jodap.stats["merge_fast_successes"] += 1
                    self.jodap._diag(
                        "merge_fast_success", component=component.component_id,
                        event_id=merge_eid, composition_index=comp_index,
                        upper_bound=float(state.upper_bound),
                        lower_bound=float(state.proven_prefix_lower_bound),
                        via="one_step_repair_prioritized",
                    )
                    break

                # A co-optimal orphan parent may have been represented only as
                # a log move at its earlier prefix (e.g. create-package(p) after
                # the sole ITEM relation was removed).  The current merge event
                # can now reveal the missing relation. Rewrite that parent to an
                # equally costly model-side firing and retry the current bridge
                # before opening any generic repair/A* search.
                rewritten_parent = self._try_cooptimal_orphan_parent_rewrite(
                    component, state, bridge_moves, bridge_assignment, warm, event)
                if rewritten_parent is not None:
                    _rw_moves, _rw_assignment, _rw_warm = rewritten_parent
                    if self._try_zero_cost_sync_extension(
                            component, state, _rw_warm, merge_eid):
                        self.jodap.stats["merge_bridge_hits"] += 1
                        self.jodap.stats["merge_fast_successes"] += 1
                        self.jodap._diag(
                            "merge_fast_success", component=component.component_id,
                            event_id=merge_eid, composition_index=comp_index,
                            upper_bound=float(state.upper_bound),
                            lower_bound=float(state.proven_prefix_lower_bound),
                            via="cooptimal_parent_rewrite")
                        if state.upper_bound <= state.proven_prefix_lower_bound + 1e-9:
                            break
                        continue

                # Before interpreting a failed aggregate as a data error or a
                # missing token as missing control-flow, try a one-relation
                # object repair.  This covers both an extra observed ITEM and a
                # missing observed ITEM while keeping the event itself intact.
                object_rel_handled, object_rel_proven = \
                    self._try_object_relation_deviation_sync_extension(
                        component, state, warm, merge_eid)
                if object_rel_proven:
                    self.jodap.stats["merge_bridge_hits"] += 1
                    self.jodap.stats["merge_fast_successes"] += 1
                    self.jodap._diag(
                        "merge_fast_success", component=component.component_id,
                        event_id=merge_eid, composition_index=comp_index,
                        upper_bound=float(state.upper_bound),
                        lower_bound=float(state.proven_prefix_lower_bound),
                        via="object_relation_repair",
                    )
                    break

                # If the bridge is structurally enabled and the only local
                # failure is a concrete false aggregate guard, try a single
                # soft object-attribute/data repair before reopening A*.  Do not
                # reinterpret a recognized relation mismatch as a data repair.
                if (not object_rel_handled) and self._try_guard_data_deviation_sync_extension(
                        component, state, warm, merge_eid):
                    self.jodap.stats["merge_bridge_hits"] += 1
                    self.jodap.stats["merge_fast_successes"] += 1
                    self.jodap._diag(
                        "merge_fast_success", component=component.component_id,
                        event_id=merge_eid, composition_index=comp_index,
                        upper_bound=float(state.upper_bound),
                        lower_bound=float(state.proven_prefix_lower_bound),
                        via="guard_data_repair",
                    )
                    break

                # A common real deviation is a missing visible preparation
                # (e.g. one omitted ``pick item``) immediately before the bridge.
                # The local sync checker has already identified the concrete
                # missing token/place, so try exactly one model repair before
                # opening the unrestricted merged A*.
                if (not prioritized_missing_input) and (not object_rel_handled) \
                        and self._try_one_step_missing_input_repair(
                            component, state, warm, merge_eid):
                    self.jodap.stats["merge_bridge_hits"] += 1
                    self.jodap.stats["merge_fast_successes"] += 1
                    self.jodap._diag(
                        "merge_fast_success", component=component.component_id,
                        event_id=merge_eid, composition_index=comp_index,
                        upper_bound=float(state.upper_bound),
                        lower_bound=float(state.proven_prefix_lower_bound),
                        via="one_step_repair",
                    )
                    break
                if self._try_cost_guided_local_repair(
                        component, state, warm, merge_eid):
                    self.jodap.stats["merge_bridge_hits"] += 1
                    self.jodap.stats["merge_fast_successes"] += 1
                    self.jodap._diag(
                        "merge_fast_success", component=component.component_id,
                        event_id=merge_eid, composition_index=comp_index,
                        upper_bound=float(state.upper_bound),
                        lower_bound=float(state.proven_prefix_lower_bound),
                        via="cost_guided_local_repair",
                    )
                    break

                self.jodap.stats["merge_fast_sync_failures"] += 1
                self.jodap._diag(
                    "merge_fast_decline", component=component.component_id, event_id=merge_eid,
                    composition_index=comp_index, reason="zero_cost_sync_extension_failed",
                    detail_reason=self.jodap._last_zero_cost_sync_decline_reason,
                )
            else:
                # The bridge is provably a log move.  Extend the certified
                # boundary analytically; no solver or joint-history search is
                # required.
                child = self._add_candidate(
                    component, state, warm, SymbolicMove("log", event_id=merge_eid),
                    evaluate=False)
                if child is None:
                    continue
                ext = self._log_extended_assignment(assignment, len(event.objects))
                child.g = ext.total_cost
                child.assignment_cost = ext.total_cost
                state.assignments_by_node[child.node_id] = ext
                self.jodap.stats["merge_bridge_log_hits"] += 1
                if ext.total_cost < state.upper_bound - 1e-9:
                    state.upper_bound = ext.total_cost
                    state.incumbent_moves = tuple(self._path(state, child.node_id))
                    state.incumbent_assignment = ext
                    state.incumbent_offline = False
                if state.upper_bound <= state.proven_prefix_lower_bound + 1e-9:
                    break

        # If no local bridge proof reached the lower bound, the normal merged
        # A* root remains available.  Count this explicitly; there is no replay
        # warm-start/JODAP validation layer in between anymore.
        if not (state.incumbent_assignment is not None
                and state.proven_prefix_lower_bound > 0
                and state.upper_bound <= state.proven_prefix_lower_bound + 1e-9):
            self.jodap.stats["merge_full_search_fallbacks"] += 1
        return installed

    def _parent_checkpoint_moves(self, component: ComponentState) -> List[SymbolicMove]:
        """Compose the retained optimal parent prefixes up to a merge checkpoint.

        Before the connecting observable unit, the parent components were
        monitored independently.  Their retained optimal paths therefore form a
        cheap, already-solved checkpoint candidate in the merged search.  The
        ordinary root is kept as an exact fallback, so this warm start cannot
        remove a potentially better merged explanation.
        """
        if not getattr(component, "merged_from", ()):
            return []
        parts = []
        for cid in sorted(component.merged_from):
            pst = self.searches.get(cid)
            if pst is None or pst.current_goal is None or pst.current_goal not in pst.nodes:
                return []
            pmoves = self._path(pst, pst.current_goal)
            first_pos = 10**12
            for m in pmoves:
                if m.kind in ("log", "sync") and m.event_id is not None:
                    u = next((u for u in component.units if u.event and u.event.event_id == m.event_id), None)
                    if u is not None:
                        first_pos = min(first_pos, u.position)
            parts.append((first_pos, cid, pmoves))
        parts.sort(key=lambda x: (x[0], x[1]))
        out: List[SymbolicMove] = []
        for _pos, _cid, pmoves in parts:
            out.extend(pmoves)
        return out


    def _parent_frontier_compositions(self, component: ComponentState,
                                      per_parent: int = 3,
                                      max_compositions: int = 12) \
            -> List[List[SymbolicMove]]:
        """Compose a small set of compatible parent frontier alternatives.

        This is a safe warm-start only: the merged search still retains its
        ordinary root, so omitted combinations cannot affect completeness. The
        goal is to reuse already explored parent alternatives instead of
        rediscovering both independent histories after a merge.
        """
        if not getattr(component, "merged_from", ()):
            return []
        parent_choices: List[Tuple[int, int, List[Tuple[float, int, List[SymbolicMove]]]]] = []
        for cid in sorted(component.merged_from):
            pst = self.searches.get(cid)
            if pst is None:
                return []
            full = frozenset(pst.current_event_ids)
            candidates: List[Tuple[float, int, List[SymbolicMove]]] = []
            seen: Set[Tuple[Any, ...]] = set()
            for node in pst.nodes.values():
                if node.consumed != full or node.g == float("inf"):
                    continue
                sig = self._canonical_move_signature(node.move_signature)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append((node.g, node.model_depth, self._path(pst, node.node_id)))
            candidates.sort(key=lambda x: (x[0], x[1], len(x[2])))
            candidates = candidates[:max(1, per_parent)]
            if not candidates and pst.current_goal is not None and pst.current_goal in pst.nodes:
                n = pst.nodes[pst.current_goal]
                candidates = [(n.g, n.model_depth, self._path(pst, n.node_id))]
            if not candidates:
                return []
            first_pos = 10**12
            for _g, _d, pmoves in candidates:
                for m in pmoves:
                    if m.kind in ("log", "sync") and m.event_id is not None:
                        u = next((u for u in component.units
                                  if u.event and u.event.event_id == m.event_id), None)
                        if u is not None:
                            first_pos = min(first_pos, u.position)
            parent_choices.append((first_pos, cid, candidates))

        parent_choices.sort(key=lambda x: (x[0], x[1]))
        pools = [x[2] for x in parent_choices]
        out: List[List[SymbolicMove]] = []
        seen_paths: Set[Tuple[Any, ...]] = set()
        for combo in itertools.product(*pools):
            moves: List[SymbolicMove] = []
            for _g, _d, pmoves in combo:
                moves.extend(pmoves)
            sig = tuple((m.kind, m.event_id, m.transition_id) for m in moves)
            if sig in seen_paths:
                continue
            seen_paths.add(sig)
            out.append(moves)
            if len(out) >= max_compositions:
                break
        self.jodap.stats["merge_frontier_compositions"] += len(out)
        return out

    def _materialize_warm_path(self, component: ComponentState, state: SearchState,
                               moves: Sequence[SymbolicMove],
                               assignment: JointAssignment) -> Optional[SearchNode]:
        """Insert one already-evaluated checkpoint path as an additional A* source."""
        if not moves or 0 not in state.nodes:
            return None
        parent = state.nodes[0]
        for index, move in enumerate(moves):
            consumed = parent.consumed
            event_order = parent.event_order
            model_depth = parent.model_depth
            model_sig = parent.model_signature
            move_sig = parent.move_signature + ((move.kind, move.event_id, move.transition_id),)
            if move.kind in ("log", "sync"):
                if move.event_id in consumed:
                    return None
                consumed = frozenset(set(consumed) | {move.event_id})
                event_order = event_order + (move.event_id,)
            if move.kind in ("model", "sync"):
                model_depth += 1
                if model_depth > state.model_bound:
                    return None
                model_sig = model_sig + (move.transition_id,)
            node = SearchNode(
                node_id=state.new_id(), consumed=consumed, event_order=event_order,
                model_depth=model_depth, g=float("inf"),
                h=self._heuristic(component, consumed), model_signature=model_sig,
                move_signature=move_sig
            )
            state.nodes[node.node_id] = node
            state.predecessor[node.node_id] = (parent.node_id, move)
            parent = node
        parent.g = assignment.total_cost
        parent.assignment_cost = assignment.total_cost
        sig = self._signature(parent)
        existing = state.signatures.get(sig)
        if existing is not None and state.nodes.get(existing, parent).g <= parent.g:
            return state.nodes.get(existing)
        state.signatures[sig] = parent.node_id
        state.push(parent)
        self.jodap.stats["merge_warm_starts"] += 1
        self.jodap.stats["merge_checkpoint_reused"] += 1
        return parent

    def _install_merge_checkpoint_warm_start(self, component: ComponentState,
                                             state: SearchState) -> None:
        # Start with the retained optimal parent paths, then add a small set of
        # already-explored parent frontier alternatives.  Every composition is
        # validated once against the merged JODAP before becoming an A* source.
        candidates: List[List[SymbolicMove]] = []
        moves = self._parent_checkpoint_moves(component)
        if moves:
            candidates.append(moves)
        candidates.extend(self._parent_frontier_compositions(component))
        if not candidates:
            return
        seen: Set[Tuple[Any, ...]] = set()
        merge_eid = getattr(component, "merge_event_id", None)
        for moves in candidates:
            key = tuple((m.kind, m.event_id, m.transition_id) for m in moves)
            if key in seen:
                continue
            seen.add(key)
            assignment, _ = self._evaluate_move_sequence(component, state, moves)
            if assignment is None:
                continue
            warm = self._materialize_warm_path(component, state, moves, assignment)
            if warm is None:
                continue
            self.jodap.stats["merge_frontier_hits"] += 1
            # Immediately expose the connecting observation from each compatible
            # checkpoint. The normal root remains in OPEN as an exact fallback.
            if merge_eid and merge_eid not in warm.consumed:
                self._expand(component, state, warm, include_model=False, only_event=merge_eid)

    def _parent_merge_seed(self, component: ComponentState) -> List[SymbolicMove]:
        if not getattr(component, "merged_from", ()):
            return []
        seed: List[SymbolicMove] = []
        consumed: Set[str] = set()
        for cid in sorted(component.merged_from):
            pst = self.searches.get(cid)
            if pst is None or pst.current_goal is None:
                return []
            pmoves = self._path(pst, pst.current_goal)
            seed.extend(pmoves)
            for m in pmoves:
                if m.kind in ("log", "sync") and m.event_id is not None:
                    consumed.add(m.event_id)
        # The observation that caused the merge (and any other not-yet-covered
        # events) is conservatively explained as a log move.
        for eid in self._topological_event_order(component):
            if eid not in consumed:
                seed.append(SymbolicMove("log", event_id=eid))
        return seed

    def _seed_incumbent(self, component: ComponentState, state: SearchState) -> None:
        # Merged components are no longer replayed through JODAP here. Their
        # certified parent boundary witnesses are composed directly by
        # _install_direct_merge_compositions(). This avoids the expensive joint
        # encoding before we know that any re-optimization is necessary.

        # The all-log prefix is analytically feasible: log moves do not change
        # the model marking or impose data/guard constraints.  Its cost is just
        # the configured log-move cost, which in this implementation equals the
        # number of participating objects.  Evaluating this seed through JODAP
        # previously caused the *only* eager fallback in long fitting traces.
        all_log = [SymbolicMove("log", event_id=eid)
                   for eid in self._topological_event_order(component)]
        all_log_cost = sum(
            len(component.observation_formula.events[m.event_id].objects)
            for m in all_log
        )
        if float(all_log_cost) < state.upper_bound - 1e-9:
            state.upper_bound = float(all_log_cost)
            state.incumbent_moves = tuple(all_log)
            state.incumbent_assignment = JointAssignment(total_cost=float(all_log_cost))
            state.incumbent_offline = False
            self.jodap.stats["upper_bound_seeded"] += 1
            self.jodap.stats["analytic_all_log_seeds"] += 1

    def _seed_offline_incumbent(self, component: ComponentState, state: SearchState) -> None:
        """Find a cheap feasible final completion for branch-and-bound.

        Starting from the current optimal prefix explanation, perform a small
        beam search over trailing model-only moves.  This is not used to prove
        optimality; it only supplies a feasible offline upper bound. The main
        A* search still proves/returns the optimum.
        """
        base = list(state.incumbent_moves) if state.incumbent_assignment is not None \
            and not state.incumbent_offline else []
        if not base and state.current_goal is not None and state.current_goal in state.nodes:
            base = self._path(state, state.current_goal)
        if not base:
            base = [SymbolicMove("log", event_id=eid)
                    for eid in self._topological_event_order(component)]

        # The prefix itself may already end in a final marking.
        final_assignment, _ = self._evaluate_move_sequence(
            component, state, base, require_final=True
        )
        if final_assignment is not None:
            state.upper_bound = final_assignment.total_cost
            state.incumbent_moves = tuple(base)
            state.incumbent_assignment = final_assignment
            state.incumbent_offline = True
            self.jodap.stats["offline_seeded"] += 1
            return

        max_extra = min(6, max(2, self.model_depth_margin + 1))
        beam_width = 24
        frontier: List[Tuple[float, List[SymbolicMove]]] = [(0.0, base)]
        best: Optional[Tuple[float, List[SymbolicMove], JointAssignment]] = None
        for _depth in range(1, max_extra + 1):
            nxt: List[Tuple[float, List[SymbolicMove]]] = []
            for _score, seq in frontier:
                last_model_tid = None
                if seq and seq[-1].kind == "model":
                    last_model_tid = seq[-1].transition_id
                for t in self.transitions:
                    tid = t["id"]
                    if last_model_tid is not None and last_model_tid > tid \
                            and self._independent(last_model_tid, tid):
                        continue
                    cand = seq + [SymbolicMove(
                        "model", transition_id=tid, transition_label=t.get("label")
                    )]
                    assn, _ = self._evaluate_move_sequence(component, state, cand)
                    if assn is None:
                        continue
                    nxt.append((assn.total_cost, cand))
                    fin, _ = self._evaluate_move_sequence(
                        component, state, cand, require_final=True
                    )
                    if fin is not None and (best is None or fin.total_cost < best[0]):
                        best = (fin.total_cost, cand, fin)
            if best is not None:
                break
            nxt.sort(key=lambda x: x[0])
            frontier = nxt[:beam_width]
            if not frontier:
                break

        if best is not None:
            state.upper_bound = best[0]
            state.incumbent_moves = tuple(best[1])
            state.incumbent_assignment = best[2]
            state.incumbent_offline = True
            self.jodap.stats["offline_seeded"] += 1

    def _expand(self, component: ComponentState, state: SearchState, node: SearchNode,
                *, include_model: bool = True, only_event: Optional[str] = None,
                allowed_transition_ids: Optional[Set[int]] = None) -> None:
        enabled = self._enabled_events(component, node)
        if only_event is not None:
            enabled = [e for e in enabled if e.event_id == only_event]

        # If a feasible incumbent exists, exact certification is a decision
        # problem: can anything with integer cost <= UB-1 still exist?  Carry
        # that budget into successor generation itself so operators whose
        # admissible minimum already exceeds the budget never become nodes.
        strict_cap = self._strict_improvement_cap(state)
        if strict_cap is not None:
            node_lb = (float(node.g) if node.g != float("inf") else 0.0) + float(node.h)
            if node_lb > float(strict_cap) + 1e-9:
                self.jodap.stats["strict_improvement_generation_node_closed"] += 1
                self.jodap._diag(
                    "strict_improvement_generation_node_closed",
                    component=component.component_id, node=node.node_id,
                    node_lower_bound=node_lb, improvement_cap=int(strict_cap),
                )
                return

        def generation_allows(move: SymbolicMove, consumed_after: FrozenSet[str]) -> bool:
            if strict_cap is None:
                return True
            lb = self._strict_generation_lower_bound(component, node, move, consumed_after)
            if lb <= float(strict_cap) + 1e-9:
                return True
            self.jodap.stats["strict_improvement_generation_pruned"] += 1
            if move.kind == "log":
                self.jodap.stats["strict_improvement_generation_log_pruned"] += 1
            elif move.kind == "model":
                self.jodap.stats["strict_improvement_generation_model_pruned"] += 1
            self.jodap._diag(
                "strict_improvement_generation_pruned",
                component=component.component_id, parent=node.node_id,
                move_kind=move.kind, event_id=move.event_id,
                transition_id=move.transition_id, lower_bound=lb,
                improvement_cap=int(strict_cap),
            )
            return False

        # Cheap structural filter before invoking JODAP. CoCoMoT's reachability
        # table ignores concrete data/object assignments and is therefore an
        # over-approximation: removing transitions absent from it is safe.
        reachable_ids = self.jodap.structurally_reachable_transition_ids(
            component, state, node
        )

        def zero_goal_proved() -> bool:
            return (not state.search_offline
                    and state.incumbent_assignment is not None
                    and state.incumbent_offline is False
                    and state.upper_bound <= 1e-9
                    and state.proven_prefix_lower_bound <= 1e-9)

        # Log and synchronous extensions consume one causally enabled event.
        # Log moves do not require a model transition and are always retained.
        for e in enabled:
            log_move = SymbolicMove("log", event_id=e.event_id)
            log_consumed = frozenset(set(node.consumed) | {e.event_id})
            if generation_allows(log_move, log_consumed):
                self._add_candidate(component, state, node, log_move)
            if zero_goal_proved():
                self.jodap.stats["exact_zero_cost_expansion_stops"] += 1
                return
            for t in self.visible_by_label.get(e.activity, []):
                if allowed_transition_ids is not None and t["id"] not in allowed_transition_ids:
                    continue
                if t["id"] not in reachable_ids:
                    self.jodap.stats["filtered_transitions"] += 1
                    continue
                sync_move = SymbolicMove(
                    "sync", event_id=e.event_id, transition_id=t["id"],
                    transition_label=t.get("label")
                )
                sync_consumed = frozenset(set(node.consumed) | {e.event_id})
                if generation_allows(sync_move, sync_consumed):
                    self._add_candidate(component, state, node, sync_move)
                if zero_goal_proved():
                    self.jodap.stats["exact_zero_cost_expansion_stops"] += 1
                    return

        # A model move changes only the model execution. Concrete object
        # bindings remain symbolic, but structurally unreachable transitions do
        # not need an expensive assignment query.
        if include_model and only_event is None and node.model_depth < state.model_bound:
            for t in self.transitions:
                if allowed_transition_ids is not None and t["id"] not in allowed_transition_ids:
                    continue
                if t["id"] not in reachable_ids:
                    self.jodap.stats["filtered_transitions"] += 1
                    continue
                model_move = SymbolicMove(
                    "model", transition_id=t["id"], transition_label=t.get("label")
                )
                if generation_allows(model_move, node.consumed):
                    self._add_candidate(component, state, node, model_move)
                if zero_goal_proved():
                    self.jodap.stats["exact_zero_cost_expansion_stops"] += 1
                    return

    def _path_relevant_object_attributes(self, state: SearchState, node: SearchNode,
                                         available: Set[str]) -> Set[str]:
        attrs: Set[str] = set()
        for move in self._path(state, node.node_id):
            if move.kind not in ("model", "sync") or move.transition_id is None:
                continue
            t = next((t for t in self.transitions if t["id"] == move.transition_id), None)
            if t is not None:
                attrs.update(self.jodap._guard_object_attribute_names(
                    t.get("constraint"), available
                ))
        return attrs

    def _rebuild_open_for_changed_observations(
            self, component: ComponentState, state: SearchState,
            affected_attributes: Optional[Set[str]] = None) -> None:
        """Revalidate only paths that can be affected by changed observations.

        A newly observed object can change the binding domain and therefore
        requires full revalidation.  A pure object-attribute update is local: a
        path whose selected transition guards never mention the changed
        attribute keeps exactly the same JODAP optimum and can be reopened
        without another solver query.
        """
        state.open_heap.clear()
        state.open_ids.clear()
        state.closed_ids.clear()
        valid_ids = []
        available = {a for vals in component.observation_formula.current_object_attributes().values()
                     for a in vals}
        for nid in sorted(state.nodes):
            node = state.nodes[nid]
            if nid == 0:
                node.g = 0.0
                valid_ids.append(nid)
                continue
            if affected_attributes is not None:
                relevant = self._path_relevant_object_attributes(state, node, available)
                if not (relevant & affected_attributes):
                    valid_ids.append(nid)
                    continue
            assignment = self.jodap.solve(component, state, node, require_final=False)
            if assignment is None:
                continue
            node.g = assignment.total_cost
            node.assignment_cost = assignment.total_cost
            valid_ids.append(nid)
        # Remove infeasible nodes and predecessor descendants whose parent vanished.
        valid = set(valid_ids)
        changed = True
        while changed:
            changed = False
            for nid in list(valid):
                if nid in state.predecessor and state.predecessor[nid][0] not in valid:
                    valid.remove(nid); changed = True
        state.nodes = {nid: n for nid, n in state.nodes.items() if nid in valid}
        state.predecessor = {nid: x for nid, x in state.predecessor.items() if nid in valid}
        state.signatures = {self._signature(n): nid for nid, n in state.nodes.items()}
        for nid in sorted(state.nodes):
            state.push(state.nodes[nid])
        state.current_goal = None
        state.current_assignment = None

    def _relation_repair_metadata(self, component: ComponentState, observed_objects: Iterable[str],
                                  model_objects: Iterable[str]) -> Dict[str, Any]:
        """Build persistent provenance for an observed/model object-relation mismatch.

        The mismatch is attached to stable non-ITEM anchor objects (normally the
        PACKAGE).  Later lifecycle events carrying the same anchor can reuse the
        model-side relation set without paying the same object deviation again.
        """
        observed = {str(o) for o in observed_objects}
        model = {str(o) for o in model_objects}
        extras = observed - model
        missing = model - observed
        changed = extras | missing
        changed_types = {
            component.observation_formula.object_types.get(o) for o in changed
        }
        anchors = {
            o for o in (observed & model)
            if component.observation_formula.object_types.get(o) not in changed_types
        }
        if not anchors:
            # Fallback: any common non-ITEM object is a safer lifecycle anchor
            # than an ITEM relation itself.
            anchors = {
                o for o in (observed & model)
                if component.observation_formula.object_types.get(o) != "ITEM"
            }
        return {
            "relation_repair": True,
            "relation_repair_anchor_objects": tuple(sorted(anchors)),
            "relation_repair_extra_objects": tuple(sorted(extras)),
            "relation_repair_missing_objects": tuple(sorted(missing)),
            "observed_objects": tuple(sorted(observed)),
            "model_objects": tuple(sorted(model)),
            "object_cost": len(extras) + len(missing),
        }

    def _persistent_relation_override(self, component: ComponentState, event: StreamEvent,
                                      transition_id: int, parent_assignment: JointAssignment):
        """Return an inherited model-side relation set for a later lifecycle event.

        A prior object-relation repair is a persistent model/observation fact, not
        a fresh deviation at every subsequent event.  This method transports the
        same extra/missing relation set across events sharing the repaired anchor.
        """
        self.jodap.stats["object_relation_carry_attempts"] += 1
        observed = {str(o) for o in event.objects}
        try:
            net = self.jodap._query_slice_static_net()
            nt = next(t for t in net._transitions if t.get("id") == transition_id)
        except Exception:
            net = self.jodap._static_net or self.net_metadata
            nt = next((t for t in net._transitions if t.get("id") == transition_id), None)
        if nt is None:
            self.jodap.stats["object_relation_carry_misses"] += 1
            return None

        for row in reversed(parent_assignment.object_bindings):
            if not row.get("relation_repair"):
                continue
            anchors = {str(o) for o in row.get("relation_repair_anchor_objects", ())}
            if not anchors or not anchors.issubset(observed):
                continue
            extras = {str(o) for o in row.get("relation_repair_extra_objects", ())}
            missing = {str(o) for o in row.get("relation_repair_missing_objects", ())}
            candidate = frozenset((observed - extras) | missing)
            if candidate == frozenset(observed):
                continue
            self.jodap.stats["object_relation_provenance_candidates"] += 1
            try:
                binding = self.jodap._lazy_binding(component, net, nt, tuple(sorted(candidate)))
            except Exception:
                binding = None
            if binding is None:
                continue
            self.jodap.stats["object_relation_carry_hits"] += 1
            self.jodap._diag(
                "object_relation_carry_forward", component=component.component_id,
                event_id=event.event_id, activity=event.activity,
                transition_id=transition_id, anchors=sorted(anchors),
                extra_relations=sorted(extras), missing_relations=sorted(missing),
                observed_objects=sorted(observed), model_objects=sorted(candidate),
            )
            return tuple(sorted(candidate)), row
        self.jodap.stats["object_relation_carry_misses"] += 1
        return None

    def _try_zero_cost_sync_extension(self, component: ComponentState, state: SearchState,
                                      previous_goal: Optional[SearchNode],
                                      event_id: str, *,
                                      allow_fixed_fallback: bool = True) -> bool:
        """Try a true incremental zero-cost synchronous extension.

        The old fast path still called the full JODAP optimizer for the complete
        extended path.  This version first asks JODAP to *extend the retained
        optimal witness* with one synchronous firing using a single SAT check.
        Only if that local check fails does normal A*+JODAP resume and permit
        re-optimization of earlier assignments.
        """
        if previous_goal is None or previous_goal.node_id not in state.nodes:
            return False
        if event_id in previous_goal.consumed:
            return False
        enabled_ids = {e.event_id for e in self._enabled_events(component, previous_goal)}
        if event_id not in enabled_ids:
            return False
        event = component.observation_formula.events[event_id]
        transitions = self.visible_by_label.get(event.activity, ())
        if not transitions:
            return False
        previous_cost = previous_goal.g
        if previous_cost == float("inf"):
            return False
        parent_assignment = state.assignments_by_node.get(previous_goal.node_id)
        if parent_assignment is None:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return False

        reachable = self.jodap.structurally_reachable_transition_ids(
            component, state, previous_goal
        )
        for t in transitions:
            if t["id"] not in reachable:
                continue
            self.jodap.stats["zero_cost_sync_attempts"] += 1

            temp = SearchNode(
                node_id=-1,
                consumed=frozenset(set(previous_goal.consumed) | {event_id}),
                event_order=previous_goal.event_order + (event_id,),
                model_depth=previous_goal.model_depth + 1,
                g=previous_cost,
                h=0.0,
                assignment_cost=previous_cost,
                model_signature=previous_goal.model_signature + (t["id"],),
                move_signature=previous_goal.move_signature +
                    (("sync", event_id, t["id"]),),
            )
            inherited_relation = self._persistent_relation_override(
                component, event, t["id"], parent_assignment
            )
            inherited_model_objects = inherited_relation[0] if inherited_relation else None
            assignment = self.jodap.check_zero_cost_sync_extension(
                component, state, previous_goal, temp, event, t["id"], parent_assignment,
                model_objects_override=inherited_model_objects,
                extra_cost=0.0,
                repair_tag="object_relation_carry" if inherited_model_objects is not None else None,
            )
            if assignment is not None and inherited_relation is not None:
                # Preserve the original relation provenance on the new binding,
                # but do not pay the already-accounted-for object mismatch again.
                meta = dict(inherited_relation[1])
                meta["step"] = assignment.object_bindings[-1].get("step")
                meta["transition_id"] = assignment.object_bindings[-1].get("transition_id")
                meta["transition"] = assignment.object_bindings[-1].get("transition")
                meta["objects"] = tuple(sorted(inherited_model_objects))
                meta["observed_objects"] = tuple(sorted(event.objects))
                meta["model_objects"] = tuple(sorted(inherited_model_objects))
                meta["object_cost"] = 0
                meta["relation_repair_inherited"] = True
                assignment.object_bindings[-1] = meta
                self.jodap.stats["object_relation_carry_free_extensions"] += 1
            if assignment is None and allow_fixed_fallback:
                # Whether the local check returned UNKNOWN or found a genuine
                # soft mismatch, evaluate exactly this *one* fixed extension
                # before reopening A*.  The lazy fixed-path witness may now
                # return a certified positive-cost incumbent as well.
                private_id = state.new_id()
                private = SearchNode(
                    node_id=private_id, consumed=temp.consumed,
                    event_order=temp.event_order, model_depth=temp.model_depth,
                    g=float("inf"), h=0.0, assignment_cost=previous_cost,
                    model_signature=temp.model_signature, move_signature=temp.move_signature,
                )
                state.nodes[private_id] = private
                state.predecessor[private_id] = (
                    previous_goal.node_id,
                    SymbolicMove("sync", event_id=event_id, transition_id=t["id"],
                                 transition_label=t.get("label"))
                )
                try:
                    assignment = self.jodap.check_fixed_sync_extension(
                        component, state, private, event, t["id"], parent_assignment
                    )
                finally:
                    state.nodes.pop(private_id, None)
                    state.predecessor.pop(private_id, None)
                    state.assignments_by_node.pop(private_id, None)
            if assignment is None:
                continue
            # Never accept a witness below an independently proven prefix LB.
            if assignment.total_cost + 1e-9 < state.proven_prefix_lower_bound:
                continue

            # Materialize the proven-optimal extension in the retained search.
            cand = self._add_candidate(
                component, state, previous_goal,
                SymbolicMove("sync", event_id=event_id,
                             transition_id=t["id"], transition_label=t.get("label")),
                evaluate=False,
            )
            if cand is None:
                continue
            cand.g = assignment.total_cost
            cand.assignment_cost = assignment.total_cost
            cand.h = self._heuristic(component, cand.consumed)
            state.assignments_by_node[cand.node_id] = assignment
            state.push(cand)  # refresh heap with the now-known finite f-value

            if cand.consumed != frozenset(component.execution.event_ids):
                continue
            state.current_goal = cand.node_id
            state.current_assignment = assignment
            state.upper_bound = assignment.total_cost
            state.incumbent_moves = tuple(self._path(state, cand.node_id))
            state.incumbent_assignment = assignment
            state.incumbent_offline = False
            if assignment.total_cost <= previous_cost + 1e-9:
                self.jodap.stats["zero_cost_sync_hits"] += 1
            # A positive extension is immediately optimal only when the
            # independent prefix lower bound reaches its feasible cost.
            if assignment.total_cost <= state.proven_prefix_lower_bound + 1e-9:
                if assignment.total_cost > previous_cost + 1e-9:
                    self.jodap.stats["positive_lower_bound_terminations"] += 1
                return True
            # Keep the feasible fixed-path incumbent to prune the subsequent A*.
            # The caller will continue normal search because optimality is not
            # yet proved.
            return False
        return False


    def _try_guard_data_deviation_sync_extension(
            self, component: ComponentState, state: SearchState,
            parent: Optional[SearchNode], event_id: str) -> bool:
        """Repair one concrete false aggregate guard as a unit data deviation.

        This path is attempted only after the local zero-cost synchronous check
        has established ``guard_false``.  It keeps the same activity, object
        binding, marking transition and observed event, but changes exactly one
        model-side object attribute to the value forced by a supported aggregate
        equality.  A cost of one is attached to that soft mismatch.

        The result is final only when an independent zero-deviation test proves
        that the new event must add at least one unit.  Otherwise no shortcut is
        installed and unrestricted exact search remains available.
        """
        if parent is None or parent.node_id not in state.nodes:
            return False
        detail = getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
        if detail.get("reason") != "guard_false":
            return False
        event = component.observation_formula.events.get(event_id)
        if event is None or event_id in parent.consumed:
            return False
        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return False

        tid = detail.get("transition_id")
        transition = self.transition_by_id.get(tid) if tid is not None else None
        if transition is None:
            # Diagnostics from older/local paths may not carry the id.  The
            # activity/object binding still selects a unique supported candidate.
            candidates = self.visible_by_label.get(event.activity, ())
            transition = next((t for t in candidates if t.get("constraint") is not None), None)
        if transition is None:
            return False

        try:
            net = self.jodap._static_net or self.jodap.new_net()
            self.jodap._static_net = net
            net_transition = next(t for t in net._transitions if t.get("id") == transition.get("id"))
            binding = self.jodap._lazy_binding(component, net, net_transition, tuple(event.objects))
        except Exception:
            binding = None
            net_transition = None
        if binding is None or net_transition is None:
            return False

        self.jodap.stats["guard_data_repair_attempts"] += 1
        overrides = self.jodap.infer_single_object_attribute_guard_repair(
            component, net_transition, binding
        )
        if not overrides:
            self.jodap.stats["guard_data_repair_fallbacks"] += 1
            self.jodap._diag(
                "guard_data_repair_decline", component=component.component_id,
                event_id=event_id, activity=event.activity,
                transition_id=transition.get("id"), reason="unsupported_or_nonunique_guard_repair",
            )
            return False
        self.jodap.stats["guard_data_repair_candidates"] += 1

        # Zero additional cost must be impossible before a unit repair can be
        # declared optimal.  This checker considers every structurally compatible
        # transition, so alternative zero-cost synchronous explanations are not
        # accidentally pruned.
        inc_lb = self._event_zero_deviation_increment_lb(component, event)
        if inc_lb < 1:
            self.jodap.stats["guard_data_repair_fallbacks"] += 1
            self.jodap._diag(
                "guard_data_repair_decline", component=component.component_id,
                event_id=event_id, activity=event.activity,
                transition_id=transition.get("id"), reason="positive_lower_bound_not_proven",
                overrides={f"{k[0]}.{k[1]}": v for k, v in overrides.items()},
            )
            return False

        lower_bound = float(parent_assignment.total_cost) + 1.0
        state.proven_prefix_lower_bound = max(state.proven_prefix_lower_bound, lower_bound)
        self.jodap.stats["guard_data_repair_lb_proofs"] += 1
        self.jodap._diag(
            "guard_data_repair_attempt", component=component.component_id,
            event_id=event_id, activity=event.activity, parent_node=parent.node_id,
            parent_cost=float(parent_assignment.total_cost), lower_bound=lower_bound,
            transition_id=transition.get("id"), transition=transition.get("label"),
            binding={str(k): (list(v) if isinstance(v, (list, tuple, set)) else v)
                     for k, v in binding.items()},
            overrides={f"{k[0]}.{k[1]}": v for k, v in overrides.items()},
        )

        temp = SearchNode(
            node_id=-1,
            consumed=frozenset(set(parent.consumed) | {event_id}),
            event_order=parent.event_order + (event_id,),
            model_depth=parent.model_depth + 1,
            g=lower_bound, h=0.0, assignment_cost=lower_bound,
            model_signature=parent.model_signature + (transition["id"],),
            move_signature=parent.move_signature + (("sync", event_id, transition["id"]),),
        )
        assignment = self.jodap.check_zero_cost_sync_extension(
            component, state, parent, temp, event, transition["id"], parent_assignment,
            model_object_attribute_overrides=overrides,
            extra_cost=1.0, repair_tag="single_object_attribute_guard",
        )
        if assignment is None:
            self.jodap.stats["guard_data_repair_fallbacks"] += 1
            self.jodap._diag(
                "guard_data_repair_decline", component=component.component_id,
                event_id=event_id, activity=event.activity,
                transition_id=transition.get("id"), reason="repaired_sync_not_locally_certified",
            )
            return False

        cand = self._add_candidate(
            component, state, parent,
            SymbolicMove("sync", event_id=event_id, transition_id=transition["id"],
                         transition_label=transition.get("label")),
            evaluate=False,
        )
        if cand is None:
            self.jodap.stats["guard_data_repair_fallbacks"] += 1
            return False
        cand.g = float(assignment.total_cost)
        cand.assignment_cost = float(assignment.total_cost)
        cand.h = self._heuristic(component, cand.consumed)
        state.assignments_by_node[cand.node_id] = assignment
        state.push(cand)
        self.jodap.stats["guard_data_repair_hits"] += 1

        if cand.consumed != frozenset(component.execution.event_ids):
            # In normal online use the newly enabled event completes the current
            # prefix.  Keep the exact fallback for unusual partial candidates.
            return False
        state.current_goal = cand.node_id
        state.current_assignment = assignment
        state.upper_bound = float(assignment.total_cost)
        state.incumbent_moves = tuple(self._path(state, cand.node_id))
        state.incumbent_assignment = assignment
        state.incumbent_offline = False
        if assignment.total_cost <= state.proven_prefix_lower_bound + 1e-9:
            self.jodap.stats["guard_data_repair_proven"] += 1
            self.jodap.stats["positive_lower_bound_terminations"] += 1
            self.jodap._diag(
                "guard_data_repair_success", component=component.component_id,
                event_id=event_id, activity=event.activity,
                transition_id=transition.get("id"), total_cost=float(assignment.total_cost),
                lower_bound=float(state.proven_prefix_lower_bound), proven=True,
                overrides={f"{k[0]}.{k[1]}": v for k, v in overrides.items()},
            )
            return True
        return False


    def _register_active_component(self, component: ComponentState) -> None:
        """Track the latest observation components without looking ahead.

        ComponentManager removes parent components when an observed relation
        actually merges them.  Mirror that behavior here so missing-relation
        inference can inspect only components that are independently active at
        the current stream prefix.
        """
        merged = tuple(int(cid) for cid in (getattr(component, "merged_from", ()) or ()))
        inherited: Dict[int, Set[str]] = {}
        for cid in merged:
            self._active_components.pop(cid, None)
            for src, objs in self._virtual_component_dependencies.pop(cid, {}).items():
                if src in merged:
                    self.jodap.stats["object_relation_virtual_dependency_reuses"] += 1
                    continue
                inherited.setdefault(int(src), set()).update(str(o) for o in objs)
        self._active_components[int(component.component_id)] = component
        if inherited:
            self._virtual_component_dependencies[int(component.component_id)] = inherited

    def _current_assignment_for_component(self, cid: int):
        st = self.searches.get(int(cid))
        if st is None:
            return None, None
        assn = st.current_assignment or st.incumbent_assignment
        node = st.nodes.get(st.current_goal) if st.current_goal is not None else None
        if assn is None and node is not None:
            assn = st.assignments_by_node.get(node.node_id)
        if assn is None:
            # Prefer a certified full-prefix node when current_goal was cleared
            # after an incremental update.
            full = frozenset(st.current_event_ids)
            choices = []
            for nid, n in st.nodes.items():
                if n.consumed != full:
                    continue
                a = st.assignments_by_node.get(nid)
                if a is not None:
                    choices.append((float(a.total_cost), nid, n, a))
            if choices:
                choices.sort(key=lambda x: (x[0], x[1]))
                _, _nid, node, assn = choices[0]
        return node, assn

    def _component_object_type(self, obj: str) -> Optional[str]:
        for comp in self._active_components.values():
            typ = comp.observation_formula.object_types.get(obj)
            if typ is not None:
                return typ
        return None

    def _component_object_attributes(self, obj: str) -> Dict[str, Any]:
        for comp in self._active_components.values():
            attrs = comp.observation_formula.current_object_attributes().get(obj)
            if attrs is not None:
                return dict(attrs)
        return {}

    def _borrow_object_state(self, base: JointAssignment, source: JointAssignment,
                             obj: str) -> JointAssignment:
        """Add only the already-certified live state of ``obj`` to ``base``.

        The source component must have zero accumulated deviation for this fast
        path.  We therefore do not import its objective cost or unrelated
        history.  Only marking/token entries that contain the concrete object
        are borrowed, enough to test whether the current model transition can
        consume it.  The resulting repair metadata later lets merge composition
        suppress the stale copy from the source parent.
        """
        known = {str(obj)}
        marks = list(base.marking_signature or ())
        for item in source.marking_signature or ():
            if self._objects_in_value(item, known) and item not in marks:
                marks.append(item)
        token = list(base.token_data_signature or ())
        for item in source.token_data_signature or ():
            if self._objects_in_value(item, known) and item not in token:
                token.append(item)
        attrs = list(base.object_attribute_assignments or ())
        for row in source.object_attribute_assignments or ():
            if str(row.get("object", row.get("object_id", ""))) == str(obj):
                if row not in attrs:
                    attrs.append(dict(row))
        return JointAssignment(
            total_cost=float(base.total_cost),
            object_bindings=list(base.object_bindings),
            data_assignments=list(base.data_assignments),
            object_attribute_assignments=attrs,
            marking_signature=tuple(sorted(marks, key=repr)),
            data_state_signature=base.data_state_signature,
            token_data_signature=tuple(sorted(token, key=repr)),
            data_provenance_signature=base.data_provenance_signature,
            solve_seconds=base.solve_seconds,
            encode_seconds=base.encode_seconds,
        )

    def _composite_component_for_object(self, component: ComponentState,
                                        source: ComponentState, obj: str) -> ComponentState:
        """Lightweight current-prefix view exposing one external known object."""
        combined = ComponentState(component.component_id)
        combined.objects.update(component.objects)
        combined.objects.add(str(obj))
        combined.observation_formula.merge(component.observation_formula)
        combined.observation_formula.merge(source.observation_formula)
        combined.execution = component.execution
        combined.units = list(component.units)
        combined.current_alignment = component.current_alignment
        combined.checkpoint_position = component.checkpoint_position
        combined.merged_from = component.merged_from
        combined.merge_position = component.merge_position
        combined.merge_event_id = component.merge_event_id
        combined.parent_checkpoint_positions = component.parent_checkpoint_positions
        return combined

    def _try_object_relation_deviation_sync_extension(
            self, component: ComponentState, state: SearchState,
            parent: Optional[SearchNode], event_id: str) -> Tuple[bool, bool]:
        """Try a bounded object-relation repair before data/control fallback.

        The observed event stays unchanged.  The model-side LIST binding may
        differ by up to ``max_object_relation_fast_cardinality`` objects.
        Candidates are checked in increasing symmetric-difference cardinality
        against the same concrete marking/data/guard evaluator used by normal
        synchronous extensions.  Missing relations may borrow only objects from
        already-observed certified active components; no future information is
        consulted.
        """
        if parent is None or parent.node_id not in state.nodes:
            return False, False
        event = component.observation_formula.events.get(event_id)
        if event is None or event_id in parent.consumed:
            return False, False
        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return False, False

        transitions = list(self.visible_by_label.get(event.activity, ()))
        if not transitions:
            return False, False

        observed = {str(o) for o in event.objects}
        detail = getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
        required_missing = {str(o) for o in detail.get("required_objects", ()) if o is not None}
        marked_objects = self._marked_objects_from_assignment(component, parent_assignment)
        try:
            net = self.jodap._query_slice_static_net()
        except Exception:
            try:
                net = self.jodap._static_net or self.jodap.new_net()
                self.jodap._static_net = net
            except Exception:
                return False, False

        self.jodap.stats["object_relation_repair_attempts"] += 1
        candidates = []
        latent_seen = False
        for t in transitions:
            try:
                nt = next(x for x in net._transitions if x.get("id") == t.get("id"))
                decls = self.jodap._lazy_unique_object_decls(net, nt)
            except Exception:
                continue
            list_types = {typ[:typ.rfind(" LIST")] for typ in decls.values() if "LIST" in typ}
            if not list_types:
                continue

            removable = [o for o in observed
                         if component.observation_formula.object_types.get(o) in list_types]
            removable.sort(key=lambda o: (o not in required_missing, o))
            for k in range(1, min(self.max_object_relation_fast_cardinality, len(removable)) + 1):
                for combo in itertools.combinations(removable, k):
                    candidates.append(("extra", tuple(combo), t, nt,
                                       frozenset(observed - set(combo)), ()))
                    if k > 1:
                        self.jodap.stats["object_relation_multirelation_candidates"] += 1

            local_addable: Set[str] = {
                o for o in marked_objects - observed
                if component.observation_formula.object_types.get(o) in list_types
            }
            for other_state in self.searches.values():
                assn = other_state.current_assignment or other_state.incumbent_assignment
                if assn is None:
                    continue
                for row in reversed(assn.object_bindings):
                    for o in row.get("model_objects", row.get("objects", ())):
                        o = str(o)
                        if o not in observed and component.observation_formula.object_types.get(o) in list_types:
                            local_addable.add(o)

            source_by_object: Dict[str, Tuple[int, JointAssignment]] = {}
            for source_cid, source_comp in sorted(self._active_components.items()):
                if int(source_cid) == int(component.component_id):
                    continue
                _source_node, source_assn = self._current_assignment_for_component(source_cid)
                if source_assn is None or abs(float(source_assn.total_cost)) > 1e-9:
                    continue
                for obj in sorted(source_comp.objects):
                    obj = str(obj)
                    if obj in observed or obj in local_addable:
                        continue
                    if source_comp.observation_formula.object_types.get(obj) not in list_types:
                        continue
                    if not any(self._objects_in_value(mark, {obj})
                               for mark in (source_assn.marking_signature or ())):
                        continue
                    source_by_object.setdefault(obj, (int(source_cid), source_assn))
                    self.jodap.stats["object_relation_provenance_candidates"] += 1

            pool = sorted(local_addable | set(source_by_object))
            observed_list_objects = {o for o in observed
                                     if component.observation_formula.object_types.get(o) in list_types}
            if not observed_list_objects and pool and not latent_seen:
                latent_seen = True
                self.jodap.stats["object_relation_latent_completion_attempts"] += 1
                self.jodap._diag("object_relation_latent_completion_attempt",
                                 component=component.component_id, event_id=event_id,
                                 activity=event.activity, candidate_pool=len(pool))
            for k in range(1, min(self.max_object_relation_fast_cardinality, len(pool)) + 1):
                for combo in itertools.combinations(pool, k):
                    specs = []
                    for obj in combo:
                        if obj in source_by_object:
                            cid, assn = source_by_object[obj]
                            specs.append((obj, cid, assn))
                    candidates.append(("missing", tuple(combo), t, nt,
                                       frozenset(observed | set(combo)), tuple(specs)))
                    if k > 1:
                        self.jodap.stats["object_relation_multirelation_candidates"] += 1

        if not candidates:
            self.jodap.stats["object_relation_repair_fallbacks"] += 1
            self.jodap._diag("object_relation_repair_decline", component=component.component_id,
                             event_id=event_id, activity=event.activity,
                             reason="no_bounded_list_relation_candidate")
            return False, False

        def rank(row):
            kind, changed, t, _nt, _objs, source_specs = row
            hinted = kind == "extra" and bool(set(changed) & required_missing)
            return (len(changed), not hinted, 0 if kind == "extra" else 1,
                    0 if source_specs else 1, t.get("id", 0), tuple(changed))

        candidates.sort(key=rank)

        # Marking-aware LIST-domain reduction.  Before trying any concrete
        # relation binding, eliminate local candidates whose actual LIST members
        # cannot satisfy the transition's required input-place marking.  Borrowed
        # cross-component objects are intentionally left untouched because the
        # existing provenance path supplies their certified marking separately.
        marking_filtered = []
        marking_pruned = 0
        marking_supported = False
        for cand in candidates:
            kind, changed, t, nt, model_objects, source_specs = cand
            if source_specs:
                marking_filtered.append(cand)
                continue
            domain = self.jodap.marking_feasible_list_domain(
                component, parent_assignment, nt, tuple(sorted(model_objects)),
                tuple(sorted(model_objects))
            )
            if domain is None:
                marking_filtered.append(cand)
                continue
            marking_supported = True
            if domain.get("infeasible"):
                marking_pruned += 1
            else:
                marking_filtered.append(cand)
        if marking_supported:
            self.jodap.stats["marking_domain_candidates_pruned"] += marking_pruned
            self.jodap._diag(
                "marking_domain_candidate_filter", component=component.component_id,
                event_id=event_id, activity=event.activity,
                candidates_before=len(candidates), candidates_after=len(marking_filtered),
                candidates_pruned=marking_pruned, path="object_relation_fast_path",
            )
            candidates = marking_filtered

        # Guard-directed LIST binding.  When the zero-cost probe diagnosed a
        # concrete aggregate guard failure, cheaply evaluate the same parsed
        # guard for each relation candidate before invoking the expensive local
        # synchronization checker.  Only a *decidable false* candidate is
        # removed; unknown/unsupported guards and cross-component candidates
        # remain untouched, so this cannot remove a valid exact explanation.
        if detail.get("reason") == "guard_false":
            self.jodap.stats["guard_directed_list_attempts"] += 1
            filtered = []
            supported = False
            pruned = 0
            for cand in candidates:
                kind, changed, t, nt, model_objects, source_specs = cand
                if source_specs:
                    filtered.append(cand)
                    continue
                try:
                    binding = self.jodap._lazy_binding(
                        component, net, nt, tuple(sorted(model_objects))
                    )
                    verdict = (
                        self.jodap.evaluate_guard_for_concrete_object_binding(
                            component, nt, binding
                        ) if binding is not None else None
                    )
                except Exception:
                    verdict = None
                if verdict is None:
                    filtered.append(cand)
                    continue
                supported = True
                if verdict is True:
                    filtered.append(cand)
                else:
                    pruned += 1
            if supported:
                self.jodap.stats["guard_directed_list_supported"] += 1
                self.jodap.stats["guard_directed_list_candidates_pruned"] += pruned
                self.jodap.stats["guard_directed_list_candidates_kept"] += len(filtered)
                self.jodap._diag(
                    "guard_directed_list_filter", component=component.component_id,
                    event_id=event_id, activity=event.activity,
                    candidates_before=len(candidates), candidates_after=len(filtered),
                    candidates_pruned=pruned,
                )
                candidates = filtered
            else:
                self.jodap.stats["guard_directed_list_fallbacks"] += 1

        if not candidates:
            self.jodap.stats["object_relation_repair_fallbacks"] += 1
            self.jodap._diag(
                "object_relation_repair_decline", component=component.component_id,
                event_id=event_id, activity=event.activity,
                reason="guard_directed_all_candidates_false",
            )
            return False, False

        best = None
        best_cardinality = None
        for kind, changed, t, nt, model_objects, source_specs in candidates:
            cardinality = len(changed)
            if best_cardinality is not None and cardinality > best_cardinality:
                break
            self.jodap.stats["object_relation_repair_candidates"] += 1
            repair_cost = float(cardinality)
            temp = SearchNode(
                node_id=-1,
                consumed=frozenset(set(parent.consumed) | {event_id}),
                event_order=parent.event_order + (event_id,),
                model_depth=parent.model_depth + 1,
                g=float(parent_assignment.total_cost) + repair_cost, h=0.0,
                assignment_cost=float(parent_assignment.total_cost) + repair_cost,
                model_signature=parent.model_signature + (t["id"],),
                move_signature=parent.move_signature + (("sync", event_id, t["id"]),),
            )
            eval_component = component
            eval_parent_assignment = parent_assignment
            borrowed_sources: List[Tuple[str, int]] = []
            valid_sources = True
            for obj, source_cid, source_assn in source_specs:
                source_comp = self._active_components.get(int(source_cid))
                if source_comp is None:
                    valid_sources = False
                    break
                eval_component = self._composite_component_for_object(eval_component, source_comp, obj)
                eval_parent_assignment = self._borrow_object_state(eval_parent_assignment, source_assn, obj)
                borrowed_sources.append((str(obj), int(source_cid)))
            if not valid_sources:
                continue
            assignment = self.jodap.check_zero_cost_sync_extension(
                eval_component, state, parent, temp, event, t["id"], eval_parent_assignment,
                model_objects_override=tuple(sorted(model_objects)),
                extra_cost=repair_cost, repair_tag="object_relation_repair")
            if assignment is None:
                continue
            best = (assignment, kind, tuple(changed), t, model_objects,
                    tuple(borrowed_sources), repair_cost)
            best_cardinality = cardinality
            break

        if best is None:
            self.jodap.stats["object_relation_repair_fallbacks"] += 1
            self.jodap._diag("object_relation_repair_decline", component=component.component_id,
                             event_id=event_id, activity=event.activity,
                             reason="all_bounded_relation_candidates_invalid",
                             candidate_count=len(candidates))
            return False, False

        assignment, kind, changed_objects, transition, model_objects, borrowed_sources, repair_cost = best
        changed_object = changed_objects[0] if len(changed_objects) == 1 else tuple(changed_objects)
        if assignment.object_bindings:
            meta = self._relation_repair_metadata(component, observed, model_objects)
            meta.update({
                "step": assignment.object_bindings[-1].get("step"),
                "transition_id": assignment.object_bindings[-1].get("transition_id"),
                "transition": assignment.object_bindings[-1].get("transition"),
                "objects": tuple(sorted(model_objects)),
            })
            if borrowed_sources:
                meta["borrowed_object_sources"] = tuple(borrowed_sources)
                meta["relation_claimed_objects"] = tuple(sorted(obj for obj, _cid in borrowed_sources))
            assignment.object_bindings[-1] = meta

        self.jodap.stats["object_relation_repair_hits"] += 1
        if kind == "extra":
            self.jodap.stats["object_relation_extra_hits"] += 1
        else:
            self.jodap.stats["object_relation_missing_hits"] += 1
        if borrowed_sources:
            self.jodap.stats["object_relation_cross_component_hits"] += 1
            deps = self._virtual_component_dependencies.setdefault(int(component.component_id), {})
            for obj, source_cid in borrowed_sources:
                deps.setdefault(int(source_cid), set()).add(str(obj))
                self.jodap.stats["object_relation_virtual_dependencies"] += 1
                self.jodap._diag("object_relation_virtual_dependency",
                                 component=component.component_id, source_component=source_cid,
                                 claimed_object=obj, event_id=event_id)
        if len(changed_objects) > 1:
            self.jodap.stats["object_relation_multirelation_hits"] += 1
        self.jodap.stats["object_relation_max_repair_cardinality"] = max(
            int(self.jodap.stats.get("object_relation_max_repair_cardinality", 0)),
            len(changed_objects))
        if latent_seen and kind == "missing":
            self.jodap.stats["object_relation_latent_completion_hits"] += 1

        inc_lb = self._event_zero_deviation_increment_lb(component, event)
        positive_lb = inc_lb >= 1 and len(changed_objects) == 1
        if positive_lb:
            lb = float(parent_assignment.total_cost) + repair_cost
            state.proven_prefix_lower_bound = max(state.proven_prefix_lower_bound, lb)
            self.jodap.stats["object_relation_repair_lb_proofs"] += 1
            self.jodap._diag("object_relation_repair_lower_bound",
                             component=component.component_id, event_id=event_id,
                             activity=event.activity, relation_kind=kind,
                             changed_object=changed_object, lower_bound=lb)

        cand = self._add_candidate(
            component, state, parent,
            SymbolicMove("sync", event_id=event_id, transition_id=transition["id"],
                         transition_label=transition.get("label")),
            evaluate=False)
        if cand is None:
            self.jodap.stats["object_relation_repair_fallbacks"] += 1
            return True, False
        cand.g = float(assignment.total_cost)
        cand.assignment_cost = float(assignment.total_cost)
        cand.h = self._heuristic(component, cand.consumed)
        state.assignments_by_node[cand.node_id] = assignment
        state.push(cand)
        if assignment.total_cost < state.upper_bound - 1e-9:
            state.upper_bound = float(assignment.total_cost)
            state.incumbent_moves = tuple(self._path(state, cand.node_id))
            state.incumbent_assignment = assignment
            state.incumbent_offline = False

        complete = cand.consumed == frozenset(component.execution.event_ids)
        # Preserve an equally optimal but operationally different complete
        # witness for future prefixes.  This is essential for orphan first
        # events: a cost-1 log explanation is a valid optimum, but a cost-1
        # relation-repair explanation may be the only optimum that can extend
        # the model state cheaply when a later observation connects the objects.
        if complete and assignment.total_cost <= state.upper_bound + 1e-9:
            moves_for_boundary = tuple(self._path(state, cand.node_id))
            bsig = (
                round(float(assignment.total_cost), 9),
                tuple((m.kind, m.event_id, m.transition_id) for m in moves_for_boundary),
                tuple(sorted(assignment.marking_signature or (), key=repr)),
            )
            existing = set()
            for _c, _moves, _a in state.cooptimal_continuation_boundaries:
                existing.add((
                    round(float(_c), 9),
                    tuple((m.kind, m.event_id, m.transition_id) for m in _moves),
                    tuple(sorted(_a.marking_signature or (), key=repr)),
                ))
            if bsig not in existing:
                state.cooptimal_continuation_boundaries.append(
                    (float(assignment.total_cost), moves_for_boundary, assignment))
                # Keep only the lowest-cost small frontier; these are concrete
                # certified boundaries, not search nodes.
                state.cooptimal_continuation_boundaries.sort(
                    key=lambda row: (row[0], len(row[1])))
                del state.cooptimal_continuation_boundaries[8:]
                self.jodap.stats["cooptimal_continuation_boundaries"] += 1
                self.jodap._diag(
                    "cooptimal_continuation_boundary",
                    component=component.component_id, event_id=event_id,
                    activity=event.activity, total_cost=float(assignment.total_cost),
                    move_count=len(moves_for_boundary), relation_kind=kind)

        proven = bool(complete and positive_lb and
                      assignment.total_cost <= state.proven_prefix_lower_bound + 1e-9)
        if proven:
            state.current_goal = cand.node_id
            state.current_assignment = assignment
            self.jodap.stats["object_relation_repair_proven"] += 1
            self.jodap.stats["positive_lower_bound_terminations"] += 1
        else:
            self.jodap.stats["object_relation_repair_incumbents"] += 1

        self.jodap._diag(
            "object_relation_repair_success", component=component.component_id,
            event_id=event_id, activity=event.activity, relation_kind=kind,
            changed_object=changed_object, changed_objects=list(changed_objects),
            repair_cardinality=len(changed_objects),
            source_components=sorted({cid for _obj, cid in borrowed_sources}),
            observed_objects=sorted(observed), model_objects=sorted(model_objects),
            transition_id=transition.get("id"), transition=transition.get("label"),
            parent_cost=float(parent_assignment.total_cost), total_cost=float(assignment.total_cost),
            lower_bound=float(state.proven_prefix_lower_bound), proven=proven)
        return True, proven


    @staticmethod
    def _assignment_with_total_cost(parent: JointAssignment, total_cost: float) -> JointAssignment:
        """Reuse a certified concrete state with a different accumulated cost.

        A log move consumes an observation but does not change model markings,
        data, bindings, or provenance.  Reconstructing the small dataclass is
        therefore sufficient and deliberately avoids deepcopying retained
        symbolic/native solver state.
        """
        return JointAssignment(
            total_cost=float(total_cost),
            object_bindings=parent.object_bindings,
            data_assignments=parent.data_assignments,
            object_attribute_assignments=parent.object_attribute_assignments,
            marking_signature=parent.marking_signature,
            data_state_signature=parent.data_state_signature,
            token_data_signature=parent.token_data_signature,
            data_provenance_signature=parent.data_provenance_signature,
            solve_seconds=parent.solve_seconds,
            encode_seconds=parent.encode_seconds,
        )

    def _try_extra_event_log_fast_path(
            self, component: ComponentState, state: SearchState,
            parent: Optional[SearchNode], event_id: str) -> Tuple[bool, bool]:
        """Recognize a duplicated observed event and try its direct log move.

        Returns ``(recognized, proven)``.  The fast path is intentionally
        conservative: it is considered an ``extra event`` only when the local
        synchronous checker has just failed because a concrete input token is
        absent *and* the same activity over the same object set already occurs
        in the certified prefix.  This separates the controlled extra-pick case
        from the missing-pick case, where ``create package`` fails because a
        preceding producer event is absent.

        A direct log move is always a feasible explanation and leaves the
        certified model state unchanged.  For the one-object lifecycle events
        used in the benchmark its cost is one.  If the missing input place has
        no compatible invisible producer for the missing object, zero
        additional cost is impossible: synchronization requires a positive-cost
        visible model repair, while consuming the event as a log move also costs
        one.  Thus the inherited optimum + 1 is a valid lower bound, matching
        the log-move upper bound and proving the new prefix optimal.
        """
        if parent is None or parent.node_id not in state.nodes:
            return False, False
        detail = getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
        if detail.get("reason") != "required_input_token_not_marked":
            return False, False
        event = component.observation_formula.events.get(event_id)
        if event is None or len(event.objects) != 1:
            return False, False
        event_objects = frozenset(str(o) for o in event.objects)
        missing_objects = frozenset(
            str(o) for o in detail.get("required_objects", ()) if o is not None
        )
        if not missing_objects or not missing_objects.issubset(event_objects):
            return False, False

        # A controlled duplicate has a prior consumed event with the same
        # lifecycle label and concrete object set.  Do not infer "extra" merely
        # from a missing token; that pattern is also produced by missing-event
        # deviations and belongs to the one-step model-repair path.
        duplicate_of = None
        for old_eid in parent.event_order:
            if old_eid not in parent.consumed:
                continue
            old = component.observation_formula.events.get(old_eid)
            if old is None:
                continue
            if old.activity == event.activity \
                    and frozenset(str(o) for o in old.objects) == event_objects:
                duplicate_of = old_eid
                break
        if duplicate_of is None:
            return False, False

        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return True, False

        self.jodap.stats["extra_event_fast_attempts"] += 1
        self.jodap.stats["extra_event_fast_recognized"] += 1
        self.jodap._diag(
            "extra_event_fast_attempt", component=component.component_id,
            event_id=event_id, activity=event.activity,
            duplicate_of=duplicate_of, objects=sorted(event_objects),
            parent_node=parent.node_id, parent_cost=float(parent_assignment.total_cost),
            required_place_id=detail.get("place_id"),
            required_place=detail.get("place_name"),
            missing_objects=sorted(missing_objects),
        )

        # The direct log move is exact and needs no JODAP call: it consumes the
        # observation and leaves the certified concrete model state untouched.
        log_cost = float(len(event.objects))
        total_cost = float(parent_assignment.total_cost) + log_cost
        log_node = self._add_candidate(
            component, state, parent,
            SymbolicMove("log", event_id=event_id), evaluate=False,
        )
        if log_node is None:
            self.jodap.stats["extra_event_fast_fallbacks"] += 1
            self.jodap._diag(
                "extra_event_fast_decline", component=component.component_id,
                event_id=event_id, reason="log_candidate_not_materialized",
            )
            return True, False

        assignment = self._assignment_with_total_cost(parent_assignment, total_cost)
        log_node.g = total_cost
        log_node.assignment_cost = total_cost
        log_node.h = self._heuristic(component, log_node.consumed)
        state.assignments_by_node[log_node.node_id] = assignment
        state.push(log_node)
        self.jodap.stats["extra_event_fast_hits"] += 1

        if total_cost < state.upper_bound - 1e-9:
            state.upper_bound = total_cost
            state.incumbent_moves = tuple(self._path(state, log_node.node_id))
            state.incumbent_assignment = assignment
            state.incumbent_offline = False

        # Prove a +1 incremental LB only if no zero-cost invisible transition
        # can restore the concrete missing token to the required input place.
        # A visible producer is harmless for this proof because model-only
        # visible moves themselves carry positive control-flow cost.
        positive_lb_proved = False
        place_id = detail.get("place_id")
        if place_id is not None:
            try:
                net = self.jodap._query_slice_static_net()
                missing_obj = next(iter(missing_objects))
                invisible_producer = any(
                    t.get("invisible", False)
                    and any(a.get("source") == t["id"] and a.get("target") == place_id
                            for a in net._arcs)
                    and self.jodap._lazy_binding(component, net, t, (missing_obj,)) is not None
                    for t in net._transitions
                )
                positive_lb_proved = not invisible_producer
            except Exception as exc:
                self.jodap._diag(
                    "extra_event_fast_lb_unknown", component=component.component_id,
                    event_id=event_id, error=repr(exc),
                )

        if positive_lb_proved:
            state.proven_prefix_lower_bound = max(
                state.proven_prefix_lower_bound, total_cost
            )
            self.jodap.stats["extra_event_fast_lb_proofs"] += 1
            self.jodap._diag(
                "extra_event_fast_lower_bound", component=component.component_id,
                event_id=event_id, duplicate_of=duplicate_of,
                lower_bound=float(total_cost),
                reason="duplicate_missing_token_no_invisible_restore",
            )

        all_events = frozenset(component.execution.event_ids)
        if log_node.consumed == all_events and positive_lb_proved \
                and total_cost <= state.proven_prefix_lower_bound + 1e-9:
            state.current_goal = log_node.node_id
            state.current_assignment = assignment
            self.jodap.stats["extra_event_fast_proven"] += 1
            self.jodap._diag(
                "extra_event_fast_success", component=component.component_id,
                event_id=event_id, duplicate_of=duplicate_of,
                total_cost=float(total_cost),
                lower_bound=float(state.proven_prefix_lower_bound), proven=True,
            )
            return True, True

        self.jodap.stats["extra_event_fast_incumbents"] += 1
        self.jodap._diag(
            "extra_event_fast_success", component=component.component_id,
            event_id=event_id, duplicate_of=duplicate_of,
            total_cost=float(total_cost),
            lower_bound=float(state.proven_prefix_lower_bound), proven=False,
        )
        return True, False



    def _try_one_step_missing_input_repair(
            self, component: ComponentState, state: SearchState,
            parent: Optional[SearchNode], event_id: str) -> bool:
        """Try ``model(t,obj) ; sync(event)`` after a local marking failure.

        The zero-cost delta checker already tells us when a synchronous firing
        fails because one concrete object token is absent from a required input
        place.  Before reopening unrestricted A*, try the smallest repair that
        can possibly restore that token: one visible model-only transition that
        produces the missing place for that same object, followed by the
        observed synchronous transition.

        The candidate is solved from the retained certified boundary with the
        lazy suffix encoder.  Thus only two new model steps are encoded; the
        historical prefix is fixed by ``boundary_assignment``.  A result is
        globally final only when a separate structural argument proves that a
        positive model move is unavoidable.  Otherwise it is installed merely
        as a feasible incumbent and ordinary A* remains free to improve it.
        """
        if parent is None or parent.node_id not in state.nodes:
            return False
        detail = getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
        if detail.get("reason") != "required_input_token_not_marked":
            return False
        missing_objects = [str(o) for o in detail.get("required_objects", ()) if o is not None]
        if len(missing_objects) != 1:
            return False
        missing_obj = missing_objects[0]
        place_id = detail.get("place_id")
        if place_id is None:
            return False
        event = component.observation_formula.events.get(event_id)
        if event is None:
            return False
        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return False

        self.jodap.stats["one_step_repair_attempts"] += 1
        self.jodap._diag(
            "one_step_repair_attempt", component=component.component_id,
            event_id=event_id, activity=event.activity, parent_node=parent.node_id,
            parent_cost=float(parent_assignment.total_cost), missing_object=missing_obj,
            required_place_id=place_id, required_place=detail.get("place_name"),
        )

        # Find transitions that can put this concrete object into the missing
        # place.  Binding compatibility is checked with the same logical
        # inscription parser used by the local synchronous fast path.
        try:
            # Producer discovery is purely structural. Reuse the single
            # parsed static model instead of reparsing PNML for every repair.
            net = self.jodap._query_slice_static_net()
            candidates = []
            for t in net._transitions:
                if t.get("invisible", False):
                    continue
                if not any(a.get("source") == t["id"] and a.get("target") == place_id
                           for a in net._arcs):
                    continue
                if self.jodap._lazy_binding(component, net, t, (missing_obj,)) is None:
                    continue
                candidates.append(t)
        except Exception as exc:
            self.jodap.stats["one_step_repair_fallbacks"] += 1
            self.jodap._diag(
                "one_step_repair_decline", component=component.component_id,
                event_id=event_id, reason="producer_discovery_failed", error=repr(exc),
            )
            return False

        if not candidates:
            self.jodap.stats["one_step_repair_fallbacks"] += 1
            self.jodap._diag(
                "one_step_repair_decline", component=component.component_id,
                event_id=event_id, reason="no_visible_single_object_producer",
                missing_object=missing_obj, required_place_id=place_id,
            )
            return False

        # A +1 lower bound is safe when the missing place has no invisible
        # producer and no already-observed zero-cost synchronous producer for
        # this object. In that case restoring the token necessarily requires at
        # least one visible model-only move, whose one-object cost is 1.
        invisible_producer = any(
            t.get("invisible", False)
            and any(a.get("source") == t["id"] and a.get("target") == place_id for a in net._arcs)
            and self.jodap._lazy_binding(component, net, t, (missing_obj,)) is not None
            for t in net._transitions
        )
        producer_labels = {str(t.get("label")) for t in candidates}
        observed_producer = any(
            e.activity in producer_labels and missing_obj in set(e.objects)
            for eid, e in component.observation_formula.events.items()
            if eid in parent.consumed
        )
        positive_lb_proved = not invisible_producer and not observed_producer
        repair_lb = float(parent_assignment.total_cost) + 1.0
        if positive_lb_proved:
            state.proven_prefix_lower_bound = max(state.proven_prefix_lower_bound, repair_lb)
            self.jodap.stats["one_step_repair_lb_proofs"] += 1
            self.jodap._diag(
                "one_step_repair_lower_bound", component=component.component_id,
                event_id=event_id, lower_bound=repair_lb,
                reason="missing_token_requires_visible_model_producer",
                missing_object=missing_obj, required_place_id=place_id,
            )

        sync_transitions = list(self.visible_by_label.get(event.activity, ()))
        if not sync_transitions:
            return False

        best = None
        for producer in sorted(candidates, key=lambda t: t["id"]):
            for sync_t in sync_transitions:
                self.jodap.stats["one_step_repair_candidates"] += 1
                try:
                    focused = self._focused_state_from_boundary(state, parent.node_id)
                    root = focused.nodes[0]
                    model_move = SymbolicMove(
                        "model", transition_id=producer["id"],
                        transition_label=producer.get("label")
                    )
                    model_node = self._add_candidate(
                        component, focused, root, model_move, evaluate=False
                    )
                    if model_node is None:
                        continue
                    sync_move = SymbolicMove(
                        "sync", event_id=event_id, transition_id=sync_t["id"],
                        transition_label=sync_t.get("label")
                    )
                    sync_node = self._add_candidate(
                        component, focused, model_node, sync_move, evaluate=False
                    )
                    if sync_node is None:
                        continue
                    # The lazy suffix encoder indexes model steps in global
                    # model-depth coordinates when a certified boundary exists.
                    fixed_step = int(parent.model_depth)
                    assignment = self.jodap._lazy_solve_fixed_path(
                        component, focused, sync_node,
                        lower_bound=int(math.floor(repair_lb + 1e-9)),
                        fixed_binding_by_step={fixed_step: (missing_obj,)},
                        accept_certified_nonzero=True,
                    )
                except Exception as exc:
                    self.jodap._diag(
                        "one_step_repair_candidate_error", component=component.component_id,
                        event_id=event_id, producer_transition=producer.get("label"),
                        producer_transition_id=producer.get("id"), error=repr(exc),
                    )
                    continue
                if assignment is None:
                    continue
                if best is None or assignment.total_cost < best[0].total_cost - 1e-9:
                    best = (assignment, producer, sync_t)

        if best is None:
            self.jodap.stats["one_step_repair_fallbacks"] += 1
            self.jodap._diag(
                "one_step_repair_decline", component=component.component_id,
                event_id=event_id, reason="all_two_step_candidates_infeasible",
                candidate_count=len(candidates) * len(sync_transitions),
            )
            return False

        assignment, producer, sync_t = best
        # Materialize only the two suffix moves in the retained search graph.
        model_node = self._add_candidate(
            component, state, parent,
            SymbolicMove("model", transition_id=producer["id"],
                         transition_label=producer.get("label")),
            evaluate=False,
        )
        if model_node is None:
            return False
        sync_node = self._add_candidate(
            component, state, model_node,
            SymbolicMove("sync", event_id=event_id, transition_id=sync_t["id"],
                         transition_label=sync_t.get("label")),
            evaluate=False,
        )
        if sync_node is None:
            return False
        sync_node.g = float(assignment.total_cost)
        sync_node.assignment_cost = float(assignment.total_cost)
        sync_node.h = self._heuristic(component, sync_node.consumed)
        state.assignments_by_node[sync_node.node_id] = assignment
        state.push(sync_node)
        self.jodap.stats["one_step_repair_hits"] += 1

        if assignment.total_cost < state.upper_bound - 1e-9:
            state.upper_bound = float(assignment.total_cost)
            state.incumbent_moves = tuple(self._path(state, sync_node.node_id))
            state.incumbent_assignment = assignment
            state.incumbent_offline = False
        all_events = frozenset(component.execution.event_ids)
        if sync_node.consumed == all_events and positive_lb_proved \
                and assignment.total_cost <= state.proven_prefix_lower_bound + 1e-9:
            state.current_goal = sync_node.node_id
            state.current_assignment = assignment
            self.jodap.stats["one_step_repair_proven"] += 1
            self.jodap._diag(
                "one_step_repair_success", component=component.component_id,
                event_id=event_id, missing_object=missing_obj,
                producer_transition=producer.get("label"),
                producer_transition_id=producer.get("id"),
                sync_transition=sync_t.get("label"),
                total_cost=float(assignment.total_cost),
                lower_bound=float(state.proven_prefix_lower_bound), proven=True,
            )
            return True

        self.jodap.stats["one_step_repair_incumbents"] += 1
        self.jodap._diag(
            "one_step_repair_success", component=component.component_id,
            event_id=event_id, missing_object=missing_obj,
            producer_transition=producer.get("label"),
            producer_transition_id=producer.get("id"),
            sync_transition=sync_t.get("label"),
            total_cost=float(assignment.total_cost),
            lower_bound=float(state.proven_prefix_lower_bound), proven=False,
        )
        return False


    def _try_fresh_object_zero_cost_sync_extension(
            self, component: ComponentState, state: SearchState,
            parent: Optional[SearchNode], event_id: str) -> bool:
        """Extend a certified prefix by ``nu* ; sync`` without reopening history.

        This is the non-merge counterpart of the direct merge bridge fast path.
        A newly observed object (for example a PACKAGE) changes the component
        object domain, so the old implementation classified the update as
        ``observations_changed`` and rebuilt/re-solved the whole retained search
        *before* trying the existing silent-preparation macro.  For a simple
        invisible fresh-object transition that work is unnecessary: the retained
        assignment is already a certified boundary.

        We therefore apply each supported ``nu`` creation analytically to that
        concrete marking, materialize only private zero-cost model predecessors,
        and run the same local delta synchronous check used by merge composition.
        No full fixed-path/JODAP fallback is allowed inside this fast path; if any
        local condition is unknown or false we remove the private preparation and
        continue with the original exact rebuild/search unchanged.
        """
        if parent is None or parent.node_id not in state.nodes:
            return False
        if event_id in parent.consumed:
            return False
        if event_id not in {e.event_id for e in self._enabled_events(component, parent)}:
            return False
        event = component.observation_formula.events.get(event_id)
        if event is None or not self.visible_by_label.get(event.activity):
            return False

        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return False

        marked = self._marked_objects_from_assignment(component, parent_assignment)
        missing = sorted(
            (o for o in event.objects if o not in marked),
            key=lambda o: (component.observation_formula.object_types.get(o, ""), o),
        )
        if not missing:
            return False

        self.jodap.stats["fresh_object_sync_attempts"] += 1
        self.jodap._diag(
            "fresh_object_sync_attempt", component=component.component_id,
            event_id=event_id, activity=event.activity, parent_node=parent.node_id,
            parent_cost=float(parent_assignment.total_cost),
            missing_objects=list(missing),
        )

        current_node = parent
        current_assignment = parent_assignment
        created_ids: List[int] = []

        def cleanup() -> None:
            # These nodes were never placed in OPEN/signatures.  Stale references
            # therefore cannot survive after a failed local proof.
            for nid in reversed(created_ids):
                state.open_ids.discard(nid)
                state.closed_ids.discard(nid)
                state.assignments_by_node.pop(nid, None)
                state.predecessor.pop(nid, None)
                state.nodes.pop(nid, None)

        for obj in missing:
            typ = component.observation_formula.object_types.get(obj)
            transition = self._nu_transition_for_type(typ) if typ is not None else None
            if transition is None:
                self.jodap.stats["fresh_object_sync_prepare_failures"] += 1
                self.jodap.stats["fresh_object_sync_failures"] += 1
                self.jodap._diag(
                    "fresh_object_sync_decline", component=component.component_id,
                    event_id=event_id, activity=event.activity, object_id=obj,
                    object_type=typ, reason="no_simple_nu_transition_for_type",
                )
                cleanup()
                return False

            next_assignment = self._apply_zero_cost_fresh_creation(
                component, current_assignment, obj, transition, current_node.model_depth
            )
            if next_assignment is None:
                self.jodap.stats["fresh_object_sync_prepare_failures"] += 1
                self.jodap.stats["fresh_object_sync_failures"] += 1
                self.jodap._diag(
                    "fresh_object_sync_decline", component=component.component_id,
                    event_id=event_id, activity=event.activity, object_id=obj,
                    object_type=typ, transition_id=transition.get("id"),
                    transition=transition.get("label"),
                    reason="simple_nu_creation_rejected",
                )
                cleanup()
                return False

            depth = current_node.model_depth + 1
            if depth > state.model_bound:
                self.jodap.stats["fresh_object_sync_prepare_failures"] += 1
                self.jodap.stats["fresh_object_sync_failures"] += 1
                self.jodap._diag(
                    "fresh_object_sync_decline", component=component.component_id,
                    event_id=event_id, activity=event.activity, object_id=obj,
                    reason="model_bound_exceeded", requested_depth=depth,
                    model_bound=state.model_bound,
                )
                cleanup()
                return False

            move = SymbolicMove(
                "model", transition_id=transition["id"],
                transition_label=transition.get("label")
            )
            nid = state.new_id()
            node = SearchNode(
                node_id=nid,
                consumed=current_node.consumed,
                event_order=current_node.event_order,
                model_depth=depth,
                g=float(next_assignment.total_cost),
                h=self._heuristic(component, current_node.consumed),
                assignment_cost=float(next_assignment.total_cost),
                model_signature=current_node.model_signature + (transition["id"],),
                move_signature=current_node.move_signature +
                    (("model", None, transition["id"]),),
            )
            state.nodes[nid] = node
            state.predecessor[nid] = (current_node.node_id, move)
            state.assignments_by_node[nid] = next_assignment
            created_ids.append(nid)
            current_node = node
            current_assignment = next_assignment
            self.jodap.stats["fresh_object_sync_created_objects"] += 1
            self.jodap._diag(
                "fresh_object_sync_creation", component=component.component_id,
                event_id=event_id, activity=event.activity, object_id=obj,
                object_type=typ, transition_id=transition.get("id"),
                transition=transition.get("label"), node=nid, model_depth=depth,
            )

        self.jodap.stats["fresh_object_sync_preparations"] += 1
        # The prepared assignment is already installed on current_node.  Use only
        # the local delta check here: falling back to check_fixed_sync_extension
        # would re-encode the historical path and defeat the purpose of this fast
        # path.  The ordinary exact search remains untouched below on failure.
        if self._try_zero_cost_sync_extension(
                component, state, current_node, event_id, allow_fixed_fallback=False):
            self.jodap.stats["fresh_object_sync_successes"] += 1
            self.jodap._diag(
                "fresh_object_sync_success", component=component.component_id,
                event_id=event_id, activity=event.activity,
                created_objects=list(missing),
                upper_bound=float(state.upper_bound),
                lower_bound=float(state.proven_prefix_lower_bound),
            )
            return True

        # If the local zero-cost checker has already diagnosed a missing
        # required input token, prioritize the dedicated one-step producer
        # repair before the more general object-relation machinery.  The latter
        # performs substantially heavier LIST-domain / guard-directed analysis
        # in the current implementation and can delay the simple missing-pick
        # case until the observation timeout.  The one-step repair is exact: it
        # returns True only when its +1 candidate meets the proven lower bound;
        # otherwise any feasible candidate is kept merely as an incumbent and
        # the general repair paths below remain available.
        decline = getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
        if decline.get("reason") == "required_input_token_not_marked":
            if self._try_one_step_missing_input_repair(
                    component, state, current_node, event_id):
                self.jodap.stats["fresh_object_sync_successes"] += 1
                self.jodap._diag(
                    "fresh_object_sync_success", component=component.component_id,
                    event_id=event_id, activity=event.activity,
                    created_objects=list(missing), via="one_step_repair_prioritized",
                    upper_bound=float(state.upper_bound),
                    lower_bound=float(state.proven_prefix_lower_bound),
                )
                return True

        # A fresh-object event may also carry one wrong object relation.
        object_rel_handled, object_rel_proven = \
            self._try_object_relation_deviation_sync_extension(
                component, state, current_node, event_id)
        if object_rel_proven:
            self.jodap.stats["fresh_object_sync_successes"] += 1
            self.jodap._diag(
                "fresh_object_sync_success", component=component.component_id,
                event_id=event_id, activity=event.activity,
                created_objects=list(missing), via="object_relation_repair",
                upper_bound=float(state.upper_bound),
                lower_bound=float(state.proven_prefix_lower_bound),
            )
            return True

        # The fresh object may be structurally correct while one observed
        # object attribute violates the new guard (e.g. mutated PACKAGE weight).
        if (not object_rel_handled) and self._try_guard_data_deviation_sync_extension(
                component, state, current_node, event_id):
            self.jodap.stats["fresh_object_sync_successes"] += 1
            self.jodap._diag(
                "fresh_object_sync_success", component=component.component_id,
                event_id=event_id, activity=event.activity,
                created_objects=list(missing), via="guard_data_repair",
                upper_bound=float(state.upper_bound),
                lower_bound=float(state.proven_prefix_lower_bound),
            )
            return True

        # For non-missing-token declines (or when the prioritized attempt did
        # not prove the prefix), retain the original final one-step fallback.
        if (not object_rel_handled) and decline.get("reason") != "required_input_token_not_marked" \
                and self._try_one_step_missing_input_repair(
                    component, state, current_node, event_id):
            self.jodap.stats["fresh_object_sync_successes"] += 1
            self.jodap._diag(
                "fresh_object_sync_success", component=component.component_id,
                event_id=event_id, activity=event.activity,
                created_objects=list(missing), via="one_step_repair",
                upper_bound=float(state.upper_bound),
                lower_bound=float(state.proven_prefix_lower_bound),
            )
            return True

        detail = self.jodap._last_zero_cost_sync_decline_reason
        self.jodap.stats["fresh_object_sync_failures"] += 1
        self.jodap._diag(
            "fresh_object_sync_decline", component=component.component_id,
            event_id=event_id, activity=event.activity,
            created_objects=list(missing),
            reason="zero_cost_sync_extension_failed", detail_reason=detail,
        )
        cleanup()
        return False


    def _try_first_event_latent_object_relation_repair(
            self, component: ComponentState, state: SearchState,
            root: Optional[SearchNode], event_id: str) -> bool:
        """Try an object-relation explanation for an orphan first observation.

        Removing the only ITEM relation from ``create package`` can make that
        event form its own observation component.  At that prefix a one-unit log
        move may already be optimal, and that alignment is perfectly valid.
        However, retaining *only* the log explanation discards a co-optimal model
        state that later ``send package`` can extend cheaply.  This helper keeps
        an equally optimal relation-repair witness whenever it can be certified
        using only already-observed active component state.

        No future event is consulted.  We start from an empty local model state,
        create any newly observed object through supported zero-cost ``nu``
        transitions, borrow only certified zero-cost live ITEM state from other
        currently active components, and invoke the ordinary bounded relation
        repair.  The result is accepted solely by total cost/lower-bound equality;
        no particular cost decomposition is required for correctness.
        """
        if root is None or root.node_id not in state.nodes:
            return False
        if root.consumed or len(component.execution.event_ids) != 1:
            return False
        event = component.observation_formula.events.get(event_id)
        if event is None or not self.visible_by_label.get(event.activity):
            return False
        # This path is useful only for transitions with a LIST object parameter.
        try:
            net = self.jodap._query_slice_static_net()
        except Exception:
            net = self.jodap._static_net or self.net_metadata
        has_list = False
        for t in self.visible_by_label.get(event.activity, ()):
            try:
                nt = next(x for x in net._transitions if x.get("id") == t.get("id"))
                decls = self.jodap._lazy_unique_object_decls(net, nt)
                if any("LIST" in str(tp) for tp in decls.values()):
                    has_list = True
                    break
            except Exception:
                continue
        if not has_list:
            return False

        self.jodap.stats["first_event_latent_relation_attempts"] += 1
        self.jodap._diag(
            "first_event_latent_relation_attempt",
            component=component.component_id, event_id=event_id,
            activity=event.activity, observed_objects=sorted(event.objects))

        # Construct the smallest concrete local boundary.  The relevant creation
        # transitions used by this benchmark are source-less and zero cost; if a
        # different model needs additional initial state this conservative path
        # simply declines and the exact search remains available.
        current_assignment = JointAssignment(total_cost=0.0)
        current_node = root
        created_ids: List[int] = []

        def cleanup() -> None:
            for nid in reversed(created_ids):
                state.open_ids.discard(nid)
                state.closed_ids.discard(nid)
                state.assignments_by_node.pop(nid, None)
                state.predecessor.pop(nid, None)
                state.nodes.pop(nid, None)

        marked = self._marked_objects_from_assignment(component, current_assignment)
        missing = sorted(
            (o for o in event.objects if o not in marked),
            key=lambda o: (component.observation_formula.object_types.get(o, ""), o))
        for obj in missing:
            typ = component.observation_formula.object_types.get(obj)
            transition = self._nu_transition_for_type(typ) if typ is not None else None
            if transition is None:
                cleanup()
                return False
            next_assignment = self._apply_zero_cost_fresh_creation(
                component, current_assignment, obj, transition, current_node.model_depth)
            if next_assignment is None:
                cleanup()
                return False
            depth = current_node.model_depth + 1
            if depth > state.model_bound:
                cleanup()
                return False
            move = SymbolicMove("model", transition_id=transition["id"],
                                transition_label=transition.get("label"))
            nid = state.new_id()
            node = SearchNode(
                node_id=nid, consumed=current_node.consumed,
                event_order=current_node.event_order, model_depth=depth,
                g=float(next_assignment.total_cost),
                h=self._heuristic(component, current_node.consumed),
                assignment_cost=float(next_assignment.total_cost),
                model_signature=current_node.model_signature + (transition["id"],),
                move_signature=current_node.move_signature +
                    (("model", None, transition["id"]),))
            state.nodes[nid] = node
            state.predecessor[nid] = (current_node.node_id, move)
            state.assignments_by_node[nid] = next_assignment
            created_ids.append(nid)
            current_node = node
            current_assignment = next_assignment

        handled, proven = self._try_object_relation_deviation_sync_extension(
            component, state, current_node, event_id)
        if proven:
            self.jodap.stats["first_event_latent_relation_hits"] += 1
            self.jodap._diag(
                "first_event_latent_relation_success",
                component=component.component_id, event_id=event_id,
                activity=event.activity,
                upper_bound=float(state.upper_bound),
                lower_bound=float(state.proven_prefix_lower_bound))
            return True

        # A non-proven candidate remains a valid co-optimal/upper-bound witness
        # if the helper installed one.  Keep it in the state; otherwise remove
        # the private preparation and let normal A* proceed.
        if handled and state.incumbent_assignment is not None:
            if state.incumbent_assignment.total_cost <= state.upper_bound + 1e-9:
                self.jodap.stats["first_event_latent_relation_hits"] += 1
                return False
        cleanup()
        return False

    def _event_zero_deviation_increment_lb(
            self, component: ComponentState, event: StreamEvent) -> int:
        """Prove whether the new event necessarily adds positive cost.

        A zero-cost extension is possible only through a structurally compatible
        synchronous transition whose guard is satisfied when *all recorded
        observations are kept unchanged on the model side*.  We therefore bind
        the transition to the observed objects and directly evaluate its guard
        with the observed event values and currently observed object attributes.

        The previous implementation first inspected ``constraint.vars()`` and
        required every non-object variable to occur in ``event.attributes``.
        CoCoMoT's expression API may report object variables occurring inside
        functions (e.g. ``P`` in ``sum(cost(P))`` or ``o`` in ``budget(o)``) in
        a form that does not exactly match the binding dictionary.  Such guards
        were consequently classified as having an unknown historical scalar and
        the positive lower bound was lost.  The expression evaluator already has
        the sound behavior we need: missing event/process values, missing object
        bindings, or unsupported constructs evaluate to ``None``.  Hence we let
        it decide completeness instead of pre-classifying variables.

        If every structurally compatible transition evaluates to ``False``, no
        zero-deviation synchronous explanation exists, so at least one unit of
        non-negative alignment cost is unavoidable.  ``True`` or ``None`` for
        any candidate conservatively leaves the lower bound at zero.
        """
        transitions = self.visible_by_label.get(event.activity, ())
        if not transitions:
            return max(1, len(event.objects))

        observed_oa = component.observation_formula.current_object_attributes()
        object_values = {
            (str(o), str(a)): v
            for o, vals in observed_oa.items() for a, v in vals.items()
        }
        event_values = dict(event.attributes)
        saw_structural = False
        saw_unknown = False

        for t in transitions:
            binding = self.jodap._lazy_binding(
                component, self.net_metadata, t, tuple(event.objects)
            )
            if binding is None:
                continue
            saw_structural = True

            # An unguarded structurally compatible synchronous transition is a
            # zero-deviation explanation, so no positive increment is provable.
            constraint = t.get("constraint")
            if constraint is None:
                self.jodap.stats["zero_deviation_guard_true"] += 1
                return 0

            self.jodap.stats["zero_deviation_guard_checks"] += 1
            value = self.jodap._lazy_eval_expr(
                constraint, event_values, binding, object_values
            )
            if value is True:
                self.jodap.stats["zero_deviation_guard_true"] += 1
                return 0
            if value is None:
                # Missing historical/process data or an unsupported expression
                # means that this transition *might* still admit a zero-cost
                # explanation.  Remember the uncertainty but continue checking
                # other candidates for diagnostics.
                self.jodap.stats["zero_deviation_guard_unknown"] += 1
                saw_unknown = True
                continue

            # False means this concrete synchronous explanation requires at
            # least one data/object observation repair.
            self.jodap.stats["zero_deviation_guard_false"] += 1

        if not saw_structural:
            # No synchronous move can even match the activity/object structure;
            # the event must be consumed by a positive-cost deviation.
            return max(1, len(event.objects))
        if saw_unknown:
            return 0

        # Every structurally compatible synchronous transition was false under
        # the zero-deviation observed valuation.  Since deviation costs are
        # non-negative and unit mismatch costs are at least one, the extended
        # prefix has an incremental lower bound of one.
        self.jodap.stats["positive_lower_bound_proofs"] += 1
        return 1

    def _marked_objects_from_assignment(self, component: ComponentState,
                                        assignment: Optional[JointAssignment]) -> Set[str]:
        """Extract concrete process objects currently represented in the model marking."""
        if assignment is None:
            return set()
        known = set(component.objects)
        marked: Set[str] = set()
        for item in assignment.marking_signature:
            marked.update(self._objects_in_value(item, known))
        return marked

    def _nu_transition_for_type(self, object_type: str) -> Optional[Dict[str, Any]]:
        """Choose one canonical invisible fresh-object transition for a type.

        This is used only for a speculative macro successor. If the chosen
        preparation is not feasible, ordinary A* expansion remains available.
        """
        candidates = [
            self.transition_by_id[tid]
            for tid, typ in self.nu_transition_types.items()
            if typ == object_type and tid in self.transition_by_id
               and self.transition_by_id[tid].get("invisible", False)
        ]
        return min(candidates, key=lambda t: t["id"]) if candidates else None

    def _silent_preparation_for_event(self, component: ComponentState,
                                      state: SearchState, parent: SearchNode,
                                      event: StreamEvent) \
            -> Optional[Tuple[List[SymbolicMove], Dict[int, str]]]:
        """Build a conservative zero-cost creation macro for an observed event.

        Only missing observed objects that can be introduced by a single
        invisible fresh-object transition are included. The macro is merely a
        shortcut: it is accepted only after one exact JODAP check and normal A*
        remains available when it fails.
        """
        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        marked = self._marked_objects_from_assignment(component, parent_assignment)
        missing = sorted(
            (o for o in event.objects if o not in marked),
            key=lambda o: (component.observation_formula.object_types.get(o, ""), o),
        )
        if not missing:
            return None

        moves: List[SymbolicMove] = []
        bindings: Dict[int, str] = {}
        model_step = parent.model_depth
        for obj in missing:
            typ = component.observation_formula.object_types.get(obj)
            if typ is None:
                return None
            t = self._nu_transition_for_type(typ)
            if t is None:
                return None
            moves.append(SymbolicMove(
                "model", transition_id=t["id"], transition_label=t.get("label")
            ))
            bindings[model_step] = obj
            model_step += 1
        return moves, bindings

    def _try_zero_cost_prepared_sync_extension(
            self, component: ComponentState, state: SearchState,
            parent: SearchNode, event_id: str) -> bool:
        """Try ``tau* ; sync`` as one macro successor with a single JODAP query.

        The macro collapses mandatory invisible fresh-object preparation before
        a synchronous event. Intermediate invisible moves remain in the
        reconstructed alignment, but they are not independently inserted into
        OPEN or optimized one by one.
        """
        if event_id in parent.consumed:
            return False
        if event_id not in {e.event_id for e in self._enabled_events(component, parent)}:
            return False
        event = component.observation_formula.events[event_id]
        transitions = self.visible_by_label.get(event.activity, ())
        if not transitions:
            return False
        prep = self._silent_preparation_for_event(component, state, parent, event)
        if prep is None:
            return False
        prep_moves, fresh_bindings = prep
        if parent.model_depth + len(prep_moves) + 1 > state.model_bound:
            return False

        lower_bound = 0 if parent.g == float("inf") else int(parent.g)
        lower_bound = max(lower_bound, int(state.proven_prefix_lower_bound))
        for t in transitions:
            self.jodap.stats["silent_macro_attempts"] += 1
            created_ids: List[int] = []
            cur = parent
            # Materialize a private predecessor chain, but do not place any of
            # the intermediate nodes in OPEN.
            all_moves = list(prep_moves) + [SymbolicMove(
                "sync", event_id=event_id,
                transition_id=t["id"], transition_label=t.get("label")
            )]
            failed = False
            for move in all_moves:
                consumed = cur.consumed
                event_order = cur.event_order
                depth = cur.model_depth
                msig = cur.model_signature
                mvsig = cur.move_signature + ((move.kind, move.event_id, move.transition_id),)
                if move.kind in ("log", "sync"):
                    if move.event_id in consumed:
                        failed = True
                        break
                    consumed = frozenset(set(consumed) | {move.event_id})
                    event_order = event_order + (move.event_id,)
                if move.kind in ("model", "sync"):
                    depth += 1
                    msig = msig + (move.transition_id,)
                nid = state.new_id()
                node = SearchNode(
                    nid, consumed, event_order, depth, float("inf"),
                    self._heuristic(component, consumed), 0.0, msig, mvsig
                )
                state.nodes[nid] = node
                state.predecessor[nid] = (cur.node_id, move)
                created_ids.append(nid)
                cur = node
            if failed:
                for nid in created_ids:
                    state.nodes.pop(nid, None); state.predecessor.pop(nid, None)
                continue

            assignment = self.jodap.solve(
                component, state, cur, require_final=False,
                lower_bound=lower_bound,
                canonical_fresh_bindings=fresh_bindings,
            )
            # A zero-cost preparation from an already optimal parent proves the
            # new prefix optimum. For the initial root, the global lower bound
            # is also zero.
            if assignment is None \
                    or assignment.total_cost > lower_bound + 1e-9 \
                    or assignment.total_cost + 1e-9 < state.proven_prefix_lower_bound:
                for nid in created_ids:
                    state.nodes.pop(nid, None); state.predecessor.pop(nid, None)
                continue

            cur.g = assignment.total_cost
            cur.assignment_cost = assignment.total_cost
            state.assignments_by_node[cur.node_id] = assignment
            sig = self._signature(cur)
            existing = state.signatures.get(sig)
            if existing is not None and existing != cur.node_id:
                ex = state.nodes.get(existing)
                if ex is not None and ex.g <= cur.g + 1e-9:
                    for nid in created_ids:
                        state.nodes.pop(nid, None); state.predecessor.pop(nid, None)
                    continue
            state.signatures[sig] = cur.node_id
            state.push(cur)
            self.jodap.stats["silent_macro_hits"] += 1
            self.jodap.stats["silent_macro_moves"] += len(prep_moves)
            if cur.consumed == frozenset(component.execution.event_ids):
                state.current_goal = cur.node_id
                state.current_assignment = assignment
                state.upper_bound = cur.g
                state.incumbent_moves = tuple(self._path(state, cur.node_id))
                state.incumbent_assignment = assignment
                state.incumbent_offline = False
            return True
        return False

    def _transition_data_dependencies(self, transition_id: int) -> Tuple[Set[str], Set[str]]:
        """Return conservative scalar data reads/writes for one transition.

        Guard variables and data carried on incoming arcs are reads. Explicit
        transition writes and data carried on outgoing arcs are writes. Object
        variables are harmless here because only names present in the model's
        data-type table are retained.
        """
        t = self.transition_by_id.get(transition_id)
        if t is None:
            return set(), set()
        raw_data_types = getattr(self.net_metadata, "_data_types", {})
        if isinstance(raw_data_types, dict):
            data_names = set(raw_data_types.keys())
        elif isinstance(raw_data_types, (list, tuple, set, frozenset)):
            # CoCoMoT's parsed net stores data types as a list in the current
            # model representation; membership tests below only need the names.
            data_names = set(raw_data_types)
        else:
            # Be conservative for unexpected metadata representations instead
            # of failing the provenance path.
            try:
                data_names = set(raw_data_types)
            except TypeError:
                data_names = set()
        reads: Set[str] = set()
        writes: Set[str] = {self._base_var(v) for v in t.get("write", [])}
        guard = t.get("constraint")
        if guard is not None:
            try:
                reads.update(self._base_var(v) for v in guard.vars())
            except Exception:
                reads.update(data_names)
        for arc in getattr(self.net_metadata, "_arcs", []):
            if arc.get("target") == transition_id:
                reads.update(str(n) for n, typ in arc.get("inscription", []) if typ in getattr(self.net_metadata, "_data_types", {}))
            if arc.get("source") == transition_id:
                writes.update(str(n) for n, typ in arc.get("inscription", []) if typ in getattr(self.net_metadata, "_data_types", {}))
        if data_names:
            reads.intersection_update(data_names)
            writes.intersection_update(data_names)
        return reads, writes

    def _ancestor_before_model_step(self, state: SearchState, goal: SearchNode,
                                    model_step: int) -> SearchNode:
        """Return the latest incumbent-path node immediately before model_step.

        A node with model_depth == model_step has executed exactly the model
        steps [0, model_step). Choosing the latest such node preserves any log
        moves that occurred before the defining model transition while reopening
        the defining transition itself and everything after it.
        """
        cur = goal
        candidate = goal
        while True:
            if cur.model_depth <= model_step:
                candidate = cur
                break
            pred = state.predecessor.get(cur.node_id)
            if pred is None:
                break
            cur = state.nodes[pred[0]]
        # If several log moves have the same model depth, walk forward is not
        # readily available; the backward traversal already stops at the latest
        # node at/below the requested depth, which is the desired boundary.
        return candidate

    def _provenance_dependency_slice(self, component: ComponentState, state: SearchState,
                                     previous_goal: SearchNode, event: StreamEvent) -> Optional[Dict[str, Any]]:
        """Build a conservative backward repair slice from retained provenance.

        The slice is *search focused*, not a semantic relaxation. It identifies
        the earliest retained model step that produced a scalar needed by the
        new event's candidate guards, recursively follows dependencies of those
        defining transitions, and keeps transitions touching those values plus
        the incumbent suffix and one-hop structural neighbours. If the focused
        search cannot prove the global lower bound, normal unrestricted A* is
        resumed, preserving exactness.
        """
        parent_assignment = state.assignments_by_node.get(previous_goal.node_id)
        if parent_assignment is None:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return None

        candidates = list(self.visible_by_label.get(event.activity, ()))
        if not candidates:
            return None

        provenance: Dict[str, Tuple[Any, int, str]] = {}
        for item in parent_assignment.data_provenance_signature or ():
            if len(item) >= 4:
                name, value, src_step, src_kind = item[:4]
                provenance[self._base_var(name)] = (value, int(src_step), str(src_kind))

        relevant_vars: Set[str] = set()
        relevant_object_attrs: Set[str] = set()
        candidate_ids: Set[int] = set()
        observed_oa = component.observation_formula.current_object_attributes()
        available_object_attrs = {a for vals in observed_oa.values() for a in vals}
        for t in candidates:
            candidate_ids.add(int(t["id"]))
            reads, _writes = self._transition_data_dependencies(int(t["id"]))
            relevant_vars.update(reads)
            relevant_object_attrs.update(
                self.jodap._guard_object_attribute_names(
                    t.get("constraint"), available_object_attrs
                )
            )

        # Observed values at the new event do not require reopening history.
        observed_now = {self._base_var(k) for k in event.attributes}
        pending = list(sorted(relevant_vars - observed_now))
        source_steps: Set[int] = set()
        dependency_transition_ids: Set[int] = set(candidate_ids)
        seen_vars: Set[str] = set()

        path = self._path(state, previous_goal.node_id)
        model_moves = [m for m in path if m.kind in ("model", "sync")]

        # Object attributes do not currently have scalar token provenance, but
        # their *guard-use provenance* is still available from the incumbent
        # path. Reopen from the earliest selected transition whose guard depends
        # on one of the same object attributes (e.g. vip/priority/budget/cost).
        if relevant_object_attrs:
            for step, move in enumerate(model_moves):
                if move.transition_id is None:
                    continue
                mt = self.transition_by_id.get(int(move.transition_id))
                if mt is None:
                    continue
                attrs = self.jodap._guard_object_attribute_names(
                    mt.get("constraint"), available_object_attrs
                )
                if attrs & relevant_object_attrs:
                    source_steps.add(step)
                    dependency_transition_ids.add(int(move.transition_id))

        while pending:
            name = pending.pop()
            if name in seen_vars:
                continue
            seen_vars.add(name)
            prov = provenance.get(name)
            if prov is None:
                continue
            src_step = int(prov[1])
            if src_step < 0 or src_step >= len(model_moves):
                continue
            source_steps.add(src_step)
            src_move = model_moves[src_step]
            if src_move.transition_id is None:
                continue
            tid = int(src_move.transition_id)
            dependency_transition_ids.add(tid)
            reads, _writes = self._transition_data_dependencies(tid)
            for dep in reads:
                relevant_vars.add(dep)
                if dep not in observed_now and dep not in seen_vars:
                    pending.append(dep)

        if not source_steps:
            # Static object attributes and already-certified current data do not
            # require replaying their writers.  They are exactly the case where
            # a boundary-state slice is most useful: keep the entire old optimum
            # fixed and solve only the new repair.  If a scalar is required but
            # is absent from both provenance and the retained data state, the
            # compact boundary is insufficient and we conservatively skip it.
            current_data_names = {self._base_var(str(x[0])) for x in (parent_assignment.data_state_signature or ()) if x}
            missing = {v for v in (relevant_vars - observed_now) if v not in current_data_names}
            if missing and not relevant_object_attrs:
                return None
            if not relevant_vars and not relevant_object_attrs:
                return None
            start_step = int(previous_goal.model_depth)
            boundary = previous_goal
        else:
            start_step = max(0, min(source_steps))
            boundary = self._ancestor_before_model_step(state, previous_goal, start_step)

        # Preserve the incumbent structural suffix from the defining write to
        # the previous goal. This ensures at least the known route remains in the
        # focused transition alphabet.
        for step, move in enumerate(model_moves):
            if step >= start_step and move.transition_id is not None:
                dependency_transition_ids.add(int(move.transition_id))

        # Add all transitions whose scalar/object-attribute footprint touches
        # the dependency set.
        for t in self.transitions:
            tid = int(t["id"])
            reads, writes = self._transition_data_dependencies(tid)
            attrs = self.jodap._guard_object_attribute_names(
                t.get("constraint"), available_object_attrs
            ) if relevant_object_attrs else set()
            if (reads | writes) & relevant_vars or attrs & relevant_object_attrs:
                dependency_transition_ids.add(tid)

        # Add invisible transitions and one-hop structural neighbours of the
        # retained set. This is conservative enough for preparation/routing while
        # still removing unrelated visible branches in data-heavy models.
        dependency_transition_ids.update(
            int(t["id"]) for t in self.transitions if t.get("invisible", False)
        )
        places_by_tid: Dict[int, Set[Any]] = {}
        for t in self.transitions:
            tid = int(t["id"])
            places_by_tid[tid] = {
                (a.get("source") if a.get("target") == tid else a.get("target"))
                for a in getattr(self.net_metadata, "_arcs", [])
                if a.get("source") == tid or a.get("target") == tid
            }
        touched_places: Set[Any] = set()
        for tid in list(dependency_transition_ids):
            touched_places.update(places_by_tid.get(tid, set()))
        for tid, places in places_by_tid.items():
            if places & touched_places:
                dependency_transition_ids.add(tid)

        kept = dependency_transition_ids & self.all_transition_ids
        if not kept or len(kept) >= len(self.all_transition_ids):
            return None

        return {
            "start_step": start_step,
            "boundary_node_id": boundary.node_id,
            "relevant_vars": tuple(sorted(relevant_vars)),
            "relevant_object_attributes": tuple(sorted(relevant_object_attrs)),
            "source_steps": tuple(sorted(source_steps)),
            "allowed_transition_ids": frozenset(kept),
            "total_transition_count": len(self.all_transition_ids),
            "reopened_model_steps": max(0, previous_goal.model_depth - start_step),
        }

    def _focused_state_from_boundary(self, state: SearchState, boundary_node_id: int) -> SearchState:
        """Create a temporary search whose semantic root is a certified witness.

        Unlike the first provenance experiment, the predecessor chain before the
        boundary is *not* copied into the temporary state.  Consequently
        ``JODAPSolver._path`` contains only reopened moves, while
        ``boundary_assignment`` supplies the concrete marking/data facts fixed by
        the certified prefix.  If this compact solver cannot handle a candidate,
        the focused experiment simply fails and the original full search resumes.
        """
        boundary_src = state.nodes[boundary_node_id]
        boundary_assn = state.assignments_by_node.get(boundary_node_id)
        if boundary_assn is None and state.current_goal == boundary_node_id:
            boundary_assn = state.current_assignment
        if boundary_assn is None:
            raise ValueError("provenance boundary has no certified assignment")

        focused = SearchState(state.component_id)
        focused.model_bound = state.model_bound
        focused.search_offline = False
        focused.current_event_ids = state.current_event_ids
        focused.current_objects = state.current_objects
        focused.current_attribute_observations = state.current_attribute_observations
        focused.current_object_attribute_snapshot = state.current_object_attribute_snapshot
        focused.proven_prefix_lower_bound = state.proven_prefix_lower_bound
        focused.upper_bound = float("inf")
        focused.boundary_assignment = boundary_assn
        focused.boundary_prefix_moves = tuple(self._path(state, boundary_node_id))
        focused.boundary_model_depth = int(boundary_src.model_depth)
        focused.boundary_node_original = boundary_node_id

        # Keep global structural signatures/depth/consumed events on the root so
        # successor generation remains identical to the original search, but cut
        # the predecessor link so JODAP sees only the suffix.
        root = _clone_search_node(boundary_src, node_id=0)
        root.g = float(boundary_assn.total_cost)
        root.assignment_cost = float(boundary_assn.total_cost)
        focused.nodes[0] = root
        focused.assignments_by_node[0] = boundary_assn
        focused.signatures[self._signature(root)] = 0
        focused.next_id = 1
        focused.push(root)
        return focused

    def _try_provenance_focused_repair(self, component: ComponentState, state: SearchState,
                                       previous_goal: SearchNode, event: StreamEvent) \
            -> Optional[Tuple[Tuple[SymbolicMove, ...], JointAssignment, bool, Dict[str, Any]]]:
        """Speculatively search only a provenance-derived repair suffix.

        The result is globally certified only when its cost reaches the
        independent prefix lower bound. Otherwise it is merely a feasible
        incumbent that may seed the subsequent unrestricted A* search.
        """
        if self.provenance_slicing != "focus":
            return None
        slice_info = self._provenance_dependency_slice(component, state, previous_goal, event)
        if slice_info is None:
            return None

        self.jodap.stats["provenance_slice_attempts"] += 1
        self.jodap.stats["provenance_slice_guard_vars"] += (
            len(slice_info["relevant_vars"]) + len(slice_info.get("relevant_object_attributes", ()))
        )
        self.jodap.stats["provenance_slice_transitions_total"] += int(slice_info["total_transition_count"])
        self.jodap.stats["provenance_slice_transitions_kept"] += len(slice_info["allowed_transition_ids"])
        self.jodap.stats["provenance_slice_model_steps_reopened"] += int(slice_info["reopened_model_steps"])
        t0 = time.perf_counter()

        focused = self._focused_state_from_boundary(state, int(slice_info["boundary_node_id"]))
        allowed = set(slice_info["allowed_transition_ids"])
        all_events = frozenset(component.execution.event_ids)

        # Do not copy a full-prefix incumbent into the suffix state: its move
        # sequence lives in a different coordinate system.  The original state
        # still retains that incumbent and will use it if focused repair fails.
        best_assignment = None
        best_moves: Tuple[SymbolicMove, ...] = ()
        best_cost = float("inf")
        expansions = 0

        while expansions < self.provenance_slice_max_expansions:
            node = focused.pop()
            if node is None:
                break
            if best_assignment is not None and node.f >= best_cost - 1e-9:
                break
            if node.consumed == all_events:
                assignment = self.jodap.solve(component, focused, node, require_final=False)
                if assignment is not None and assignment.total_cost < best_cost + 1e-9:
                    best_assignment = assignment
                    best_cost = assignment.total_cost
                    suffix_moves = tuple(self._path(focused, node.node_id))
                    best_moves = tuple(focused.boundary_prefix_moves) + suffix_moves
                    focused.upper_bound = best_cost
                    focused.incumbent_assignment = assignment
                    focused.incumbent_moves = best_moves
                if best_assignment is not None \
                        and best_cost <= state.proven_prefix_lower_bound + 1e-9:
                    self.jodap.stats["provenance_slice_hits"] += 1
                    self.jodap.stats["provenance_slice_proven"] += 1
                    self.jodap.stats["provenance_slice_nodes_expanded"] += expansions
                    self.jodap.stats["provenance_slice_seconds_ms"] += int((time.perf_counter() - t0) * 1000)
                    return best_moves, best_assignment, True, slice_info

            focused.closed_ids.add(node.node_id)
            self._expand(component, focused, node, allowed_transition_ids=allowed)
            expansions += 1

        self.jodap.stats["provenance_slice_nodes_expanded"] += expansions
        self.jodap.stats["provenance_slice_seconds_ms"] += int((time.perf_counter() - t0) * 1000)
        if best_assignment is not None:
            self.jodap.stats["provenance_slice_hits"] += 1
            improved = state.incumbent_assignment is None or best_cost < state.upper_bound - 1e-9
            if improved:
                self.jodap.stats["provenance_slice_incumbent_improvements"] += 1
            return best_moves, best_assignment, False, slice_info
        self.jodap.stats["provenance_slice_fallbacks"] += 1
        return None


    def _local_repair_transition_pool(
            self, component: ComponentState, state: SearchState,
            parent: SearchNode, event: StreamEvent) -> List[Dict[str, Any]]:
        """Return a small structurally relevant model-repair transition pool.

        This is deliberately an over-approximate performance heuristic, not a
        semantic restriction: if the bounded local search misses the optimum,
        unrestricted A* remains the exact fallback.
        """
        net = self.jodap._query_slice_static_net()
        sync_ids = {t["id"] for t in self.visible_by_label.get(event.activity, ())}
        required_places: Set[int] = set()

        # Places required by the current synchronous transition(s).
        for arc in getattr(net, "_arcs", ()):
            if arc.get("target") in sync_ids:
                required_places.add(arc.get("source"))

        # If the local checker has already identified a missing token, prioritize
        # transitions that can restore exactly that place.
        detail = getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
        place_id = detail.get("place_id")
        if place_id is not None:
            required_places.add(place_id)

        scored = []
        for t in getattr(net, "_transitions", ()):
            tid = t.get("id")
            if tid in sync_ids:
                continue
            outputs = {
                a.get("target") for a in getattr(net, "_arcs", ())
                if a.get("source") == tid
            }
            feeds_current = bool(outputs & required_places)
            invisible = bool(t.get("invisible", False))
            move = SymbolicMove(
                "model", transition_id=tid, transition_label=t.get("label")
            )
            lb = self._minimum_model_move_cost(move)
            # Cheap relevant producers first, then other silent transitions,
            # then other visible transitions.  Cost is the primary key.
            relevance = 0 if feeds_current else (1 if invisible else 2)
            scored.append((float(lb), relevance, str(t.get("label") or ""), int(tid), t))

        scored.sort(key=lambda row: row[:4])
        return [row[-1] for row in scored[:self.local_repair_transition_cap]]

    def _guard_directed_relation_object_sets(
            self, component: ComponentState, parent_assignment: JointAssignment,
            event: StreamEvent, transition: Dict[str, Any], limit: int
    ) -> Optional[List[Tuple[float, Tuple[str, ...]]]]:
        """Generate only bounded relation edits satisfying a simple aggregate guard.

        This is used *before* ordinary relation-candidate enumeration.  For a
        recognized guard such as ``weight(p) == sum(weight(I))`` we derive the
        required LIST aggregate from the concrete scalar object and search only
        add/remove edits whose observed numeric attributes can meet it.  The
        search is bounded by the existing object-relation cost limit and uses
        integerized decimal values, so it does not depend on floating-point
        equality.  Every returned candidate is rechecked with the existing
        independent guard evaluator.  Unsupported/incomplete cases return
        ``None`` and fall back to the old exact path.
        """
        spec = self.jodap.aggregate_list_guard_spec(transition)
        if spec is None:
            return None
        attr, scalar_var, list_var, offset = spec
        try:
            net = self.jodap._query_slice_static_net()
            nt = next(x for x in net._transitions if x.get("id") == transition.get("id"))
            decls = self.jodap._lazy_unique_object_decls(net, nt)
            observed_binding = self.jodap._lazy_binding(
                component, net, nt, tuple(sorted(str(o) for o in event.objects))
            )
        except Exception:
            return None
        if observed_binding is None:
            return None
        scalar_obj = observed_binding.get(scalar_var)
        observed_list = observed_binding.get(list_var)
        if scalar_obj is None or isinstance(scalar_obj, (list, tuple, set)):
            return None
        if not isinstance(observed_list, (list, tuple, set)):
            return None
        list_type = decls.get(list_var)
        if not list_type or "LIST" not in str(list_type):
            return None
        base_type = str(list_type)[:str(list_type).rfind(" LIST")]

        attrs = component.observation_formula.current_object_attributes()
        scalar_value = attrs.get(str(scalar_obj), {}).get(attr)
        if not isinstance(scalar_value, (int, float)) or isinstance(scalar_value, bool):
            return None

        marked = self._marked_objects_from_assignment(component, parent_assignment)
        observed_set = {str(o) for o in observed_list}
        addable = {
            str(o) for o in marked - set(map(str, event.objects))
            if component.observation_formula.object_types.get(str(o)) == base_type
        }
        universe = sorted(observed_set | addable)
        if not universe:
            return None
        numeric = {}
        raw_values = [scalar_value, offset]
        for obj in universe:
            value = attrs.get(obj, {}).get(attr)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            numeric[obj] = value
            raw_values.append(value)

        # Convert decimals to exact-ish integers using their written decimal
        # precision.  This avoids 7.249999999 comparisons while preserving the
        # concrete OCEL values used by the guard evaluator.
        try:
            from decimal import Decimal, InvalidOperation
            decimals = [Decimal(str(v)) for v in raw_values]
            scale_digits = max(max(0, -d.as_tuple().exponent) for d in decimals)
            scale = Decimal(10) ** scale_digits
            to_int = lambda v: int(Decimal(str(v)) * scale)
            target = to_int(scalar_value) - to_int(offset)
            values = {o: to_int(v) for o, v in numeric.items()}
        except Exception:
            return None

        current_sum = sum(values[o] for o in observed_set)
        delta_needed = target - current_sum
        removable = sorted(observed_set)
        addable = sorted(addable)

        # Build small change-combination maps rather than the full LIST
        # powerset.  ``limit`` is the existing relation-repair cardinality bound
        # (normally one or two), so complexity is polynomial in the candidate
        # pool for the supported fast path.
        rem_by_sum: Dict[Tuple[int, int], List[Tuple[str, ...]]] = {}
        add_by_sum: Dict[Tuple[int, int], List[Tuple[str, ...]]] = {}
        for r in range(0, min(limit, len(removable)) + 1):
            for combo in itertools.combinations(removable, r):
                rem_by_sum.setdefault((r, sum(values[o] for o in combo)), []).append(combo)
        for a in range(0, min(limit, len(addable)) + 1):
            for combo in itertools.combinations(addable, a):
                add_by_sum.setdefault((a, sum(values[o] for o in combo)), []).append(combo)

        out: Set[Tuple[float, Tuple[str, ...]]] = set()
        for (r, rem_sum), rem_combos in rem_by_sum.items():
            for a in range(0, limit - r + 1):
                # new_sum = current - removed + added = target
                required_add_sum = delta_needed + rem_sum
                add_combos = add_by_sum.get((a, required_add_sum), ())
                if not add_combos:
                    continue
                for rem in rem_combos:
                    for add in add_combos:
                        if r + a == 0:
                            continue
                        model_list = (observed_set - set(rem)) | set(add)
                        model_objects = {
                            str(o) for o in event.objects
                            if str(o) not in observed_set
                        } | model_list
                        try:
                            binding = self.jodap._lazy_binding(
                                component, net, nt, tuple(sorted(model_objects))
                            )
                            verdict = (
                                self.jodap.evaluate_guard_for_concrete_object_binding(
                                    component, nt, binding
                                ) if binding is not None else None
                            )
                        except Exception:
                            verdict = None
                        if verdict is True:
                            out.add((float(r + a), tuple(sorted(model_objects))))
                        elif verdict is None:
                            # Syntax recognition succeeded but independent
                            # semantic validation did not.  Do not prune.
                            return None

        rows = sorted(out, key=lambda row: (row[0], row[1]))
        # Approximate the old bounded candidate count for instrumentation only.
        generic = 0
        for k in range(1, min(limit, len(removable)) + 1):
            generic += math.comb(len(removable), k)
        for k in range(1, min(limit, len(addable)) + 1):
            generic += math.comb(len(addable), k)
        self.jodap.stats["guard_directed_binding_generation_attempts"] += 1
        self.jodap.stats["guard_directed_binding_generation_supported"] += 1
        self.jodap.stats["guard_directed_binding_generation_candidates"] += len(rows)
        self.jodap.stats["guard_directed_binding_generation_subsets_avoided"] += max(
            0, generic - len(rows)
        )
        self.jodap._diag(
            "guard_directed_binding_generation",
            component=component.component_id, event_id=event.event_id,
            activity=event.activity, transition_id=transition.get("id"),
            attribute=attr, scalar_object=str(scalar_obj), target=target,
            observed_list_size=len(observed_set), addable_size=len(addable),
            relation_limit=int(limit), generic_candidates=generic,
            generated_candidates=len(rows), subsets_avoided=max(0, generic - len(rows)),
        )
        return rows

    def _local_relation_repair_candidates(
            self, component: ComponentState, parent_assignment: JointAssignment,
            event: StreamEvent, transition: Dict[str, Any],
            diagnostic: Optional[Dict[str, Any]] = None) -> List[Tuple[float, Tuple[str, ...]]]:
        """Generate bounded model-side object sets for a local relation repair.

        This helper is intentionally side-effect free.  Persistent/cross-component
        relation handling remains owned by the established specialized fast path;
        the compositional search uses the same current-component semantics so it
        can combine a relation repair with preceding model repairs.
        """
        observed = {str(o) for o in event.objects}
        try:
            net = self.jodap._query_slice_static_net()
            nt = next(x for x in net._transitions if x.get("id") == transition.get("id"))
            decls = self.jodap._lazy_unique_object_decls(net, nt)
        except Exception:
            return []
        list_types = {typ[:typ.rfind(" LIST")] for typ in decls.values() if "LIST" in typ}
        if not list_types:
            return []
        marked = self._marked_objects_from_assignment(component, parent_assignment)
        removable = sorted(o for o in observed
                           if component.observation_formula.object_types.get(o) in list_types)
        addable = sorted(o for o in marked - observed
                         if component.observation_formula.object_types.get(o) in list_types)
        out: List[Tuple[float, Tuple[str, ...]]] = []
        limit = self.max_object_relation_fast_cardinality

        def marking_filter(rows: List[Tuple[float, Tuple[str, ...]]]) \
                -> List[Tuple[float, Tuple[str, ...]]]:
            kept: List[Tuple[float, Tuple[str, ...]]] = []
            pruned = 0
            supported = False
            for cost, model_objects in rows:
                domain = self.jodap.marking_feasible_list_domain(
                    component, parent_assignment, nt, tuple(sorted(model_objects)),
                    tuple(sorted(model_objects))
                )
                if domain is None:
                    kept.append((cost, model_objects))
                    continue
                supported = True
                if domain.get("infeasible"):
                    pruned += 1
                else:
                    kept.append((cost, model_objects))
            if supported:
                self.jodap.stats["marking_domain_candidates_pruned"] += pruned
                self.jodap._diag(
                    "marking_domain_candidate_filter", component=component.component_id,
                    event_id=event.event_id, activity=event.activity,
                    transition_id=transition.get("id"), candidates_before=len(rows),
                    candidates_after=len(kept), candidates_pruned=pruned,
                    path="compositional_relation_candidates",
                )
            return kept

        diagnostic = diagnostic or {}
        if diagnostic.get("reason") == "guard_false":
            directed = self._guard_directed_relation_object_sets(
                component, parent_assignment, event, transition, limit
            )
            if directed is not None:
                return marking_filter(directed)
            self.jodap.stats["guard_directed_binding_generation_fallbacks"] += 1

        # A missing concrete input token is a much stronger diagnostic than a
        # generic relation mismatch.  Changing unrelated members of a LIST
        # binding cannot make that token become marked.  The only relation-side
        # edit that can directly remove the diagnosed obligation is removal of
        # the missing observed object itself.  Keep those candidates (they may
        # be co-optimal with a model repair), but do not enumerate the powerset
        # of unrelated ITEM relations first.
        if diagnostic.get("reason") == "required_input_token_not_marked":
            missing = {
                str(o) for o in diagnostic.get("required_objects", ())
                if o is not None
            }
            direct = sorted(missing & set(removable))
            if direct:
                # One candidate per directly implicated object, plus the joint
                # removal when a token contains several missing list members.
                for obj in direct:
                    out.append((1.0, tuple(sorted(observed - {obj}))))
                if 1 < len(direct) <= limit:
                    out.append((float(len(direct)), tuple(sorted(observed - set(direct)))))
            # Report how much generic relation enumeration was deliberately
            # avoided.  This is diagnostic only; exact A* is still the fallback.
            generic = 0
            for k in range(1, min(limit, len(removable)) + 1):
                generic += math.comb(len(removable), k)
            for k in range(1, min(limit, len(addable)) + 1):
                generic += math.comb(len(addable), k)
            self.jodap.stats["local_repair_relation_candidates_pruned"] += max(
                0, generic - len(out)
            )
            out = sorted(set(out), key=lambda row: (row[0], row[1]))
            return marking_filter(out)

        for k in range(1, min(limit, len(removable)) + 1):
            for combo in itertools.combinations(removable, k):
                out.append((float(k), tuple(sorted(observed - set(combo)))))
        for k in range(1, min(limit, len(addable)) + 1):
            for combo in itertools.combinations(addable, k):
                out.append((float(k), tuple(sorted(observed | set(combo)))))
        out.sort(key=lambda row: (row[0], row[1]))
        return marking_filter(out)

    def _local_missing_token_producers(
            self, component: ComponentState, event: StreamEvent,
            diagnostic: Dict[str, Any], sync_transition: Dict[str, Any]) \
            -> List[Tuple[Dict[str, Any], Tuple[str, ...]]]:
        """Return producer moves bound to the concrete missing token objects.

        The returned binding is carried into the lazy local suffix solver.  This
        avoids losing the diagnostic information and reopening a symbolic object
        binding problem immediately after discovering exactly which token is
        missing.  The helper is intentionally conservative: if the producer
        cannot be concretely bound by the diagnosed objects, it is not used by
        this local fast path and exact A* remains available.
        """
        if diagnostic.get("reason") != "required_input_token_not_marked":
            return []
        place_id = diagnostic.get("place_id")
        missing = tuple(sorted(
            str(o) for o in diagnostic.get("required_objects", ()) if o is not None
        ))
        if place_id is None or not missing:
            return []
        try:
            net = self.jodap._query_slice_static_net()
        except Exception:
            return []

        sync_id = sync_transition.get("id")
        out: List[Tuple[Dict[str, Any], Tuple[str, ...]]] = []
        for t in getattr(net, "_transitions", ()):
            if t.get("id") == sync_id:
                continue
            if not any(a.get("source") == t.get("id") and a.get("target") == place_id
                       for a in getattr(net, "_arcs", ())):
                continue
            # Concrete-binding-first: only admit a local producer when the
            # diagnosed object tuple itself fully determines its object binding.
            if self.jodap._lazy_binding(component, net, t, missing) is None:
                continue
            out.append((t, missing))

        def key(row):
            t, binding = row
            mv = SymbolicMove("model", transition_id=t.get("id"),
                              transition_label=t.get("label"))
            return (self._minimum_model_move_cost(mv),
                    0 if t.get("invisible", False) else 1,
                    str(t.get("label") or ""), int(t.get("id")))

        out.sort(key=key)
        self.jodap.stats["local_repair_missing_token_producers"] += len(out)
        return out

    @staticmethod
    def _local_failure_signature(detail: Dict[str, Any]) -> Tuple[Any, ...]:
        """Stable signature for diagnostics whose truth is binding-invariant."""
        reason = detail.get("reason")
        if reason == "required_input_token_not_marked":
            return (
                reason,
                detail.get("place_id"),
                tuple(sorted(str(o) for o in detail.get("required_objects", ()) if o is not None)),
            )
        if reason == "guard_false":
            return (reason, tuple(sorted(detail.get("guard_variables", ()) or ())))
        return (reason,)

    def _try_cost_guided_local_repair(
            self, component: ComponentState, state: SearchState,
            parent: Optional[SearchNode], event_id: str) -> bool:
        """Compose local repairs in nondecreasing alignment cost before A*.

        The search starts at the *certified optimal prefix boundary*.  A queue
        contains model-repair prefixes and is ordered by an admissible structural
        cost lower bound, not by the number of edits.  At every popped boundary
        synchronization is retried first.  Its concrete failure diagnostic then
        generates only relevant data/object repair operators.  Model operators
        are added afterwards and can expose a different failure on the next pop,
        enabling chains such as ``missing pick -> guard repair -> sync``.

        A feasible result is final only when it reaches the independently proven
        prefix lower bound.  Otherwise it is installed solely as an upper-bound
        incumbent and unrestricted A* remains the exact correctness fallback.
        No future observation is inspected.
        """
        if parent is None or parent.node_id not in state.nodes:
            return False
        event = component.observation_formula.events.get(event_id)
        if event is None or event_id in parent.consumed:
            return False
        parent_assignment = state.assignments_by_node.get(parent.node_id)
        if parent_assignment is None and state.current_goal == parent.node_id:
            parent_assignment = state.current_assignment
        if parent_assignment is None:
            return False
        sync_transitions = list(self.visible_by_label.get(event.activity, ()))
        if not sync_transitions:
            return False

        self.jodap.stats["local_repair_search_attempts"] += 1
        base_cost = float(parent_assignment.total_cost)
        budget_limit = base_cost + self.local_repair_cost_budget
        model_pool = self._local_repair_transition_pool(component, state, parent, event)

        # (admissible additional cost, tie, transition-id sequence,
        #  concrete object bindings aligned with that sequence).  Carrying the
        # binding is essential for diagnostic-directed producer repairs: once a
        # missing token identifies the concrete object, the local search must
        # not throw that information away and reopen generic binding search.
        queue: List[Tuple[
            float, int, Tuple[int, ...], Tuple[Optional[Tuple[str, ...]], ...]
        ]] = []
        tie = itertools.count()
        heapq.heappush(queue, (0.0, next(tie), (), ()))
        seen: Set[Tuple[Tuple[int, ...], Tuple[Optional[Tuple[str, ...]], ...]]] = set()
        best = None
        generated = solver_calls = 0
        max_depth = 0

        # A log move is a legitimate local operator and gives an immediate,
        # solver-free incumbent.  It does not change the retained model state.
        log_extra = float(len(event.objects))
        if base_cost + log_extra <= budget_limit + 1e-9:
            log_assignment = self._assignment_with_total_cost(parent_assignment, base_cost + log_extra)
            best = (float(log_assignment.total_cost), (SymbolicMove("log", event_id=event_id),), log_assignment)

        self.jodap._diag(
            "local_repair_search_start", component=component.component_id,
            event_id=event_id, activity=event.activity, parent_node=parent.node_id,
            parent_cost=base_cost, prefix_lower_bound=float(state.proven_prefix_lower_bound),
            cost_budget=float(self.local_repair_cost_budget),
            candidate_cap=int(self.local_repair_max_candidates))

        while queue and generated < self.local_repair_max_candidates:
            add_lb, _tie, seq, seq_bindings = heapq.heappop(queue)
            state_key = (seq, seq_bindings)
            if state_key in seen:
                continue
            seen.add(state_key)
            max_depth = max(max_depth, len(seq))
            if base_cost + add_lb > budget_limit + 1e-9:
                self.jodap.stats["local_repair_search_pruned_budget"] += 1
                continue

            # Materialize only this local model-repair prefix in a focused state.
            focused = self._focused_state_from_boundary(state, parent.node_id)
            local_parent = focused.nodes[0]
            local_assignment = parent_assignment
            suffix_prefix: List[SymbolicMove] = []
            feasible_prefix = True
            fixed_binding_by_step: Dict[int, Sequence[str]] = {}
            for local_index, tid in enumerate(seq):
                t = self.transition_by_id.get(tid)
                if t is None:
                    feasible_prefix = False
                    break
                mv = SymbolicMove("model", transition_id=tid, transition_label=t.get("label"))
                nxt = self._add_candidate(component, focused, local_parent, mv, evaluate=False)
                if nxt is None:
                    feasible_prefix = False
                    break
                local_parent = nxt
                suffix_prefix.append(mv)
                concrete_binding = seq_bindings[local_index]
                if concrete_binding is not None:
                    fixed_binding_by_step[int(parent.model_depth) + local_index] = concrete_binding
            if not feasible_prefix:
                continue
            if seq:
                solver_calls += 1
                self.jodap.stats["local_repair_search_solver_calls"] += 1
                local_assignment = self.jodap._lazy_solve_fixed_path(
                    component, focused, local_parent,
                    lower_bound=int(math.floor(base_cost + add_lb + 1e-9)),
                    fixed_binding_by_step=fixed_binding_by_step or None,
                    accept_certified_nonzero=True)
                if local_assignment is None or float(local_assignment.total_cost) > budget_limit + 1e-9:
                    continue
                focused.assignments_by_node[local_parent.node_id] = local_assignment

            # Retry synchronization and let its failure reason guide explicit
            # data/object operator generation at this repaired boundary.
            targeted_model_enqueued = False
            decisive_missing_token = False
            for sync_t in sync_transitions:
                if generated >= self.local_repair_max_candidates:
                    break
                generated += 1
                self.jodap.stats["local_repair_search_candidates"] += 1
                self.jodap.stats["local_repair_event_binding_probes"] += 1
                sync_move = SymbolicMove("sync", event_id=event_id,
                                         transition_id=sync_t["id"],
                                         transition_label=sync_t.get("label"))
                temp = SearchNode(
                    node_id=-1,
                    consumed=frozenset(set(local_parent.consumed) | {event_id}),
                    event_order=local_parent.event_order + (event_id,),
                    model_depth=local_parent.model_depth + 1,
                    g=float(local_assignment.total_cost), h=0.0,
                    assignment_cost=float(local_assignment.total_cost),
                    model_signature=local_parent.model_signature + (sync_t["id"],),
                    move_signature=local_parent.move_signature + (("sync", event_id, sync_t["id"]),))

                assignment = self.jodap.check_zero_cost_sync_extension(
                    component, focused, local_parent, temp, event, sync_t["id"], local_assignment)
                if assignment is not None:
                    total = float(assignment.total_cost)
                    if total <= budget_limit + 1e-9 and (best is None or total < best[0] - 1e-9):
                        best = (total, tuple(suffix_prefix + [sync_move]), assignment)
                else:
                    detail = dict(getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {})
                    reason = detail.get("reason")

                    # Missing-token diagnostics name both the exact required
                    # place and the concrete object(s).  Generate producers for
                    # that obstruction immediately, with the diagnosed object
                    # binding fixed in the queued state.  This is the key
                    # failure -> repair -> retry composition and prevents LIST
                    # subset exploration from becoming the next search axis.
                    if reason == "required_input_token_not_marked":
                        decisive_missing_token = True
                        for producer, producer_binding in self._local_missing_token_producers(
                                component, event, detail, sync_t):
                            tid = int(producer["id"])
                            new_seq = seq + (tid,)
                            new_bindings = seq_bindings + (tuple(producer_binding),)
                            new_key = (new_seq, new_bindings)
                            if new_key in seen:
                                continue
                            if producer.get("invisible", False):
                                repair_cost = 0.0
                            else:
                                # For a fully concrete producer the visible
                                # model-move charge is the number of participating
                                # objects, exactly matching the alignment cost.
                                repair_cost = float(len(set(producer_binding)))
                            new_lb = add_lb + repair_cost
                            if base_cost + new_lb > budget_limit + 1e-9:
                                self.jodap.stats["local_repair_search_pruned_budget"] += 1
                                continue
                            heapq.heappush(
                                queue, (new_lb, next(tie), new_seq, new_bindings)
                            )
                            targeted_model_enqueued = True
                            self.jodap.stats["local_repair_missing_token_targeted_enqueues"] += 1
                            self.jodap._diag(
                                "local_repair_targeted_producer", component=component.component_id,
                                event_id=event_id, activity=event.activity,
                                required_place_id=detail.get("place_id"),
                                required_place=detail.get("place_name"),
                                missing_objects=list(producer_binding),
                                producer_transition=producer.get("label"),
                                producer_transition_id=producer.get("id"),
                                producer_cost=repair_cost,
                                accumulated_additional_lb=new_lb,
                            )

                    if reason == "guard_false":
                        try:
                            net = self.jodap._query_slice_static_net()
                            nt = next(t for t in net._transitions if t.get("id") == sync_t.get("id"))
                            binding = self.jodap._lazy_binding(component, net, nt, tuple(event.objects))
                            overrides = (self.jodap.infer_single_object_attribute_guard_repair(
                                component, nt, binding) if binding is not None else None)
                        except Exception:
                            overrides = None
                        if overrides:
                            repair_cost = float(len(overrides))
                            if float(local_assignment.total_cost) + repair_cost <= budget_limit + 1e-9:
                                repaired = self.jodap.check_zero_cost_sync_extension(
                                    component, focused, local_parent, temp, event, sync_t["id"], local_assignment,
                                    model_object_attribute_overrides=overrides,
                                    extra_cost=repair_cost, repair_tag="compositional_data_repair")
                                if repaired is not None:
                                    total = float(repaired.total_cost)
                                    if best is None or total < best[0] - 1e-9:
                                        best = (total, tuple(suffix_prefix + [sync_move]), repaired)

                    if reason in {"object_binding_failed", "required_input_token_not_marked", "guard_false"}:
                        original_failure = self._local_failure_signature(detail)
                        for repair_cost, model_objects in self._local_relation_repair_candidates(
                                component, local_assignment, event, sync_t, diagnostic=detail):
                            if float(local_assignment.total_cost) + repair_cost > budget_limit + 1e-9:
                                continue
                            repaired = self.jodap.check_zero_cost_sync_extension(
                                component, focused, local_parent, temp, event, sync_t["id"], local_assignment,
                                model_objects_override=model_objects, extra_cost=repair_cost,
                                repair_tag="compositional_object_relation_repair")
                            if repaired is None:
                                repaired_detail = dict(
                                    getattr(self.jodap, "_last_zero_cost_sync_decline_detail", None) or {}
                                )
                                if self._local_failure_signature(repaired_detail) == original_failure:
                                    self.jodap.stats["local_repair_invariant_failures_pruned"] += 1
                                    self.jodap._diag(
                                        "local_repair_relation_invariant_prune",
                                        component=component.component_id, event_id=event_id,
                                        activity=event.activity, reason=reason,
                                        failure_signature=repr(original_failure),
                                        model_objects=list(model_objects),
                                    )
                                continue
                            total = float(repaired.total_cost)
                            if best is None or total < best[0] - 1e-9:
                                # Preserve explicit relation metadata for continuation semantics.
                                if repaired.object_bindings:
                                    meta = self._relation_repair_metadata(
                                        component, set(map(str, event.objects)), set(model_objects))
                                    meta.update(repaired.object_bindings[-1])
                                    repaired.object_bindings[-1] = meta
                                best = (total, tuple(suffix_prefix + [sync_move]), repaired)
                            break

                if best is not None and best[0] <= state.proven_prefix_lower_bound + 1e-9:
                    queue.clear()
                    break

            if best is not None and best[0] <= state.proven_prefix_lower_bound + 1e-9:
                break

            # Generate further model/control-flow repairs.  There is deliberately
            # no operation-count/depth bound; termination is controlled by the
            # alignment-cost budget plus the global candidate cap.  Zero-cost
            # silent sequences remain safe because the cap is a performance-only
            # fallback to exact A*.
            #
            # When a concrete missing-token diagnostic produced targeted
            # producers, do *not* also fan out over unrelated model moves from
            # this state.  They cannot directly change the diagnosed marking
            # fact.  If the targeted path exposes a different failure, the next
            # queue pop will generate operators for that new diagnostic.
            if decisive_missing_token and targeted_model_enqueued:
                continue
            for t in model_pool:
                tid = int(t["id"])
                new_seq = seq + (tid,)
                concrete_binding: Optional[Tuple[str, ...]] = None
                try:
                    net = self.jodap._query_slice_static_net()
                    nt = next(x for x in net._transitions if x.get("id") == tid)
                    if not self.jodap._lazy_unique_object_decls(net, nt):
                        concrete_binding = ()
                except Exception:
                    pass
                # An unbound object-carrying model move cannot be represented by
                # the lazy local suffix solver.  Enqueuing it only guarantees a
                # local fallback query with no useful diagnostic.  Leave such
                # ambiguous moves to exact A*; diagnostic-directed producers
                # above are the mechanism that supplies concrete bindings here.
                if concrete_binding is None:
                    continue
                new_bindings = seq_bindings + (concrete_binding,)
                if (new_seq, new_bindings) in seen:
                    continue
                mv_lb = self._minimum_model_move_cost(
                    SymbolicMove("model", transition_id=tid, transition_label=t.get("label")))
                new_lb = add_lb + float(mv_lb)
                if base_cost + new_lb > budget_limit + 1e-9:
                    self.jodap.stats["local_repair_search_pruned_budget"] += 1
                    continue
                heapq.heappush(queue, (new_lb, next(tie), new_seq, new_bindings))

        self.jodap.stats["local_repair_search_max_depth"] = max(
            int(self.jodap.stats.get("local_repair_search_max_depth", 0)), max_depth)
        if best is None:
            self.jodap.stats["local_repair_search_fallbacks"] += 1
            return False

        total, suffix_moves, assignment = best
        # Keep the witness path explicitly.  ``_add_candidate`` canonicalizes
        # by signature and may return an already-existing node; in that case
        # ``_path(state, node)`` follows the old node's predecessor chain and is
        # not necessarily the repair path that produced ``assignment``.
        # Correctness of the upper bound depends on carrying the actual complete
        # witness, so compose it directly from the certified boundary path.
        explicit_witness_moves = tuple(self._path(state, parent.node_id)) + tuple(suffix_moves)
        node = parent
        for mv in suffix_moves:
            nxt = self._add_candidate(component, state, node, mv, evaluate=False)
            if nxt is None:
                self.jodap.stats["local_repair_search_fallbacks"] += 1
                return False
            node = nxt
        node.g = total
        node.assignment_cost = total
        node.h = self._heuristic(component, node.consumed)
        state.assignments_by_node[node.node_id] = assignment
        state.push(node)
        self.jodap.stats["local_repair_search_hits"] += 1

        if total < state.upper_bound - 1e-9:
            state.upper_bound = total
            state.incumbent_moves = explicit_witness_moves
            state.incumbent_assignment = assignment
            state.incumbent_offline = False

        complete = node.consumed == frozenset(component.execution.event_ids)
        proven = bool(complete and total <= state.proven_prefix_lower_bound + 1e-9)
        if proven:
            state.current_goal = node.node_id
            state.current_assignment = assignment
            self.jodap.stats["local_repair_search_proven"] += 1
            self.jodap.stats["positive_lower_bound_terminations"] += 1
        else:
            self.jodap.stats["local_repair_search_incumbents"] += 1
            # Record the local witness directly for the current online increment.
            # Do not rely on the generic incumbent fields surviving the remainder
            # of ``_sync_increment``: the incremental graph is deliberately
            # rebuilt/reseeded before unrestricted A*.  The explicit witness is
            # only an upper bound; it never certifies optimality by itself.
            if complete and total < state.pending_increment_incumbent_cost - 1e-9:
                consumed = frozenset(
                    m.event_id for m in explicit_witness_moves
                    if m.kind in ("log", "sync") and m.event_id is not None
                )
                if consumed == frozenset(component.execution.event_ids):
                    state.pending_increment_incumbent_moves = explicit_witness_moves
                    state.pending_increment_incumbent_assignment = assignment
                    state.pending_increment_incumbent_cost = float(total)
                    state.pending_increment_incumbent_event_ids = consumed
                    self.jodap._diag(
                        "local_repair_incumbent_handoff_recorded",
                        component=component.component_id, event_id=event_id,
                        upper_bound=float(total), move_count=len(explicit_witness_moves),
                    )
        self.jodap._diag(
            "local_repair_search_success", component=component.component_id,
            event_id=event_id, activity=event.activity, parent_cost=base_cost,
            total_cost=total, additional_cost=total-base_cost,
            prefix_lower_bound=float(state.proven_prefix_lower_bound), proven=proven,
            candidate_count=generated, solver_calls=solver_calls,
            sequence=[{"kind": m.kind, "transition_id": m.transition_id,
                       "transition": m.transition_label, "event_id": m.event_id}
                      for m in suffix_moves])
        return proven


    @staticmethod
    def _certified_boundary_assignment_for_goal(
            state: SearchState, previous_goal: Optional[SearchNode]
    ) -> Tuple[Optional[JointAssignment], bool]:
        """Recover the exact assignment of the last certified online goal.

        ``current_assignment`` is intentionally not treated as authoritative: it
        is a transient convenience pointer and can be cleared by observation-domain
        revalidation while the certified goal node and its exact JODAP assignment
        remain in ``assignments_by_node``.  The incumbent fallback is accepted only
        when it consumes exactly the previous observed prefix and has the same
        certified cost as the goal.

        Returns ``(assignment, recovered)`` where ``recovered`` is true when the
        assignment came from durable node/incumbent storage rather than the transient
        ``current_assignment`` pointer.
        """
        if previous_goal is None:
            return None, False
        if state.current_assignment is not None:
            return state.current_assignment, False
        assignment = state.assignments_by_node.get(previous_goal.node_id)
        if assignment is not None:
            return assignment, True
        if state.incumbent_assignment is not None and not state.incumbent_offline:
            incumbent_events = {
                m.event_id for m in state.incumbent_moves
                if m.kind in ("log", "sync") and m.event_id is not None
            }
            if incumbent_events == set(state.current_event_ids) \
                    and previous_goal.g != float("inf") \
                    and abs(float(state.incumbent_assignment.total_cost)
                            - float(previous_goal.g)) <= 1e-9:
                return state.incumbent_assignment, True
        return None, False

    def _sync_increment(self, component: ComponentState, state: SearchState) -> None:
        # A pending local witness belongs to exactly one observation increment.
        # Clear any stale value before processing the newly enlarged prefix.
        state.pending_increment_incumbent_moves = ()
        state.pending_increment_incumbent_assignment = None
        state.pending_increment_incumbent_cost = float("inf")
        state.pending_increment_incumbent_event_ids = frozenset()

        previous_goal = (state.nodes.get(state.current_goal)
                         if state.current_goal is not None else None)
        # Make the last certified online prefix a persistent boundary on the
        # search node itself.  Successors inherit this descriptor independently
        # of the canonical predecessor graph, so later exact queries can always
        # recover a checkpoint-relative suffix.
        if previous_goal is not None:
            # ``current_assignment`` is a convenience pointer, not the
            # authoritative storage for a certified goal.  Observation-domain
            # revalidation and several fast paths may clear that pointer while
            # retaining the certified goal node and its exact assignment in
            # ``assignments_by_node``.  Requiring both used to drop the
            # checkpoint lineage in the _025/_028-style hard cases and sent the
            # subsequent eager query straight back to absolute depth.
            boundary_assignment, recovered_assignment = \
                self._certified_boundary_assignment_for_goal(state, previous_goal)

            if boundary_assignment is not None:
                boundary_snapshot = _clone_search_node(previous_goal)
                # A boundary snapshot is a root descriptor, not another lineage.
                boundary_snapshot.checkpoint_boundary_assignment = None
                boundary_snapshot.checkpoint_boundary_snapshot = None
                boundary_snapshot.checkpoint_boundary_model_depth = 0
                boundary_snapshot.checkpoint_boundary_prefix_moves = ()
                boundary_snapshot.checkpoint_suffix_moves = ()
                previous_goal.checkpoint_boundary_assignment = boundary_assignment
                previous_goal.checkpoint_boundary_snapshot = boundary_snapshot
                previous_goal.checkpoint_boundary_model_depth = int(previous_goal.model_depth)
                previous_goal.checkpoint_boundary_prefix_moves = tuple(
                    self._path(state, previous_goal.node_id)
                )
                previous_goal.checkpoint_suffix_moves = ()
                self.jodap.stats["persistent_checkpoint_lineage_seeded"] += 1
                if recovered_assignment:
                    self.jodap.stats["persistent_checkpoint_lineage_assignment_recovered"] += 1
                    self.jodap._diag(
                        "persistent_checkpoint_lineage_assignment_recovered",
                        component=component.component_id, node=previous_goal.node_id,
                        boundary_model_depth=int(previous_goal.model_depth),
                        boundary_cost=float(boundary_assignment.total_cost),
                    )
                self.jodap._diag(
                    "persistent_checkpoint_lineage_seeded",
                    component=component.component_id, node=previous_goal.node_id,
                    boundary_model_depth=int(previous_goal.model_depth),
                    boundary_cost=float(boundary_assignment.total_cost),
                )
        new_events = set(component.execution.event_ids) - set(state.current_event_ids)
        new_objects = frozenset(component.objects) != state.current_objects
        attr_count = sum(len(v) for v in component.observation_formula.attribute_history.values())
        attrs_now = component.observation_formula.current_object_attributes()
        attr_snapshot = tuple(
            sorted((o, a, self.jodap._freeze_value(v))
                   for o, vals in attrs_now.items() for a, v in vals.items())
        )
        old_attr_map = {(o, a): v for o, a, v in state.current_object_attribute_snapshot}
        new_attr_map = {(o, a): v for o, a, v in attr_snapshot}
        changed_attr_names = {a for (o, a) in set(old_attr_map) | set(new_attr_map)
                              if old_attr_map.get((o, a)) != new_attr_map.get((o, a))}
        observations_changed = new_objects or bool(changed_attr_names) \
            or attr_count != state.current_attribute_observations
        old_bound = state.model_bound
        new_bound = self._bound(component)
        bound_grew = new_bound > old_bound
        state.model_bound = max(old_bound, new_bound)

        # Prefix costs are monotone.  If the new event has no zero-deviation
        # synchronous explanation under fully observed guard inputs, it adds at
        # least one unit to *every* alignment, independently of the retained
        # model marking.  This gives the positive-cost analogue of the familiar
        # global zero lower bound.
        if len(new_events) == 1 and previous_goal is not None \
                and previous_goal.g != float("inf"):
            eid_lb = next(iter(new_events))
            ev_lb = component.observation_formula.events.get(eid_lb)
            if ev_lb is not None:
                inc_lb = self._event_zero_deviation_increment_lb(component, ev_lb)
                state.proven_prefix_lower_bound = max(
                    state.proven_prefix_lower_bound,
                    float(previous_goal.g) + float(inc_lb)
                )

        # Fresh observed objects (for example a new PACKAGE on ``create package``)
        # make ``observations_changed`` true.  Try the certified-boundary local
        # ``nu* ; sync`` proof *before* invalidating/re-solving the retained
        # search.  On success the new prefix is proved optimal at the previous
        # cost and no historical JODAP context is needed.
        if len(new_events) == 1 and previous_goal is not None and new_objects:
            eid_fresh = next(iter(new_events))
            if self._try_fresh_object_zero_cost_sync_extension(
                    component, state, previous_goal, eid_fresh):
                state.current_event_ids = frozenset(component.execution.event_ids)
                state.current_objects = frozenset(component.objects)
                state.current_attribute_observations = attr_count
                state.current_object_attribute_snapshot = attr_snapshot
                for node in state.nodes.values():
                    node.h = self._heuristic(component, node.consumed)
                return

        # The previous incumbent is complete only for the previous observed
        # prefix. Do not let it prune extensions for the newly enlarged prefix.
        if new_events or observations_changed:
            state.upper_bound = float("inf")
            state.incumbent_moves = ()
            state.incumbent_assignment = None
            state.incumbent_offline = False
        if observations_changed or new_events:
            # New observations may split object-symmetry classes or make a
            # previously equivalent operational state distinguishable.
            state.state_dominance.clear()

        # Expanding the known object domain or changing observed object data can
        # change feasibility/cost of already discovered symbolic paths. Re-solve
        # those JODAPs conservatively while retaining the path graph itself.
        if observations_changed:
            self._rebuild_open_for_changed_observations(
                component, state, None if new_objects else changed_attr_names
            )

        # Fast path for the common conforming case.  It is sound only when the
        # new unit did not enlarge/change the previously observed object/data
        # domain; in that case the old optimum remains a lower bound for the new
        # prefix.  A zero-cost synchronous extension proves optimality directly.
        focused_seed = None
        # A non-proven local repair can already be a complete feasible alignment
        # for the enlarged current prefix.  Preserve that witness across the
        # bookkeeping reset later in this method; otherwise the exact fallback
        # would discard the strongest known upper bound and revert to the much
        # weaker analytic all-log seed.
        increment_incumbent_seed = None
        if len(new_events) == 1 and not observations_changed:
            eid = next(iter(new_events))
            fast_ok = self._try_zero_cost_sync_extension(component, state, previous_goal, eid)
            object_rel_handled = False
            if (not fast_ok) and previous_goal is not None:
                object_rel_handled, object_rel_proven = \
                    self._try_object_relation_deviation_sync_extension(
                        component, state, previous_goal, eid
                    )
                fast_ok = object_rel_proven
            if (not fast_ok) and previous_goal is not None and not object_rel_handled:
                fast_ok = self._try_guard_data_deviation_sync_extension(
                    component, state, previous_goal, eid
                )
            extra_event_handled = False
            if (not fast_ok) and previous_goal is not None and not object_rel_handled:
                extra_event_handled, extra_event_proven = self._try_extra_event_log_fast_path(
                    component, state, previous_goal, eid
                )
                fast_ok = extra_event_proven
            if (not fast_ok) and previous_goal is not None and not object_rel_handled:
                # If the duplicate-event lower-bound proof was unavailable, an
                # exact fixed synchronous extension may still find a zero-cost
                # invisible restoration.  Keep that correctness fallback.
                fast_ok = self._try_zero_cost_prepared_sync_extension(
                    component, state, previous_goal, eid
                )
            if (not fast_ok) and previous_goal is not None \
                    and not extra_event_handled and not object_rel_handled:
                # Missing-event deviations benefit from a one-model-step repair.
                # A recognized duplicate event does not: trying to manufacture
                # its already-consumed input token merely opens a large,
                # unnecessary repair search.
                fast_ok = self._try_one_step_missing_input_repair(
                    component, state, previous_goal, eid
                )
            if (not fast_ok) and previous_goal is not None:
                # Multi-deviation case: combine the already-defined repair
                # semantics in a bounded cost-directed neighbourhood before
                # reopening unrestricted A*.  This does not use the benchmark's
                # injected mutation count; it follows the actual conformance
                # objective and retains any non-proven local solution as a UB.
                fast_ok = self._try_cost_guided_local_repair(
                    component, state, previous_goal, eid
                )
                if not fast_ok:
                    # Prefer the explicit witness handed off by the local search.
                    # This remains valid even when canonical node reuse makes the
                    # generic predecessor-derived ``incumbent_moves`` unsuitable
                    # for current-prefix validation.
                    pending = state.pending_increment_incumbent_assignment
                    pending_cost = float(state.pending_increment_incumbent_cost)
                    if pending is not None \
                            and state.pending_increment_incumbent_event_ids == frozenset(component.execution.event_ids) \
                            and pending_cost < float("inf"):
                        increment_incumbent_seed = (
                            tuple(state.pending_increment_incumbent_moves), pending, pending_cost
                        )
                        self.jodap.stats["local_repair_incumbents_preserved"] += 1
                        self.jodap._diag(
                            "local_repair_incumbent_preserved",
                            component=component.component_id, event_id=eid,
                            upper_bound=pending_cost, source="explicit_local_handoff",
                        )
                    else:
                        # Backward-compatible fallback for any non-local fast path
                        # that installed a complete current-prefix incumbent.
                        local_ub = self.jodap._current_prefix_upper_bound(component, state)
                        if local_ub < float("inf") and state.incumbent_assignment is not None:
                            increment_incumbent_seed = (
                                tuple(state.incumbent_moves), state.incumbent_assignment, float(local_ub)
                            )
                            self.jodap.stats["local_repair_incumbents_preserved"] += 1
                            self.jodap._diag(
                                "local_repair_incumbent_preserved",
                                component=component.component_id, event_id=eid,
                                upper_bound=float(local_ub), source="generic_current_prefix",
                            )
            if fast_ok:
                state.current_event_ids = frozenset(component.execution.event_ids)
                state.current_objects = frozenset(component.objects)
                state.current_attribute_observations = attr_count
                state.current_object_attribute_snapshot = attr_snapshot
                for node in state.nodes.values():
                    node.h = self._heuristic(component, node.consumed)
                return

            # If the cheap extension mechanisms cannot certify the new prefix,
            # try a provenance-guided suffix search before reopening unrestricted
            # A*. This never changes semantics: a focused result is final only
            # when it reaches the independently proven global lower bound.
            if previous_goal is not None and self.provenance_slicing == "focus":
                ev = component.observation_formula.events.get(eid)
                if ev is not None:
                    focused_seed = self._try_provenance_focused_repair(
                        component, state, previous_goal, ev
                    )
                    if focused_seed is not None and focused_seed[2]:
                        moves, assignment, _proven, _info = focused_seed
                        warm = self._materialize_warm_path(component, state, moves, assignment)
                        if warm is not None:
                            state.current_goal = warm.node_id
                            state.current_assignment = assignment
                        state.upper_bound = assignment.total_cost
                        state.incumbent_moves = tuple(moves)
                        state.incumbent_assignment = assignment
                        state.incumbent_offline = False
                        state.current_event_ids = frozenset(component.execution.event_ids)
                        state.current_objects = frozenset(component.objects)
                        state.current_attribute_observations = attr_count
                        state.current_object_attribute_snapshot = attr_snapshot
                        for node in state.nodes.values():
                            node.h = self._heuristic(component, node.consumed)
                        return

        # A newly observed event creates log/sync successors from every already
        # discovered node for which its causal predecessors have been consumed.
        for eid in sorted(new_events):
            for nid in list(sorted(state.nodes)):
                node = state.nodes.get(nid)
                if node is None:
                    continue
                self._expand(component, state, node, include_model=False, only_event=eid)

        # If the model-depth bound grew, nodes that previously could not be
        # extended by a model move may now gain such successors. Duplicate
        # signatures prevent already generated paths from being inserted twice.
        if bound_grew:
            for node in list(state.nodes.values()):
                if node.model_depth < state.model_bound:
                    self._expand(component, state, node, include_model=True, only_event=None)

        state.current_event_ids = frozenset(component.execution.event_ids)
        state.current_objects = frozenset(component.objects)
        state.current_attribute_observations = attr_count
        state.current_object_attribute_snapshot = attr_snapshot

        # A changed prefix invalidates the previous incumbent as a complete goal.
        # Recompute all heuristic values and install a fresh feasible upper bound.
        state.upper_bound = float("inf")
        state.incumbent_moves = ()
        state.incumbent_assignment = None
        state.incumbent_offline = False
        for node in state.nodes.values():
            node.h = self._heuristic(component, node.consumed)
        # Rebuild heap priorities for currently open nodes.
        old_open = list(state.open_ids)
        state.open_heap.clear()
        state.open_ids.clear()
        for nid in old_open:
            if nid in state.nodes:
                state.push(state.nodes[nid])
        self._seed_incumbent(component, state)
        # Restore a complete feasible witness found by the compositional local
        # search if it beats the generic all-log seed.  This is only an upper
        # bound: unrestricted A* remains responsible for proving whether a
        # cheaper alignment exists.  Keeping it here is what lets the exact
        # fallback cap minimization and prune dominated siblings immediately.
        if increment_incumbent_seed is not None:
            moves, assignment, ub = increment_incumbent_seed
            if state.incumbent_assignment is None or ub < state.upper_bound - 1e-9:
                state.upper_bound = float(ub)
                state.incumbent_moves = tuple(moves)
                state.incumbent_assignment = assignment
                state.incumbent_offline = False
                self.jodap.stats["local_repair_incumbents_restored"] += 1
                self.jodap._diag(
                    "local_repair_incumbent_restored",
                    component=component.component_id, upper_bound=float(ub),
                )
        # A non-proving provenance-focused result is still a feasible global
        # incumbent because every candidate was checked by exact JODAP on the
        # full selected path. Reuse it only as an upper bound; unrestricted A*
        # remains responsible for proving optimality.
        if focused_seed is not None and not focused_seed[2]:
            moves, assignment, _proven, _info = focused_seed
            if state.incumbent_assignment is None or assignment.total_cost < state.upper_bound - 1e-9:
                state.upper_bound = assignment.total_cost
                state.incumbent_moves = tuple(moves)
                state.incumbent_assignment = assignment
                state.incumbent_offline = False

    def _make_result(self, component: ComponentState, state: SearchState,
                     node: Optional[SearchNode], assignment: JointAssignment, mode: str,
                     moves_override: Optional[Sequence[SymbolicMove]] = None) -> AlignmentResult:
        moves = list(moves_override) if moves_override is not None else self._path(state, node.node_id)
        bindings_by_step = {b["step"]: b for b in assignment.object_bindings}
        data_by_step = {d["step"]: d for d in assignment.data_assignments}
        model_i = 0
        out_moves: List[AlignmentMove] = []
        model_run: List[Dict[str, Any]] = []
        for m in moves:
            if m.kind == "log":
                e = component.observation_formula.events[m.event_id]
                lc = len(e.objects)
                out_moves.append(AlignmentMove(
                    kind="log", cost=lc, activity=e.activity,
                    event_id=e.event_id, transition=None, objects=tuple(sorted(e.objects)),
                    observed_objects=tuple(sorted(e.objects)), control_flow_cost=lc
                ))
                continue
            bind = bindings_by_step.get(model_i, {})
            objs = tuple(bind.get("objects", ()))
            if m.kind == "sync":
                e = component.observation_formula.events[m.event_id]
                # Per-move data cost is reconstructed from the optimized values.
                vals = data_by_step.get(model_i, {}).get("values", {})
                obs = dict(e.attributes)
                dc = sum(1 for k, v in obs.items() if k in vals and vals[k] != v)
                chosen = tuple(bind.get("model_objects", ())) or objs or tuple(sorted(e.objects))
                observed_for_move = tuple(bind.get("observed_objects", ())) or tuple(sorted(e.objects))
                object_delta = int(bind.get(
                    "object_cost",
                    len(set(chosen).symmetric_difference(set(observed_for_move)))
                ))
                out_moves.append(AlignmentMove(
                    kind="sync", cost=dc + object_delta, activity=e.activity, event_id=e.event_id,
                    transition=m.transition_label, objects=chosen,
                    observed_objects=observed_for_move, model_objects=tuple(chosen),
                    object_match=(set(chosen) == set(observed_for_move)),
                    observed_data=dict(obs), model_data=dict(vals),
                    data_mismatches=tuple(sorted(k for k, v in obs.items() if k in vals and vals[k] != v)),
                    data_cost=dc, object_cost=object_delta
                ))
            else:
                t = next(t for t in self.transitions if t["id"] == m.transition_id)
                mc = 0 if t.get("invisible", False) else len(objs)
                out_moves.append(AlignmentMove(
                    kind="model", cost=mc, activity=t.get("label"), event_id=None,
                    transition=t.get("label"), objects=objs, model_objects=tuple(objs),
                    silent=bool(t.get("invisible", False)), control_flow_cost=mc
                ))
            model_run.append({
                "step": model_i,
                "transition": m.transition_label,
                "objects": list(objs),
                "model_state_data": data_by_step.get(model_i, {}).get("values", {}),
            })
            model_i += 1

        cf_cost = sum(m.control_flow_cost for m in out_moves)
        event_data_cost = sum(m.data_cost for m in out_moves)
        object_attribute_cost = sum(int(a.get("cost", 0)) for a in assignment.object_attribute_assignments)
        data_cost = event_data_cost + object_attribute_cost
        object_cost = sum(m.object_cost for m in out_moves)
        pos = max((u.position for u in component.units), default=0)
        return AlignmentResult(
            component_id=component.component_id,
            prefix_position=pos,
            mode=mode,
            cost=int(assignment.total_cost),
            feasible=True,
            moves=out_moves,
            model_run=model_run,
            raw="symbolic incremental A* + JODAP",
            encode_seconds=assignment.encode_seconds,
            solve_seconds=assignment.solve_seconds,
            cost_breakdown={
                "total": int(assignment.total_cost),
                "control_flow": int(cf_cost),
                "data": int(data_cost),
                "object": int(object_cost),
            },
            joint_assignment={
                "object_bindings": assignment.object_bindings,
                "data_assignments": assignment.data_assignments,
                "object_attribute_assignments": assignment.object_attribute_assignments,
                "token_data_state": assignment.token_data_signature,
                "data_provenance": assignment.data_provenance_signature,
            },
        )

    def solve(self, component: ComponentState, *, offline: bool = False) -> AlignmentResult:
        if not offline:
            self._register_active_component(component)
        if offline:
            # Offline completion uses a fresh search. Online branch-and-bound and
            # dominance pruning are valid for the current prefix objective but
            # may discard paths that later provide a cheaper route to the final
            # marking. Reusing that pruned frontier would therefore be unsound.
            state = SearchState(component.component_id)
            state.model_bound = self._bound(component)
            state.search_offline = True
            root = SearchNode(
                node_id=state.new_id(), consumed=frozenset(), event_order=(),
                model_depth=0, g=0.0, h=self._heuristic(component, frozenset()),
                model_signature=(), move_signature=()
            )
            state.signatures[self._signature(root, canonical=False)] = root.node_id
            state.push(root)
            state.current_event_ids = frozenset(component.execution.event_ids)
            state.current_objects = frozenset(component.objects)
            state.current_attribute_observations = sum(
                len(v) for v in component.observation_formula.attribute_history.values())
            self._seed_offline_incumbent(component, state)
            if component.execution.event_ids:
                self._expand(component, state, root)
        else:
            # Component merges create a new component id, hence naturally create
            # a fresh joint online search from the conservative safe checkpoint.
            state = self.searches.get(component.component_id)
            if state is None:
                state = self._initialize(component)
            else:
                self._sync_increment(component, state)
            state.search_offline = False
            merge_proved = (state.incumbent_assignment is not None
                            and state.proven_prefix_lower_bound > 0
                            and state.upper_bound <= state.proven_prefix_lower_bound + 1e-9)
            if (not state.root_expanded) and component.execution.event_ids and not merge_proved:
                root = state.nodes.get(0)
                macro_hit = False
                # On the first observed event, try to collapse mandatory silent
                # object creation and the synchronous firing into one exact
                # macro query before exposing intermediate creation states.
                if root is not None and len(component.execution.event_ids) == 1:
                    eid0 = next(iter(component.execution.event_ids))
                    prepared_hit = self._try_zero_cost_prepared_sync_extension(
                        component, state, root, eid0
                    )
                    macro_hit = prepared_hit
                    # Do not let a positive-cost prepared/log optimum suppress
                    # construction of an equally optimal extendable model-side
                    # witness.  The previous code only called the latent repair
                    # when the prepared macro failed, which is exactly why an
                    # orphan create-package often retained only its cost-1 log
                    # explanation.  Keep that reported optimum, but also build
                    # a co-optimal relation-repair boundary for continuation.
                    should_try_latent = (not prepared_hit) or (
                        state.incumbent_assignment is not None and
                        not state.incumbent_offline and
                        float(state.upper_bound) > 1e-9 and
                        math.isfinite(float(state.upper_bound)))
                    if should_try_latent:
                        if prepared_hit:
                            self.jodap.stats["first_event_latent_after_macro_attempts"] += 1
                        latent_proved = self._try_first_event_latent_object_relation_repair(
                            component, state, root, eid0
                        )
                        if prepared_hit and (latent_proved or
                                bool(state.cooptimal_continuation_boundaries)):
                            self.jodap.stats["first_event_latent_after_macro_hits"] += 1
                        # A proven latent witness can terminate the first prefix;
                        # an unproven co-optimal one is retained for continuation
                        # while the ordinary optimum remains valid.
                        macro_hit = prepared_hit or latent_proved
                if root is not None and not macro_hit:
                    self._expand(component, state, root)
                state.root_expanded = True

        all_events = frozenset(component.execution.event_ids)

        # Zero is the global lower bound of the non-negative alignment cost. If
        # the fast path/macro has already produced a feasible current-prefix
        # incumbent with cost zero, optimality is proved immediately; do not
        # generate or certify competing paths.
        if not offline and state.incumbent_assignment is not None \
                and not state.incumbent_offline and state.upper_bound <= 1e-9 \
                and state.proven_prefix_lower_bound <= 1e-9:
            incumbent_events = {
                m.event_id for m in state.incumbent_moves
                if m.kind in ("log", "sync") and m.event_id is not None
            }
            if incumbent_events == set(component.execution.event_ids):
                self.jodap.stats["zero_cost_early_terminations"] += 1
                return self._make_result(
                    component, state, None, state.incumbent_assignment, "prefix",
                    moves_override=state.incumbent_moves
                )

        # Merge-specific exact termination: if hard synchronization structure
        # proves an unavoidable positive cost and the composed parent incumbent
        # reaches that lower bound, no joint-history replay can improve it.
        if not offline and state.incumbent_assignment is not None \
                and state.proven_prefix_lower_bound > 0 \
                and state.upper_bound <= state.proven_prefix_lower_bound + 1e-9:
            incumbent_events = {
                m.event_id for m in state.incumbent_moves
                if m.kind in ("log", "sync") and m.event_id is not None
            }
            if incumbent_events == set(component.execution.event_ids):
                self.jodap.stats["positive_lower_bound_terminations"] += 1
                if getattr(component, "merged_from", ()):
                    self.jodap.stats["merge_lower_bound_terminations"] += 1
                return self._make_result(
                    component, state, None, state.incumbent_assignment, "prefix",
                    moves_override=state.incumbent_moves
                )

        # An online goal is returned before it is expanded. For a later offline
        # request the same node must be reconsidered: it may already be final, or
        # it may need trailing model moves to reach the final marking.
        if offline and state.current_goal is not None and state.current_goal in state.nodes:
            goal = state.nodes[state.current_goal]
            if goal.node_id not in state.open_ids:
                state.push(goal)

        while True:
            node = state.pop()
            if node is None:
                # If OPEN was exhausted by incumbent pruning, the stored feasible
                # prefix is optimal under the admissible lower bound.
                if state.incumbent_assignment is not None \
                        and state.incumbent_offline == offline:
                    return self._make_result(
                        component, state, None, state.incumbent_assignment,
                        "offline" if offline else "prefix",
                        moves_override=state.incumbent_moves
                    )
                pos = max((u.position for u in component.units), default=0)
                return AlignmentResult(component.component_id, pos,
                                       "offline" if offline else "prefix", None, False,
                                       raw="search exhausted within model-depth bound")

            # Branch-and-bound termination. The popped node has minimum f in
            # OPEN; if it cannot beat the incumbent, no remaining node can.
            if state.incumbent_assignment is not None \
                    and state.incumbent_offline == offline \
                    and node.f >= state.upper_bound - 1e-9:
                self.jodap.stats["upper_bound_pruned"] += 1
                return self._make_result(
                    component, state, None, state.incumbent_assignment,
                    "offline" if offline else "prefix",
                    moves_override=state.incumbent_moves
                )

            if node.consumed == all_events:
                assignment = self.jodap.solve(component, state, node, require_final=offline)
                if assignment is not None:
                    state.assignments_by_node[node.node_id] = assignment
                    state.current_goal = node.node_id
                    state.current_assignment = assignment
                    if assignment.total_cost < state.upper_bound + 1e-9:
                        state.upper_bound = assignment.total_cost
                        state.incumbent_moves = tuple(self._path(state, node.node_id))
                        state.incumbent_assignment = assignment
                        state.incumbent_offline = offline
                    if not offline:
                        # For a prefix objective, the first popped goal is optimal
                        # under the admissible heuristic.
                        return self._make_result(component, state, node, assignment, "prefix")
                    # For offline completion, finality can add cost not reflected
                    # in the node's prefix g-value. Keep searching until the
                    # branch-and-bound condition proves the incumbent optimal.
                # Whether already final or not, trailing/alternative model moves
                # may yield a cheaper offline completion.

            state.closed_ids.add(node.node_id)
            self._expand(component, state, node)

            # _expand evaluates successors eagerly.  A complete successor may
            # therefore prove a zero-cost optimum before it is ever popped from
            # OPEN.  Return immediately instead of finishing sibling generation
            # or waiting for another A* iteration.
            if not offline and state.incumbent_assignment is not None \
                    and not state.incumbent_offline and state.upper_bound <= 1e-9 \
                    and state.proven_prefix_lower_bound <= 1e-9:
                incumbent_events = {
                    m.event_id for m in state.incumbent_moves
                    if m.kind in ("log", "sync") and m.event_id is not None
                }
                if incumbent_events == set(component.execution.event_ids):
                    self.jodap.stats["zero_cost_early_terminations"] += 1
                    return self._make_result(
                        component, state, None, state.incumbent_assignment, "prefix",
                        moves_override=state.incumbent_moves
                    )

