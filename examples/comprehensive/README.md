# Comprehensive correctness benchmarks

This directory contains the compact benchmark suite used to exercise the main semantic features of ODACC. The suite includes conforming behavior, control-flow deviations, data deviations, object-relation deviations, incomplete prefixes, independent object components, and a later component merge.

The DOPID model in `net.pnml` is based on CoCoMoT's `otests/13` example and is included here so that the benchmark can be executed directly.

## Contents

- `net.pnml` — DOPID model used by all traces in this directory.
- `00_fit.jsonocel` — fully conforming baseline; expected total alignment cost 0.
- `01_data_place_order_guard_violation.jsonocel` — `d=2` violates the `place order` guard `d > 2`.
- `02_data_ship_guard_violation.jsonocel` — `d=3, m=1` violates the applicable `ship` guard.
- `03_data_ship_combined_guard_violation.jsonocel` — `d=7, m=0` violates the alternative `ship` branch.
- `04_object_missing_product_at_ship.jsonocel` — one product is missing from the `ship` event.
- `05_object_extra_product_at_ship.jsonocel` — an additional product participates only in the `ship` event.
- `06_incomplete_after_place_order.jsonocel` — incomplete prefix used to distinguish online prefix semantics from offline completion to a final marking.
- `07_extra_ship_event.jsonocel` — duplicate visible behavior, exercising a control-flow/log deviation.
- `08_wrong_order_ship_before_place_order.jsonocel` — ordering/control-flow deviation.
- `09_unmodeled_cancel_event.jsonocel` — unmodeled visible event requiring a log move.
- `10_object_attributes_and_unreferenced_object.jsonocel` — schema/stream handling for object attributes and an update-only object. The model does not constrain these object attributes, so this is not an object-attribute guard benchmark.
- `11_two_independent_components.jsonocel` — two independent object components processed within the same stream.
- `12_component_merge_stress.jsonocel` — two initially independent components become connected by a later observation, exercising component merging and checkpoint handling.

## Example

From the repository root:

```bash
python odacc.py \
  --cocomot-root ../cocomot-main \
  --model examples/comprehensive/net.pnml \
  --log examples/comprehensive/00_fit.jsonocel \
  --mode both \
  --quiet-solver
```

## Interpretation

Multiple co-optimal alignments may exist. Correctness is therefore determined primarily by feasibility and globally minimal total cost rather than by requiring a unique move sequence or a unique decomposition of the total cost into control-flow, data, and object-related contributions.
