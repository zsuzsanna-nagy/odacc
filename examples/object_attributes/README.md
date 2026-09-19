# Object-attribute benchmarks

This directory contains focused benchmarks for DOPID guards over OCEL object attributes. The model extends the order-to-ship example used by CoCoMoT with explicit guards over ORDER and PRODUCT attributes and aggregate expressions over sets of related objects.

The model in `net_object_attrs.pnml` also serves as the base model for the synthetic scalability benchmark generator in `benchmarks/scalability/`.

## Object attributes used by the model

The benchmark model contains, among others, the following conditions:

- `vip(o)` on the ORDER object in `place order`:
  `d > 2 && vip(o) == 1`
- `budget(o)` on the ORDER object and `cost(P)` on PRODUCT objects in `pay bank transfer`:
  `sum(cost(P)) <= budget(o)`
- `priority(o)` on the ORDER object in `ship`:
  `(((d <= 5 && m == 0) || (d > 5 && m == 1)) && priority(o) >= 2)`

Object-attribute values are stored in the OCEL object valuation map and are resolved by the DOPID encoding when evaluating expressions such as `vip(o)`, `budget(o)`, and `cost(P)`.

## Contents

- `net_object_attrs.pnml` — DOPID model with object-attribute and aggregate guards.
- `00_fit_all_object_guards.jsonocel` — all event-data and object-data guards hold; expected total cost 0.
- `01_violate_order_vip.jsonocel` — `vip(o)=0`, preventing synchronous satisfaction of the `place order` guard.
- `02_violate_order_budget.jsonocel` — the PRODUCT cost sum exceeds the ORDER budget for bank-transfer payment.
- `03_violate_order_priority.jsonocel` — `priority(o)=1`, violating the `ship` object guard.
- `04_violate_product_cost_sum.jsonocel` — aggregate PRODUCT cost exceeds the available budget.
- `05_fit_credit_card_despite_high_product_cost.jsonocel` — high PRODUCT cost does not violate the credit-card branch; expected total cost 0.
- `06_violate_event_and_object_data.jsonocel` — combines a ship event-data violation with an ORDER priority violation.
- `07_violate_place_event_and_object_guard.jsonocel` — combines an event-data and ORDER-object violation at `place order`.

## Example

From the repository root:

```bash
python odacc.py \
  --cocomot-root ../cocomot-main \
  --model examples/object_attributes/net_object_attrs.pnml \
  --log examples/object_attributes/00_fit_all_object_guards.jsonocel \
  --mode both \
  --quiet-solver
```

## Scope

These examples focus on object attributes available when an object is observed. They do not exercise timestamped changes of an object's attributes during execution.

For systematically generated scaling workloads derived from this model, see `examples/scalability/` and the generator in `benchmarks/scalability/`.
