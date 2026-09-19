# Synthetic scalability benchmarks

This directory contains the synthetic benchmark instances used to study how ODACC and the SMT reference behave as selected workload dimensions increase. The suites are designed to vary one main characteristic at a time while keeping the remaining structure as stable as possible.

The instances can be regenerated with `benchmarks/scalability/generate_scalability_benchmarks.py`. The generator accepts the object-attribute model in `examples/object_attributes/net_object_attrs.pnml` as its base model; see the generator README for the exact command and generation procedure.

## Models

- `models/net_guard_simple.pnml` — same basic control/object flow with simple guards.
- `models/net_guard_medium.pnml` — medium-complexity object-attribute guards.
- `models/net_guard_complex.pnml` — additional conjunctions and aggregate object-attribute conditions.

For the experiments, the simple model is used for the scaling suites unless guard complexity itself is the factor under study; `guard_complexity` is evaluated against all three model variants.

## Benchmark families

- `length_scale/` — one ORDER associated with 1, 2, 5, 10, 20, 50, or 100 PRODUCT objects, increasing the trace and binding size.
- `component_scale/` — increasing numbers of independent interleaved object components.
- `merge_scale/` — routine component-merging configurations: `(2 components, 1 product each)`, `(2,2)`, and `(3,1)`.
- `merge_scale_extended/` — more demanding merge configurations, up to 5 components and 5 products per component.
- `deviation_scale/` — fitting and controlled data/object-attribute deviations at 1, 2, 5, and 10 products.
- `deviation_scale_extended/` — the corresponding 20-product workloads, intended to stress repair and exact fallback behavior.
- `guard_complexity/` — common workloads evaluated with the simple, medium, and complex guard models to isolate formula complexity.

## Example

From the repository root:

```bash
python batch_folder.py \
  --mode online \
  --backend symbolic \
  --cocomot-root ../cocomot-main \
  --folder examples/scalability/deviation_scale \
  --model examples/scalability/models/net_guard_simple.pnml \
  --out results/deviation_symbolic \
  --progress-updates
```

For guard-complexity experiments, run the same workloads against each model in `models/`.

## Regeneration

See:

```text
benchmarks/scalability/README.md
benchmarks/scalability/generate_scalability_benchmarks.py
```

The copies in this directory are the frozen benchmark instances used for the experiments reported with this repository.
