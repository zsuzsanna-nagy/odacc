# Order Management benchmarks

This directory contains the Order Management benchmark instances used for the larger object-centric experiments. They were constructed from the published OCEL 2.0 Order Management simulation and its corresponding process-model artifacts rather than by evaluating the original complete log directly.

The benchmark construction pipeline is provided in `benchmarks/order_management/`. That directory documents the external source files, conversion procedure, component extraction, and mutation generators used to obtain the frozen benchmark instances included here.

## External source

The benchmark construction is based on the published Order Management OCEL 2.0 dataset and simulation artifacts:

- Order Management OCEL 2.0 dataset and description:
  https://www.ocel-standard.org/event-logs/simulations/order-management/
- Published process/simulation model archive:
  https://www.ocel-standard.org/event-logs/simulations/order-management/data/order-management-model.zip

The original external input files are not reproduced in this directory. See `benchmarks/order_management/README.md` for acquisition and regeneration instructions.

## Model

- `order_management_dopid.pnml` — DOPID benchmark model derived from the published Order Management process model and augmented with the data guards and object-binding semantics used in the experiments.

## Benchmark families

### `components/`

Contains 34 extracted connected components from the generated Order Management benchmark stream. The file names encode the component identifier and its number of events and objects. These traces provide structurally realistic fitting examples of different sizes and object compositions.

### `controlled_mutations/`

Contains systematic single-deviation variants derived from the 34 source components:

- `fit/` — unchanged source components.
- `cf_extra_pick/` — duplicated `pick item` behavior.
- `cf_missing_pick/` — removed `pick item` behavior.
- `data_order_price/` — modified ORDER price data.
- `data_package_weight/` — modified PACKAGE weight data.
- `object_missing_item_relation/` — removed ITEM relation at package creation.
- `object_extra_item_relation/` — added unrelated ITEM relation at package creation where applicable.

`mutation_manifest.csv` and `mutation_manifest.json` record the generated instances and their injected mutation type. The injected mutation count is a benchmark-construction property, not a proof of the globally optimal alignment cost.

### `multi_deviation_benchmark/`

Contains combinations of multiple controlled deviations:

- `two_same_dimension/` — two deviations from the same deviation dimension.
- `two_mixed/` — two deviations from different dimensions.
- `three_mixed/` — three mixed deviations.
- `stress_4_5/` — four- and five-deviation stress configurations.

The corresponding mutation manifests describe the injected combinations. As with the single-deviation suite, correctness is determined by globally minimal total alignment cost; co-optimal alignments may have different control-flow/data/object cost decompositions.

## Regeneration

The scripts and documentation used to derive these benchmarks are located in:

```text
benchmarks/order_management/
```

The instances in this `examples/order_management/` directory are the frozen files used by the experiments and should therefore be used when reproducing the reported measurements. Regeneration is provided for transparency and extensibility, but regenerated files may differ if external source data or generator settings are changed.

## Example

From the repository root, a component can be evaluated with:

```bash
python odacc.py \
  --cocomot-root ../cocomot-main \
  --model examples/order_management/order_management_dopid.pnml \
  --log examples/order_management/components/component_001_events_0030_objects_0020.jsonocel \
  --mode online \
  --backend symbolic \
  --quiet-solver
```
