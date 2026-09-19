# Order Management benchmark generator

This directory contains the scripts used to construct the ODACC Order Management benchmark from the published **OCEL 2.0 Order Management** simulation artifacts.

The original Order Management files are **not redistributed** in this repository. They are external research artifacts and should be downloaded from their original sources before running the generator.

## External sources

- OCEL 2.0 Order Management dataset and description:  
  https://www.ocel-standard.org/event-logs/simulations/order-management/
- Published process/simulation model archive:  
  https://www.ocel-standard.org/event-logs/simulations/order-management/data/order-management-model.zip
- Dataset record cited in the accompanying paper:  
  https://zenodo.org/records/18373906

The benchmark construction uses the published OCEL and process model as source material; the original event log is **not evaluated directly**. The scripts derive ODACC-compatible benchmark executions and a curated DOPID model from these artifacts.

## Required external input files

After downloading/extracting the original artifacts, provide:

- `order-management.xml` -- the OCEL 2.0 Order Management event log;
- `order-management.cpn` -- the corresponding CPN simulation/process model.

The input paths are supplied on the command line, so the files do not need to be copied into this directory.

## Included files

- `build_order_management_benchmark.py` -- runs the complete conversion pipeline;
- `cpn_to_dopid.py` -- derives the curated DOPID benchmark model from the CPN model;
- `ocel2_to_odacc.py` -- converts the OCEL 2.0 source log into ODACC-compatible object-centric components and a stream representation;
- `generate_controlled_mutations.py` -- generates the systematic single-deviation benchmark families;
- `generate_multideviation_mutations.py` -- generates benchmark instances containing multiple deviations;
- `validate_order_management_conversion.py` -- validates the generated model/log conversion;
- `order_management_profile.json` -- defines the object types, activity mappings, attribute mappings, ignored simulation transitions, and data-aware guards used by the conversion.

## Benchmark perspective

The CPN model is intentionally **not translated literally**. Simulator bookkeeping, stochastic routing, resource initialization, and logging transitions are not treated as business conformance constraints.

The curated DOPID retains the observable lifecycle activities:

- `place order`
- `confirm order`
- `pick item`
- `item out of stock`
- `reorder item`
- `pay order`
- `payment reminder`
- `create package`
- `send package`
- `package delivered`
- `failed delivery`

The conformance perspective is restricted to the following core object types:

- `orders -> ORDER`
- `items -> ITEM`
- `packages -> PACKAGE`

Products, customers, and employees are treated as contextual/shared objects. Including them in the component key would connect otherwise unrelated orders merely because they share a product or resource. ORDER/ITEM/PACKAGE relations still preserve genuine object-centric component merging, since a package may contain items associated with multiple orders.

## Data-aware guards

Two data guards are derived from the published simulation/log semantics:

```text
pay order:
    price(o) == sum(price(I)) + 5.0

create package:
    weight(p) == sum(weight(I))
```

These guards provide controlled targets for data-aware conformance deviations in the generated benchmarks.

## Generate the base benchmark

Example on Windows/PowerShell:

```powershell
py .\build_order_management_benchmark.py `
  --cpn ..\order-management.cpn `
  --ocel ..\order-management.xml `
  --out .\generated_order_management `
  --min-events 3 `
  --max-events 120
```

The generated directory contains, among other files:

- `order_management_dopid.pnml` -- curated DOPID model;
- `cpn_mapping_report.json` -- extracted CPN transition/guard inventory;
- `components/*.jsonocel` -- ODACC-compatible component traces;
- `components/manifest.csv` -- component/event/object statistics;
- `order_management.stream.jsonl` -- OCEL 2.0 stream representation preserving qualified E2O/O2O relations and timestamped object-attribute updates.

The component JSON files are compatible with the ODACC JSON-OCEL loader used in the experiments. The JSONL stream retains information needed for future/native OCEL 2.0 streaming support.

## Controlled-deviation benchmarks

After constructing the base benchmark, the mutation generators can be used to create the benchmark families used to evaluate control-flow, data, and object-related deviations. See the command-line help of each script for the available options:

```bash
python generate_controlled_mutations.py --help
python generate_multideviation_mutations.py --help
```

## Why connected components are used

The ORDER/ITEM/PACKAGE object graph contains connected components of different sizes. Package creation can connect objects originating from previously independent order-related components and therefore provides genuine object-centric merge behavior.

Processing these components separately provides realistic object-centric executions while avoiding artificial global connectivity through shared contextual objects such as products or employees.

## Reproducibility note

The generated benchmark files used in the paper may be distributed separately under the repository's benchmark/example directories. This directory contains the **generation pipeline** needed to reconstruct them from the published external Order Management artifacts.
