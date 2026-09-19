# Synthetic scalability benchmark generator

This directory contains the generator for the controlled synthetic scalability benchmarks used by ODACC.

Unlike the Order Management benchmarks, these workloads are generated entirely from a supplied ODACC/DOPID PNML model. The generator varies one principal characteristic at a time so that the effect of trace/object size, component count, component merging, deviations, and guard complexity can be studied independently.

## Included file

- `generate_scalability_benchmarks.py` -- generates all scalability benchmark families and the simple/medium/complex guard-model variants.

## Required input

The script requires a **base PNML model**. In the ODACC repository, the object-attribute benchmark model can be used as the base model, for example:

```text
examples/object_attributes/net_object_attrs.pnml
```

The input model is supplied explicitly with `--model`; it is therefore not duplicated in this directory.

## Usage

From the repository root:

```bash
python benchmarks/scalability/generate_scalability_benchmarks.py \
    --model examples/object_attributes/net_object_attrs.pnml \
    --out examples/scalability
```

On Windows/PowerShell, the same command can be written as:

```powershell
py .\benchmarks\scalability\generate_scalability_benchmarks.py `
  --model .\examples\object_attributes\net_object_attrs.pnml `
  --out .\examples\scalability
```

Use `--help` for the current command-line options:

```bash
python benchmarks/scalability/generate_scalability_benchmarks.py --help
```

## Generated model variants

The generator derives three PNML variants from the supplied base model:

- `models/net_guard_simple.pnml` -- reduced object-attribute guard complexity;
- `models/net_guard_medium.pnml` -- the base guard structure;
- `models/net_guard_complex.pnml` -- additional conjunctions and aggregate object-attribute conditions.

These variants are used only for the guard-complexity experiments. The other scalability suites are intended to be run with the selected common benchmark model so that the varied factor remains isolated.

## Generated benchmark families

### `length_scale`

A single connected component containing one ORDER and an increasing number of PRODUCT objects. The generated sizes use 1, 2, 5, 10, 20, 50, and 100 products.

This family primarily varies prefix length and object cardinality within one component.

### `component_scale`

An increasing number of independent order/product components whose events are interleaved in the observation stream.

Generated component counts: 1, 2, 5, 10, and 20.

### `merge_scale`

Routine component-merging benchmarks with deliberately cross-component observations. The default cases are kept small enough for regular experiment runs:

- 2 components / 1 product per component;
- 2 components / 2 products per component;
- 3 components / 1 product per component.

### `merge_scale_extended`

Harder merge configurations:

- 2 components / 5 products per component;
- 3 components / 2 products per component;
- 3 components / 5 products per component;
- 5 components / 1 product per component;
- 5 components / 2 products per component;
- 5 components / 5 products per component.

### `deviation_scale`

Controlled fitting and single-deviation instances at 1, 2, 5, and 10 products. The generated variants include:

- fitting behavior;
- early event-data deviation (`place_data`);
- late event-data deviation (`ship_data`);
- object-attribute deviations affecting `vip`, `priority`, `budget`, and `cost`.

### `deviation_scale_extended`

The same deviation types with 20 products. These instances are separated because they can trigger substantially more expensive repair/fallback behavior.

### `guard_complexity`

A fixed 20-product workload generated in fitting and budget-violating variants. The same traces are evaluated against the simple, medium, and complex PNML model variants to isolate guard-formula complexity.

## Example experiment commands

Routine merge benchmark:

```bash
python batch_folder.py \
    --mode online \
    --backend symbolic \
    --cocomot-root ../cocomot-main \
    --folder examples/scalability/merge_scale \
    --model examples/scalability/models/net_guard_simple.pnml \
    --out results/merge_symbolic \
    --progress-updates
```

Deviation benchmark:

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

## Reproducibility note

The generated benchmark files used in the paper can be committed under `examples/scalability/` so that the exact evaluated inputs are preserved. This generator is provided in addition to those frozen benchmark files to document how the controlled scalability suites were constructed.
