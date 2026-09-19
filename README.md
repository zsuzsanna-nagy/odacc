# ODACC

**ODACC** is a research implementation of exact **online data-aware object-centric conformance checking**. It incrementally computes certified optimal prefix alignments for object-centric event streams using component-aware processing, certified continuation and repair paths, symbolic A* search, and exact constraint reasoning over DOPID models.

This repository accompanies the paper **Exact Online Data-Aware Object-Centric Conformance Checking with Incremental Symbolic Alignments** and contains the implementation, benchmark instances, benchmark-generation tools, modified CoCoMoT files, experiment configurations, and result artifacts used in the evaluation.

## Repository structure

```text
odacc/
├── odacc.py                 # main command-line entry point
├── batch_folder.py          # run ODACC on all OCEL files in a folder
├── requirements.txt
├── src/odacc/               # ODACC implementation
├── examples/                # benchmark instances used in the evaluation
├── benchmarks/              # benchmark-generation tools
├── external/                # modified external source files (CoCoMoT)
├── experiments/             # experiment runners/configurations
└── results/                 # experimental results
```

The individual benchmark folders contain additional README files describing their purpose and provenance.

## Requirements

ODACC is implemented in Python and uses [CoCoMoT](https://github.com/bytekid/cocomot) for DOPID/OCEL constraint encoding and exact SMT reasoning.

The experiments use the Z3 backend. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

A local CoCoMoT checkout is also required. The ODACC command-line tools receive its location through `--cocomot-root`.

Example layout:

```text
workspace/
├── odacc/
└── cocomot-main/
```

### Modified CoCoMoT files

The experiments use two modified CoCoMoT source files. They are provided under `external/cocomot/` together with a README explaining the changes and how to copy them into the upstream CoCoMoT checkout.

The modifications add:

- exact symbolic handling of supported LIST-valued aggregate guards without explicit powerset enumeration, while retaining the original encoding as a fallback; and
- Z3 evaluation of parsed aggregate expressions such as `sum(...)`.

No other CoCoMoT source files are required to be replaced.

## Basic usage

Run ODACC on one DOPID model and one OCEL trace from the repository root:

```bash
python odacc.py \
  --cocomot-root ../cocomot-main \
  --model examples/comprehensive/net.pnml \
  --log examples/comprehensive/00_fit.jsonocel \
  --mode online \
  --backend symbolic \
  --quiet-solver
```

Important modes include:

- `online` -- compute an optimal prefix alignment as observations arrive;
- `offline` -- compute an alignment only after the complete input has been consumed;
- `both` -- execute both online and offline processing;
- `stream` -- inspect the generated observation stream without invoking the solver.

For direct online comparison with the SMT backend, use the corresponding backend option instead of `symbolic`.

Full command-line options are available with:

```bash
python odacc.py --help
```

## Batch execution

`batch_folder.py` runs all `*.jsonocel` files in a folder against a common model.

```bash
python batch_folder.py \
  --cocomot-root ../cocomot-main \
  --folder examples/comprehensive \
  --model examples/comprehensive/net.pnml \
  --backend symbolic \
  --mode online \
  --out results/comprehensive
```

If `--model` is omitted, `batch_folder.py` uses `net.pnml` from the selected folder when available.

## Benchmark suites

The frozen benchmark instances used in the evaluation are stored under `examples/`.

### Comprehensive

`examples/comprehensive/` contains compact correctness-oriented examples covering fitting behavior, control-flow deviations, data deviations, incomplete prefixes, independent object components, and component merging.

Some of the smaller examples are based on or adapted from the `otests/13` object-centric example distributed with CoCoMoT.

### Object attributes

`examples/object_attributes/` contains targeted tests for object attributes, aggregate guards, event/object data interactions, and related DOPID semantics.

### Scalability

`examples/scalability/` contains synthetic benchmark families varying factors such as trace length, component count, merge complexity, deviation density, and guard complexity.

The corresponding generator is provided under:

```text
benchmarks/scalability/
```

### Order Management-derived benchmarks

`examples/order_management/` contains the frozen benchmark instances derived from the published OCEL 2.0 Order Management example. The original complete event log is not evaluated directly; it is used as source material for constructing component-level and controlled-deviation benchmarks.

The benchmark-generation tools are provided under:

```text
benchmarks/order_management/
```

The external source material is available from:

- OCEL 2.0 Order Management dataset: https://www.ocel-standard.org/event-logs/simulations/order-management/
- published process/simulation model: https://www.ocel-standard.org/event-logs/simulations/order-management/data/order-management-model.zip

The generated DOPID benchmark model augments the published process model with the data guards and object-binding semantics required by the experiments.

## Benchmark generators

The `benchmarks/` directory contains the scripts used to construct the main generated benchmark families.

```text
benchmarks/
├── scalability/
└── order_management/
```

Each subdirectory contains its own README with the required inputs, generated outputs, and usage instructions.

The original third-party Order Management source files are not redistributed here; the generator README points to their public sources.

## Output

ODACC can write detailed machine-readable JSON output containing alignment costs, move information, timing measurements, and symbolic-search diagnostics.

Use:

```bash
--output path/to/result.json
```

For compact terminal output, combine this with `--quiet-solver`. The batch runner additionally creates summary files suitable for experiment aggregation.

## Reproducibility

The repository preserves the implementation, benchmark instances, benchmark-generation tools, experiment configurations, and result files used for the accompanying study. The frozen benchmark inputs are stored under `examples/`, while the corresponding experiment definitions and recorded outputs are stored under `experiments/` and `results/`, respectively.

The published result artifacts retain successful runs, timeouts, and solver failures. This allows the reported completion behavior, alignment costs, runtime measurements, and search diagnostics to be traced back to the recorded experiment outputs rather than only to aggregated tables in the paper.

The repository also contains the modified CoCoMoT files used in both the proposed implementation and the CoCoMoT-based SMT reference configuration, so that the solver setup used for the evaluation can be reconstructed consistently.

## External software

ODACC builds on:

- **CoCoMoT** -- SMT-based conformance checking for data Petri nets and OPID/DOPID models: https://github.com/bytekid/cocomot
- **Z3** -- SMT solver used in the reported ODACC experiments.

See `external/README.md` and `external/cocomot/README.md` for details about the modified upstream files used by ODACC.

## Citation

Citation information will be added after publication metadata becomes available.
