# Experimental Results

This directory contains the result files used for the evaluation of ODACC.

The experiments are organized into three groups:

## `online_symbolic_outputs/`

Results of the proposed symbolic online method.

Important files:

- `all_runs_summary.csv`  
  One row per benchmark case with the final run status, total alignment cost,
  timing information, and other case-level measurements.

- `all_runs_prefix_timings.csv`  
  Per-observation results for the online symbolic method. This file also
  contains the certification route used for each successfully processed
  observation.

- `aggregate_summary.csv`  
  Case-level aggregate statistics produced by the experiment runner.

- `aggregate_summary.json`  
  JSON representation of the aggregate case-level results.

- `aggregate_prefix_timings.csv`  
  Aggregated per-observation timing information.

- `certification_routes_results_summary.csv`  
  Summary of the certification-route distribution used for the corresponding
  analysis in the paper/repository.

- `experiment_config.json`  
  Configuration recorded for the experiment run.

## `online_smt_outputs/`

Results of the exact SMT-based online reference implementation.

The files follow the same structure as the symbolic results where applicable:

- `all_runs_summary.csv`
- `all_runs_prefix_timings.csv`
- `aggregate_summary.csv`
- `aggregate_summary.json`
- `aggregate_prefix_timings.csv`
- `experiment_config.json`

## `validation_outputs/`

Results of the combined online/offline validation experiments on the
non-Order-Management benchmark suites.

These runs are primarily used to compare the online prefix-alignment results
with the corresponding exact offline computations.

## Relation to the submitted paper

The manuscript was submitted using an earlier complete experiment run.

The result files currently provided in this repository were generated
subsequently on a faster machine using the same benchmark definitions,
algorithms, timeout settings, and experimental protocol.

The newer runs leave the results of the proposed symbolic method essentially
unchanged, but the faster hardware allows the SMT reference to complete more
benchmark cases within the prescribed timeout. Consequently, some aggregate
completion counts and runtime statistics in this repository differ from those
reported in the submitted manuscript.

The submitted manuscript reports:

- symbolic completion: 373 / 409 cases (91.2%)
- SMT completion: 136 / 409 cases (33.3%)
- jointly completed cases with identical minimum cost: 107
- symbolic faster in: 103 / 107 such cases

The newer repository results report:

- symbolic completion: 373 / 409 cases (91.2%)
- SMT completion: 179 / 409 cases (43.8%)
- jointly completed cases with identical minimum cost: 134
- symbolic faster in: 130 / 134 such cases

The newer result files should therefore be regarded as the most recent
experimental measurements available in the repository, while the published or
submitted manuscript should be consulted for the exact values reported in that
version of the paper.

## Reproducing the experiments

The corresponding experiment runners are available in the `experiments/`
directory:

- `run_validation.py`
- `run_online_symbolic.py`
- `run_online_smt.py`

The experiment protocol and runner configuration are documented in
`experiments/README.md`.