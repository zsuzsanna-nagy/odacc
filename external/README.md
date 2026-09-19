# External dependency: CoCoMoT

This directory contains the modified CoCoMoT source files used by ODACC.

Upstream project:
https://github.com/bytekid/cocomot

CoCoMoT provides the DOPID/OCEL encoding and SMT infrastructure used by the implementation.

## Modified files

The following CoCoMoT files were modified:

- `src/objectcentric/encoding.py`
- `src/smt/z3solver.py`

### `encoding.py`

The object-centric guard encoding was extended with an exact symbolic encoding for supported LIST-valued aggregate guards. This avoids explicit powerset enumeration for these guards while retaining the original CoCoMoT encoding as a fallback for unsupported expressions.

### `z3solver.py`

The Z3 backend was extended to evaluate parsed aggregate expressions, including `sum(...)`, which is required by the aggregate guards used in the benchmark models.

## Usage

To reproduce the implementation, clone the upstream CoCoMoT repository and replace the corresponding files with the modified versions provided in this directory.

Example directory mapping:

```text
external/cocomot/encoding.py
    -> cocomot/src/objectcentric/encoding.py

external/cocomot/z3solver.py
    -> cocomot/src/smt/z3solver.py