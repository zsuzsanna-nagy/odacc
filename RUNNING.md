# Running ODACC

## 1. Setup

- Python 3.10 or newer, with the packages in `requirements.txt` (including
  `z3-solver` and `scipy`).
- CoCoMoT next to this folder (`../cocomot-main`), with the modified files from
  `external/` (see `external/README.md`). Another location can be given with the
  environment variable `ODACC_COCOMOT_ROOT`.

## 2. Check the installation

```
python -m unittest tests.test_regressions -v
```

All tests should pass (about two minutes).

## 3. Experiments of the paper

Run from the repository root (on Windows, use `\` in paths):

```
python experiments/run_online_smt.py --observation-timeout 120 --online-timeout 0
python experiments/run_online_symbolic.py --variant pure-jointlb-lp --observation-timeout 120 --online-timeout 0
python experiments/run_online_symbolic.py --variant anytime-smt-jointlb-lp --deadline 100 --observation-timeout 120 --online-timeout 0 --on-observation-timeout continue
```

Each run creates a session folder under `experiments/experiment_runs/`, named
with a timestamp, the variant and the time limits. Cases that did not complete
can be rerun without starting over:

```
python experiments/rerun_failed.py --session experiments/experiment_runs/<session> --variant anytime-smt-jointlb-lp --deadline 100 --observation-timeout 120 --online-timeout 0 --on-observation-timeout continue
```

The `merged` subfolder of the rerun session holds the complete results.

## 4. Comparisons and tables

Per-prefix comparison of two sessions (completion, agreement of certified
costs, running times):

```
python tools/compare_sessions.py <session-a> <session-b> --label-a a --label-b b --out-dir report/a_vs_b
```

`prefix_disagreements.csv` lists every certified prefix whose cost differs; it
is empty for all pairs of the recorded results. The tables of the paper:

```
python tools/make_results_tables.py --smt <smt-session> --exact <exact-session> --anytime <anytime-session> --out report/tables
```

With the recorded results: `--smt results/smt --exact results/exact --anytime results/anytime`.

## 5. Single cases

```
python odacc.py --cocomot-root ../cocomot-main --model <model.pnml> --log <log.jsonocel> --mode online --backend symbolic --joint-lb on --lp-guided-candidates on --lp-heuristic on --output out.json
```

Add `--anytime-deadline S` for anytime mode and
`--symbolic-smt-fallback-after S` for the SMT check; `--backend smt` runs the
SMT reference. `python odacc.py --help` lists all options.

## 6. Option reference (runners)

| Option | Meaning |
|---|---|
| `--observation-timeout S` | limit per update (0 = none) |
| `--on-observation-timeout stop\|continue` | symbolic runner: stop the case, or record the update and continue |
| `--online-timeout S` | limit for the whole online run of a case (0 = none) |
| `--deadline S` | anytime deadline (anytime variants) |
| `--smt-after S` | start of the SMT check (variants that use it) |
| `--only a,b` | run only suites whose name contains `a` or `b` |
