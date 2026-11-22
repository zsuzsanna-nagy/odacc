# Recorded results

The results reported in the paper, one folder per method:

| Folder | Method | Configuration |
|---|---|---|
| `smt/` | SMT reference | re-solves every prefix; 120 s per update, timed-out components given up |
| `exact/` | proposed method, exact mode | variant `pure-jointlb-lp`; 120 s per update, case stopped at the first update over it |
| `anytime/` | proposed method, anytime mode | variant `anytime-smt-jointlb-lp`; deadline 100 s, SMT check after 10 s |

Each folder contains `all_runs_summary.csv` (one row per case: status,
completed updates, times) and `all_runs_prefix_timings.csv` (one row per
update: cost, proven lower bound, certification flag and route, wall time).
For `anytime/`, the 17 largest cases were rerun without the per-update limit
after a deadline-handling fix; the folder contains the merged results.

All methods ran on the same machine, one case per process. `tables/` holds the
tables of the paper, generated with

```
python tools/make_results_tables.py --smt results/smt --exact results/exact --anytime results/anytime --out results/tables
```

Case statuses: `ONLINE_OK` (all updates solved), `ONLINE_PARTIAL` (SMT: some
components given up after a timeout, the others completed),
`OBSERVATION_TIMEOUT` (stopped at an update over the limit).
