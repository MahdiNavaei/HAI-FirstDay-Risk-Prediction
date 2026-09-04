# Reproducibility

## Environment

Use Python 3.10 or newer and install the pinned dependencies from
`requirements.txt` in an isolated virtual environment.

## Workflow

1. Obtain the source CSV from the original Zenodo record.
2. Run the aggregate cohort check:

   ```bash
   python scripts/prepare_data.py --data /path/to/Hospital_infection_data.csv
   ```

3. Reproduce development model comparison and robustness preparation:

   ```bash
   python scripts/reproduce_development.py --data /path/to/Hospital_infection_data.csv
   ```

4. Reproduce the frozen held-out evaluation:

   ```bash
   python scripts/reproduce_heldout.py --data /path/to/Hospital_infection_data.csv
   ```

5. Optionally create aggregate tables and a figure from the local result:

   ```bash
   python scripts/reproduce_tables.py
   python scripts/reproduce_figures.py
   ```

The development workflow is the computationally heavier step. It writes
local model, table, figure, report, log, and development-prediction artifacts;
these paths are ignored by Git. The held-out script fits only on the
development pool, keeps held-out predictions in memory, and writes an
aggregate JSON result. It does not write encounter-level predictions.

## Frozen reference

`results/expected_results.json` contains the already reported aggregate values.
It is a reference for checking a reproduction and is never replaced by a
newly generated result. Small numerical differences can occur across supported
library/platform combinations; they should be investigated, not silently
used to revise the frozen manuscript values.

## Reproduction boundary

The released workflow is an encounter-level retrospective analysis. The
`1404-2` period is a pre-specified held-out-period evaluation boundary. It is
not presented as true temporal validation because the chronology is not
independently verified. No external, prospective, or patient-independent
validation is claimed.
