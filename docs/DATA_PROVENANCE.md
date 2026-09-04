# Data provenance

## Source

This repository does not redistribute the source data. Obtain the
Hospital-Acquired Infection (HAI) Prediction Dataset directly from its source:

- Release: HAI-release 1.0.1
- Zenodo DOI: [10.5281/zenodo.21831931](https://doi.org/10.5281/zenodo.21831931)
- Source-reported license: CC BY-NC 4.0

The source HAI dataset is not part of this repository and remains subject to
its original CC BY-NC 4.0 license.

## Analyzed artifact

The locally analyzed raw CSV had SHA-256:

```text
e60a63201eceb772329247b561a8dd996d40e33750abfc333e845d6598b08060
```

The study artifact is strongly associated with HAI-release 1.0.1. Exact
byte-level identity between the local file and the Zenodo archive was not
independently verified. The hash is provided for provenance checking, not as a
claim that every downloaded archive will be byte-identical.

## Input handling

Pass the local CSV path explicitly with `--data`. The code does not require or
copy the raw file into this repository. Deterministic parsing preserves the
source label, records quality flags, and excludes no row solely because a
predictor is missing. Generated local row-level artifacts remain ignored by
Git and must not be uploaded.
