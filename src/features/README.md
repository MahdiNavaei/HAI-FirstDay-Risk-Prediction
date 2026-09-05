# Feature construction

Feature-set definitions and fold-safe missingness augmentation are implemented
in `src/modeling/modeling_runtime.py`. Feature Set C is the frozen primary
representation: authorized first-day clinical variables plus explicit
missingness indicators.
