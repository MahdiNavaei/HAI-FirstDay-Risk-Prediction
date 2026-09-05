# Evaluation

Metric and fixed-alert-budget implementations are in
`src/modeling/modeling_runtime.py`. The primary scalar metric is
`sklearn.metrics.average_precision_score`; held-out evaluation uses natural
prevalence and does not tune after observing held-out outcomes.
