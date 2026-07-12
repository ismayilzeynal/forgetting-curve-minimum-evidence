# Minimum-Evidence Boundaries for Personalized Spaced Repetition

Reproducibility bundle for *"How Many Reviews Before We Know Your Forgetting Curve? Minimum-Evidence Boundaries for Personalized Spaced Repetition."*

The study asks how much review history is needed before a learner's forgetting rate can be estimated reliably enough to schedule on, using the public Duolingo spaced-repetition dataset (12.9 million traces, 115,222 learners). It measures the split-half reliability of a per-learner forgetting rate as a function of review-history length (a *reliability horizon*), certifies it against a within-user permutation null with a location-invariant estimator, and tests whether a personalized rate — naive or empirical-Bayes partial-pooled — beats the population default in held-out prediction.

## Data

The raw traces are **not** included (≈380 MB). Obtain them from Harvard Dataverse:

- Duolingo Spaced Repetition Data — doi:10.7910/DVN/N8XJME

Place the file at `data/duolingo_hlr.csv.gz` (columns: `p_recall, timestamp, delta, user_id, learning_language, ui_language, lexeme_id, lexeme_string, history_seen, history_correct, session_seen, session_correct`).

## Reproduce

```
pip install pandas numpy scipy matplotlib
python scripts/killcheck.py        # 10k-learner pilot (feasibility gate)
python scripts/results_gate.py     # full confirmatory analysis -> results/RESULTS.json
python scripts/figures.py          # Figure 1 (reliability horizon) + Figure 2 (personalization)
```

`results/RESULTS.json` is the single source of truth for every number in the manuscript. The seed (20260712), the estimator (within-learner-centered log-lag, weighted binomial, unpenalized intercept), the review-count bins, the per-bin permutation null, and the personalization test are all in `scripts/results_gate.py`.

## Estimator note (location invariance)

The confirmatory estimator centers log-lag *within each learner* and leaves the intercept *unpenalized*. A location-non-invariant alternative — centering on the global mean and penalizing the intercept — inflates both the split-half reliability (to ~0.56) and its within-user permutation null (to ~0.43) on this data; `scripts/diag2.py` demonstrates the difference. This is why the reliability is certified against a within-user null.

## Pre-registration

The analysis plan (`analysis_plan.md`) was frozen before the confirmatory runs. A prior pre-registration on the same dataset (OSF 368jq, registered 2026-07-03) established that user-level half-life reliability is non-zero; this work builds on and extends it.

## License

Code released under the MIT License. The dataset is subject to its original Dataverse terms.
