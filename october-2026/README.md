# Is a learner's rate of forgetting worth knowing? Code, plan and results

This folder is the code and results bundle for the manuscript "Is a Learner's Rate of Forgetting Worth Knowing? Reliability, Drift and Predictive Headroom in Naturalistic Spaced Repetition". Every number in the manuscript and in its Online Resource 1 is read from a file in `results/` or `results_v4/`, from the July result file of the earlier bundle (`../results/RESULTS.json`) or, for the feasibility check, from `plan/killcheck.json`.

It extends the earlier bundle of 12 July 2026 in this repository (the confirmatory analysis): that bundle's plan, scripts and results stay as they were (only its README is replaced), and this folder adds the exploratory analyses of October 2026.

## Layout

- `plan/analysis_plan.md`: the confirmatory analysis plan, dated 12 July 2026. It was written after a feasibility check on 10,000 learners and before the confirmatory runs, and it was not entered in a public registry.
- `plan/killcheck.json`: the output of that feasibility check (`scripts/v2_reference/killcheck.py`), which the plan's first paragraph reports on. Its 10,000 learners were drawn with the same seed from the same pool of learners with at least 20 traces as the confirmatory sample, and it estimated the split-half reliability of a cruder slope (least squares on session recall) at 0.42 overall.
- `plan/analysis_addendum_v3_FROZEN.md`: the exploratory addendum, dated 14 October 2026 and written before any of its analyses was run. It was not registered or publicly time-stamped before the analyses; the hash identifies the file, and the date is the author's record. SHA-256: `c26b19265f4615764f1b516be99d97d05c1f7c7216cf2b0718e868a6eaddf180`.
- `scripts/`: all analysis code, in run order below. `scripts/v2_reference/` holds the July scripts for comparison only; `diag2.py`, `figures.py`, `killcheck.py`, `r2_fixes.py` and `results_gate.py` there are identical to those in `../scripts/`. `diag_null.py` there is not part of the July bundle: it is a July diagnostic of the high permutation-null reliability of the uncorrected per-learner estimator, written while that estimator was in use (null reliability 0.396 overall and 0.433 at 20 to 29 traces in `../results/estimator_comparison.json`, against 0.026 overall for the corrected estimator); the correction is the first deviation listed below.
- `results/`: 18 JSON result files and 7 per-learner `.csv.gz` files written by the scripts.
- `results_v4/RESULTS_v4_derived.json`: quantities added in the revision of 27 October 2026 (stability coefficients from both splits and the drift-aware predictions they imply, variance ratios, simulation ratios, sign counts, held-out trace counts, the difference between the two simulated datasets at the estimated variance, the ratio of the raw between-learner variance in the 50-79 and 250+ bins, the variance factor of a common scaling of the raw slopes from the 20-29 to the 250+ bin, the concentration of the shrunk gains in every bin together with its expected value under the stationary model, from 2,000 seeded simulated data sets per bin, with the Monte Carlo precision of its percentiles (Table 4 of the manuscript), and unrounded recomputations of printed cells whose stored value sits exactly on a rounding boundary), computed by `scripts/16_derived_v4.py` from the files in `results/` without re-estimating anything.
- `MANIFEST.sha256`: SHA-256 of every other file in this folder.
- `FROZEN_INPUTS.sha256`: the author's hash record of the inputs before the October revision, with paths relative to the author's working folder. Its 42 entries for the plan, the addendum, the result files and the analysis scripts match files in this folder; its 22 entries for the manuscript sources and for earlier versions of the tooling scripts `11_resolve.py` to `14_refcheck.py` refer to files that are not included here.

Comments, file names and result keys use the author's internal version labels: v2 is the confirmatory run of 12 July 2026, v3 the exploratory analyses of October 2026 and v4 the revision of 27 October 2026.

## Data

The input is the public Duolingo spaced-repetition trace file at Harvard Dataverse (Settles, 2017; https://doi.org/10.7910/DVN/N8XJME), a gzip-compressed CSV of 379,004,009 bytes with SHA-256 `26cf9b33f2e9b2e8aa9a8f262a017b30699ca2f625c56a12ce699d4ea7076873`. It is not copied here. The scripts use a seeded sample of 40,000 learners with at least 20 traces (seed 20260712).

## Run order

```
python scripts/00_load_sample.py <csv.gz> <cache>       # seeded 40,000-learner sample
python scripts/01_reproduce_v2.py <cache>               # gate: the July numbers reproduced (47 checks)
python scripts/02_population_models.py <cache>          # cross-fitted strength offset and half-life regression
python scripts/03_main.py <cache>                       # addendum A1 to A8 as specified (moment-based)
python scripts/04_items.py <cache>                      # addendum item side (moment-based)
python scripts/05b_separation_diagnostics.py <cache>    # why the moments are not used for inference
python scripts/06_hier_learners.py <cache> raw|adj|hlr  # hierarchical variance, reliability, n*, splits, nulls
python scripts/05c_rank_null.py <cache>                 # permutation null for the rank-based split-half
python scripts/07_hier_items.py <cache>                 # item side, hierarchical
python scripts/08_theory_test.py <cache> raw|adj|hlr    # personalization and theory test
python scripts/09_semisynthetic.py <cache> grid|probe|hlr
python scripts/10_derived.py                            # derived quantities quoted in the text
python scripts/16_derived_v4.py                         # revision of 27 October 2026: stability coefficients, ratios, Table 4 benchmark
python scripts/12_figures.py                            # Fig. 1 to 6 from the result files
```

`scripts/05_semisynthetic.py` is the addendum's moment-based simulation; it was superseded by `09_semisynthetic.py` and not run.

The manuscript tooling is included for transparency: `11_resolve.py` replaces every number token in the manuscript sources with its value from `results/`, `results_v4/`, the July result file `../results/RESULTS.json` of the earlier bundle or `plan/killcheck.json` (and refuses a stored value that sits on a rounding boundary of its printed precision), `13_build_package.py` builds the documents and `14_refcheck.py` checks the references against Crossref and DataCite. The manuscript sources themselves are not part of this folder, so these three scripts do not run on their own here.

## Deviations

From the plan: the per-learner estimator was corrected after the first confirmatory run on 12 July 2026 (within-learner centring of log lag with an unpenalized intercept replaced global centring with a penalized intercept); the confirmatory permutation null used 8 permutations instead of 200, each on a random subsample of 6,000 learners per bin; the Brier score and the secondary half-life estimator are not reported; the weak-word flag and Leitner analyses were not carried out; the decision table was not produced because no bin reached the reliability bars; hypothesis H1 is judged not supported although its operational criterion, a positive Spearman correlation across bins, is met nominally, because the reliabilities show no monotone trend. Before the July bundle was published, the held-out test was re-implemented (`scripts/r2_fixes.py` of that bundle): the population slope was fitted to the training portions only instead of to all traces, the partially pooled rate, an arm the plan does not name, received the bin's split-half reliability as its weight, and a robustness variant that personalizes intercept and rate was added; all H3 values come from this implementation. Hypothesis H3 was judged by the plan's interval criterion, which the confirmatory code applied to the median per-learner held-out gain; it was not met in any bin.

From the addendum: (a) the moment-based variance decomposition (A3, A4) was dominated by quasi-separated per-learner fits (`results/RESULTS_v3_separation.json`), so the hierarchical estimators in `scripts/common_eb.py` replace it for inference. The hierarchical estimators, the four-dimensional fits behind the stability coefficients and the drift-aware prediction, and the level positive control were chosen after the moment-based results had been seen, and the gain-concentration analysis with its stationary benchmark after the held-out results had been seen; they are not pre-specified. Hyperparameter intervals use 500 learner bootstrap resamples instead of 1,000. In the simulation the model-based reliability is compared with the true reliability in place of the split-half reliability that A6 names. The results as specified are in `results/RESULTS_v3_main.json` and are reported in Online Resource 1. (b) By the decision rule of A1 the raw rate is the headline, because the raw and adjusted minimum-evidence horizons do not differ by more than their bootstrap intervals in any bin; the manuscript shows the personalization test for the adjusted rate in its Table 3 and for the raw rate in Online Resource 1. (c) The drift-aware prediction uses the latent covariance of a four-dimensional fit instead of the covariance of individual estimates.

## Requirements

Python 3.12 with numpy, pandas, scipy, pyarrow and matplotlib. The manuscript tooling also uses pandoc 3, python-docx, pywin32 with Microsoft Word, and PyMuPDF.

## License

The code is released under the MIT License (`../LICENSE`). The Duolingo spaced-repetition data (Settles, 2017; https://doi.org/10.7910/DVN/N8XJME) are released by Duolingo under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The files in `results/`, `results_v4/` and `plan/killcheck.json` are derived from those data: summary statistics, and per-learner estimates keyed by the de-identified learner identifiers of the original file. They contain no raw traces and are shared under the same license, for non-commercial use.
