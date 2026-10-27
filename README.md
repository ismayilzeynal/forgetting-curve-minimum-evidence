# Is a Learner's Rate of Forgetting Worth Knowing? Analysis plan, code and results

This repository holds the dated analysis plan and the confirmatory analysis behind the manuscript "Is a Learner's Rate of Forgetting Worth Knowing? Reliability, Drift and Predictive Headroom in Naturalistic Spaced Repetition" and, in the folder `october-2026/`, the code and result files of the analyses that the manuscript reports. An earlier working title of the same study was "How Many Reviews Before We Know Your Forgetting Curve? Minimum-Evidence Boundaries for Personalized Spaced Repetition".

## Contents

- `analysis_plan.md`: the analysis plan, dated 12 July 2026. It was written after a feasibility check on 10,000 learners of the same corpus and before the confirmatory runs, and it was made public here, with the code and results of those runs, on 12 July 2026. It was not entered in a public registry.
- `scripts/killcheck.py`: the feasibility check on 10,000 learners, drawn with the same seed from the same pool of learners with at least 20 traces as the confirmatory sample. It estimated the split-half reliability of a cruder slope (least squares on session recall) at 0.42 overall, which the plan reports as low and history-dependent; the manuscript notes that the support for H2 is therefore not independent of the information that shaped the plan.
- `scripts/results_gate.py` and `scripts/r2_fixes.py`: the confirmatory analysis (split-half reliability by bin with a within-learner permutation null, the estimator comparison, and the held-out personalization test).
- `scripts/diag2.py`: the diagnostic that compares the corrected and the uncorrected per-learner estimator.
- `scripts/figures.py`: the figures of the confirmatory run.
- `results/RESULTS.json` and `results/estimator_comparison.json`: the results of the confirmatory run.
- `october-2026/`: the code, the result files and the exploratory addendum of 14 October 2026 behind every result reported in the manuscript and its Online Resource 1, with its own README and `MANIFEST.sha256` (the SHA-256 of every other file in the folder).

The plan, the scripts and the result files of 12 July 2026 are kept exactly as they were published. Later commits replaced this README and added the folder `october-2026/`, together with a `.gitattributes` rule that keeps that folder's files byte for byte as listed in its manifest (see the commit history).

The manuscript reports the confirmatory results in its Online Resource 1 (Section S10) and lists every deviation from the plan in its Section 4.7. Among them: the per-learner estimator was corrected after the first confirmatory run (log lag centred within each learner, unpenalized intercept), the permutation null used 8 permutations instead of the planned 200, and, before publication, the held-out test was re-implemented in `scripts/r2_fixes.py` (population slope fitted to the training portions only, the partially pooled rate weighted by the bin's split-half reliability, and a robustness variant that personalizes intercept and rate); the partially pooled rate and the robustness variant are not in the plan. The manuscript also reports exploratory analyses of October 2026; their code, result files and the exploratory addendum of 14 October 2026 are in `october-2026/`.

## Related registration

A separate preregistration on the same corpus (Jeong, 2026; https://doi.org/10.17605/OSF.IO/368JQ, registered 3 July 2026) states hypotheses about the between-learner variance and split-half reliability of learner half-lives and about held-out gains from personalizing them. As of 27 October 2026 no results or other outputs were attached to it. The positioning paragraph of `analysis_plan.md` says that this preregistration tests whether user-level half-life reliability exists and credits it "as establishing the precondition"; that wording refers to what the registration sets out to test, not to a reported result, and none of the analyses in this repository relies on it.

## Data

The raw traces are not included. Obtain the Duolingo spaced-repetition data from Harvard Dataverse (Settles, 2017; https://doi.org/10.7910/DVN/N8XJME) and save the file as `data/duolingo_hlr.csv.gz`.

## Reproduce the confirmatory run

```
pip install pandas numpy scipy matplotlib
python scripts/killcheck.py        # feasibility check on 10,000 learners
python scripts/results_gate.py     # confirmatory analysis, writes results/RESULTS.json
python scripts/r2_fixes.py         # estimator comparison and held-out test, updates results/RESULTS.json
python scripts/figures.py          # figures of the confirmatory run
```

The seed is 20260712. The scripts are kept as published, so they read and write relative to the folder named in their `BASE` constant (`17 - Forgetting Curve Minimum Evidence - 07.12.2026`): either set `BASE` to `"."` and run the commands above from the repository root, or clone the repository into a folder of that name and run the scripts from its parent folder.

To reproduce the analyses reported in the manuscript, follow `october-2026/README.md`.

## License

Code released under the MIT License. The Duolingo dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/). The result files in this repository are derived from it (summary statistics, and in `october-2026/results/` per-learner estimates keyed by the de-identified learner identifiers of the original file; no raw traces) and are shared under the same license, for non-commercial use.
