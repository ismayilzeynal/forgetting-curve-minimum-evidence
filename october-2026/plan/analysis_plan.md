# Frozen Analysis Plan - Paper 17 (Minimum-Evidence Boundaries for Personalized Spaced Repetition)

FROZEN 2026-07-12, BEFORE any confirmatory run. A 10k-user pilot (kill-check) confirmed forgetting is
present (61% of users have a negative recall-vs-log-lag slope) and that user-level reliability is LOW
and history-dependent; the pilot's crude OLS-on-p_recall estimator is replaced below by a weighted
binomial GLM. Deviations after this point are logged in the manuscript.

## Positioning (vs the concurrent prereg osf.io/368jq)
The concurrent, independent pre-registration 368jq (2026-07-03, same dataset) already tests whether
user-level half-life reliability EXISTS (a single split-half coefficient at a fixed threshold). We do
NOT contest that. We credit it as establishing the precondition and lead on what it does not do: the
reliability HORIZON (reliability as a function of review history) and the minimum-review decision table.

## Decision the paper informs
An adaptive-learning platform designer (and a self-directed learner) deciding, after how many reviews,
to switch from a population-default forgetting curve to a personalized one - and whether a personalized
schedule even disagrees with a cheap heuristic (Leitner) enough to matter.

## Data (public, checksummed)
Duolingo spaced-repetition traces (Settles & Meeder 2016; Harvard Dataverse doi:10.7910/DVN/N8XJME):
12,854,226 traces, 115,222 users, median 34 reviews/user; 73,521 users with >=20 reviews. Columns
used: p_recall (session recall fraction), delta (lag seconds), user_id, lexeme_id, history_seen,
session_seen, session_correct.

## Forgetting-rate estimator (per user)
Weighted binomial GLM (logistic): for each user, logit(P(recall)) = a + b * log(delta), fit on the
user's trials with each row weighted by session_seen (successes = session_correct). The slope b is the
user's forgetting rate (b < 0 = recall decays with lag). Ridge-stabilized (small L2) for short
histories. Corroborated by an empirical log2 half-life per user (median over trials of delta scaled to
recall) as a secondary estimator; the reliability horizon is reported for the primary GLM slope.

## Confirmatory analyses
1. **Reliability-horizon curve (primary).** Split each user's trials into odd/even by chronological
   order; estimate b on each half. Within each review-count bin, Pearson-correlate (b_odd, b_even) and
   Spearman-Brown correct -> split-half reliability. Bins: [20,30), [30,50), [50,80), [80,130),
   [130,250), [250,+). Report reliability vs bin with bootstrap 95% CIs (resample users within bin).
2. **Minimum-review horizon + decision table.** The smallest history length at which reliability first
   reaches REL_STAR = 0.70 (individual-decision bar) and 0.50 (group bar). If never reached, report the
   ceiling reliability and declare personalization-from-history not certifiable (honest null).
3. **Per-history-length personalization certification.** Hold out each user's chronologically last 20%
   of trials; predict held-out recall with (a) the user's personalized b (from the first 80%) and (b)
   the population-mean b; compare held-out log-loss and Brier PER review-count bin; the personalization
   horizon = the smallest bin at which the personalized gain > 0 with a bootstrap 95% CI excluding 0.
4. **Weak-word flag churn.** Flag a user's "weak" lexemes (bottom-tercile estimated retention) in the
   first vs second half; report flag RETENTION (fraction of first-half weak items still weak) against
   the arbitrary-reflag rate (the tercile base rate), by history length.
5. **HLR-vs-Leitner disagreement.** Assign each item a Leitner box from its correct-streak; compare the
   "due now" set implied by personalized half-lives vs Leitner boxes; report disagreement (Cohen kappa /
   Jaccard) as a function of history length.

## Certification against a null
- **Within-user permutation null:** shuffle log(delta) across a user's trials, re-estimate the
   reliability horizon; the null reliability must be ~0 (near the no-signal floor). Certifies signal.
- Bootstrap 95% CIs on every horizon and gain. The population-default model is the baseline throughout.

## Pre-registered hypotheses (directional)
- H1: split-half reliability of the user forgetting rate INCREASES with review count (monotone trend,
  Spearman rho > 0 across bins).
- H2: reliability reaches REL_STAR = 0.70 only at a HIGH review count (the minimum-review horizon is
  large relative to the median user's 34 reviews) - i.e., most users never reach individual-decision
  reliability. (We pre-commit to reporting a refuted H2, i.e. an easily-reached horizon, with equal
  prominence.)
- H3: personalized beats the population default in held-out log-loss only above a review-count horizon,
  not for short histories.

## Honest-null contingency
If reliability never reaches 0.50 in any bin, report with full prominence: "user-level forgetting rates
on Duolingo do not become split-half reliable within the observed review range; personalization from
review history alone is not certifiable, and the population prior (or richer features) should be used."
Never spin a null.

## Page-one deliverables
1. The reliability-horizon curve (reliability vs review count, with the REL_STAR crossings).
2. The minimum-review decision table (personalize vs default, by review count, with the certified horizon).
3. A short reporting note on how many reviews a personalization claim needs.

## Thresholds
REL_STAR = 0.70 (individual) and 0.50 (group); MIN_REVIEWS = 20; HOLDOUT = last 20%; N_BOOT = 1000;
N_PERM = 200; SAMPLE = 40,000 users (>=20 reviews) for the confirmatory run; SEED = 20260712.
