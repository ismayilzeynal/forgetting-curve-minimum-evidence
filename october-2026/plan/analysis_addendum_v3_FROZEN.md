# Exploratory analysis addendum v3 (FROZEN before running)

Paper: sheet_no 17.2 (papers.id 10). Frozen 2026-10-14, after the confirmatory analysis of
2026-07-12 (analysis_plan.md, unchanged) and BEFORE any of the analyses below was run.
Everything here is exploratory and is labelled so in the manuscript. Outcomes are reported
whatever they are; no analysis below is dropped because of its result.

Data, sample and seed are unchanged: public Duolingo HLR traces (doi:10.7910/DVN/N8XJME),
the same 40,000-learner sample (learners with at least 20 traces), seed 20260712, the same six
review-count bins (20-29, 30-49, 50-79, 80-129, 130-249, 250+).

## A0. Reproduction gate

Re-run the v2 estimators on the same sample with a batched (all learners at once) Newton solver
that implements the same per-learner IRLS (ridge 1e-2 on the slope, unpenalized intercept, 30
iterations, eta clipped at +-30). Gate: per-bin split-half reliabilities, n_users, median and
pooled slopes, estimator-comparison reliabilities and the v2 per-bin median gains must equal the
v2 RESULTS.json / estimator_comparison.json values at 3 decimals (gains at 4). Random quantities
(bootstrap CIs, permutation nulls) are not part of the gate; they are replaced by A8.

## A1. Strength-adjusted forgetting rate (construct validity)

Two learner folds (random half of the sample each). On fold F a population model is fit:
logit P = c0 + c1 d_lex + c2 log(1+history_seen) + c3 (history_correct+1)/(history_seen+2)
+ language-pair indicators + c4 log(lag), where d_lex is the lexeme difficulty (smoothed logit of
mean session recall of the lexeme computed on fold F only, 20 pseudo-trials toward the fold mean).
The offset for a learner in the other fold is the fitted value without c0 and without the lag term.
The learner slope b_adj is then estimated on within-learner-centered log lag with that offset and a
free learner intercept. Reported for b_adj: reliability horizon, null, temporal split,
personalization test, theory test. Decision rule: if raw and adjusted horizons differ in any bin by
more than their bootstrap CIs, the adjusted rate is the headline and raw is the sensitivity analysis.

## A2. HLR-native learner parameter

Half-life regression (Settles & Meeder 2016 parameterization p = 2^(-lag/h), log2 h = theta . x,
features: bias, sqrt(1+history_correct), sqrt(1+history_wrong), lexeme indicators with L2 0.1,
h clipped to [15 min, 274 days]) fit by binomial likelihood on fold F and applied to the other fold.
Per learner a half-life multiplier u (log2 h = theta . x + u, ridge 1e-2) is estimated by Fisher
scoring. Reported for u: reliability horizon, within-learner lag-permutation null (diagnoses how
much of the reliability of u comes from lag-dependent signal rather than recall level),
personalization test, scheduling consequences (A7).

## A3. Variance decomposition and model-based reliability

Per bin and per parameter: tau2 = cov(b_odd, b_even) (true-score variance), half-error variance
var(b_half) - tau2, model-based per-learner error variance from the inverse Fisher information,
dispersion factor phi = empirical half-error variance / mean model-based half variance, per-learner
reliability rho_i = tau2 / (tau2 + phi sigma_i^2). Bootstrap CIs (1,000 learner resamples).

## A4. Theory test: predicted vs observed personalization gain

Second-order expansion of held-out log-loss around the learner's true parameter. For an estimate
b_hat with shrinkage weight w toward the population value beta, expected held-out gain over the
population value = 1/2 J (2 w C - w^2 V), where J is the learner's held-out Fisher information for
the parameter, C = cov(b_hat, true held-out parameter) and V = var(b_hat). Under stationarity C = tau2
and V = tau2 + sigma2, so naive (w=1) gain = 1/2 J (tau2 - sigma2), optimal shrinkage (w = rho) gain
= 1/2 J tau2 rho, oracle headroom = 1/2 J tau2. Predictions use ONLY each learner's training
portion (first 80%): tau2 and phi from odd/even halves of the training portions, sigma_i^2 from the
training fit, J_i from the held-out design with population-model probabilities. Observed gains are
the mean per-trial held-out log-loss gain (trial-weighted), CI by learner-cluster bootstrap (1,000).
The intercept is re-estimated on the training portion for every candidate slope; lag is centered
on the training-portion mean. The v2 per-learner median gain goes to the supplement. A drift-aware
prediction uses C = cov(b_train, b_heldout) as a diagnostic (it uses held-out data and is labelled so).

## A5. Minimum evidence and the flat curve

Per bin: per-review error variance sigma1^2 = (half-error variance) x (mean half length);
n* = (sigma1^2 / tau2) rho* / (1 - rho*) for rho* = 0.5 and 0.7, bootstrap CIs. Mechanism for the flat
curve: per-bin within-learner SD of log lag, mean p(1-p), mean session size, and the regression of
log(n_i x model sigma_i^2) on log SD(log lag) across learners.

## A6. Semi-synthetic recovery (simulation, labelled as such)

Real designs (each learner's lags, session sizes, order) and real fitted intercepts; slopes drawn
from N(beta, tau2_sim) with tau2_sim in {0, 0.5, 1, 2, 4, 8, 16} x the estimated overall tau2;
outcomes binomial. Rerun: split-half reliability vs true reliability, tau2 recovery, predicted vs
observed gains (naive, shrunk, oracle). Counterfactual probing design: same learners and n with log
lag spread doubled (z -> 2z) at tau2_sim = 1x; report n* under observed and probing designs.

## A7. Scheduling consequence (model-based, labelled)

With HLR (A2): target recall r* in {0.85, 0.90}; interval Delta* = -h log2 r*, so the
personalized/population interval ratio is 2^u. Report the distribution of 2^u_naive and
2^u_shrunk over learners (median, 5th and 95th percentile, share outside [0.8, 1.25]); expected
recall at the population-scheduled interval if the learner's multiplier equals u_shrunk; and, in a
semi-synthetic HLR world (u_true ~ N(0, tau2_u) on real designs), the mean absolute deviation of
achieved recall from r* and the reviews per 30 days for a reference item (population median
half-life, held fixed) under population, naive, shrunk and oracle policies.

## A8. Robustness

Chronological first/second-half split; mean of 20 random half splits; learner intercept (recall
level) horizon; lexeme-side slope and intercept horizons (same estimator, bins by reviews per
lexeme); a single permutation-null run (20 within-half lag permutations, all learners) per
estimator, which replaces the two v2 null values.
