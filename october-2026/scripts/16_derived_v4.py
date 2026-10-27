# -*- coding: utf-8 -*-
# Derived quantities added in the v4 revision (27 October 2026). Every value is arithmetic on the
# frozen v3 result files; nothing is re-estimated. Writes results_v4/RESULTS_v4_derived.json next
# to scripts/. Usage: python 16_derived_v4.py
#  theory_drift_<kind>: per bin, the stability coefficient kappa = C / tau2_t of Eq. (2a) as coded in
#    08_theory_test.py (C the latent training/held-out covariance, tau2_t the training variance of the
#    four-dimensional fit), the latent correlation, and the implied ratio of held-out to training
#    between-learner variance (kappa / r)^2
#  stationary_cells_<kind>: of the 12 cells (naive and shrunk x 6 bins), how many stationary
#    predictions lie outside the observed 95% interval
#  sim: observed/predicted ratios over the non-zero grid points, variance recovery ratios, the
#    probe's predicted headroom ratio
#  a1_bins_ci_nonoverlap: bins in which the raw and adjusted n* intervals do not overlap (addendum A1)
#  conc_null_<kind>: the concentration of the shrunk gains and its stationary benchmark (below)
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
_local = os.path.join(BASE, "results")
RES = os.environ.get("CBB_RESULTS") or (_local if os.path.isdir(_local) else
                                        os.path.join(os.path.dirname(BASE), "v3-cbb-2026-10", "results"))
OUTD = os.path.join(BASE, "results_v4")
os.makedirs(OUTD, exist_ok=True)
J = lambda f: json.load(open(os.path.join(RES, f), encoding="utf-8"))
LAB = ["20-29", "30-49", "50-79", "80-129", "130-249", "250+"]


def r(x, nd):
    return None if x is None else round(float(x), nd)


OUT = dict(note="v4 derived quantities: arithmetic on the frozen v3 result files (16_derived_v4.py)")
for kind in ("adj", "raw"):
    T = J("RESULTS_v3_theory_%s.json" % kind)
    rows, outside, inside_drift = [], 0, 0
    for lab in LAB:
        b = [x for x in T["bins"] if x["bin"] == lab][0]
        C, t2, rr = b["latent_cov_train_heldout"], b["latent_tau2_train_4d"], b["latent_corr_train_heldout"]
        k = C / t2
        rows.append(dict(bin=lab, kappa=r(k, 4), latent_corr=r(rr, 4), var_ratio_heldout_train=r((k / rr) ** 2, 4)))
        for key in ("naive", "shrunk"):
            lo, hi = b["observed_%s_ci" % key]
            if not (lo <= b["predicted_%s" % key] <= hi):
                outside += 1
        lo, hi = b["observed_shrunk_ci"]
        inside_drift += int(lo <= b["predicted_shrunk_drift"] <= hi)
    OUT["theory_drift_" + kind] = rows
    OUT["stationary_cells_outside_obs_ci_" + kind] = outside
    OUT["drift_shrunk_inside_obs_ci_" + kind] = inside_drift
    ks = [x["kappa"] for x in rows[2:]]
    OUT["kappa_longest4_min_" + kind] = min(ks)
    OUT["kappa_longest4_max_" + kind] = max(ks)
    OUT["n_bins_kappa_below_half_" + kind] = int(sum(x["kappa"] < 0.5 for x in rows))
    vr = [x["var_ratio_heldout_train"] for x in rows]
    OUT["var_ratio_shortest4_min_" + kind] = min(vr[:4])
    OUT["var_ratio_shortest4_max_" + kind] = max(vr[:4])

G = [g for g in J("RESULTS_v3_sim_grid.json")["grid"] if float(g["multiplier"]) > 0]
sr = [g["personalization"]["observed_shrunk"] / g["personalization"]["predicted_shrunk"] for g in G]
orr = [g["personalization"]["observed_oracle"] / g["personalization"]["predicted_oracle"] for g in G]
tr_ = [g["tau2_est"] / g["tau2_true_sample"] for g in G]
g1 = [g for g in G if float(g["multiplier"]) == 1.0][0]
ball = [b for b in g1["bins"] if b["bin"] == "all"][0]
OUT["sim"] = dict(
    n_grid_nonzero=len(G),
    shrunk_obs_over_pred_min=r(min(sr), 3), shrunk_obs_over_pred_max=r(max(sr), 3),
    oracle_obs_over_pred_min=r(min(orr), 3), oracle_obs_over_pred_max=r(max(orr), 3),
    n_shrunk_obs_gt_pred=int(sum(x > 1 for x in sr)), n_oracle_obs_gt_pred=int(sum(x > 1 for x in orr)),
    tau2_est_over_true_min=r(min(tr_), 3), tau2_est_over_true_max=r(max(tr_), 3),
    nstar_est_over_true_1x=r(ball["n_star_050_est"] / ball["n_star_050_true"], 3),
    naive_sign_failure_multipliers=[float(g["multiplier"]) for g in G
                                    if g["personalization"]["predicted_naive"] > 0 > g["personalization"]["observed_naive"]],
)
P = J("RESULTS_v3_sim_probe_m1.json")
OUT["probe_predicted_headroom_ratio"] = r(P["probing_design_2x_spread"]["personalization"]["predicted_oracle"]
                                          / P["observed_design"]["personalization"]["predicted_oracle"], 3)
raw, adj = J("RESULTS_v3_hier_raw.json"), J("RESULTS_v3_hier_adj.json")
nov = 0
for a_, b_ in zip(raw["bins"], adj["bins"]):
    lo1, hi1 = a_["n_star_050_ci"]; lo2, hi2 = b_["n_star_050_ci"]
    nov += int(hi1 < lo2 or hi2 < lo1)
OUT["a1_bins_ci_nonoverlap"] = nov
th = J("RESULTS_v3_theory_hlr.json")
b0 = [x for x in th["bins"] if x["bin"] == "20-29"][0]
OUT["hlr_pred_shrunk_share_of_base_20_29"] = r(b0["predicted_shrunk"] / b0["base_logloss_per_trial"], 4)

# Revision of 27 October 2026, arithmetic only:
#  boundary: cells whose stored value sits exactly on a rounding boundary of the printed precision
#    (the stored value is already rounded and its last digit is 5), recomputed without that rounding.
#    These values are kept unrounded. n* = s1^2/tau^2 (rho* = 0.5, Eq. 3) per bin from the stored
#    components, with the interval implied by their stored precision (lo, hi); the bin mean
#    reliability from the per-learner file; the 20-29 raw baseline log-loss and the median learner
#    gains from the per-learner theory files.
#  sign_match_*: cells in which a prediction has the sign of the observed gain (point estimates)
#  heldout_traces_shortest3: traces in the held-out 20% (n - floor(0.8 n)) in the three shortest bins
#  chrono_rate_se_raw_30plus: jackknife SE of the first-to-second-half latent rate correlation (raw)
import pandas as pd
from decimal import Decimal


def half_unit(x):
    return 0.5 * 10 ** Decimal(repr(float(x))).as_tuple().exponent


BND = dict()
rows = []
for b in raw["bins"]:
    s1, t2 = b["per_review_error_var"], b["tau2"]
    rows.append(dict(bin=b["label"], value=s1 / t2, lo=(s1 - half_unit(s1)) / (t2 + half_unit(t2)),
                     hi=(s1 + half_unit(s1)) / (t2 - half_unit(t2))))
BND["raw_nstar050"] = rows
PLr = pd.read_csv(os.path.join(RES, "per_learner_hier_raw.csv.gz"))
BND["raw_mean_rho_30_49"] = float(np.nanmean(PLr["rho_bin"].values[PLr["bin"].values == 1]))
PTr = pd.read_csv(os.path.join(RES, "per_learner_theory_raw.csv.gz"))
m0 = (PTr["bin"].values == 0) & (PTr["heldout_trials"].values > 0)
BND["raw_base_logloss_20_29"] = float(PTr["L_pop"].values[m0].sum() / PTr["heldout_trials"].values[m0].sum())
PTa = pd.read_csv(os.path.join(RES, "per_learner_theory_adj.csv.gz"))
meds = []
for i in range(6):
    mm = (PTa["bin"].values == i) & (PTa["heldout_trials"].values > 0)
    meds.append(float(np.median((PTa["L_pop"] - PTa["L_shrunk"]).values[mm] / PTa["heldout_trials"].values[mm])))
BND["adj_max_median_learner_gain_shrunk"] = max(meds)
OUT["boundary"] = BND


def sgn(x):
    return (x > 0) - (x < 0)


for kind in ("adj", "raw"):
    T = J("RESULTS_v3_theory_%s.json" % kind)
    nv = sum(sgn(b["predicted_naive"]) == sgn(b["observed_naive"]) for b in T["bins"])
    sh = sum(sgn(b["predicted_shrunk"]) == sgn(b["observed_shrunk"]) for b in T["bins"])
    dr = sum(sgn(b["predicted_shrunk_drift"]) == sgn(b["observed_shrunk"]) for b in T["bins"])
    OUT["sign_match_" + kind] = dict(stationary_naive=int(nv), stationary_shrunk=int(sh), stationary_all12=int(nv + sh),
                                     drift_shrunk=int(dr), drift_shrunk_wrong=int(6 - dr))
nh = PTr["n"].values[(PTr["bin"].values >= 0) & (PTr["bin"].values <= 2)]
ho_ = nh - np.floor(0.8 * nh)
OUT["heldout_traces_shortest3"] = dict(min=int(ho_.min()), max=int(ho_.max()))
se = [x["latent_corr_rate_se"] for x in raw["latent_split"]["chrono"] if x["bin"] in LAB[1:]]
OUT["chrono_rate_se_raw_30plus"] = dict(min=min(se), max=max(se))

# Revision of 27 October 2026 (continued), arithmetic only:
#  drift_first_second_<kind>: per bin, the stability coefficient implied by the first-to-second-half
#    four-dimensional fit (latent_split.chrono of the hierarchical result file): kappa_fs = covariance
#    of the two half rates / variance of the first-half rate; and Eq. (2a) evaluated exactly as for the
#    printed drift-aware prediction, sum(1/2 J w tau2_t (2 kappa - 1)) / held-out trials with
#    w = clip(w_eb, 0, 1) and tau2_t the training variance of the training/held-out fit, but with
#    kappa_fs in place of kappa = C / tau2_t. pred_below_obs_ci: the prediction lies below the
#    observed 95% interval of the shrunk gain. Values are kept unrounded.
#  fs_drift_30plus_<kind>: over the five bins from 30 traces, the range of those predictions and the
#    number of bins in which they lie outside the observed interval
#  chrono_rate_se_adj_30plus: jackknife SE range of the first-to-second-half latent rate correlation (adjusted)
for kind, H_ in (("adj", adj), ("raw", raw)):
    T = J("RESULTS_v3_theory_%s.json" % kind)
    P_ = PTa if kind == "adj" else PTr
    ch = {x["bin"]: x for x in H_["latent_split"]["chrono"]}
    rows = []
    for i, lab in enumerate(LAB):
        b = [x for x in T["bins"] if x["bin"] == lab][0]
        kfs = ch[lab]["cov_rate_halves"] / ch[lab]["tau2_rate_half"][0]
        mm = (P_["bin"].values == i) & (P_["heldout_trials"].values > 0)
        w = np.clip(P_["w_eb"].values[mm], 0, 1)
        pr = float((0.5 * P_["J"].values[mm] * w * b["latent_tau2_train_4d"] * (2 * kfs - 1)).sum()
                   / P_["heldout_trials"].values[mm].sum())
        lo, hi = b["observed_shrunk_ci"]
        rows.append(dict(bin=lab, kappa_fs=float(kfs), pred_shrunk_drift_fs=pr, converged=bool(ch[lab]["converged"]),
                         pred_below_obs_ci=bool(pr < lo), pred_outside_obs_ci=bool(not (lo <= pr <= hi))))
    OUT["drift_first_second_" + kind] = rows
    r30 = rows[1:]
    OUT["fs_drift_30plus_" + kind] = dict(min=min(x["pred_shrunk_drift_fs"] for x in r30),
                                          max=max(x["pred_shrunk_drift_fs"] for x in r30),
                                          n_outside_obs_ci=int(sum(x["pred_outside_obs_ci"] for x in r30)),
                                          n_below_obs_ci=int(sum(x["pred_below_obs_ci"] for x in r30)))
se = [x["latent_corr_rate_se"] for x in adj["latent_split"]["chrono"] if x["bin"] in LAB[1:]]
OUT["chrono_rate_se_adj_30plus"] = dict(min=min(se), max=max(se))

# Revision of 27 October 2026 (continued), arithmetic only, values kept unrounded:
#  sim.replicate_pair_shrunk_diff: absolute difference of the observed shrunk gains of the two simulated
#    datasets at the estimated variance (the 1x grid point and the observed design of the probe draw)
#  raw_tau2_ratio_50_79_to_250: between-learner variance of the raw rate, 50-79 bin over 250+ bin
pr_ = J("RESULTS_v3_sim_probe_m1.json")["observed_design"]["personalization"]["observed_shrunk"]
OUT["sim"]["replicate_pair_shrunk_diff"] = abs(pr_ - g1["personalization"]["observed_shrunk"])
rbin = {b["label"]: b for b in raw["bins"]}
OUT["raw_tau2_ratio_50_79_to_250"] = rbin["50-79"]["tau2"] / rbin["250+"]["tau2"]

# Revision of 27 October 2026 (continued): the concentration of the shrunk gains and its stationary benchmark.
#  conc_null_<kind>: per bin, the 1% of learners with the largest absolute contributions to the summed
#    shrunk gain, chosen exactly as in 10_derived.py (k = max(1, round(0.01 m)) of the m learners with
#    held-out trials). Observed: the share of the summed gain they carry and the gain of the others and of
#    all learners per held-out trial (the first two equal RESULTS_v3_derived.json at its precision).
#    Benchmark (arithmetic and seeded random numbers on the per-learner file, nothing re-estimated): the
#    same quantities in NDRAW data sets simulated from the stationary model behind Eq. (2), with each
#    learner's J, w = clip(w_eb, 0, 1) and tau2_bin: a true deviation d ~ N(0, tau2), a shrunk estimate
#    deviation x = w d + N(0, w (1 - w) tau2) (a posterior mean with reliability w), held-out score noise
#    s ~ N(0, J) and the second-order gain s x + 1/2 J (d^2 - (x - d)^2), whose mean is
#    pred_shrunk = 1/2 J tau2 w; the same top-1% rule is applied to each data set. One generator per cell,
#    np.random.default_rng((SEED_NULL, rate index, bin index)), rates in the order adj, raw; in each data
#    set d, the error of x and s are drawn in that order (standard normals, scaled). Stored: median, 2.5th
#    and 97.5th percentiles of the share and of the gain without the top 1% (share_exp, gain_wo_exp); the
#    fraction of data sets whose share is at least the observed one and whose gain without the top 1% is at
#    most the observed one; the fraction in which the gain without the top 1% lies below the summed
#    pred_shrunk of the same remaining learners (the selection on the outcome lowers it by construction);
#    and the mean summed gain per trial over the data sets (close to the bin's predicted_shrunk).
NDRAW, SEED_NULL = 2000, 20261027
OUT["conc_null_draws"], OUT["conc_null_seed"] = NDRAW, SEED_NULL
for ki, kind in enumerate(("adj", "raw")):
    P_ = PTa if kind == "adj" else PTr
    rows = []
    for i, lab in enumerate(LAB):
        mm = (P_["bin"].values == i) & (P_["heldout_trials"].values > 0)
        m = int(mm.sum())
        k = max(1, int(round(0.01 * m)))
        g = (P_["L_pop"] - P_["L_shrunk"]).values[mm]
        t_ = P_["heldout_trials"].values[mm]
        top = np.argsort(-np.abs(g))[:k]
        keep = np.ones(m, bool)
        keep[top] = False
        Jm, t2m = P_["J"].values[mm], P_["tau2_bin"].values[mm]
        w = np.clip(P_["w_eb"].values[mm], 0, 1)
        ps = P_["pred_shrunk"].values[mm]
        sd_d, sd_e, sd_s = np.sqrt(t2m), np.sqrt(w * (1 - w) * t2m), np.sqrt(Jm)
        rng = np.random.default_rng((SEED_NULL, ki, i))
        sh, wo, full = np.empty(NDRAW), np.empty(NDRAW), np.empty(NDRAW)
        below = np.zeros(NDRAW, bool)
        for rr in range(NDRAW):
            d = sd_d * rng.standard_normal(m)
            x = w * d + sd_e * rng.standard_normal(m)
            s = sd_s * rng.standard_normal(m)
            gs = s * x + 0.5 * Jm * (d * d - (x - d) ** 2)
            tp = np.argsort(-np.abs(gs))[:k]
            kp = np.ones(m, bool)
            kp[tp] = False
            sh[rr] = gs[tp].sum() / gs.sum()
            wo[rr] = gs[kp].sum() / t_[kp].sum()
            full[rr] = gs.sum() / t_.sum()
            below[rr] = wo[rr] < ps[kp].sum() / t_[kp].sum()
        sh_o, wo_o = float(g[top].sum() / g.sum()), float(g[keep].sum() / t_[keep].sum())
        rows.append(dict(bin=lab, n_learners=m, k_top1pct=k, share_obs=sh_o, gain_wo_obs=wo_o,
                         gain_all_obs=float(g.sum() / t_.sum()),
                         share_exp=[float(v) for v in np.percentile(sh, [50, 2.5, 97.5])],
                         gain_wo_exp=[float(v) for v in np.percentile(wo, [50, 2.5, 97.5])],
                         frac_share_ge_obs=float(np.mean(sh >= sh_o)), frac_gain_wo_le_obs=float(np.mean(wo <= wo_o)),
                         frac_gain_wo_below_pred_kept=float(np.mean(below)), mean_gain_all=float(full.mean())))
    OUT["conc_null_" + kind] = rows
json.dump(OUT, open(os.path.join(OUTD, "RESULTS_v4_derived.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(OUT, indent=1))
