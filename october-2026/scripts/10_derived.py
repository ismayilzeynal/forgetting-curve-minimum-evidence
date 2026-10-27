# -*- coding: utf-8 -*-
# Derived quantities quoted in the text (ratios, percentages, shares), computed from the result
# files and the per-learner outputs so that no number is computed by hand.
# Writes results/RESULTS_v3_derived.json. Usage: python 10_derived.py
import os, json, math
import numpy as np, pandas as pd
from common import RESULTS, BINLAB, r

J = lambda f: json.load(open(os.path.join(RESULTS, f), encoding="utf-8"))
raw, adj, hlr = J("RESULTS_v3_hier_raw.json"), J("RESULTS_v3_hier_adj.json"), J("RESULTS_v3_hier_hlr.json")
thr, tha, thh = J("RESULTS_v3_theory_raw.json"), J("RESULTS_v3_theory_adj.json"), J("RESULTS_v3_theory_hlr.json")
sep = J("RESULTS_v3_separation.json")
OUT = {}

# rate: spread relative to the mean, and the size of one between-learner SD in recall-odds terms
mu_b = raw["overall"]["mu"][1]; tau_b = raw["overall"]["tau"]
OUT["raw_recall_at_mu_intercept"] = r(1 / (1 + math.exp(-raw["overall"]["mu"][0])), 4)
OUT["raw_tau_over_abs_mu"] = r(tau_b / abs(mu_b), 3)
OUT["raw_share_learners_positive_true_rate"] = r(0.5 * math.erfc((0 - mu_b) / (tau_b * math.sqrt(2))), 4)
OUT["raw_tau2_ratio_first_last_bin"] = r(raw["bins"][0]["tau2"] / raw["bins"][-1]["tau2"], 2)
OUT["hlr_one_sd_halflife_factor"] = r(2 ** hlr["overall"]["tau"], 2)
OUT["hlr_null_tau2_ratio"] = r(np.mean([p["tau2"] for p in hlr["permutation_null"]]) / hlr["overall"]["tau2"], 3)
OUT["raw_null_tau2_mean"] = r(float(np.mean([p["tau2"] for p in raw["permutation_null"]])), 7)
OUT["adj_null_tau2_mean"] = r(float(np.mean([p["tau2"] for p in adj["permutation_null"]])), 7)
OUT["raw_null_tau2_max"] = r(float(np.max([p["tau2"] for p in raw["permutation_null"]])), 7)
OUT["raw_null_share_of_tau2"] = r(float(np.mean([p["tau2"] for p in raw["permutation_null"]])) / raw["overall"]["tau2"], 4)

# share of sampled learners whose history reaches n* for the overall model
PL = pd.read_csv(os.path.join(RESULTS, "per_learner_hier_raw.csv.gz"))
for k, key in ((0.5, "n_star_050"), (0.7, "n_star_070")):
    OUT["share_sample_n_ge_" + key] = r(float(np.mean(PL["n"].values >= raw["overall"][key])), 4)
OUT["ratio_nstar050_to_median_n"] = [dict(bin=b["label"], ratio=r(b["n_star_050"] / b["median_n"], 2)) for b in raw["bins"]]

# gains as a share of the baseline log-loss, per bin (raw and adjusted rate, HLR multiplier)
for nm, T in (("raw", thr), ("adj", tha), ("hlr", thh)):
    rows = []
    for b in T["bins"]:
        base = b["base_logloss_per_trial"]
        row = dict(bin=b["bin"])
        for k in ("observed_naive", "observed_shrunk", "predicted_oracle", "predicted_shrunk"):
            row[k + "_pct_of_base"] = r(100 * b[k] / base, 3)
        rows.append(row)
    OUT["gain_pct_" + nm] = rows
    OUT["gain_pct_" + nm + "_max_oracle"] = r(max(x["predicted_oracle_pct_of_base"] for x in rows), 3)
    OUT["gain_pct_" + nm + "_max_obs_shrunk"] = r(max(x["observed_shrunk_pct_of_base"] for x in rows), 3)
lv = thr.get("level_control", [])
if lv:
    OUT["level_vs_rate_shrunk_ratio"] = [dict(bin=a["bin"], ratio=r(a["observed_level_gain"] / b["observed_shrunk"], 1) if b["observed_shrunk"] > 0 else None)
                                         for a, b in zip(lv, thr["bins"])]
    OUT["level_gain_pct_of_base"] = [dict(bin=a["bin"], pct=r(100 * a["observed_level_gain"] / b["base_logloss_per_trial"], 2)) for a, b in zip(lv, thr["bins"])]
# ratio observed/predicted for the shrunk policy (raw)
OUT["raw_obs_over_pred_shrunk"] = [dict(bin=b["bin"], ratio=r(b["observed_shrunk"] / b["predicted_shrunk"], 2) if b["predicted_shrunk"] else None) for b in thr["bins"]]
# how concentrated the observed shrunk-rate gain is: share of the bin's summed gain carried by the
# 1% of learners with the largest absolute contribution, and the gain after removing them
for kind in ("raw", "adj"):
    PT = pd.read_csv(os.path.join(RESULTS, "per_learner_theory_%s.csv.gz" % kind))
    conc = []
    for i, lab in enumerate(BINLAB):
        m = (PT["bin"].values == i) & (PT["heldout_trials"].values > 0)
        gain = (PT["L_pop"] - PT["L_shrunk"]).values[m]; tr_ = PT["heldout_trials"].values[m]
        k = max(1, int(round(0.01 * m.sum())))
        top = np.argsort(-np.abs(gain))[:k]
        keep = np.ones(m.sum(), bool); keep[top] = False
        conc.append(dict(bin=lab, share_of_gain_top1pct=r(gain[top].sum() / gain.sum(), 3) if gain.sum() != 0 else None,
                         gain_without_top1pct=r(gain[keep].sum() / tr_[keep].sum(), 7),
                         naive_gain_without_top1pct=r(((PT["L_pop"] - PT["L_naive"]).values[m][keep]).sum() / tr_[keep].sum(), 7)))
    OUT[kind + "_shrunk_gain_concentration"] = conc
# level control with drift: Eq. (2a) using the first/second-half latent correlation of the level as kappa
if lv:
    OUT["level_drift_adjusted_prediction"] = [
        dict(bin=a["bin"], kappa=c["latent_corr_intercept"],
             predicted=r(a["predicted_level_gain"] * (2 * c["latent_corr_intercept"] - 1), 6))
        for a, c in zip(lv, [x for x in raw["latent_split"]["chrono"] if x["bin"] != "all"])]
OUT["adj_max_median_learner_gain_shrunk"] = r(max(b["median_learner_gain_shrunk"] for b in tha["bins"]), 7)
PR = J("RESULTS_v3_sim_probe_m1.json")
OUT["probe_nstar_ratio"] = r(PR["observed_design"]["bins"][-1]["n_star_050_est"] / PR["probing_design_2x_spread"]["bins"][-1]["n_star_050_est"], 2)
OUT["probe_headroom_ratio"] = r(PR["probing_design_2x_spread"]["personalization"]["observed_oracle"] / PR["observed_design"]["personalization"]["observed_oracle"], 2)
OUT["bins_min_learners"] = int(min(b["n_learners"] for b in raw["bins"])); OUT["bins_max_learners"] = int(max(b["n_learners"] for b in raw["bins"]))
OUT["sep_pearson_drop_without_top1pct"] = r(sep["v2_estimator"]["pearson_sb"] - sep["v2_estimator"]["pearson_sb_without_top1pct"], 4)
json.dump(OUT, open(os.path.join(RESULTS, "RESULTS_v3_derived.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(OUT, indent=1)[:3000])
