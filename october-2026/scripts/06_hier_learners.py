# -*- coding: utf-8 -*-
# Hierarchical (empirical Bayes) analysis of learner parameters, added after the addendum's
# moment-based decomposition proved outlier-driven (see README and BUILD-REPORT). For one
# parameter family per run (raw or strength-adjusted logistic rate with its intercept, or the
# HLR half-life multiplier) it estimates: the between-learner variance (overall and per bin, with
# learner-bootstrap CIs), model-based reliability of each learner's full-history estimate, the
# per-review error variance and the minimum evidence n* for a target reliability, latent
# (error-free) split-half correlations for odd/even and first/second-half splits, a within-learner
# lag-permutation null for the variance, and robust observed split-half reliabilities.
# Usage: python 06_hier_learners.py <cache_dir> <raw|adj|hlr>
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import *
from common_eb import *

CACHE, KIND = sys.argv[1], sys.argv[2]
D = load(CACHE); G = D["G"]; g = D["g"]; n = D["n"]; pos = D["pos"]
PM = np.load(os.path.join(CACHE, "popmodels.npz"))
sc, ss, zc, dd = D["sc"], D["ss"], D["zc"], D["delta_days"]
bins = binof(n)
rng = np.random.default_rng(SEED + 11 + ["raw", "adj", "hlr"].index(KIND))
NB = 500
PRIOR_SD = dict(a=2.5, b=1.0, u=2.0)   # fixed weak priors for the individual ("naive") estimates
off = PM["off_adj"] if KIND == "adj" else np.zeros(len(g))
s_pop = PM["s_pop"]
OUT = dict(kind=KIND, n_learners=int(G), n_traces=int(len(g)))
T0 = time.time()


def log(*a):
    print("[%7.0fs]" % (time.time() - T0), *a, flush=True)


def sbr(x, y, method="spearman"):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return None
    c = spearmanr(x[m], y[m]).correlation if method == "spearman" else np.corrcoef(x[m], y[m])[0, 1]
    return 2 * c / (1 + c) if c > -1 else None


def ci(vals):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    return [r(np.percentile(v, 2.5), 6), r(np.percentile(v, 97.5), 6)]


def fit(rows, groups=None, z=None, init=None):
    z = zc if z is None else z
    if KIND == "hlr":
        R = eb_hlr(g[rows], s_pop[rows], dd[rows], sc[rows], ss[rows], G, groups=groups,
                   mu0=0.0 if init is None else float(init["mu"][0]),
                   tau2_0=0.25 if init is None else float(init["Sigma"][0, 0]))
        return R
    X = np.column_stack([np.ones(rows.sum()), z[rows]])
    return eb_logistic(g[rows], X, sc[rows], ss[rows], G, off=off[rows], groups=groups,
                       mu0=None if init is None else init["mu"], Sigma0=None if init is None else init["Sigma"],
                       theta0=None if init is None else init["theta"])


IDX = 0 if KIND == "hlr" else 1     # index of the rate-like parameter


def summarize(R, mask, label, boot=True):
    S = R["Sigma"]; tau2 = float(S[IDX, IDX])
    lv = lik_var(R["Hlik"], IDX) if KIND != "hlr" else np.where(R["Hlik"][:, 0, 0] > 1e-12, 1.0 / np.maximum(R["Hlik"][:, 0, 0], 1e-300), np.inf)
    rho = 1.0 - R["V"][:, IDX, IDX] / tau2
    m = mask & R["use"]
    s1 = float(np.median((n * lv)[m & np.isfinite(lv)]))
    res = dict(label=label, n_learners=int(m.sum()), mu=[r(x, 6) for x in R["mu"]], Sigma=[[r(x, 7) for x in row] for row in S],
               tau2=r(tau2, 7), tau=r(np.sqrt(tau2), 6), converged=bool(R["converged"]),
               mean_rho=r(np.mean(rho[m]), 4), median_rho=r(np.median(rho[m]), 4),
               share_rho_ge_050=r(np.mean(rho[m] >= 0.5), 4), share_rho_ge_070=r(np.mean(rho[m] >= 0.7), 4),
               per_review_error_var=r(s1, 5), share_finite_info=r(np.mean(np.isfinite(lv[m])), 4),
               n_star_050=r(s1 / tau2 * 1.0, 1), n_star_070=r(s1 / tau2 * 0.7 / 0.3, 1),
               median_n=r(np.median(n[m]), 1))
    if KIND != "hlr":
        res["tau2_intercept"] = r(float(S[0, 0]), 6)
        res["corr_intercept_rate"] = r(latent_corr(S, 0, 1), 4)
        rho_a = 1.0 - R["V"][:, 0, 0] / S[0, 0]
        res["mean_rho_intercept"] = r(np.mean(rho_a[m]), 4); res["median_rho_intercept"] = r(np.median(rho_a[m]), 4)
    if boot:
        bs = bootstrap_hyper(R, rng, nb=NB, groups_mask=mask)
        t2 = np.array([b[1][IDX, IDX] for b in bs])
        res["tau2_ci"] = ci(t2)
        res["n_star_050_ci"] = ci(s1 / t2); res["n_star_070_ci"] = ci(s1 / t2 * 0.7 / 0.3)
        res["mu_rate_ci"] = ci([b[0][IDX] for b in bs])
        if KIND != "hlr":
            res["tau2_intercept_ci"] = ci([b[1][0, 0] for b in bs])
            res["corr_intercept_rate_ci"] = ci([latent_corr(b[1], 0, 1) for b in bs])
    return res, rho, lv


# ---------- 1. overall and per-bin fits ----------
allrows = np.ones(len(g), bool)
R = fit(allrows)
log("overall fit", R["mu"], np.diag(R["Sigma"]), R["converged"])
res_all, rho_all, lv_all = summarize(R, np.ones(G, bool), "all")
OUT["overall"] = res_all
PL = pd.DataFrame(dict(uid=D["uid"], n=n, bin=bins, rho_overall=rho_all, lik_var=lv_all,
                       mode_rate=R["theta"][:, IDX]))
if KIND != "hlr":
    PL["mode_intercept"] = R["theta"][:, 0]
OUT["bins"] = []
for i, lab in enumerate(BINLAB):
    gm = bins == i
    Rb = fit(gm[g], groups=gm, init=R)
    rb, rho_b, _ = summarize(Rb, gm, lab)
    PL.loc[gm, "rho_bin"] = rho_b[gm]
    OUT["bins"].append(rb)
    log("bin", lab, rb["tau2"], rb["tau2_ci"], rb["mean_rho"], rb["n_star_050"])
dump(OUT, "RESULTS_v3_hier_%s.json" % KIND)

PL.to_csv(os.path.join(RESULTS, "per_learner_hier_%s.csv.gz" % KIND), index=False, compression="gzip")


def jackknife_ci(R4, gm, fn, k=10):
    """Delete-a-group jackknife (k random learner groups) of fn(Sigma), re-estimated from the
    stored likelihood expansions; normal-theory 95% interval."""
    idx = np.where(gm & R4["use"])[0]
    grp = rng.integers(0, k, len(idx))
    vals = []
    for j in range(k):
        keep = np.zeros(G, bool); keep[idx[grp != j]] = True
        _, S = subset_hyper(R4, keep)
        vals.append(fn(S))
    vals = np.array(vals); est_ = fn(R4["Sigma"])
    se = np.sqrt((k - 1) / k * np.sum((vals - vals.mean()) ** 2))
    return [r(est_ - 1.96 * se, 4), r(min(est_ + 1.96 * se, 1.0), 4)], r(se, 5)


# ---------- 2. latent split-half correlations (logistic families) ----------
# odd/even halves share the calendar period, so their latent correlation checks that the
# between-learner variance is not fitted noise; first/second halves test stability over time.
if KIND != "hlr":
    OUT["latent_split"] = {}
    for split in ("oddeven", "chrono"):
        if split == "oddeven":
            m0 = pos % 2 == 0
        else:
            m0 = pos < (n // 2)[g]
        X4 = np.column_stack([m0 * 1.0, m0 * zc, (~m0) * 1.0, (~m0) * zc])
        rows_out = []
        labels = ["all"] if split == "oddeven" else BINLAB + ["all"]
        for lab in labels:
            gm = (bins == BINLAB.index(lab)) if lab != "all" else np.ones(G, bool)
            rr = gm[g]
            R4 = eb_logistic(g[rr], X4[rr], sc[rr], ss[rr], G, off=off[rr], groups=gm, switch=1e-2)
            S4 = R4["Sigma"]
            ci_b, se_b = jackknife_ci(R4, gm, lambda S: latent_corr(S, 1, 3))
            ci_a, se_a = jackknife_ci(R4, gm, lambda S: latent_corr(S, 0, 2))
            row = dict(bin=lab, n_learners=int((gm & R4["use"]).sum()), converged=bool(R4["converged"]),
                       latent_corr_rate=r(latent_corr(S4, 1, 3), 4), latent_corr_rate_ci=ci_b, latent_corr_rate_se=se_b,
                       latent_corr_intercept=r(latent_corr(S4, 0, 2), 4), latent_corr_intercept_ci=ci_a,
                       latent_corr_intercept_se=se_a,
                       tau2_rate_half=[r(S4[1, 1], 7), r(S4[3, 3], 7)], cov_rate_halves=r(S4[1, 3], 7),
                       tau2_intercept_half=[r(S4[0, 0], 6), r(S4[2, 2], 6)])
            rows_out.append(row)
            log("latent", split, lab, row["latent_corr_rate"], row["latent_corr_rate_ci"], row["latent_corr_intercept"])
            OUT["latent_split"][split] = rows_out
            dump(OUT, "RESULTS_v3_hier_%s.json" % KIND)

# ---------- 3. lag-permutation null for the between-learner variance ----------
OUT["permutation_null"] = []
for k in range(3):
    if KIND == "hlr":
        dp = permute_within(g, dd, rng)
        Rp = eb_hlr(g, s_pop, dp, sc, ss, G, mu0=float(R["mu"][0]), tau2_0=float(R["Sigma"][0, 0]))
    else:
        zp = permute_within(g, zc, rng)
        X = np.column_stack([np.ones(len(g)), zp])
        Rp = eb_logistic(g, X, sc, ss, G, off=off)
    rowp = dict(rep=k, tau2=r(float(Rp["Sigma"][IDX, IDX]), 7), mu_rate=r(float(Rp["mu"][IDX]), 6),
                bins_tau2_approx=[r(float(subset_hyper(Rp, bins == i)[1][IDX, IDX]), 7) for i in range(len(BINS))])
    if KIND != "hlr":
        rowp["tau2_intercept"] = r(float(Rp["Sigma"][0, 0]), 6)
    OUT["permutation_null"].append(rowp)
    log("null", rowp)
dump(OUT, "RESULTS_v3_hier_%s.json" % KIND)

# ---------- 4. robust observed split-half reliability of weak-prior individual estimates ----------
OUT["observed_split_half"] = {}
for split in ("oddeven", "chrono"):
    m0 = (pos % 2 == 0) if split == "oddeven" else (pos < (n // 2)[g])
    ests = []
    for m in (m0, ~m0):
        if KIND == "hlr":
            u, _ = weak_prior_hlr(g[m], s_pop[m], dd[m], sc[m], ss[m], G, float(R["mu"][0]), PRIOR_SD["u"])
            ests.append(u)
        else:
            X = np.column_stack([np.ones(m.sum()), zc[m]])
            th, _ = weak_prior_fit(g[m], X, sc[m], ss[m], G, R["mu"], [PRIOR_SD["a"], PRIOR_SD["b"]], off=off[m])
            ests.append(th)
    rows_out = []
    for i, lab in enumerate(BINLAB + ["all"]):
        gm = (bins == i) if lab != "all" else np.ones(G, bool)
        if KIND == "hlr":
            row = dict(bin=lab, spearman_sb=r(sbr(ests[0][gm], ests[1][gm]), 4), pearson_sb=r(sbr(ests[0][gm], ests[1][gm], "pearson"), 4))
        else:
            row = dict(bin=lab, spearman_sb_rate=r(sbr(ests[0][gm, 1], ests[1][gm, 1]), 4),
                       pearson_sb_rate=r(sbr(ests[0][gm, 1], ests[1][gm, 1], "pearson"), 4),
                       spearman_sb_intercept=r(sbr(ests[0][gm, 0], ests[1][gm, 0]), 4))
        rows_out.append(row)
    OUT["observed_split_half"][split] = rows_out
    log("observed", split, rows_out[-1])
OUT["weak_prior_sd"] = PRIOR_SD

# ---------- 5. design mechanism: lag spread and per-review information ----------
if KIND != "hlr":
    sdz = np.sqrt(np.bincount(g, zc * zc, G) / n)
    mss = np.bincount(g, ss, G) / n
    eta = R["theta"][g, 0] + R["theta"][g, 1] * zc + off
    p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    mpq = np.bincount(g, p * (1 - p), G) / n
    ok = np.isfinite(lv_all) & (sdz > 0)
    Xr = np.column_stack([np.ones(ok.sum()), np.log(sdz[ok]), np.log(mss[ok]), np.log(np.maximum(mpq[ok], 1e-6))])
    coef = np.linalg.lstsq(Xr, np.log(n[ok] * lv_all[ok]), rcond=None)[0]
    OUT["design_mechanism"] = dict(
        per_bin=[dict(bin=lab, median_sd_loglag=r(np.median(sdz[bins == i]), 4), mean_p1mp=r(np.mean(mpq[bins == i]), 4),
                      mean_session_size=r(np.mean(mss[bins == i]), 3),
                      median_per_review_error_var=r(np.median((n * lv_all)[(bins == i) & ok]), 4)) for i, lab in enumerate(BINLAB)],
        regression_log_per_review_var=dict(log_sd_loglag=r(coef[1], 4), log_session_size=r(coef[2], 4),
                                           log_p1mp=r(coef[3], 4), n=int(ok.sum())),
        spearman_sd_loglag_vs_n=r(spearmanr(sdz, n).correlation, 4))
    PL["sd_loglag"] = sdz
dump(OUT, "RESULTS_v3_hier_%s.json" % KIND)
PL.to_csv(os.path.join(RESULTS, "per_learner_hier_%s.csv.gz" % KIND), index=False, compression="gzip")
log("done")
