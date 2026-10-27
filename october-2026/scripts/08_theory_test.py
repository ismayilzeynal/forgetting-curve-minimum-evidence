# -*- coding: utf-8 -*-
# Personalization test and theory test with hierarchical estimators. Each learner's first 80% of
# traces is training, the last 20% held out. Per review-count bin, an empirical-Bayes model is fit
# on the training portions; three policies are compared on held-out binomial log-loss:
#   population (bin mean), naive (individual estimate under a fixed weak prior), shrunk (EB mode).
# The theory predicts each learner's expected gain 1/2 J (2 w tau2 - w^2 (tau2 + sigma2)) from
# training-only quantities (tau2, sigma2, shrinkage weight w) and the held-out Fisher information J.
# A drift-aware variant uses the latent covariance between training and held-out rates (it uses
# held-out data and is labelled as a diagnostic). For the logistic families the run also reports the
# level (intercept) personalization as a positive control. For HLR it adds scheduling consequences.
# Usage: python 08_theory_test.py <cache_dir> <raw|adj|hlr>
import sys, os, time
import numpy as np, pandas as pd
from common import *
from common_eb import *
from common_eb import hlr_terms  # unclipped multiplier likelihood

CACHE, KIND = sys.argv[1], sys.argv[2]
D = load(CACHE); G = D["G"]; g = D["g"]; n = D["n"]; pos = D["pos"]
PM = np.load(os.path.join(CACHE, "popmodels.npz")); s_pop = PM["s_pop"]
sc, ss, dd = D["sc"], D["ss"], D["delta_days"]
bins = binof(n)
rng = np.random.default_rng(SEED + 31 + ["raw", "adj", "hlr"].index(KIND))
NB = 1000
off = PM["off_adj"] if KIND == "adj" else np.zeros(len(g))
k = np.floor(n * 0.8).astype(int); tr = pos < k[g]; ho = ~tr
mtr = np.bincount(g[tr], D["ll"][tr], G) / k
zt = D["ll"] - mtr[g]
nn = np.bincount(g[ho], ss[ho], G)
T0 = time.time()
OUT = dict(kind=KIND, note="gains are held-out log-loss reductions per trial (nats) relative to the population policy")
PL = pd.DataFrame(dict(uid=D["uid"], n=n, bin=bins, heldout_trials=nn))


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


def ratio_ci(num, den, idx):
    vals = []
    for _ in range(NB):
        j = rng.choice(idx, len(idx), replace=True)
        vals.append(num[j].sum() / den[j].sum())
    return [r(np.percentile(vals, 2.5), 7), r(np.percentile(vals, 97.5), 7)]


def gain_pred(J, w, tau2, s2, C=None, tau2_t=None):
    C = tau2 if C is None else C
    t = tau2 if tau2_t is None else tau2_t
    with np.errstate(invalid="ignore"):
        s2f = np.where(np.isfinite(s2), s2, 0.0)
        val = 0.5 * J * (2 * w * C - w ** 2 * (t + s2f))
    return np.where(np.isfinite(s2), val, 0.0)


def intercept_given_slope(rows, s, mu, S, it=50):
    """Training intercept for slope s under the conditional prior a | b = s from N(mu, S)."""
    c = S[0, 1] / S[1, 1]; pa = 1.0 / (S[0, 0] - S[0, 1] ** 2 / S[1, 1])
    m_a = mu[0] + c * (s - mu[1])
    a = m_a.copy()
    gg = g[rows]; z = zt[rows]; o = off[rows]; c_ = sc[rows]; s_ = ss[rows]
    for _ in range(it):
        eta = np.clip(a[gg] + s[gg] * z + o, -30, 30); p = 1 / (1 + np.exp(-eta))
        W = s_ * p * (1 - p)
        step = (np.bincount(gg, c_ - s_ * p, G) - pa * (a - m_a)) / (np.bincount(gg, W, G) + pa)
        a += step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(a[gg] + s[gg] * z + o, -30, 30); p = 1 / (1 + np.exp(-eta)); W = s_ * p * (1 - p)
    dads = -(np.bincount(gg, W * z, G) - pa * c) / (np.bincount(gg, W, G) + pa)
    return a, dads, pa


def heldout_loss(a, s):
    eta = a[g[ho]] + s[g[ho]] * zt[ho] + off[ho]
    return np.bincount(g[ho], logloss_rows(eta, sc[ho], ss[ho]), G)


rows_out = []; level_rows = []; sched = None
for i, lab in enumerate(BINLAB):
    gm = bins == i
    rr = tr & gm[g]
    if KIND == "hlr":
        R = eb_hlr(g[rr], s_pop[rr], dd[rr], sc[rr], ss[rr], G, groups=gm)
        mu = float(R["mu"][0]); tau2 = float(R["Sigma"][0, 0])
        u_eb = R["theta"][:, 0]
        u_nv, I_nv = weak_prior_hlr(g[rr], s_pop[rr], dd[rr], sc[rr], ss[rr], G, mu, 2.0)
        I_tr = R["Hlik"][:, 0, 0]
        s2 = np.where(I_tr > 1e-12, 1.0 / np.maximum(I_tr, 1e-300), np.inf)
        w_eb = 1.0 - R["V"][:, 0, 0] / tau2
        w_nv = I_nv / (I_nv + 0.25)

        def L(u):
            loss, _, _, _ = hlr_terms(s_pop[ho] + u[g[ho]], dd[ho], sc[ho], ss[ho])
            return np.bincount(g[ho], loss, G)
        _, _, info, _ = hlr_terms(s_pop[ho] + mu, dd[ho], sc[ho], ss[ho])
        J = np.bincount(g[ho], info, G)
        Lp = L(np.full(G, mu)); Ln = L(u_nv); Ls = L(u_eb)
        C = tau2_t = None
        PL.loc[gm, "u_eb"] = u_eb[gm]; PL.loc[gm, "u_naive"] = u_nv[gm]; PL.loc[gm, "u_pop"] = mu
        PL.loc[gm, "w_eb"] = w_eb[gm]; PL.loc[gm, "tau2_bin"] = tau2
    else:
        X = np.column_stack([np.ones(rr.sum()), zt[rr]])
        R = eb_logistic(g[rr], X, sc[rr], ss[rr], G, off=off[rr], groups=gm)
        mu, S = R["mu"], R["Sigma"]; tau2 = float(S[1, 1])
        th_nv, H_nv = weak_prior_fit(g[rr], X, sc[rr], ss[rr], G, mu, [2.5, 1.0], off=off[rr])
        b_eb = R["theta"][:, 1]; b_nv = th_nv[:, 1]; b_pop = np.full(G, mu[1])
        s2 = lik_var(R["Hlik"], 1)
        w_eb = 1.0 - R["V"][:, 1, 1] / tau2
        I_nv = 1.0 / lik_var(H_nv, 1)
        w_nv = I_nv / (I_nv + 1.0)
        a_p, dads, pa = intercept_given_slope(rr, b_pop, mu, S)
        a_n, _, _ = intercept_given_slope(rr, b_nv, mu, S)
        a_s, _, _ = intercept_given_slope(rr, b_eb, mu, S)
        Lp = heldout_loss(a_p, b_pop); Ln = heldout_loss(a_n, b_nv); Ls = heldout_loss(a_s, b_eb)
        eta = np.clip(a_p[g[ho]] + mu[1] * zt[ho] + off[ho], -30, 30); p = 1 / (1 + np.exp(-eta))
        J = np.bincount(g[ho], ss[ho] * p * (1 - p) * (zt[ho] + dads[g[ho]]) ** 2, G)
        # level (intercept) personalization as a positive control: fully pooled (mu_a, mu_b)
        # versus the learner's conditional EB intercept with the population slope
        Lpool = heldout_loss(np.full(G, mu[0]), b_pop)
        Ja = np.bincount(g[ho], ss[ho] * p * (1 - p), G)
        Haa = R["Hlik"][:, 0, 0]
        tau2a = float(S[0, 0] - S[0, 1] ** 2 / S[1, 1])
        wa = Haa / (Haa + 1.0 / tau2a)
        s2a = np.where(Haa > 1e-12, 1.0 / np.maximum(Haa, 1e-300), np.inf)
        idx = np.where(gm & (nn > 0))[0]
        pa_pred = gain_pred(Ja, wa, tau2a, s2a)
        level_rows.append(dict(bin=lab, n_learners=int(len(idx)),
                               observed_level_gain=r((Lpool[idx] - Lp[idx]).sum() / nn[idx].sum(), 6),
                               observed_level_gain_ci=ratio_ci(Lpool - Lp, nn, idx),
                               predicted_level_gain=r(pa_pred[idx].sum() / nn[idx].sum(), 6),
                               predicted_level_oracle=r((0.5 * Ja[idx] * tau2a).sum() / nn[idx].sum(), 6),
                               tau2_intercept_conditional=r(tau2a, 6)))
        # drift-aware diagnostic: latent covariance of training and held-out rates
        m0 = tr
        X4 = np.column_stack([m0 * 1.0, m0 * zt, (~m0) * 1.0, (~m0) * zt])
        rr4 = gm[g]
        R4 = eb_logistic(g[rr4], X4[rr4], sc[rr4], ss[rr4], G, off=off[rr4], groups=gm, switch=1e-2)
        C = float(R4["Sigma"][1, 3]); tau2_t = float(R4["Sigma"][1, 1])
        PL.loc[gm, "b_eb"] = b_eb[gm]; PL.loc[gm, "b_naive"] = b_nv[gm]; PL.loc[gm, "b_pop"] = mu[1]
        PL.loc[gm, "w_eb"] = w_eb[gm]; PL.loc[gm, "tau2_bin"] = tau2
    idx = np.where(gm & (nn > 0))[0]
    # shrunk policy: the EB estimate is a posterior mean, so its expected squared error is the
    # posterior variance and the predicted gain is 1/2 J tau2 rho_i with rho_i = 1 - V_i/tau2
    # (this also covers information the rate borrows from the intercept through the prior)
    w_eb = np.clip(w_eb, 0.0, 1.0)
    pn = gain_pred(J, w_nv, tau2, s2); pe = 0.5 * J * tau2 * w_eb; po = 0.5 * J * tau2
    PL.loc[gm, "J"] = J[gm]; PL.loc[gm, "sigma2"] = s2[gm]; PL.loc[gm, "w_naive"] = w_nv[gm]
    PL.loc[gm, "L_pop"] = Lp[gm]; PL.loc[gm, "L_naive"] = Ln[gm]; PL.loc[gm, "L_shrunk"] = Ls[gm]
    PL.loc[gm, "pred_naive"] = pn[gm]; PL.loc[gm, "pred_shrunk"] = pe[gm]; PL.loc[gm, "pred_oracle"] = po[gm]
    tot = nn[idx].sum()
    row = dict(bin=lab, n_learners=int(len(idx)), heldout_trials=int(tot), tau2_train=r(tau2, 7),
               population_value=r(float(mu) if KIND == "hlr" else float(mu[1]), 6),
               base_logloss_per_trial=r(Lp[idx].sum() / tot, 5),
               observed_naive=r((Lp[idx] - Ln[idx]).sum() / tot, 7), observed_naive_ci=ratio_ci(Lp - Ln, nn, idx),
               observed_shrunk=r((Lp[idx] - Ls[idx]).sum() / tot, 7), observed_shrunk_ci=ratio_ci(Lp - Ls, nn, idx),
               predicted_naive=r(pn[idx].sum() / tot, 7), predicted_naive_ci=ratio_ci(pn, nn, idx),
               predicted_shrunk=r(pe[idx].sum() / tot, 7), predicted_shrunk_ci=ratio_ci(pe, nn, idx),
               predicted_oracle=r(po[idx].sum() / tot, 7), predicted_oracle_ci=ratio_ci(po, nn, idx),
               mean_J_per_trial=r(J[idx].sum() / tot, 6),
               median_w_naive=r(np.median(w_nv[idx]), 4), median_w_shrunk=r(np.median(w_eb[idx]), 4),
               median_learner_gain_naive=r(np.median((Lp[idx] - Ln[idx]) / nn[idx]), 7),
               median_learner_gain_shrunk=r(np.median((Lp[idx] - Ls[idx]) / nn[idx]), 7))
    if C is not None:
        row["latent_cov_train_heldout"] = r(C, 7); row["latent_tau2_train_4d"] = r(tau2_t, 7)
        row["latent_corr_train_heldout"] = r(C / np.sqrt(tau2_t * float(R4["Sigma"][3, 3])), 4)
        # with drift, b_h = kappa b + noise and kappa = C / tau2_t; for a posterior mean the gain is
        # 1/2 J rho (2 C - tau2_t), negative when kappa < 1/2
        pnd = gain_pred(J, w_nv, tau2, s2, C=C, tau2_t=tau2_t); ped = 0.5 * J * w_eb * (2 * C - tau2_t)
        row["predicted_naive_drift"] = r(pnd[idx].sum() / tot, 7); row["predicted_shrunk_drift"] = r(ped[idx].sum() / tot, 7)
    rows_out.append(row)
    log(lab, {kk: row[kk] for kk in ("tau2_train", "observed_naive", "predicted_naive", "observed_shrunk", "predicted_shrunk", "predicted_oracle")})
    OUT["bins"] = rows_out
    if level_rows:
        OUT["level_control"] = level_rows
        log("level", level_rows[-1])
    dump(OUT, "RESULTS_v3_theory_%s.json" % KIND)

# pooled over bins (trial-weighted) and the predicted-versus-observed agreement
v = PL["heldout_trials"].values > 0
v &= PL["bin"].values >= 0
tot = PL.loc[v, "heldout_trials"].sum()
allrow = dict(bin="all", n_learners=int(v.sum()))
for nm, col in (("observed_naive", "L_naive"), ("observed_shrunk", "L_shrunk")):
    allrow[nm] = r((PL.loc[v, "L_pop"] - PL.loc[v, col]).sum() / tot, 7)
    allrow[nm + "_ci"] = ratio_ci((PL["L_pop"] - PL[col]).values, PL["heldout_trials"].values, np.where(v)[0])
for nm in ("pred_naive", "pred_shrunk", "pred_oracle"):
    allrow[nm.replace("pred", "predicted")] = r(PL.loc[v, nm].sum() / tot, 7)
OUT["all"] = allrow
obs = np.array([[x["observed_naive"], x["observed_shrunk"]] for x in rows_out]).ravel()
prd = np.array([[x["predicted_naive"], x["predicted_shrunk"]] for x in rows_out]).ravel()
OUT["agreement"] = dict(pearson_pred_obs_12cells=r(np.corrcoef(prd, obs)[0, 1], 4),
                       mean_abs_diff=r(np.mean(np.abs(prd - obs)), 7),
                       share_obs_within_pred_ci=r(np.mean([
                           (x["predicted_naive_ci"][0] <= x["observed_naive"] <= x["predicted_naive_ci"][1]) for x in rows_out] + [
                           (x["predicted_shrunk_ci"][0] <= x["observed_shrunk"] <= x["predicted_shrunk_ci"][1]) for x in rows_out]), 4))
log("all", allrow, OUT["agreement"])

if KIND == "hlr":
    vv = (PL["bin"].values >= 0) & np.isfinite(PL["u_eb"].values)
    d_eb = (PL["u_eb"] - PL["u_pop"]).values[vv]; d_nv = (PL["u_naive"] - PL["u_pop"]).values[vv]

    def dist(x):
        return dict(median=r(np.median(x), 4), p5=r(np.percentile(x, 5), 4), p95=r(np.percentile(x, 95), 4),
                    share_outside_0_8_1_25=r(np.mean((x < 0.8) | (x > 1.25)), 4))
    sched = dict(n_learners=int(vv.sum()), interval_ratio_shrunk=dist(np.exp2(d_eb)), interval_ratio_naive=dist(np.exp2(d_nv)))
    for rs_ in (0.85, 0.90):
        # recall reached at the population interval if the learner's true multiplier equals the estimate
        rec_eb = rs_ ** np.exp2(-d_eb); rec_nv = rs_ ** np.exp2(-d_nv)
        sched["recall_at_population_interval_r%.2f" % rs_] = dict(
            shrunk_p5=r(np.percentile(rec_eb, 5), 4), shrunk_p95=r(np.percentile(rec_eb, 95), 4),
            shrunk_mean_abs_dev=r(np.mean(np.abs(rec_eb - rs_)), 4), shrunk_share_dev_gt_0_05=r(np.mean(np.abs(rec_eb - rs_) > 0.05), 4),
            naive_mean_abs_dev=r(np.mean(np.abs(rec_nv - rs_)), 4), naive_share_dev_gt_0_05=r(np.mean(np.abs(rec_nv - rs_) > 0.05), 4))
    OUT["scheduling_real"] = sched
    log("sched", sched)
dump(OUT, "RESULTS_v3_theory_%s.json" % KIND)
PL.to_csv(os.path.join(RESULTS, "per_learner_theory_%s.csv.gz" % KIND), index=False, compression="gzip")
log("done")
