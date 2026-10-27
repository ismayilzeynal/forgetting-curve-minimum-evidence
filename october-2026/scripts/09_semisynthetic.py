# -*- coding: utf-8 -*-
# Semi-synthetic validation (simulation, not data). Every sampled learner keeps the real design:
# trace order, log lags (centered as in the analysis), session sizes, and the training/held-out
# split. Outcomes are binomial draws from a known logistic model with the learner's real EB
# intercept and a slope b_i ~ N(mu_b, tau2_sim). For each tau2_sim the hierarchical pipeline is
# rerun: recovery of tau2 and of reliability (model-based versus the squared correlation of the
# estimate with the truth), predicted versus observed personalization gains (naive, shrunk, oracle),
# and n*. A probing design doubles each learner's log-lag spread (z -> 2 z) at the same n.
# A half-life world checks scheduling consequences with known multipliers.
# Usage: python 09_semisynthetic.py <cache_dir> <part: grid|probe|hlr> [multipliers...]
import sys, os, json, time
import numpy as np
from common import *
from common_eb import *
from common_eb import hlr_terms  # unclipped multiplier likelihood

CACHE, PART = sys.argv[1], sys.argv[2]
D = load(CACHE); G = D["G"]; g = D["g"]; n = D["n"]; pos = D["pos"]
PM = np.load(os.path.join(CACHE, "popmodels.npz")); s_pop = PM["s_pop"]
ss, zc, dd, ll = D["ss"], D["zc"], D["delta_days"], D["ll"]
ssi = np.round(ss).astype(np.int64)
bins = binof(n)
H = json.load(open(os.path.join(RESULTS, "RESULTS_v3_hier_raw.json")))
mu_hat = np.array(H["overall"]["mu"]); tau2_hat = float(H["overall"]["tau2"])
k = np.floor(n * 0.8).astype(int); tr = pos < k[g]; ho = ~tr
nn = np.bincount(g[ho], ss[ho], G)
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


# real intercepts from the observed-data EB fit (recall level kept realistic)
import pandas as pd
PLH = pd.read_csv(os.path.join(RESULTS, "per_learner_hier_raw.csv.gz"))
a_true = PLH["mode_intercept"].values


def simulate(btrue, z, seed):
    rg = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-np.clip(a_true[g] + btrue[g] * z, -30, 30)))
    return rg.binomial(ssi, p).astype(float)


def pipeline(sc_, z, btrue):
    out = dict()
    X = np.column_stack([np.ones(len(g)), z])
    R = eb_logistic(g, X, sc_, ss, G)
    tau2 = float(R["Sigma"][1, 1]); rho = 1 - R["V"][:, 1, 1] / tau2
    lv = lik_var(R["Hlik"], 1)
    out["tau2_est"] = r(tau2, 7)
    rows = []
    for i, lab in enumerate(BINLAB + ["all"]):
        m = (bins == i) if lab != "all" else np.ones(G, bool)
        s1 = float(np.median((n * lv)[m & np.isfinite(lv)]))
        c = np.corrcoef(R["theta"][m, 1], btrue[m])[0, 1] if np.std(btrue) > 0 else float("nan")
        rows.append(dict(bin=lab, mean_rho_model=r(np.mean(rho[m]), 4), rho_true=r(c ** 2 if c == c else None, 4),
                         per_review_error_var=r(s1, 5),
                         n_star_050_est=r(s1 / tau2, 1), n_star_050_true=r(s1 / np.var(btrue) if np.var(btrue) > 0 else None, 1)))
    out["bins"] = rows
    # personalization on the split, overall EB on training portions (zt = z re-centered on training)
    zt = z - (np.bincount(g[tr], z[tr], G) / k)[g]
    Xt = np.column_stack([np.ones(tr.sum()), zt[tr]])
    Rt = eb_logistic(g[tr], Xt, sc_[tr], ss[tr], G)
    mu, S = Rt["mu"], Rt["Sigma"]; t2 = float(S[1, 1])
    th_nv, H_nv = weak_prior_fit(g[tr], Xt, sc_[tr], ss[tr], G, mu, [2.5, 1.0])
    b_eb = Rt["theta"][:, 1]; b_nv = th_nv[:, 1]
    s2 = lik_var(Rt["Hlik"], 1); w_eb = 1 - Rt["V"][:, 1, 1] / t2
    I_nv = 1 / lik_var(H_nv, 1); w_nv = I_nv / (I_nv + 1.0)
    c = S[0, 1] / S[1, 1]; pa = 1 / (S[0, 0] - S[0, 1] ** 2 / S[1, 1])

    def icpt(s):
        m_a = mu[0] + c * (s - mu[1]); a = m_a.copy()
        gg = g[tr]; z_ = zt[tr]; c_ = sc_[tr]; s_ = ss[tr]
        for _ in range(50):
            p = 1 / (1 + np.exp(-np.clip(a[gg] + s[gg] * z_, -30, 30)))
            st = (np.bincount(gg, c_ - s_ * p, G) - pa * (a - m_a)) / (np.bincount(gg, s_ * p * (1 - p), G) + pa)
            a += st
            if np.max(np.abs(st)) < 1e-10:
                break
        p = 1 / (1 + np.exp(-np.clip(a[gg] + s[gg] * z_, -30, 30))); W = s_ * p * (1 - p)
        return a, -(np.bincount(gg, W * z_, G) - pa * c) / (np.bincount(gg, W, G) + pa)

    def Lho(s):
        a, _ = icpt(s)
        return np.bincount(g[ho], logloss_rows(a[g[ho]] + s[g[ho]] * zt[ho], sc_[ho], ss[ho]), G)
    bp = np.full(G, mu[1])
    a_p, dads = icpt(bp)
    p = 1 / (1 + np.exp(-np.clip(a_p[g[ho]] + mu[1] * zt[ho], -30, 30)))
    J = np.bincount(g[ho], ss[ho] * p * (1 - p) * (zt[ho] + dads[g[ho]]) ** 2, G)
    Lp, Ln, Ls, Lo = Lho(bp), Lho(b_nv), Lho(b_eb), Lho(btrue)

    def pred(w):
        s2f = np.where(np.isfinite(s2), s2, 0.0)
        return np.where(np.isfinite(s2), 0.5 * J * (2 * w * t2 - w ** 2 * (t2 + s2f)), 0.0)
    v = nn > 0; tot = nn[v].sum()
    out["personalization"] = dict(
        tau2_train=r(t2, 7),
        observed_naive=r((Lp[v] - Ln[v]).sum() / tot, 7), predicted_naive=r(pred(w_nv)[v].sum() / tot, 7),
        observed_shrunk=r((Lp[v] - Ls[v]).sum() / tot, 7), predicted_shrunk=r((0.5 * J * t2 * np.clip(w_eb, 0, 1))[v].sum() / tot, 7),
        observed_oracle=r((Lp[v] - Lo[v]).sum() / tot, 7), predicted_oracle=r((0.5 * J * t2)[v].sum() / tot, 7),
        n_learners=int(v.sum()))
    return out


OUTF = os.path.join(RESULTS, "RESULTS_v3_sim_%s.json" % PART)
OUT = dict(note="Simulation on real designs; outcomes are binomial draws from a known model, not data.",
           tau2_hat=tau2_hat, mu_hat=list(mu_hat))
if PART == "grid":
    mults = [float(x) for x in sys.argv[3:]] or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    OUT["grid"] = []
    for j, mult in enumerate(mults):
        rg = np.random.default_rng(SEED + 100 + int(mult * 10))
        bt = mu_hat[1] + np.sqrt(mult * tau2_hat) * rg.standard_normal(G)
        sc_ = simulate(bt, zc, SEED + 200 + int(mult * 10))
        res = pipeline(sc_, zc, bt); res["multiplier"] = mult; res["tau2_true"] = r(mult * tau2_hat, 7)
        res["tau2_true_sample"] = r(float(np.var(bt)), 7)
        OUT["grid"].append(res); log(mult, res["tau2_est"], res["bins"][-1], res["personalization"])
        dump(OUT, os.path.basename(OUTF).replace("RESULTS_v3_sim_grid", "RESULTS_v3_sim_grid_" + "_".join(str(m) for m in mults)) if len(sys.argv) > 3 else os.path.basename(OUTF))
elif PART == "probe":
    mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    rg = np.random.default_rng(SEED + 300)
    bt = mu_hat[1] + np.sqrt(mult * tau2_hat) * rg.standard_normal(G)
    for name, z in (("observed_design", zc), ("probing_design_2x_spread", 2 * zc)):
        sc_ = simulate(bt, z, SEED + 301)
        OUT[name] = pipeline(sc_, z, bt); OUT[name]["multiplier"] = mult
        log(name, OUT[name]["tau2_est"], OUT[name]["bins"][-1], OUT[name]["personalization"])
        dump(OUT, "RESULTS_v3_sim_probe_m%g.json" % mult)
elif PART == "hlr":
    HH = json.load(open(os.path.join(RESULTS, "RESULTS_v3_hier_hlr.json")))
    mu_u = float(HH["overall"]["mu"][0]); tau2_u = float(HH["overall"]["tau2"])
    rg = np.random.default_rng(SEED + 400)
    u_true = mu_u + np.sqrt(tau2_u) * rg.standard_normal(G)
    _, _, _, p_sim = hlr_terms(s_pop + u_true[g], dd, np.zeros(len(g)), ss)
    sc_h = rg.binomial(ssi, p_sim).astype(float)
    R = eb_hlr(g[tr], s_pop[tr], dd[tr], sc_h[tr], ss[tr], G)
    mu = float(R["mu"][0]); u_eb = R["theta"][:, 0]
    u_nv, _ = weak_prior_hlr(g[tr], s_pop[tr], dd[tr], sc_h[tr], ss[tr], G, mu, 2.0)
    pol = dict(population=np.full(G, mu), naive=u_nv, shrunk=u_eb, oracle=u_true)
    h_ref = float(np.median(np.exp2(np.clip(s_pop[ho] + mu_u, S_MIN, S_MAX))))
    res = dict(tau2_u_true=r(tau2_u, 6), tau2_u_est_train=r(float(R["Sigma"][0, 0]), 6),
               reference_half_life_days=r(h_ref, 3), n_learners=int(G))
    for rs_ in (0.85, 0.90):
        for name, up in pol.items():
            achieved = rs_ ** np.exp2(up - u_true)
            interval = -h_ref * np.exp2(up - mu) * np.log2(rs_)
            res["r%.2f_%s" % (rs_, name)] = dict(mean_abs_dev=r(np.mean(np.abs(achieved - rs_)), 4),
                                                share_dev_gt_0_05=r(np.mean(np.abs(achieved - rs_) > 0.05), 4),
                                                mean_achieved=r(np.mean(achieved), 4),
                                                reviews_per_30_days=r(np.mean(30.0 / interval), 3))
    OUT["hlr_world"] = res; log(res)
    dump(OUT, "RESULTS_v3_sim_hlr.json")
log("done")
