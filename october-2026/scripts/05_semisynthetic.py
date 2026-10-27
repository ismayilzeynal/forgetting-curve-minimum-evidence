# -*- coding: utf-8 -*-
# A6 and A7 (simulation part). Semi-synthetic worlds that keep every sampled learner's real lag
# design, session sizes and trace order, but draw outcomes from a known model:
#  (1) logistic world: slopes b_i ~ N(beta, m * tau2_hat), m in a grid; checks recovery of
#      reliability and tau2, and predicted vs observed personalization gain (naive, shrunk, oracle);
#      plus a probing design with the within-learner log-lag spread doubled;
#  (2) HLR world: half-life multipliers u_i ~ N(beta_u, tau2_u_hat); scheduling consequences.
# Writes results/RESULTS_v3_sim.json. Usage: python 05_semisynthetic.py <cache_dir>
import sys, os, json
import numpy as np
from common import *

CACHE = sys.argv[1]
D = load(CACHE); G = D["G"]; g = D["g"]; n = D["n"]; pos = D["pos"]
PM = np.load(os.path.join(CACHE, "popmodels.npz")); s_pop = PM["s_pop"]
M = json.load(open(os.path.join(RESULTS, "RESULTS_v3_main.json")))
ss, zc, dd, ll = D["ss"], D["zc"], D["delta_days"], D["ll"]
ssi = np.round(ss).astype(np.int64)
bins = binof(n)
rng = np.random.default_rng(SEED + 5)
tau2_hat = M["reliability_raw"]["bins"][-1]["decomposition"]["tau2"]
tau2u_hat = M["reliability_hlr"]["bins"][-1]["decomposition"]["tau2"]
beta_hat = M["personalization_raw"]["population_value"]
k = np.floor(n * 0.8).astype(int); tr = pos < k[g]; ho = ~tr
nn = np.bincount(g[ho], ss[ho], G)
a_design = fit_a_given_b(g, zc, D["sc"], ss, G, beta_hat)
OUT = dict(tau2_hat_raw=tau2_hat, tau2_hat_hlr=tau2u_hat, beta_hat_raw=beta_hat,
           note="Simulation on real designs; outcomes are binomial draws from a known model, not data.")


def sim_logistic(btrue, zdes):
    p = 1 / (1 + np.exp(-np.clip(a_design[g] + btrue[g] * zdes, -30, 30)))
    return rng.binomial(ssi, p).astype(float)


def halves_decomp(sc_, zdes, mask_learners):
    m0 = pos % 2 == 0; m1 = pos % 2 == 1
    _, eo, vo, _ = fit_ab(g[m0], zdes[m0], sc_[m0], ss[m0], G)
    _, ee, ve, _ = fit_ab(g[m1], zdes[m1], sc_[m1], ss[m1], G)
    out = []
    for i, lab in enumerate(BINLAB + ["all"]):
        m = ((bins == i) if lab != "all" else (bins >= 0)) & mask_learners & np.isfinite(eo) & np.isfinite(ee)
        tau2 = float(np.cov(eo[m], ee[m])[0, 1]); varh = (np.var(eo[m], ddof=1) + np.var(ee[m], ddof=1)) / 2
        err = varh - tau2
        inv_n = float(np.mean((1 / np.ceil(n[m] / 2) + 1 / np.floor(n[m] / 2)) / 2))
        s1 = err / inv_n
        out.append(dict(bin=lab, split_half=r(sb(eo[m], ee[m])), tau2=r(tau2, 6), err_half=r(err, 6), sigma1_sq=r(s1, 6),
                        n_star_050=r(s1 / tau2 if tau2 > 0 else None, 1), n_star_070=r(s1 / tau2 * 0.7 / 0.3 if tau2 > 0 else None, 1)))
    return out, eo, ee, vo, ve


def personalize_sim(sc_, zdes, btrue):
    mtr = np.bincount(g[tr], ll[tr], G) / k
    zt = zdes - (np.bincount(g[tr], zdes[tr], G) / k)[g]
    _, b_tr, v_tr, _ = fit_ab(g[tr], zt[tr], sc_[tr], ss[tr], G)
    beta = fit_ab(np.zeros(tr.sum(), np.int64), zt[tr], sc_[tr], ss[tr], 1)[1][0]
    t0 = tr & (pos % 2 == 0); t1 = tr & (pos % 2 == 1)
    _, eo, vo, _ = fit_ab(g[t0], zt[t0], sc_[t0], ss[t0], G)
    _, ee, ve, _ = fit_ab(g[t1], zt[t1], sc_[t1], ss[t1], G)
    a_pop = fit_a_given_b(g[tr], zt[tr], sc_[tr], ss[tr], G, beta)

    def L(bvec):
        bv = np.where(np.isfinite(bvec), bvec, beta)
        a = fit_a_given_b(g[tr], zt[tr], sc_[tr], ss[tr], G, bv)
        return np.bincount(g[ho], logloss_rows(a[g[ho]] + bv[g[ho]] * zt[ho], sc_[ho], ss[ho]), G)
    e_tr = a_pop[g[tr]] + beta * zt[tr]; ptr = 1 / (1 + np.exp(-np.clip(e_tr, -30, 30))); Wt = ss[tr] * ptr * (1 - ptr)
    zbar = np.bincount(g[tr], Wt * zt[tr], G) / np.bincount(g[tr], Wt, G)
    e_ho = a_pop[g[ho]] + beta * zt[ho]; pho = 1 / (1 + np.exp(-np.clip(e_ho, -30, 30)))
    J = np.bincount(g[ho], ss[ho] * pho * (1 - pho) * (zt[ho] - zbar[g[ho]]) ** 2, G)
    valid = np.isfinite(b_tr) & np.isfinite(v_tr) & (nn > 0)
    rho = np.full(G, np.nan); pred = np.zeros(3); tot = nn[valid].sum()
    for i in range(len(BINS)):
        idx = np.where(valid & (bins == i))[0]
        m = idx[np.isfinite(eo[idx]) & np.isfinite(ee[idx])]
        tau2 = float(np.cov(eo[m], ee[m])[0, 1])
        err = (np.var(eo[m], ddof=1) + np.var(ee[m], ddof=1)) / 2 - tau2
        phi = err / float(np.mean((vo[m] + ve[m]) / 2)); t = max(tau2, 0.0)
        s2 = phi * v_tr[idx]; rr = t / (t + s2) if t > 0 else np.zeros(len(idx)); rho[idx] = rr
        pred += np.array([np.sum(0.5 * J[idx] * (t - s2)), np.sum(0.5 * J[idx] * t * rr), np.sum(0.5 * J[idx] * t)]) / tot
    shr = beta + rho * (b_tr - beta)
    Lp = L(np.full(G, beta)); Ln = L(np.where(valid, b_tr, beta)); Ls = L(np.where(valid, shr, beta)); Lo = L(btrue)
    v = valid
    obs = [(Lp[v].sum() - X[v].sum()) / nn[v].sum() for X in (Ln, Ls, Lo)]
    return dict(predicted_naive=r(pred[0], 6), predicted_shrunk=r(pred[1], 6), predicted_oracle=r(pred[2], 6),
                observed_naive=r(obs[0], 6), observed_shrunk=r(obs[1], 6), observed_oracle=r(obs[2], 6),
                n_learners=int(v.sum()))


grid = []
for mult in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    btrue = beta_hat + np.sqrt(mult * tau2_hat) * rng.standard_normal(G)
    sc_ = sim_logistic(btrue, zc)
    hd, eo, ee, _, _ = halves_decomp(sc_, zc, np.ones(G, bool))
    _, bfull, _, _ = fit_ab(g, zc, sc_, ss, G)
    true_rel = []
    for i, lab in enumerate(BINLAB + ["all"]):
        m = ((bins == i) if lab != "all" else (bins >= 0)) & np.isfinite(bfull)
        c = np.corrcoef(bfull[m], btrue[m])[0, 1] if mult > 0 else 0.0
        true_rel.append(r(c ** 2))
    pers = personalize_sim(sc_, zc, btrue)
    row = dict(multiplier=mult, tau2_true=r(mult * tau2_hat, 6), bins=hd, true_reliability_full=true_rel, personalization=pers)
    grid.append(row)
    print(mult, hd[-1], true_rel[-1], pers, flush=True)
OUT["logistic_grid"] = grid

# probing design: within-learner log-lag spread doubled, same learners, same n, tau2 at the estimate
btrue = beta_hat + np.sqrt(tau2_hat) * rng.standard_normal(G)
for name, zdes in (("observed_design", zc), ("probing_design_2x_spread", 2 * zc)):
    sc_ = sim_logistic(btrue, zdes)
    hd, _, _, _, _ = halves_decomp(sc_, zdes, np.ones(G, bool))
    OUT[name] = dict(bins=hd, personalization=personalize_sim(sc_, zdes, btrue))
    print(name, hd[-1], OUT[name]["personalization"], flush=True)

# ---------- HLR world: scheduling consequences ----------
beta_u = M["personalization_hlr"]["population_value"]
u_true = beta_u + np.sqrt(max(tau2u_hat, 0.0)) * rng.standard_normal(G)
_, _, _, p_sim = hlr_terms(s_pop + u_true[g], dd, np.zeros(len(g)), ss)
sc_h = rng.binomial(ssi, p_sim).astype(float)
u_tr, v_tr = fit_u(g[tr], s_pop[tr], dd[tr], sc_h[tr], ss[tr], G)
bu = fit_u(np.zeros(tr.sum(), np.int64), s_pop[tr], dd[tr], sc_h[tr], ss[tr], 1, need=1)[0][0]
t0 = tr & (pos % 2 == 0); t1 = tr & (pos % 2 == 1)
eo, vo = fit_u(g[t0], s_pop[t0], dd[t0], sc_h[t0], ss[t0], G)
ee, ve = fit_u(g[t1], s_pop[t1], dd[t1], sc_h[t1], ss[t1], G)
valid = np.isfinite(u_tr) & np.isfinite(v_tr)
rho = np.full(G, np.nan)
for i in range(len(BINS)):
    idx = np.where(valid & (bins == i))[0]
    m = idx[np.isfinite(eo[idx]) & np.isfinite(ee[idx])]
    tau2 = float(np.cov(eo[m], ee[m])[0, 1]); err = (np.var(eo[m], ddof=1) + np.var(ee[m], ddof=1)) / 2 - tau2
    phi = err / float(np.mean((vo[m] + ve[m]) / 2)); t = max(tau2, 0.0)
    rho[idx] = t / (t + phi * v_tr[idx]) if t > 0 else 0.0
pol = dict(population=np.full(G, bu), naive=u_tr, shrunk=bu + rho * (u_tr - bu), oracle=u_true)
h_ref = float(np.median(np.exp2(np.clip(s_pop[ho] + beta_u, S_MIN, S_MAX))))
hl = dict(tau2_u_true=r(tau2u_hat, 6), reference_half_life_days=r(h_ref, 4), n_learners=int(valid.sum()))
for rs_ in (0.85, 0.90):
    for name, up in pol.items():
        achieved = rs_ ** np.exp2(up[valid] - u_true[valid])
        interval = -h_ref * np.exp2(up[valid] - bu) * np.log2(rs_)
        hl["r%.2f_%s" % (rs_, name)] = dict(mean_abs_dev=r(np.mean(np.abs(achieved - rs_))),
                                            share_dev_gt_0_05=r(np.mean(np.abs(achieved - rs_) > 0.05)),
                                            mean_achieved=r(np.mean(achieved)),
                                            reviews_per_30_days=r(np.mean(30.0 / interval), 3))
OUT["hlr_world"] = hl
print(hl)
dump(OUT, "RESULTS_v3_sim.json")
print("done")
