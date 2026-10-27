# -*- coding: utf-8 -*-
# A1-A5, A7, A8 (learner side): reliability horizons, permutation nulls, temporal and random
# splits, variance decomposition, minimum evidence, personalization with the theory test, and
# HLR scheduling consequences. Writes results/RESULTS_v3_main.json and results/per_learner_v3.csv.gz.
# Usage: python 03_main.py <cache_dir>
import sys, os
import numpy as np, pandas as pd
from common import *

CACHE = sys.argv[1]
D = load(CACHE); G = D["G"]; g = D["g"]; n = D["n"]; pos = D["pos"]
PM = np.load(os.path.join(CACHE, "popmodels.npz"))
off_adj, s_pop = PM["off_adj"], PM["s_pop"]
sc, ss, zc, dd = D["sc"], D["ss"], D["zc"], D["delta_days"]
bins = binof(n)
rng = np.random.default_rng(SEED + 3)
NB = 1000; NPERM = 20
OUT = dict(notes=dict(
    split_half="Spearman-Brown corrected Pearson correlation of two half estimates across learners",
    tau2="covariance of the two half estimates (true-score variance of the parameter)",
    err_half="variance of a half estimate minus tau2 (error variance of a half estimate)",
    phi="err_half divided by the mean model-based (sandwich) variance of a half estimate",
    sigma1_sq="per-trace error variance: err_half / mean(1/n_half)",
    n_star="traces needed for a full-history estimate to reach the target reliability: (sigma1_sq/tau2) rho/(1-rho)"))
PL = pd.DataFrame(dict(uid=D["uid"], n=n, bin=bins))


def est(kind, m, z=None, dvec=None):
    z = zc if z is None else z
    dvec = dd if dvec is None else dvec
    if kind == "hlr":
        return fit_u(g[m], s_pop[m], dvec[m], sc[m], ss[m], G)
    off = off_adj[m] if kind == "adj" else None
    a, b, vb, va = fit_ab(g[m], z[m], sc[m], ss[m], G, off=off)
    return (a, va) if kind == "icpt" else (b, vb)


def decomp(eo, ee, vo, ve, nho, nhe, nb=NB):
    m = np.isfinite(eo) & np.isfinite(ee) & np.isfinite(vo) & np.isfinite(ve)
    if m.sum() < 20:
        return None
    eo, ee, vo, ve, nho, nhe = eo[m], ee[m], vo[m], ve[m], nho[m], nhe[m]

    def stats(ix):
        a, b = eo[ix], ee[ix]
        tau2 = float(np.cov(a, b)[0, 1]); varh = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
        err = varh - tau2; mv = float(np.mean((vo[ix] + ve[ix]) / 2))
        inv_n = float(np.mean((1 / nho[ix] + 1 / nhe[ix]) / 2))
        s1 = err / inv_n
        ns = {k: (s1 / tau2 * k / (1 - k) if tau2 > 0 else np.inf) for k in (0.5, 0.7)}
        return tau2, err, err / mv, s1, ns[0.5], ns[0.7]

    pt = stats(np.arange(len(eo)))
    bs = np.array([stats(rng.integers(0, len(eo), len(eo))) for _ in range(nb)])
    keys = ["tau2", "err_half", "phi", "sigma1_sq", "n_star_050", "n_star_070"]
    res = {}
    for j, k in enumerate(keys):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        res[k] = r(pt[j], 6); res[k + "_ci"] = [r(lo, 6), r(hi, 6)]
    res["n_learners"] = int(len(eo))
    res["share_boot_tau2_nonpositive"] = r(float(np.mean(bs[:, 0] <= 0)), 4)
    return res


def split_masks(kind):
    if kind == "oddeven":
        return pos % 2 == 0, pos % 2 == 1
    if kind == "chrono":
        h = (n // 2)[g]
        return pos < h, pos >= h
    raise ValueError(kind)


def reliability_block(kind, with_null=True):
    res = dict()
    m0, m1 = split_masks("oddeven")
    eo, vo = est(kind, m0); ee, ve = est(kind, m1)
    nho = np.ceil(n / 2); nhe = np.floor(n / 2)
    PL[kind + "_odd"] = eo; PL[kind + "_even"] = ee; PL[kind + "_var_odd"] = vo; PL[kind + "_var_even"] = ve
    c0, c1 = split_masks("chrono")
    co, _ = est(kind, c0); ce, _ = est(kind, c1)
    PL[kind + "_first"] = co; PL[kind + "_second"] = ce
    rs = []
    for _ in range(20):
        bit = rng.random(len(g)) < 0.5
        a, _ = est(kind, bit); b, _ = est(kind, ~bit)
        rs.append([sb(a[bins == i], b[bins == i]) for i in range(len(BINS))] + [sb(a, b)])
    rs = np.array(rs)
    nulls = []
    if with_null:
        for _ in range(NPERM):
            outs = []
            for m in (m0, m1):
                if kind == "hlr":
                    dp = dd.copy(); dp[m] = permute_within(g[m], dd[m], rng)
                    outs.append(est(kind, m, dvec=dp)[0])
                else:
                    zp = zc.copy(); zp[m] = permute_within(g[m], zc[m], rng)
                    outs.append(est(kind, m, z=zp)[0])
            nulls.append([sb(outs[0][bins == i], outs[1][bins == i]) for i in range(len(BINS))] + [sb(outs[0], outs[1])])
        nulls = np.array(nulls)
    rows = []
    for i, lab in enumerate(BINLAB + ["all"]):
        m = (bins == i) if lab != "all" else (bins >= 0)
        row = dict(bin=lab, n_learners=int(m.sum()),
                   oddeven=r(sb(eo[m], ee[m]), 4), oddeven_ci=[r(x, 4) for x in boot_sb(eo[m], ee[m], rng, NB)],
                   chrono=r(sb(co[m], ce[m]), 4), chrono_ci=[r(x, 4) for x in boot_sb(co[m], ce[m], rng, NB)],
                   random20_mean=r(np.nanmean(rs[:, i]), 4), random20_sd=r(np.nanstd(rs[:, i]), 4))
        if with_null:
            row["null_median"] = r(np.nanmedian(nulls[:, i]), 4)
            row["null_p2_5"] = r(np.nanpercentile(nulls[:, i], 2.5), 4)
            row["null_p97_5"] = r(np.nanpercentile(nulls[:, i], 97.5), 4)
        row["decomposition"] = decomp(eo[m], ee[m], vo[m], ve[m], nho[m], nhe[m])
        rows.append(row)
        print(kind, lab, row["oddeven"], row["chrono"], row.get("null_median"), flush=True)
    res["bins"] = rows
    res["spearman_trend_oddeven"] = r(pd.Series([x["oddeven"] for x in rows[:-1]]).corr(pd.Series(range(6)), method="spearman"), 4)
    return res


for kind in ("raw", "adj", "hlr", "icpt"):
    OUT["reliability_" + kind] = reliability_block(kind, with_null=(kind != "icpt"))
dump(OUT, "RESULTS_v3_main.json")

# ---------- flat-curve mechanism (A5) ----------
a_f, b_f, vb_f, _ = fit_ab(g, zc, sc, ss, G)
sdz = np.sqrt(np.bincount(g, zc * zc, G) / n)
eta = a_f[g] + np.nan_to_num(b_f[g]) * zc
p = 1 / (1 + np.exp(-np.clip(np.nan_to_num(eta), -30, 30)))
mpq = np.bincount(g, p * (1 - p), G) / n
mss = np.bincount(g, ss, G) / n
mech = []
for i, lab in enumerate(BINLAB):
    m = bins == i
    mech.append(dict(bin=lab, median_sd_loglag=r(np.median(sdz[m])), mean_p1mp=r(np.nanmean(mpq[m])),
                     mean_session_size=r(np.mean(mss[m])), median_n=r(np.median(n[m]), 1),
                     median_full_var_b=r(np.nanmedian(vb_f[m]), 6)))
ok = np.isfinite(vb_f) & (vb_f > 0) & (sdz > 0)
X = np.column_stack([np.ones(ok.sum()), np.log(sdz[ok]), np.log(mss[ok])])
y = np.log(n[ok] * vb_f[ok])
coef = np.linalg.lstsq(X, y, rcond=None)[0]
OUT["flat_curve_mechanism"] = dict(per_bin=mech, regression_log_n_var_on_log_sd_loglag=r(coef[1]),
                                   regression_log_n_var_on_log_session_size=r(coef[2]), n_learners=int(ok.sum()),
                                   spearman_sd_loglag_vs_n=r(pd.Series(sdz).corr(pd.Series(n), method="spearman")))
PL["sd_loglag"] = sdz; PL["raw_full"] = b_f; PL["raw_full_var"] = vb_f

# ---------- truncation curve (added after the freeze; labelled in the paper) ----------
heavy = n >= 250
trunc = []
for m_n in (20, 30, 50, 80, 130, 250):
    keep = heavy[g] & (pos < m_n)
    mm = np.bincount(g[keep], D["ll"][keep], G) / np.maximum(np.bincount(g[keep], minlength=G), 1)
    mz = zc.copy(); mz[keep] = D["ll"][keep] - mm[g[keep]]
    a0 = keep & (pos % 2 == 0); a1 = keep & (pos % 2 == 1)
    rowt = dict(first_n=m_n, n_learners=int(heavy.sum()))
    for kind in ("raw", "adj", "icpt"):
        eo, vo = est(kind, a0, z=mz); ee, ve = est(kind, a1, z=mz)
        rowt[kind] = r(sb(eo[heavy], ee[heavy]), 4)
        rowt[kind + "_ci"] = [r(x, 4) for x in boot_sb(eo[heavy], ee[heavy], rng, NB)]
        dcp = decomp(eo[heavy], ee[heavy], vo[heavy], ve[heavy],
                     np.full(heavy.sum(), np.ceil(m_n / 2)), np.full(heavy.sum(), np.floor(m_n / 2)), nb=200)
        rowt[kind + "_tau2"] = dcp["tau2"] if dcp else None
        rowt[kind + "_err_half"] = dcp["err_half"] if dcp else None
    trunc.append(rowt); print("trunc", rowt, flush=True)
OUT["truncation_heavy_learners"] = trunc
dump(OUT, "RESULTS_v3_main.json")

# ---------- personalization and theory test (A4) ----------
k = np.floor(n * 0.8).astype(int); tr = pos < k[g]; ho = ~tr
mtr = np.bincount(g[tr], D["ll"][tr], G) / k
zt = D["ll"] - mtr[g]
nn = np.bincount(g[ho], ss[ho], G)


def cluster_ratio_ci(num, den, idx):
    vals = []
    for _ in range(NB):
        j = rng.choice(idx, len(idx), replace=True)
        vals.append(num[j].sum() / den[j].sum())
    return [r(np.percentile(vals, 2.5), 6), r(np.percentile(vals, 97.5), 6)]


def theory_pred(J, sig_model, eo, ee, vo, ve, idx, C=None):
    """Predicted per-trial gains (naive, shrunk, oracle) for learners idx; tau2 and phi from training halves."""
    m = idx[np.isfinite(eo[idx]) & np.isfinite(ee[idx]) & np.isfinite(vo[idx]) & np.isfinite(ve[idx])]
    tau2 = float(np.cov(eo[m], ee[m])[0, 1])
    err = (np.var(eo[m], ddof=1) + np.var(ee[m], ddof=1)) / 2 - tau2
    phi = err / float(np.mean((vo[m] + ve[m]) / 2))
    t = max(tau2, 0.0)
    s2 = phi * sig_model[idx]
    rho = t / (t + s2) if t > 0 else np.zeros(len(idx))
    tot = nn[idx].sum()
    cc = t if C is None else C
    naive = np.sum(0.5 * J[idx] * (2 * cc - (t + s2))) / tot
    shr = np.sum(0.5 * J[idx] * (2 * rho * cc - rho ** 2 * (t + s2))) / tot
    orc = np.sum(0.5 * J[idx] * t) / tot
    return naive, shr, orc, tau2, phi, rho


def personalize(kind):
    res = dict()
    if kind == "hlr":
        est_tr, var_tr = fit_u(g[tr], s_pop[tr], dd[tr], sc[tr], ss[tr], G)
        beta = fit_u(np.zeros(tr.sum(), np.int64), s_pop[tr], dd[tr], sc[tr], ss[tr], 1, need=1)[0][0]
        t0 = tr & (pos % 2 == 0); t1 = tr & (pos % 2 == 1)
        eo, vo = fit_u(g[t0], s_pop[t0], dd[t0], sc[t0], ss[t0], G)
        ee, ve = fit_u(g[t1], s_pop[t1], dd[t1], sc[t1], ss[t1], G)
        est_ho, _ = fit_u(g[ho], s_pop[ho], dd[ho], sc[ho], ss[ho], G)

        def L(uvec):
            loss, _, _, _ = hlr_terms(s_pop[ho] + uvec[g[ho]], dd[ho], sc[ho], ss[ho])
            return np.bincount(g[ho], loss, G)
        _, _, info, _ = hlr_terms(s_pop[ho] + beta, dd[ho], sc[ho], ss[ho])
        J = np.bincount(g[ho], info, G)
    else:
        off = off_adj if kind == "adj" else np.zeros(len(g))
        _, est_tr, var_tr, _ = fit_ab(g[tr], zt[tr], sc[tr], ss[tr], G, off=off[tr])
        beta = fit_ab(np.zeros(tr.sum(), np.int64), zt[tr], sc[tr], ss[tr], 1, off=off[tr])[1][0]
        t0 = tr & (pos % 2 == 0); t1 = tr & (pos % 2 == 1)
        _, eo, vo, _ = fit_ab(g[t0], zt[t0], sc[t0], ss[t0], G, off=off[t0])
        _, ee, ve, _ = fit_ab(g[t1], zt[t1], sc[t1], ss[t1], G, off=off[t1])
        _, est_ho, _, _ = fit_ab(g[ho], zt[ho], sc[ho], ss[ho], G, off=off[ho])
        a_pop = fit_a_given_b(g[tr], zt[tr], sc[tr], ss[tr], G, beta, off=off[tr])

        def L(bvec):
            bv = np.where(np.isfinite(bvec), bvec, beta)
            a = fit_a_given_b(g[tr], zt[tr], sc[tr], ss[tr], G, bv, off=off[tr])
            return np.bincount(g[ho], logloss_rows(a[g[ho]] + bv[g[ho]] * zt[ho] + off[ho], sc[ho], ss[ho]), G)
        e_tr = a_pop[g[tr]] + beta * zt[tr] + off[tr]; ptr = 1 / (1 + np.exp(-np.clip(e_tr, -30, 30)))
        Wt = ss[tr] * ptr * (1 - ptr)
        zbar = np.bincount(g[tr], Wt * zt[tr], G) / np.bincount(g[tr], Wt, G)
        e_ho = a_pop[g[ho]] + beta * zt[ho] + off[ho]; pho = 1 / (1 + np.exp(-np.clip(e_ho, -30, 30)))
        J = np.bincount(g[ho], ss[ho] * pho * (1 - pho) * (zt[ho] - zbar[g[ho]]) ** 2, G)
    valid = np.isfinite(est_tr) & (nn > 0) & np.isfinite(var_tr)
    rho_all = np.full(G, np.nan)
    for i in range(len(BINS)):
        idx = np.where(valid & (bins == i))[0]
        rho_all[idx] = theory_pred(J, var_tr, eo, ee, vo, ve, idx)[5]
    shr = beta + rho_all * (est_tr - beta)
    Lp = L(np.full(G, beta)); Ln = L(np.where(valid, est_tr, beta)); Ls = L(np.where(valid, shr, beta))
    for nm, val in (("tr", est_tr), ("tr_var", var_tr), ("tr_odd", eo), ("tr_even", ee), ("tr_var_odd", vo),
                    ("tr_var_even", ve), ("ho", est_ho), ("J", J), ("rho", rho_all), ("L_pop", Lp),
                    ("L_naive", Ln), ("L_shrunk", Ls), ("valid", valid)):
        PL[kind + "_" + nm] = val
    PL["heldout_trials"] = nn
    rows = []
    for i, lab in enumerate(BINLAB + ["all"]):
        idx = np.where(valid & ((bins == i) if lab != "all" else (bins >= 0)))[0]
        if lab == "all":
            pr = np.zeros(3); tot = nn[idx].sum()
            for j in range(len(BINS)):
                ij = np.where(valid & (bins == j))[0]
                pr += np.array(theory_pred(J, var_tr, eo, ee, vo, ve, ij)[:3]) * nn[ij].sum() / tot
            pnaive, pshr, porc = pr; tau2 = phi = None
        else:
            pnaive, pshr, porc, tau2, phi, _ = theory_pred(J, var_tr, eo, ee, vo, ve, idx)
        on = (Lp[idx].sum() - Ln[idx].sum()) / nn[idx].sum()
        osh = (Lp[idx].sum() - Ls[idx].sum()) / nn[idx].sum()
        row = dict(bin=lab, n_learners=int(len(idx)), heldout_trials=r(nn[idx].sum(), 0),
                   base_logloss_per_trial=r(Lp[idx].sum() / nn[idx].sum(), 5),
                   observed_naive=r(on, 6), observed_naive_ci=cluster_ratio_ci(Lp - Ln, nn, idx),
                   observed_shrunk=r(osh, 6), observed_shrunk_ci=cluster_ratio_ci(Lp - Ls, nn, idx),
                   predicted_naive=r(pnaive, 6), predicted_shrunk=r(pshr, 6), predicted_oracle=r(porc, 6),
                   tau2_train=r(tau2, 6), phi_train=r(phi, 4),
                   median_rho=r(np.nanmedian(rho_all[idx]), 4),
                   median_learner_gain_naive=r(np.median((Lp[idx] - Ln[idx]) / nn[idx]), 6),
                   median_learner_gain_shrunk=r(np.median((Lp[idx] - Ls[idx]) / nn[idx]), 6),
                   mean_J_per_trial=r(J[idx].sum() / nn[idx].sum(), 6))
        if lab != "all":
            bsp = []
            for _ in range(300):
                j = rng.choice(idx, len(idx), replace=True)
                bsp.append(theory_pred(J, var_tr, eo, ee, vo, ve, j)[:3])
            bsp = np.array(bsp)
            row["predicted_naive_ci"] = [r(x, 6) for x in np.percentile(bsp[:, 0], [2.5, 97.5])]
            row["predicted_shrunk_ci"] = [r(x, 6) for x in np.percentile(bsp[:, 1], [2.5, 97.5])]
            row["predicted_oracle_ci"] = [r(x, 6) for x in np.percentile(bsp[:, 2], [2.5, 97.5])]
            mm = np.isfinite(est_tr[idx]) & np.isfinite(est_ho[idx])
            C = float(np.cov(est_tr[idx][mm], est_ho[idx][mm])[0, 1]) if mm.sum() > 20 else float("nan")
            row["cov_train_heldout"] = r(C, 6); row["n_cov_train_heldout"] = int(mm.sum())
            if np.isfinite(C):
                dr = theory_pred(J, var_tr, eo, ee, vo, ve, idx, C=C)
                row["predicted_naive_drift"] = r(dr[0], 6); row["predicted_shrunk_drift"] = r(dr[1], 6)
        rows.append(row)
        print(kind, "pers", lab, {k2: row[k2] for k2 in ("observed_naive", "predicted_naive", "observed_shrunk", "predicted_shrunk", "predicted_oracle")}, flush=True)
    res["population_value"] = r(beta, 6)
    res["bins"] = rows
    return res, est_tr, rho_all, beta, valid


for kind in ("raw", "adj"):
    OUT["personalization_" + kind] = personalize(kind)[0]
    dump(OUT, "RESULTS_v3_main.json")
resH, u_tr, rhoH, betaH, validH = personalize("hlr")
OUT["personalization_hlr"] = resH

# ---------- scheduling consequences with HLR (A7, real-data part) ----------
sched = dict(population_u=r(betaH, 6))
v = validH & np.isfinite(rhoH)
ratio_naive = np.exp2(u_tr[v] - betaH); ratio_shr = np.exp2(rhoH[v] * (u_tr[v] - betaH))


def dist(x):
    return dict(median=r(np.median(x)), p5=r(np.percentile(x, 5)), p95=r(np.percentile(x, 95)),
                share_outside_0_8_1_25=r(np.mean((x < 0.8) | (x > 1.25))))


sched["interval_ratio_naive"] = dist(ratio_naive); sched["interval_ratio_shrunk"] = dist(ratio_shr)
for rs_ in (0.85, 0.90):
    rec = rs_ ** (1.0 / ratio_shr)
    recn = rs_ ** (1.0 / ratio_naive)
    sched["recall_at_population_interval_r%.2f" % rs_] = dict(
        shrunk_median=r(np.median(rec)), shrunk_p5=r(np.percentile(rec, 5)), shrunk_p95=r(np.percentile(rec, 95)),
        shrunk_mean_abs_dev=r(np.mean(np.abs(rec - rs_))), shrunk_share_dev_gt_0_05=r(np.mean(np.abs(rec - rs_) > 0.05)),
        naive_mean_abs_dev=r(np.mean(np.abs(recn - rs_))), naive_share_dev_gt_0_05=r(np.mean(np.abs(recn - rs_) > 0.05)))
sched["n_learners"] = int(v.sum())
OUT["scheduling_hlr_real"] = sched

dump(OUT, "RESULTS_v3_main.json")
PL.to_csv(os.path.join(RESULTS, "per_learner_v3.csv.gz"), index=False, compression="gzip")
print("done")
