# -*- coding: utf-8 -*-
# A0 reproduction gate: recompute the deterministic v2 quantities with the batched solver and
# compare them with the v2 result files. Usage: python 01_reproduce_v2.py <cache_dir>
import sys, os, json, math
import numpy as np
from common import *

D = load(sys.argv[1]); G = D["G"]; g = D["g"]; n = D["n"]; pos = D["pos"]
V2 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
R2 = json.load(open(os.path.join(V2, "RESULTS.json")))
E2 = json.load(open(os.path.join(V2, "estimator_comparison.json")))
zc, sc, ss = D["zc"], D["sc"], D["ss"]
bins = binof(n)

_, b_full, _, _ = fit_ab(g, zc, sc, ss, G)
pa, pb, _, _ = fit_ab(np.zeros(len(g), np.int64), zc, sc, ss, 1)
oe = pos % 2
def halves(z, ridge_a=0.0):
    m0 = oe == 0; m1 = oe == 1
    bo = fit_ab(g[m0], z[m0], sc[m0], ss[m0], G, ridge_a=ridge_a)[1]
    be = fit_ab(g[m1], z[m1], sc[m1], ss[m1], G, ridge_a=ridge_a)[1]
    return bo, be
bo, be = halves(zc)
zg = D["ll"] - D["ll"].mean()
bo_n, be_n = halves(zg, ridge_a=1e-2)

out = dict(forgetting=dict(pct_negative_slope=float(np.mean(np.nan_to_num(b_full, nan=0.0) < 0)),
                           median_slope=float(np.nanmedian(b_full)), pooled_within_user_slope=float(pb[0])),
           curve=[], inv_overall=sb(bo, be), noninv_overall=sb(bo_n, be_n), noninv_bins=[])
for i, lab in enumerate(BINLAB):
    m = bins == i
    out["curve"].append(dict(bin=lab, n_users=int(m.sum()), reliability=sb(bo[m], be[m])))
    out["noninv_bins"].append(dict(bin=lab, reliability=sb(bo_n[m], be_n[m])))

# v2 personalization (r2_fixes spec)
k = np.floor(n * 0.8).astype(int); tr = pos < k[g]; ho = ~tr
a_tr, b_tr, _, _ = fit_ab(g[tr], zc[tr], sc[tr], ss[tr], G)
PA, PB, _, _ = fit_ab(np.zeros(tr.sum(), np.int64), zc[tr], sc[tr], ss[tr], 1); PA, PB = PA[0], PB[0]
relbybin = {c["bin"]: c["reliability"] for c in R2["H1_reliability_horizon"]["curve"]}
w = np.array([max(0.0, min(1.0, relbybin.get(BINLAB[bi], 0.0))) if bi >= 0 else 0.0 for bi in bins])
b_eb = np.where(np.isfinite(b_tr), w * b_tr + (1 - w) * PB, PB)
gh = g[ho]; zh = zc[ho]; sch = sc[ho]; ssh = ss[ho]
def hl(a, bvec):
    eta = a[gh] + bvec[gh] * zh
    return np.bincount(gh, logloss_rows(eta, sch, ssh), G)
nn = np.bincount(gh, ssh, G)
valid = np.isfinite(a_tr) & (nn > 0)
L_pop = hl(a_tr, np.full(G, PB)); L_pp = hl(a_tr, b_tr); L_eb = hl(a_tr, b_eb)
L_fpop = hl(np.full(G, PA), np.full(G, PB))
out["pers"] = []
for i, lab in enumerate(BINLAB):
    m = valid & (bins == i)
    out["pers"].append(dict(bin=lab, base=float(L_pop[m].sum() / nn[m].sum()),
                            naive=float(np.median((L_pop[m] - L_pp[m]) / nn[m])),
                            eb=float(np.median((L_pop[m] - L_eb[m]) / nn[m])),
                            full=float(np.median((L_fpop[m] - L_pp[m]) / nn[m]))))

# compare
chk = []
def c(name, new, old, nd):
    ok = (old is None and (new is None or new != new)) or (old is not None and round(new, nd) == round(old, nd))
    if not ok and old is not None and abs(new - old) <= 0.5 * 10 ** (-nd) + 1e-12:
        ok = True
    chk.append(dict(name=name, v3=None if new != new else round(new, 6), v2=old, ok=bool(ok)))
F2 = R2["forgetting"]
c("pct_negative_slope", out["forgetting"]["pct_negative_slope"], F2["pct_negative_slope"], 4)
c("median_slope", out["forgetting"]["median_slope"], F2["median_slope"], 5)
c("pooled_within_user_slope", out["forgetting"]["pooled_within_user_slope"], F2["pooled_within_user_slope"], 5)
c("overall_reliability", out["inv_overall"], R2["H1_reliability_horizon"]["overall_reliability"], 3)
c("noninv_overall", out["noninv_overall"], E2["estimators"]["noninv"]["overall_reliability"], 3)
for i, lab in enumerate(BINLAB):
    c2 = R2["H1_reliability_horizon"]["curve"][i]
    c("rel_" + lab, out["curve"][i]["reliability"], c2["reliability"], 3)
    c("nusers_" + lab, out["curve"][i]["n_users"], c2["n_users"], 0)
    c("noninv_" + lab, out["noninv_bins"][i]["reliability"], E2["estimators"]["noninv"]["per_bin"][i]["reliability"], 3)
    p2 = R2["H3_personalization"]["per_bin"][i]
    c("base_" + lab, out["pers"][i]["base"], p2["base_logloss_per_trial"], 4)
    c("naive_med_" + lab, out["pers"][i]["naive"], p2["personalized"]["median_gain"], 4)
    c("eb_med_" + lab, out["pers"][i]["eb"], p2["partial_pooled"]["median_gain"], 4)
    c("full_med_" + lab, out["pers"][i]["full"], R2["H3_personalization"]["full_personalization_robustness"][i]["gain"]["median_gain"], 4)
nbad = sum(not x["ok"] for x in chk)
dump(dict(n_checks=len(chk), n_mismatch=nbad, checks=chk, recomputed=out), "v2_reproduction.json")
for x in chk:
    if not x["ok"]:
        print("MISMATCH", x)
print("checks", len(chk), "mismatches", nbad)
