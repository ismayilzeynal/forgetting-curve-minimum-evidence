# -*- coding: utf-8 -*-
# Within-learner lag-permutation null for the rank-based (Spearman-Brown corrected Spearman)
# split-half reliability of the weak-prior individual rate estimates (odd/even halves), raw rate.
# Lags are permuted within each half, which destroys any true rate but keeps every learner's
# recall level, session sizes and lag distribution. Writes results/RESULTS_v3_rank_null.json.
# Usage: python 05c_rank_null.py <cache_dir>
import sys, json, os
import numpy as np
from scipy.stats import spearmanr
from common import *
from common_eb import weak_prior_fit

D = load(sys.argv[1]); G = D["G"]; g = D["g"]; pos = D["pos"]; n = D["n"]; sc = D["sc"]; ss = D["ss"]; zc = D["zc"]
bins = binof(n)
rng = np.random.default_rng(SEED + 51)
H = json.load(open(os.path.join(RESULTS, "RESULTS_v3_hier_raw.json")))
mu = np.array(H["overall"]["mu"])
m0 = pos % 2 == 0


def ssb(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    c = spearmanr(a[m], b[m]).correlation
    return 2 * c / (1 + c)


def est(z):
    out = []
    for m in (m0, ~m0):
        X = np.column_stack([np.ones(m.sum()), z[m]])
        th, _ = weak_prior_fit(g[m], X, sc[m], ss[m], G, mu, [2.5, 1.0])
        out.append(th[:, 1])
    return out


o, e = est(zc)
obs = [ssb(o[bins == i], e[bins == i]) for i in range(6)] + [ssb(o, e)]
nulls = []
for k in range(5):
    zp = zc.copy()
    for m in (m0, ~m0):
        zp[m] = permute_within(g[m], zc[m], rng)
    po, pe = est(zp)
    nulls.append([ssb(po[bins == i], pe[bins == i]) for i in range(6)] + [ssb(po, pe)])
    print(k, nulls[-1], flush=True)
nulls = np.array(nulls)
rows = [dict(bin=lab, observed_spearman_sb=r(obs[i], 4), null_mean=r(nulls[:, i].mean(), 4),
             null_min=r(nulls[:, i].min(), 4), null_max=r(nulls[:, i].max(), 4))
        for i, lab in enumerate(BINLAB + ["all"])]
dump(dict(n_permutations=5, bins=rows), "RESULTS_v3_rank_null.json")
print(rows)
