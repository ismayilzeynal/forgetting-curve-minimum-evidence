# -*- coding: utf-8 -*-
# Why the moment-based (addendum A3) decomposition is not used for inference: per-learner
# logistic fits on half histories are often quasi-separated, so the Pearson moments of the
# estimates are driven by a few extreme values. This script quantifies that: share of learners
# with a perfect half, share of extreme slopes, and how the Pearson split-half coefficient moves
# with the prior scale while the rank-based coefficient does not.
# Writes results/RESULTS_v3_separation.json. Usage: python 05b_separation_diagnostics.py <cache_dir>
import sys
import numpy as np
from scipy.stats import spearmanr
from common import *

D = load(sys.argv[1]); G = D["G"]; g = D["g"]; pos = D["pos"]; n = D["n"]; sc = D["sc"]; ss = D["ss"]; zc = D["zc"]
bins = binof(n)
m0 = pos % 2 == 0; m1 = ~m0
fail0 = np.bincount(g[m0], ss[m0] - sc[m0], G); fail1 = np.bincount(g[m1], ss[m1] - sc[m1], G)
pa = fit_ab(np.zeros(len(g), np.int64), zc, sc, ss, 1)[0][0]


def ssb(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    c = spearmanr(a[m], b[m]).correlation
    return 2 * c / (1 + c)


def halves(la, lb):
    out = []
    for m in (m0, m1):
        a, b, _, _ = fit_ab(g[m], zc[m], sc[m], ss[m], G, off=np.full(m.sum(), pa), ridge_a=la, ridge_b=lb, it=60)
        out.append(b)
    return out


OUT = dict(share_learners_perfect_half=r(float(np.mean((fail0 == 0) | (fail1 == 0))), 4),
           share_learners_perfect_overall=r(float(np.mean(np.bincount(g, ss - sc, G) == 0)), 4))
bo, be = halves(0.0, 1e-2)   # v2 confirmatory estimator (slope ridge only)
fin = np.isfinite(bo)
OUT["v2_estimator"] = dict(pearson_sb=r(sb(bo, be), 4), spearman_sb=r(ssb(bo, be), 4),
                           share_abs_slope_gt_5=r(float(np.mean(np.abs(bo[fin]) > 5)), 4),
                           slope_p1_p50_p99=[r(x, 3) for x in np.percentile(bo[fin], [1, 50, 99])],
                           max_abs_slope=r(float(np.nanmax(np.abs(bo))), 1))
# influence of the most extreme 1% of learners on the Pearson coefficient
ext = np.abs(bo) + np.abs(be)
keep = ext < np.nanpercentile(ext, 99)
OUT["v2_estimator"]["pearson_sb_without_top1pct"] = r(sb(bo[keep], be[keep]), 4)
OUT["prior_sensitivity"] = []
for sa, sbp in ((None, 10.0), (2.5, 2.5), (2.5, 1.0), (1.5, 1.0)):
    la = 0.0 if sa is None else 1 / sa ** 2
    o, e = halves(la, 1 / sbp ** 2)
    m = np.isfinite(o) & np.isfinite(e)
    OUT["prior_sensitivity"].append(dict(prior_sd_intercept=sa, prior_sd_rate=sbp,
                                         pearson_sb=r(sb(o, e), 4), spearman_sb=r(ssb(o, e), 4),
                                         pearson_sb_bins=[r(sb(o[bins == i], e[bins == i]), 3) for i in range(6)],
                                         spearman_sb_bins=[r(ssb(o[bins == i], e[bins == i]), 3) for i in range(6)],
                                         cov_halves=r(float(np.cov(o[m], e[m])[0, 1]), 5)))
    print(OUT["prior_sensitivity"][-1], flush=True)
dump(OUT, "RESULTS_v3_separation.json")
print(OUT)
