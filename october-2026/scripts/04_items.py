# -*- coding: utf-8 -*-
# A8 item side ("what" vs "who"): the same location-invariant slope and intercept estimators,
# applied to lexemes instead of learners, binned by reviews per lexeme, with a permutation null,
# variance decomposition and a truncation curve at matched evidence. Writes results/RESULTS_v3_items.json.
# Usage: python 04_items.py <cache_dir>
import sys
import numpy as np, pandas as pd
from common import *

D = load(sys.argv[1])
rng = np.random.default_rng(SEED + 4)
order = np.lexsort((np.arange(len(D["g"])), D["ts"], D["lex"]))
gl = D["lex"][order]; L = int(gl.max()) + 1
ll = D["ll"][order]; sc = D["sc"][order]; ss = D["ss"][order]
nl = np.bincount(gl, minlength=L)
start = np.concatenate([[0], np.cumsum(nl)[:-1]])
pos = np.arange(len(gl)) - start[gl]
z = ll - (np.bincount(gl, ll, L) / np.maximum(nl, 1))[gl]
bins = binof(nl)


def fit(m, zz, what):
    a, b, vb, va = fit_ab(gl[m], zz[m], sc[m], ss[m], L)
    return (b, vb) if what == "slope" else (a, va)


def decomp(eo, ee):
    m = np.isfinite(eo) & np.isfinite(ee)
    if m.sum() < 20:
        return None
    tau2 = float(np.cov(eo[m], ee[m])[0, 1]); varh = (np.var(eo[m], ddof=1) + np.var(ee[m], ddof=1)) / 2
    return dict(tau2=r(tau2, 6), err_half=r(varh - tau2, 6))


OUT = dict(n_lexemes=int((nl > 0).sum()), lexemes_ge20=int((nl >= 20).sum()),
           median_reviews_per_lexeme=r(np.median(nl[nl > 0]), 1))
m0 = pos % 2 == 0; m1 = pos % 2 == 1
for what in ("slope", "intercept"):
    eo, _ = fit(m0, z, what); ee, _ = fit(m1, z, what)
    nulls = []
    if what == "slope":
        for _ in range(20):
            outs = []
            for m in (m0, m1):
                zp = z.copy(); zp[m] = permute_within(gl[m], z[m], rng)
                outs.append(fit(m, zp, what)[0])
            nulls.append([sb(outs[0][bins == i], outs[1][bins == i]) for i in range(len(BINS))] + [sb(outs[0], outs[1])])
        nulls = np.array(nulls)
    rows = []
    for i, lab in enumerate(BINLAB + ["all"]):
        m = (bins == i) if lab != "all" else (bins >= 0)
        row = dict(bin=lab, n_lexemes=int(m.sum()), oddeven=r(sb(eo[m], ee[m])),
                   oddeven_ci=[r(x) for x in boot_sb(eo[m], ee[m], rng, 1000)], decomposition=decomp(eo[m], ee[m]))
        if what == "slope":
            row["null_median"] = r(np.nanmedian(nulls[:, i]))
            row["null_p2_5"] = r(np.nanpercentile(nulls[:, i], 2.5)); row["null_p97_5"] = r(np.nanpercentile(nulls[:, i], 97.5))
        rows.append(row); print(what, row, flush=True)
    OUT[what] = rows

heavy = nl >= 250
trunc = []
for m_n in (20, 30, 50, 80, 130, 250):
    keep = heavy[gl] & (pos < m_n)
    mm = np.bincount(gl[keep], ll[keep], L) / np.maximum(np.bincount(gl[keep], minlength=L), 1)
    zz = z.copy(); zz[keep] = ll[keep] - mm[gl[keep]]
    a0 = keep & m0; a1 = keep & m1
    row = dict(first_n=m_n, n_lexemes=int(heavy.sum()))
    for what in ("slope", "intercept"):
        eo, _ = fit(a0, zz, what); ee, _ = fit(a1, zz, what)
        row[what] = r(sb(eo[heavy], ee[heavy])); row[what + "_ci"] = [r(x) for x in boot_sb(eo[heavy], ee[heavy], rng, 1000)]
        dc = decomp(eo[heavy], ee[heavy]); row[what + "_tau2"] = dc["tau2"] if dc else None
    trunc.append(row); print(row, flush=True)
OUT["truncation_heavy_lexemes"] = trunc
dump(OUT, "RESULTS_v3_items.json")
print("done")
