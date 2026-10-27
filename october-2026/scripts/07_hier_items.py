# -*- coding: utf-8 -*-
# Item side ("what" versus "who") with the same hierarchical estimator as 06: lexemes as groups,
# log lag centered within lexeme, bins by reviews per lexeme. Between-lexeme variance of rate and
# intercept, model-based reliability, per-review error variance, n*, and latent odd/even split-half
# correlations. Writes results/RESULTS_v3_hier_items.json. Usage: python 07_hier_items.py <cache_dir>
import sys, time
import numpy as np
from common import *
from common_eb import *

D = load(sys.argv[1])
rng = np.random.default_rng(SEED + 21)
order = np.lexsort((np.arange(len(D["g"])), D["ts"], D["lex"]))
gl = D["lex"][order]; L = int(gl.max()) + 1
ll = D["ll"][order]; sc = D["sc"][order]; ss = D["ss"][order]
nl = np.bincount(gl, minlength=L)
start = np.concatenate([[0], np.cumsum(nl)[:-1]])
pos = np.arange(len(gl)) - start[gl]
z = ll - (np.bincount(gl, ll, L) / np.maximum(nl, 1))[gl]
bins = binof(nl)
T0 = time.time()
OUT = dict(n_lexemes=int((nl > 0).sum()), lexemes_ge20=int((nl >= 20).sum()),
           median_reviews_per_lexeme=r(np.median(nl[nl > 0]), 1))


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [r(np.percentile(v, 2.5), 6), r(np.percentile(v, 97.5), 6)]


def summarize(R, mask, lab):
    S = R["Sigma"]; m = mask & R["use"]
    out = dict(bin=lab, n_lexemes=int(m.sum()), mu=[r(x, 6) for x in R["mu"]], converged=bool(R["converged"]))
    bs = bootstrap_hyper(R, rng, nb=500, groups_mask=mask)
    for k, name in ((1, "rate"), (0, "intercept")):
        tau2 = float(S[k, k]); lv = lik_var(R["Hlik"], k); rho = 1 - R["V"][:, k, k] / tau2
        s1 = float(np.median((nl * lv)[m & np.isfinite(lv)]))
        t2 = np.array([b[1][k, k] for b in bs])
        out[name] = dict(tau2=r(tau2, 7), tau2_ci=ci(t2), mean_rho=r(np.mean(rho[m]), 4), median_rho=r(np.median(rho[m]), 4),
                         per_review_error_var=r(s1, 5), n_star_050=r(s1 / tau2, 1), n_star_050_ci=ci(s1 / t2),
                         n_star_070=r(s1 / tau2 * 7 / 3, 1))
    out["corr_intercept_rate"] = r(latent_corr(S, 0, 1), 4)
    return out


groups_ok = nl >= 20
X = np.column_stack([np.ones(len(gl)), z])
keep = groups_ok[gl]
R = eb_logistic(gl[keep], X[keep], sc[keep], ss[keep], L, groups=groups_ok)
OUT["overall"] = summarize(R, groups_ok, "all")
print("[%5.0fs]" % (time.time() - T0), OUT["overall"], flush=True)
OUT["bins"] = []
for i, lab in enumerate(BINLAB):
    gm = bins == i; rr = gm[gl]
    Rb = eb_logistic(gl[rr], X[rr], sc[rr], ss[rr], L, groups=gm, mu0=R["mu"], Sigma0=R["Sigma"], theta0=R["theta"])
    OUT["bins"].append(summarize(Rb, gm, lab))
    print("[%5.0fs]" % (time.time() - T0), OUT["bins"][-1], flush=True)
dump(OUT, "RESULTS_v3_hier_items.json")

m0 = pos % 2 == 0
X4 = np.column_stack([m0 * 1.0, m0 * z, (~m0) * 1.0, (~m0) * z])
OUT["latent_split_oddeven"] = []
for i, lab in [(6, "all")]:
    gm = groups_ok; rr = gm[gl]
    R4 = eb_logistic(gl[rr], X4[rr], sc[rr], ss[rr], L, groups=gm, switch=1e-2)
    S4 = R4["Sigma"]; bs = bootstrap_hyper(R4, rng, nb=40, groups_mask=gm)
    row = dict(bin=lab, n_lexemes=int((gm & R4["use"]).sum()), converged=bool(R4["converged"]),
               latent_corr_rate=r(latent_corr(S4, 1, 3), 4), latent_corr_rate_ci=ci([latent_corr(b[1], 1, 3) for b in bs]),
               latent_corr_intercept=r(latent_corr(S4, 0, 2), 4), latent_corr_intercept_ci=ci([latent_corr(b[1], 0, 2) for b in bs]))
    OUT["latent_split_oddeven"].append(row)
    print("[%5.0fs]" % (time.time() - T0), row, flush=True)
dump(OUT, "RESULTS_v3_hier_items.json")
print("done")
