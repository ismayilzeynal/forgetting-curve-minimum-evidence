# -*- coding: utf-8 -*-
# A1/A2 population models, cross-fitted over two learner folds: the strength offset for the
# adjusted forgetting rate, and half-life regression (HLR) for the half-life multiplier.
# Writes <cache>/popmodels.npz (per-row offsets) and results/population_models.json.
import sys, os
import numpy as np
from scipy.optimize import minimize
from common import *

CACHE = sys.argv[1]
D = load(CACHE); G = D["G"]; g = D["g"]
rng = np.random.default_rng(SEED + 1)
fold_u = rng.permutation(np.arange(G) % 2)
fold = fold_u[g]
sc, ss, ll = D["sc"], D["ss"], D["ll"]
NLEX = int(D["lex"].max()) + 1; NLP = int(D["lp"].max()) + 1
S_MIN, S_MAX = math.log2(15.0 / 1440.0), math.log2(274.0)

def lex_difficulty(m):
    pbar = sc[m].sum() / ss[m].sum()
    num = np.bincount(D["lex"][m], sc[m], NLEX) + 20 * pbar
    den = np.bincount(D["lex"][m], ss[m], NLEX) + 20
    pl = num / den
    return np.log(pl / (1 - pl))

def irls(X, sc_, ss_, it=25):
    beta = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))); W = ss_ * p * (1 - p) + 1e-9
        H = (X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (sc_ - ss_ * p)); beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta

def strength_X(m, dlex):
    lpd = np.zeros((m.sum(), NLP - 1))
    lpi = D["lp"][m]; sel = lpi > 0
    lpd[np.where(sel)[0], lpi[sel] - 1] = 1.0
    hs = D["hs"][m]; hc = D["hc"][m]
    return np.column_stack([dlex[D["lex"][m]], np.log1p(hs), (hc + 1) / (hs + 2), lpd])

def hlr_feats(m):
    hc = D["hc"][m]; hw = D["hs"][m] - hc
    return np.column_stack([np.ones(m.sum()), np.sqrt(1 + hc), np.sqrt(1 + np.maximum(hw, 0))])

def hlr_terms(s_raw, dd, sc_, ss_):
    s = np.clip(s_raw, S_MIN, S_MAX); free_s = (s_raw > S_MIN) & (s_raw < S_MAX)
    q_raw = LN2 * dd * np.exp2(-s); q = np.clip(q_raw, -math.log(0.9999), math.log(1e4))
    free_q = (q_raw > -math.log(0.9999)) & (q_raw < math.log(1e4))
    p = np.exp(-q); omp = -np.expm1(-q)
    loss = sc_ * q - (ss_ - sc_) * np.log(omp)
    dLds = LN2 * q * ((ss_ - sc_) * p / omp - sc_) * free_s * free_q
    info = ss_ * p * (LN2 * q) ** 2 / omp * free_s * free_q
    return loss, dLds, info, p

def fit_hlr(m, l2=0.1):
    Xd = hlr_feats(m); lx = D["lex"][m]; dd = D["delta_days"][m]; sc_ = sc[m]; ss_ = ss[m]
    nd = Xd.shape[1]
    def f(th):
        s_raw = Xd @ th[:nd] + th[nd:][lx]
        loss, dLds, _, _ = hlr_terms(s_raw, dd, sc_, ss_)
        gr = np.concatenate([Xd.T @ dLds, np.bincount(lx, dLds, NLEX) + 2 * l2 * th[nd:]])
        return loss.sum() + l2 * np.sum(th[nd:] ** 2), gr
    th0 = np.zeros(nd + NLEX); th0[0] = 1.0
    res = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000, maxfun=4000))
    return res.x, res

off_adj = np.zeros(len(g)); s_pop = np.zeros(len(g)); coefs = {}
for F in (0, 1):
    mF = fold == F; mO = ~mF
    dlex = lex_difficulty(mF)
    XF = np.column_stack([np.ones(mF.sum()), strength_X(mF, dlex), ll[mF]])
    beta = irls(XF, sc[mF], ss[mF])
    XO = strength_X(mO, dlex)
    off_adj[mO] = XO @ beta[1:-1]
    th, res = fit_hlr(mF)
    s_pop[mO] = hlr_feats(mO) @ th[:3] + th[3:][D["lex"][mO]]
    # fit quality on the other fold
    loss, _, _, p = hlr_terms(s_pop[mO], D["delta_days"][mO], sc[mO], ss[mO])
    coefs["fold%d" % F] = dict(
        strength_model=dict(intercept=r(beta[0]), d_lex=r(beta[1]), log1p_history_seen=r(beta[2]),
                            smoothed_accuracy=r(beta[3]), log_lag=r(beta[-1]), n_langpair_dummies=int(NLP - 1)),
        hlr=dict(bias=r(th[0]), sqrt_1p_history_correct=r(th[1]), sqrt_1p_history_wrong=r(th[2]),
                 converged=bool(res.success), iterations=int(res.nit),
                 other_fold_logloss_per_trial=r(loss.sum() / ss[mO].sum()),
                 other_fold_mae_p_recall=r(np.mean(np.abs(p - sc[mO] / ss[mO])))))
    print(F, coefs["fold%d" % F], flush=True)

np.savez_compressed(os.path.join(CACHE, "popmodels.npz"), off_adj=off_adj, s_pop=s_pop, fold_u=fold_u)
h = np.exp2(np.clip(s_pop, S_MIN, S_MAX))
dump(dict(folds="two random learner folds (seed 20260713); every learner's offsets come from the model fit on the other fold",
          lexemes_in_sample=NLEX, language_pairs=NLP, coefficients=coefs,
          hlr_population_half_life_days=dict(median=r(np.median(h)), p25=r(np.percentile(h, 25)), p75=r(np.percentile(h, 75)))),
     "population_models.json")
print("done")
