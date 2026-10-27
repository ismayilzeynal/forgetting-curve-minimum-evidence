# -*- coding: utf-8 -*-
# Hierarchical (empirical Bayes) estimators used after the addendum's moment-based variance
# decomposition proved outlier-driven (quasi-separated per-learner fits). Each group g has a
# K-vector of coefficients theta_g ~ N(mu, Sigma); rows contribute a binomial likelihood with
# logit p = X_row . theta_g + off_row. mu and Sigma are estimated by Laplace-approximate EM.
# Implementation: each outer round expands every group's log-likelihood to second order at its
# current posterior mode (one pass over the rows), then runs EM to convergence on those
# expansions (cheap, group-sized arrays). At the fixed point the expansion point equals the
# posterior mode, so the solution is the Laplace-EM solution. The same stored expansions give a
# learner bootstrap of (mu, Sigma) without further passes over the rows.
import math
import numpy as np
from common import LN2


def hlr_terms(s, dd, sc, ss):
    """Binomial half-life regression terms for log2 half-life s (days) and lag dd (days), without
    clipping s: the learner multiplier must keep a smooth likelihood (clipping s, as the population
    fit does, zeroes the information of learners near the clip and lets their multiplier drift).
    q = ln2 * dd / h is bounded only for numerical safety. Returns per-row loss, d loss / d s,
    Fisher information for a shift in s, and recall probability."""
    q = np.clip(LN2 * dd * np.exp2(-s), 1e-10, 50.0)
    p = np.exp(-q); omp = -np.expm1(-q)
    loss = sc * q - (ss - sc) * np.log(omp)
    dLds = LN2 * q * ((ss - sc) * p / omp - sc)
    info = ss * p * (LN2 * q) ** 2 / omp
    return loss, dLds, info, p


def _pairs(K):
    return [(k, l) for k in range(K) for l in range(k, K)]


def group_stats(g, X, sc, ss, theta, off, G):
    """Per-group log-likelihood gradient (K) and information (K x K) at theta."""
    K = X.shape[1]
    if K == 2:
        eta = theta[g, 0] * X[:, 0] + theta[g, 1] * X[:, 1] + off
    else:
        eta = np.einsum("ij,ij->i", X, theta[g]) + off
    eta = np.clip(eta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = ss * p * (1 - p)
    rr = sc - ss * p
    grad = np.stack([np.bincount(g, rr * X[:, k], G) for k in range(K)], axis=1)
    H = np.zeros((G, K, K))
    for k, l in _pairs(K):
        v = np.bincount(g, W * X[:, k] * X[:, l], G)
        H[:, k, l] = v; H[:, l, k] = v
    return grad, H


def _inv_small(A):
    """Batched inverse with closed forms for 1 x 1 and 2 x 2 blocks."""
    K = A.shape[1]
    if K == 1:
        return 1.0 / A
    if K == 2:
        a, b, c, d = A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1]
        det = a * d - b * c
        out = np.empty_like(A)
        out[:, 0, 0] = d / det; out[:, 0, 1] = -b / det; out[:, 1, 0] = -c / det; out[:, 1, 1] = a / det
        return out
    return np.linalg.inv(A)


def em_on_expansions(theta_c, grad, H, use, w, mu, Sigma, tol=1e-9, it_max=20000, step_max=5.0):
    """Maximize the marginal likelihood of (mu, Sigma) when each group's log-likelihood is the
    quadratic expansion b'(t) - 1/2 t' H t with b = grad + H theta_c (the Laplace-EM fixed point
    on these expansions). The Gaussian integral is closed-form:
      sum_g w_g [1/2 log|P| - 1/2 log|H_g + P| + 1/2 m_g'(H_g + P)^-1 m_g - 1/2 mu' P mu],
    with P = Sigma^-1 and m_g = b_g + P mu; it is maximized by quasi-Newton over mu and the
    Cholesky factor of Sigma (log diagonal). Returns mu, Sigma, modes, posterior covariances."""
    from scipy.optimize import minimize
    K = theta_c.shape[1]
    sel = (w * use) > 0
    tc, Hs, ww = theta_c[sel], H[sel], (w * use)[sel]
    bb = grad[sel] + np.einsum("gkl,gl->gk", Hs, tc)
    sw = ww.sum()
    tri = np.tril_indices(K)
    Lc0 = np.linalg.cholesky(Sigma)
    sc_mu = np.sqrt(np.diag(Sigma))

    def unpack(x):
        m_ = mu + x[:K] * sc_mu
        Lm = np.zeros((K, K)); Lm[tri] = x[K:]
        Lm[np.diag_indices(K)] = np.exp(Lm[np.diag_indices(K)])
        Lc = Lc0 @ Lm
        return m_, Lc

    def nll(x):
        m_, Lc = unpack(x)
        S = Lc @ Lc.T
        P = np.linalg.inv(S)
        A = Hs + P[None]
        Ai = _inv_small(A)
        m = bb + P @ m_
        quad = np.einsum("gk,gkl,gl->g", m, Ai, m)
        sign, logdetA = np.linalg.slogdet(A) if K > 2 else (None, None)
        if K == 1:
            logdetA = np.log(A[:, 0, 0])
        elif K == 2:
            logdetA = np.log(A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0])
        logdetP = -2 * np.sum(np.log(np.abs(np.diag(Lc))))
        val = ww @ (0.5 * logdetP - 0.5 * logdetA + 0.5 * quad) - 0.5 * sw * float(m_ @ P @ m_)
        return -val / sw

    x0 = np.zeros(K + len(tri[0]))
    res = minimize(nll, x0, method="BFGS", options=dict(gtol=1e-9, maxiter=2000))
    mu, Lc = unpack(res.x)
    Sigma = Lc @ Lc.T
    P = np.linalg.inv(Sigma)
    V = _inv_small(Hs + P[None])
    th = np.einsum("gkl,gl->gk", V, bb + P @ mu)
    th = tc + np.clip(th - tc, -step_max, step_max)
    th_full = np.full((len(sel), K), np.nan); th_full[sel] = th
    V_full = np.full((len(sel), K, K), np.nan); V_full[sel] = V
    return mu, Sigma, th_full, V_full, int(res.nit)


def pooled_fit(g, X, sc, ss, off):
    K = X.shape[1]
    th = np.zeros((1, K)); z = np.zeros(len(g), np.int64)
    for _ in range(40):
        gr, H = group_stats(z, X, sc, ss, th, off, 1)
        st = np.linalg.solve(H[0] + 1e-9 * np.eye(K), gr[0]); th[0] += st
        if np.max(np.abs(st)) < 1e-10:
            break
    return th[0]


def group_obj(g, X, sc, ss, theta, off, G, mu, P):
    """Per-group penalized log-likelihood (up to constants)."""
    K = X.shape[1]
    if K == 2:
        eta = theta[g, 0] * X[:, 0] + theta[g, 1] * X[:, 1] + off
    else:
        eta = np.einsum("ij,ij->i", X, theta[g]) + off
    eta = np.clip(eta, -30, 30)
    ll = sc * eta - ss * np.logaddexp(0.0, eta)
    d = theta - mu
    return np.bincount(g, ll, G) - 0.5 * np.einsum("gk,kl,gl->g", d, P, d)


def posterior_modes(g, X, sc, ss, G, mu, Sigma, off, theta, newton=1, final=False, step_max=5.0, tol=1e-9):
    """Damped Newton: a group's step is halved (up to 8 times) when it lowers that group's
    penalized log-likelihood, so separated groups cannot oscillate."""
    P = np.linalg.inv(Sigma)
    H = None
    f0 = group_obj(g, X, sc, ss, theta, off, G, mu, P)
    for _ in range(newton):
        grad, H = group_stats(g, X, sc, ss, theta, off, G)
        step = np.linalg.solve(H + P[None], (grad - (theta - mu) @ P.T)[:, :, None])[:, :, 0]
        step = np.clip(step, -step_max, step_max)
        lam = np.ones(G)
        for _h in range(9):
            cand = theta + lam[:, None] * step
            f1 = group_obj(g, X, sc, ss, cand, off, G, mu, P)
            bad = f1 < f0 - 1e-10
            if not bad.any() or _h == 8:
                lam[bad] = 0.0
                cand = theta + lam[:, None] * step
                f1 = np.where(bad, f0, f1)
                break
            lam[bad] *= 0.5
        theta = cand; f0 = f1
        if np.max(np.abs(lam[:, None] * step)) < tol:
            break
    if final or H is None:
        grad, H = group_stats(g, X, sc, ss, theta, off, G)
    V = np.linalg.inv(H + P[None])
    return theta, V, H, grad


def eb_logistic(g, X, sc, ss, G, off=None, groups=None, mu0=None, Sigma0=None, theta0=None,
                it_em=300, tol=1e-6, switch=1e-3, polish=12, verbose=False):
    """Laplace-EM for a binomial-logit model with group-level Gaussian coefficients.
    Each EM iteration takes one exact Newton step towards every group's posterior mode and
    uses the Laplace covariance; at the fixed point modes are exact. groups: mask of groups
    entering the M-step (default: all groups with rows)."""
    K = X.shape[1]
    off = np.zeros(len(g)) if off is None else off
    cnt = np.bincount(g, minlength=G)
    use = (cnt > 0) if groups is None else (groups & (cnt > 0))
    mu = pooled_fit(g, X, sc, ss, off) if mu0 is None else np.asarray(mu0, float).copy()
    Sigma = np.diag(np.full(K, 0.25)) if Sigma0 is None else np.asarray(Sigma0, float).copy()
    theta = np.tile(mu, (G, 1)) if theta0 is None else theta0.copy()
    conv = False
    sw = use.sum()
    for it in range(it_em):
        theta, V, H, _ = posterior_modes(g, X, sc, ss, G, mu, Sigma, off, theta, newton=1)
        mu_n = theta[use].sum(0) / sw
        d = theta[use] - mu_n
        S_n = (d[:, :, None] * d[:, None, :] + V[use]).sum(0) / sw
        S_n = (S_n + S_n.T) / 2
        ev, evec = np.linalg.eigh(S_n)
        S_n = (evec * np.maximum(ev, 1e-12)) @ evec.T
        sd = np.sqrt(np.diag(S_n))
        delta = max(np.max(np.abs(mu_n - mu) / sd), np.max(np.abs(S_n - Sigma) / np.outer(sd, sd)))
        mu, Sigma = mu_n, S_n
        if verbose and it % 5 == 0:
            print("  em", it, "mu", np.round(mu, 5), "diagS", np.round(np.diag(Sigma), 6), "delta", delta, flush=True)
        if delta < switch:
            break
    # phase 2: exact modes under the current prior, then EM on the local quadratic expansions
    # (accurate near the fixed point); repeat until (mu, Sigma) stop moving.
    for rnd in range(polish):
        theta, V, H, grad = posterior_modes(g, X, sc, ss, G, mu, Sigma, off, theta, newton=40, final=True)
        mu_n, S_n, _, _, nit = em_on_expansions(theta, grad, H, use, np.ones(G), mu, Sigma, tol=1e-9,
                                                it_max=200000, step_max=1.0)
        sd = np.sqrt(np.diag(S_n))
        delta = max(np.max(np.abs(mu_n - mu) / sd), np.max(np.abs(S_n - Sigma) / np.outer(sd, sd)))
        mu, Sigma = mu_n, S_n
        if verbose:
            print("  polish", rnd, "inner", nit, "mu", np.round(mu, 5), "diagS", np.round(np.diag(Sigma), 6), "delta", delta, flush=True)
        if delta < tol:
            conv = True
            break
    theta, V, H, grad = posterior_modes(g, X, sc, ss, G, mu, Sigma, off, theta, newton=40, final=True)
    theta[~use] = mu
    return dict(mu=mu, Sigma=Sigma, theta=theta, V=V, Hlik=H, grad=grad, use=use, n_iter=it + 1,
                n_polish=rnd + 1, converged=conv)


def bootstrap_hyper(R, rng, nb=1000, groups_mask=None):
    """Learner bootstrap of (mu, Sigma) from the stored expansions at the full-sample modes."""
    use = R["use"] if groups_mask is None else (R["use"] & groups_mask)
    idx = np.where(use)[0]
    theta_c = R["theta"][idx]; grad = R["grad"][idx]; H = R["Hlik"][idx]
    out = []
    for _ in range(nb):
        cnt = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
        mu, S, _, _, _ = em_on_expansions(theta_c, grad, H, np.ones(len(idx), bool), cnt, R["mu"], R["Sigma"],
                                          tol=1e-6, it_max=20000)
        out.append((mu, S))
    return out


def subset_hyper(R, mask):
    """Re-estimate (mu, Sigma) for a subset of groups from the stored expansions (for bins)."""
    idx = np.where(R["use"] & mask)[0]
    mu, S, th, V, _ = em_on_expansions(R["theta"][idx], R["grad"][idx], R["Hlik"][idx],
                                       np.ones(len(idx), bool), np.ones(len(idx)), R["mu"], R["Sigma"])
    return mu, S


def lik_var(H, k):
    """Sampling variance of coefficient k from the likelihood information alone (the other
    coefficients profiled out); inf where the information matrix is (near) singular."""
    det = np.linalg.det(H)
    K = H.shape[1]
    out = np.full(H.shape[0], np.inf)
    ok = det > 1e-10 * np.maximum(1.0, np.abs(H).max(axis=(1, 2))) ** K
    out[ok] = np.linalg.inv(H[ok])[:, k, k]
    out[~np.isfinite(out) | (out <= 0)] = np.inf
    return out


def latent_corr(Sigma, i, j):
    return float(Sigma[i, j] / math.sqrt(Sigma[i, i] * Sigma[j, j]))


def weak_prior_fit(g, X, sc, ss, G, mu, prior_sd, off=None, theta=None, newton=60):
    """Individual estimates under a fixed weakly informative prior N(mu, diag(prior_sd^2))."""
    off = np.zeros(len(g)) if off is None else off
    mu = np.asarray(mu, float)
    P = np.diag(1.0 / np.asarray(prior_sd, float) ** 2)
    theta = np.tile(mu, (G, 1)) if theta is None else theta.copy()
    for _ in range(newton):
        grad, H = group_stats(g, X, sc, ss, theta, off, G)
        step = np.linalg.solve(H + P[None], (grad - (theta - mu) @ P.T)[:, :, None])[:, :, 0]
        step = np.clip(step, -5, 5)
        theta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    _, H = group_stats(g, X, sc, ss, theta, off, G)
    return theta, H


# ---------- one-dimensional HLR multiplier ----------

def hlr_stats(g, s_base, u, dd, sc, ss, G):
    _, dLds, info, _ = hlr_terms(s_base + u[g], dd, sc, ss)
    return -np.bincount(g, dLds, G), np.bincount(g, info, G)


def _hlr_obj(g, s_base, u, dd, sc, ss, G, mu, tau2):
    loss, _, _, _ = hlr_terms(s_base + u[g], dd, sc, ss)
    return -np.bincount(g, loss, G) - 0.5 * (u - mu) ** 2 / tau2


def hlr_modes(g, s_base, dd, sc, ss, G, mu, tau2, u, newton=3):
    """Damped Fisher-scoring steps towards each group's posterior mode of u under N(mu, tau2)."""
    f0 = _hlr_obj(g, s_base, u, dd, sc, ss, G, mu, tau2)
    for _ in range(newton):
        gr, I = hlr_stats(g, s_base, u, dd, sc, ss, G)
        step = np.clip((gr - (u - mu) / tau2) / (I + 1.0 / tau2), -2, 2)
        lam = np.ones(G)
        for h in range(9):
            cand = u + lam * step
            f1 = _hlr_obj(g, s_base, cand, dd, sc, ss, G, mu, tau2)
            bad = f1 < f0 - 1e-10
            if not bad.any() or h == 8:
                lam[bad] = 0.0; cand = u + lam * step; f1 = np.where(bad, f0, f1)
                break
            lam[bad] *= 0.5
        u, f0 = cand, f1
    gr, I = hlr_stats(g, s_base, u, dd, sc, ss, G)
    return u, gr, I


def eb_hlr(g, s_base, dd, sc, ss, G, groups=None, it_em=300, tol=1e-6, mu0=0.0, tau2_0=1.0):
    """Laplace-EM for log2 h = s_base + u_g, u_g ~ N(mu, tau2), binomial HLR likelihood with
    Fisher information as the curvature; damped mode updates."""
    cnt = np.bincount(g, minlength=G)
    use = (cnt > 0) if groups is None else (groups & (cnt > 0))
    mu, tau2 = float(mu0), float(tau2_0)
    u = np.full(G, mu)
    conv = False
    for it in range(it_em):
        u, gr, I = hlr_modes(g, s_base, dd, sc, ss, G, mu, tau2, u)
        v = 1.0 / (I + 1.0 / tau2)
        mu_n = float(u[use].mean()); t_n = float(np.mean((u[use] - mu_n) ** 2 + v[use]))
        d = max(abs(mu_n - mu) / math.sqrt(t_n), abs(t_n - tau2) / t_n)
        mu, tau2 = mu_n, t_n
        if d < tol:
            conv = True
            break
    u, gr, I = hlr_modes(g, s_base, dd, sc, ss, G, mu, tau2, u, newton=10)
    v = 1.0 / (I + 1.0 / tau2)
    u[~use] = mu
    return dict(mu=np.array([mu]), Sigma=np.array([[tau2]]), theta=u[:, None], V=v[:, None, None],
                Hlik=I[:, None, None], grad=(gr - 0.0)[:, None], use=use, n_iter=it + 1, converged=conv)


def weak_prior_hlr(g, s_base, dd, sc, ss, G, mu, prior_sd, newton=60):
    u = np.full(G, float(mu)); p0 = 1.0 / prior_sd ** 2
    for _ in range(newton):
        gr, I = hlr_stats(g, s_base, u, dd, sc, ss, G)
        step = np.clip((gr - p0 * (u - mu)) / (I + p0), -5, 5)
        u += step
        if np.max(np.abs(step)) < 1e-9:
            break
    _, I = hlr_stats(g, s_base, u, dd, sc, ss, G)
    return u, I
