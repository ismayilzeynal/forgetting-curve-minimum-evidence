# -*- coding: utf-8 -*-
# Shared data loading and batched estimators for the v3 analyses. Every learner-level fit is the
# same Newton/IRLS update as the v2 per-learner loop, run for all groups at once with bincount sums.
import os, json, math
import numpy as np, pandas as pd

SEED = 20260712
BINS = [(20, 30), (30, 50), (50, 80), (80, 130), (130, 250), (250, 10**9)]
BINLAB = ["20-29", "30-49", "50-79", "80-129", "130-249", "250+"]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
LN2 = math.log(2.0)


def load(cache):
    df = pd.read_parquet(os.path.join(cache, "sample40k.parquet"))
    uid, g = np.unique(df["user_id"].values, return_inverse=True)
    G = len(uid)
    n = np.bincount(g, minlength=G)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    pos = np.arange(len(df)) - start[g]
    D = dict(
        G=G, uid=uid, g=g.astype(np.int64), n=n, pos=pos, nrow=n[g],
        ll=np.log(df["delta"].values.astype("float64")),
        delta_days=df["delta"].values.astype("float64") / 86400.0,
        ss=df["session_seen"].values.astype("float64").clip(1),
        sc=df["session_correct"].values.astype("float64").clip(0),
        hs=df["history_seen"].values.astype("float64"),
        hc=df["history_correct"].values.astype("float64"),
        lex=pd.factorize(df["lexeme_id"].values)[0].astype(np.int64),
        lp=pd.factorize(df["learning_language"].astype(str) + ">" + df["ui_language"].astype(str))[0].astype(np.int64),
        ts=df["timestamp"].values.astype("int64"),
    )
    mean_ll = np.bincount(D["g"], D["ll"], G) / n
    D["zc"] = D["ll"] - mean_ll[D["g"]]
    return D


def binof(n):
    out = np.full(len(n), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(BINS):
        out[(n >= lo) & (n < hi)] = i
    return out


def group_ok(g, z, G, need=5):
    cnt = np.bincount(g, minlength=G).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.bincount(g, z, G) / cnt
        v = np.bincount(g, (z - m[g]) ** 2, G) / cnt
    return (cnt >= need) & (np.sqrt(np.nan_to_num(v, nan=0.0)) >= 1e-6)


def fit_ab(g, z, sc, ss, G, off=None, ridge_a=0.0, ridge_b=1e-2, it=30, tol=1e-7, need=5):
    """Batched per-group weighted binomial logistic fit logit p = a + b z (+ off).
    Returns a, b, var_b (sandwich), var_a (sandwich); NaN for groups failing the v2 checks."""
    ok = group_ok(g, z, G, need)
    a = np.zeros(G); b = np.zeros(G); active = ok.copy()
    o = np.zeros(len(z)) if off is None else off
    for _ in range(it):
        ra = active[g]
        if not ra.any():
            break
        gg = g[ra]; zz = z[ra]; s = ss[ra]; c = sc[ra]
        eta = np.clip(a[gg] + b[gg] * zz + o[ra], -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta)); W = s * p * (1 - p) + 1e-9; r = c - s * p
        H00 = np.bincount(gg, W, G) + ridge_a; H01 = np.bincount(gg, W * zz, G)
        H11 = np.bincount(gg, W * zz * zz, G) + ridge_b
        g0 = np.bincount(gg, r, G) - ridge_a * a; g1 = np.bincount(gg, r * zz, G) - ridge_b * b
        det = H00 * H11 - H01 * H01
        with np.errstate(invalid="ignore", divide="ignore"):
            sa = (H11 * g0 - H01 * g1) / det; sb = (H00 * g1 - H01 * g0) / det
        upd = active & np.isfinite(sa) & np.isfinite(sb) & (det != 0)
        a[upd] += sa[upd]; b[upd] += sb[upd]
        conv = np.maximum(np.abs(sa), np.abs(sb)) < tol
        active = active & upd & ~conv
    # sandwich variance of the ridge estimator at the solution
    rr = ok[g]
    gg = g[rr]; zz = z[rr]; s = ss[rr]
    eta = np.clip(a[gg] + b[gg] * zz + o[rr], -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta)); W = s * p * (1 - p)
    h00 = np.bincount(gg, W, G); h01 = np.bincount(gg, W * zz, G); h11 = np.bincount(gg, W * zz * zz, G)
    A00 = h00 + ridge_a; A11 = h11 + ridge_b; A01 = h01
    det = A00 * A11 - A01 * A01
    with np.errstate(invalid="ignore", divide="ignore"):
        i00 = A11 / det; i01 = -A01 / det; i11 = A00 / det
        var_b = i01 * i01 * h00 + 2 * i01 * i11 * h01 + i11 * i11 * h11
        var_a = i00 * i00 * h00 + 2 * i00 * i01 * h01 + i01 * i01 * h11
    a[~ok] = np.nan; b[~ok] = np.nan; var_b[~ok] = np.nan; var_a[~ok] = np.nan
    return a, b, var_b, var_a


def fit_a_given_b(g, z, sc, ss, G, b, off=None, it=40, tol=1e-9):
    """Per-group intercept with the slope held fixed at b[g] (b may be scalar)."""
    cnt = np.bincount(g, minlength=G)
    bb = np.broadcast_to(np.asarray(b, float), (G,)) if np.ndim(b) == 0 else b
    o = np.zeros(len(z)) if off is None else off
    a = np.zeros(G)
    for _ in range(it):
        eta = np.clip(a[g] + bb[g] * z + o, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta)); W = ss * p * (1 - p) + 1e-9
        step = np.bincount(g, sc - ss * p, G) / np.bincount(g, W, G)
        step = np.nan_to_num(step)
        a += step
        if np.max(np.abs(step)) < tol:
            break
    a[cnt == 0] = np.nan
    return a


def logloss_rows(eta, sc, ss):
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))), 1e-6, 1 - 1e-6)
    return -(sc * np.log(p) + (ss - sc) * np.log(1 - p))


def sb(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 20 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    return 2 * r / (1 + r) if (1 + r) > 1e-9 else float("nan")


def boot_sb(a, b, rng, nb=1000):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 20:
        return [None, None]
    vals = []
    for _ in range(nb):
        idx = rng.integers(0, len(a), len(a))
        v = sb(a[idx], b[idx])
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def permute_within(g_rows, vals, rng):
    """Permute vals within groups; g_rows must be non-decreasing."""
    order = np.lexsort((rng.random(len(vals)), g_rows))
    return vals[order]


def r(x, k=4):
    if x is None:
        return None
    try:
        if not np.isfinite(x):
            return None
    except TypeError:
        return x
    return round(float(x), k)


def dump(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


S_MIN, S_MAX = math.log2(15.0 / 1440.0), math.log2(274.0)


def hlr_terms(s_raw, dd, sc_, ss_):
    """Binomial half-life regression terms for log2 half-life s (days) and lag dd (days):
    per-row loss, d loss / d s, Fisher information for a shift in s, and recall probability."""
    s = np.clip(s_raw, S_MIN, S_MAX); free_s = (s_raw > S_MIN) & (s_raw < S_MAX)
    q_raw = LN2 * dd * np.exp2(-s); q = np.clip(q_raw, -math.log(0.9999), math.log(1e4))
    free_q = (q_raw > -math.log(0.9999)) & (q_raw < math.log(1e4))
    p = np.exp(-q); omp = -np.expm1(-q)
    loss = sc_ * q - (ss_ - sc_) * np.log(omp)
    dLds = LN2 * q * ((ss_ - sc_) * p / omp - sc_) * free_s * free_q
    info = ss_ * p * (LN2 * q) ** 2 / omp * free_s * free_q
    return loss, dLds, info, p


def fit_u(g, s_base, dd, sc, ss, G, ridge=1e-2, it=30, tol=1e-7, need=5):
    """Batched per-group half-life multiplier u (log2 h = s_base + u) by Fisher scoring.
    Returns u and its sandwich variance; NaN for groups with fewer than `need` rows."""
    cnt = np.bincount(g, minlength=G); ok = cnt >= need
    u = np.zeros(G); active = ok.copy()
    for _ in range(it):
        ra = active[g]
        if not ra.any():
            break
        gg = g[ra]
        _, dLds, info, _ = hlr_terms(s_base[ra] + u[gg], dd[ra], sc[ra], ss[ra])
        grad = np.bincount(gg, dLds, G) + ridge * u
        I = np.bincount(gg, info, G) + ridge
        step = -grad / I
        upd = active & np.isfinite(step)
        u[upd] += step[upd]
        active = active & upd & ~(np.abs(step) < tol)
    rr = ok[g]
    _, _, info, _ = hlr_terms(s_base[rr] + u[g[rr]], dd[rr], sc[rr], ss[rr])
    I0 = np.bincount(g[rr], info, G)
    var = I0 / (I0 + ridge) ** 2
    u[~ok] = np.nan; var[~ok] = np.nan
    return u, var
