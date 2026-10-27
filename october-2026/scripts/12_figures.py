# -*- coding: utf-8 -*-
# Figures 1 to 6 from the result files only. Usage: python 12_figures.py
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, BINLAB

# v4: results are read from the frozen v3 folder unless a results/ folder sits next to scripts/
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(RESULTS):
    RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "v3-cbb-2026-10", "results")
RESULTS = os.environ.get("CBB_RESULTS") or RESULTS
FIG = os.path.join(os.path.dirname(HERE), "draft", "figures")
os.makedirs(FIG, exist_ok=True)
J = lambda f: json.load(open(os.path.join(RESULTS, f), encoding="utf-8"))
# v4 revision: lettering at least 8 pt, mathtext in Arial, plain tick labels on
# log axes (no superscripted exponents), PNG written as RGB at 600 dpi
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600, "legend.frameon": False,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.6, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
# axis labels use Arial Unicode glyphs (tau, rho, superscript two) instead of mathtext, so no
# DejaVu Sans glyphs enter the figures
C = dict(raw="#1f4e79", adj="#c55a11", icpt="#548235", hlr="#7030a0", item="#7f7f7f")
W = 6.85
x = np.arange(len(BINLAB))
from matplotlib.ticker import FuncFormatter, NullFormatter
PLAIN = FuncFormatter(lambda v, _: ("%.10f" % v).rstrip("0").rstrip("."))


def plain_log(a):
    a.yaxis.set_major_formatter(PLAIN)
    a.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name.replace(".png", ".pdf")), bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    p = os.path.join(FIG, name)
    im = Image.open(p)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
    bg.save(p, dpi=(600, 600))


def err(rows, key, cikey):
    v = np.array([r_[key] for r_ in rows], float)
    lo = np.array([r_[cikey][0] for r_ in rows], float); hi = np.array([r_[cikey][1] for r_ in rows], float)
    # point values are stored rounded (e.g. n* to 1 decimal), so clip the bar lengths at zero
    return v, np.vstack([np.maximum(v - lo, 0), np.maximum(hi - v, 0)])


raw, adj, hlr = J("RESULTS_v3_hier_raw.json"), J("RESULTS_v3_hier_adj.json"), J("RESULTS_v3_hier_hlr.json")
items = J("RESULTS_v3_hier_items.json")

# ---------- Fig 1: identifiability by evidence ----------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
for key, lab, mk, src in (("raw", "Rate (raw)", "o", raw), ("adj", "Rate (strength-adjusted)", "s", adj),
                          ("hlr", "Half-life multiplier", "D", hlr)):
    ax[0].plot(x, [b["mean_rho"] for b in src["bins"]], marker=mk, color=C[key], label=lab, lw=1.2, ms=4)
ax[0].plot(x, [b["mean_rho_intercept"] for b in raw["bins"]], marker="^", color=C["icpt"], label="Recall level (intercept)", lw=1.2, ms=4)
for yv in (0.5, 0.7):
    ax[0].axhline(yv, color="0.6", ls=":", lw=0.8)
ax[0].set_xticks(x, BINLAB, rotation=30); ax[0].set_ylim(0, 1); ax[0].set_xlabel("Traces per learner")
ax[0].set_ylabel("Mean model-based reliability"); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2); ax[0].set_title("a", loc="left", fontweight="bold")
ls_ = raw["latent_split"]
rows = [r_ for r_ in ls_["chrono"] if r_["bin"] != "all"]
v, e = err(rows, "latent_corr_rate", "latent_corr_rate_ci")
ax[1].errorbar(x - 0.08, v, yerr=e, marker="o", color=C["raw"], ms=4, lw=1, capsize=2, label="Rate")
v, e = err(rows, "latent_corr_intercept", "latent_corr_intercept_ci")
ax[1].errorbar(x + 0.08, v, yerr=e, marker="^", ls="--", color=C["icpt"], ms=4, lw=1, capsize=2, label="Level")
# v4 revision: the 20-29 four-dimensional fit did not converge; draw its points open
for j_, r_ in enumerate(rows):
    if not r_.get("converged", True):
        ax[1].plot(x[j_] - 0.08, r_["latent_corr_rate"], "o", mfc="white", mec=C["raw"], ms=4, zorder=5)
        ax[1].plot(x[j_] + 0.08, r_["latent_corr_intercept"], "^", mfc="white", mec=C["icpt"], ms=4, zorder=5)
ax[1].axhline(0, color="0.6", lw=0.6); ax[1].axhline(0.5, color="0.6", ls=":", lw=0.8)
ax[1].set_xticks(x, BINLAB, rotation=30); ax[1].set_ylim(-0.2, 1.05); ax[1].set_xlabel("Traces per learner")
ax[1].set_ylabel("First/second-half" + chr(10) + "latent correlation"); ax[1].legend(fontsize=8, loc="lower left"); ax[1].set_title("b", loc="left", fontweight="bold")
save(fig, "Fig2.png")

# ---------- Fig 2: variance, error and minimum evidence ----------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
for key, src, mk, off in (("raw", raw, "o", -0.1), ("adj", adj, "s", 0.1)):
    v, e = err(src["bins"], "tau2", "tau2_ci")
    ax[0].errorbar(x + off, v, yerr=e, marker=mk, color=C[key], ms=4, lw=1, capsize=2,
                   label="Rate, %s" % ("raw" if key == "raw" else "strength-adjusted"))
    null = np.array([[p["bins_tau2_approx"][i] for i in range(6)] for p in src["permutation_null"]])
    ax[0].plot(x + off, null.mean(0), marker="x", ls=":", color=C[key], ms=4, lw=0.8,
               label="Lag-permutation null, %s" % ("raw" if key == "raw" else "adjusted"))
ax[0].set_yscale("log"); plain_log(ax[0]); ax[0].set_xticks(x, BINLAB, rotation=30); ax[0].set_xlabel("Traces per learner")
ax[0].set_ylabel("Between-learner variance τ²"); ax[0].legend(fontsize=8); ax[0].set_title("a", loc="left", fontweight="bold")
for key, src, mk, off in (("raw", raw, "o", -0.1), ("adj", adj, "s", 0.1), ("hlr", hlr, "D", 0.0)):
    v, e = err(src["bins"], "n_star_050", "n_star_050_ci")
    ax[1].errorbar(x + off, v, yerr=e, marker=mk, color=C[key], ms=4, lw=1, capsize=2,
                   label={"raw": "Rate, raw", "adj": "Rate, adjusted", "hlr": "Half-life multiplier"}[key])
# v4 revision: the 20-29 n* of both rates rests on nearly flat likelihoods (Table 2, note b); draw it open
for key, src, mk, off in (("raw", raw, "o", -0.1), ("adj", adj, "s", 0.1)):
    ax[1].plot(x[0] + off, src["bins"][0]["n_star_050"], mk, mfc="white", mec=C[key], ms=4, zorder=5)
ax[1].plot(x, [b["median_n"] for b in raw["bins"]], color="k", ls="--", lw=1, label="Median traces")
ax[1].set_yscale("log"); plain_log(ax[1]); ax[1].set_xticks(x, BINLAB, rotation=30); ax[1].set_xlabel("Traces per learner")
ax[1].set_ylabel("Traces needed for ρ = 0.5 (n*)"); ax[1].set_ylim(0.4, 1.2e4)
ax[1].legend(fontsize=8, loc="upper right", ncol=2); ax[1].set_title("b", loc="left", fontweight="bold")
save(fig, "Fig1.png")

# ---------- predicted versus observed gains (strength-adjusted rate; raw rate in Online Resource 1) ----------
# v4: three panels, because the naive losses (tens of millinats) hide the shrunk gains (about one)
fig, ax = plt.subplots(1, 3, figsize=(W, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
T = J("RESULTS_v3_theory_adj.json")


def pv(key):
    pk = "predicted_" + key
    ok_ = "observed_" + key.replace("_drift", "")
    px = np.array([b[pk] for b in T["bins"]]) * 1000
    py = np.array([b[ok_] for b in T["bins"]]) * 1000
    lo = np.array([b[ok_ + "_ci"][0] for b in T["bins"]]) * 1000
    hi = np.array([b[ok_ + "_ci"][1] for b in T["bins"]]) * 1000
    return px, py, np.vstack([np.maximum(py - lo, 0), np.maximum(hi - py, 0)]), lo, hi


for k_ax, keys in ((0, (("naive", "Naive", "o", "none"),)),
                   (1, (("shrunk", "Shrunk, stationary", "s", "none"),
                        ("shrunk_drift", "Shrunk, drift-aware", "s", C["adj"])))):
    allv = []
    for key, lab, mk, fill in keys:
        px, py, e, lo, hi = pv(key)
        ax[k_ax].errorbar(px, py, yerr=e, fmt=mk, mfc=fill, mec=C["adj"], ecolor=C["adj"], ms=4.5, lw=0.8,
                          capsize=1.5, label=lab)
        if key == "shrunk_drift":   # v4 revision: the 20-29 drift-aware point is not interpreted; half-filled
            ax[k_ax].plot(px[0], py[0], mk, ms=4.5, mec=C["adj"], mfc=C["adj"], fillstyle="left",
                          markerfacecoloralt="white", zorder=5)
        allv += list(px) + list(lo) + list(hi)
    lo_, hi_ = min(allv + [0]), max(allv + [0])
    pad = 0.08 * (hi_ - lo_)
    lim = [lo_ - pad, hi_ + pad]
    ax[k_ax].plot(lim, lim, color="0.5", lw=0.8, ls="--")
    ax[k_ax].axhline(0, color="0.8", lw=0.6); ax[k_ax].axvline(0, color="0.8", lw=0.6)
    ax[k_ax].set_xlim(lim); ax[k_ax].set_ylim(lim)
    if k_ax == 1:   # room above the points for the two-entry legend
        ax[k_ax].set_ylim(lim[0], lim[1] + 0.4 * (lim[1] - lim[0]))
    ax[k_ax].set_xlabel("Predicted gain (millinats per trial)")
    ax[k_ax].legend(fontsize=8, loc="upper left")
ax[0].set_ylabel("Observed held-out gain (millinats per trial)")
ax[0].set_title("a", loc="left", fontweight="bold"); ax[1].set_title("b", loc="left", fontweight="bold")
po = np.array([b["predicted_oracle"] for b in T["bins"]]) * 1000
os_ = np.array([b["observed_shrunk"] for b in T["bins"]]) * 1000
lv = np.array([b["observed_level_gain"] for b in T["level_control"]]) * 1000
ax[2].bar(x - 0.27, po, width=0.27, color="white", edgecolor=C["adj"], hatch="///", label="Rate: headroom (oracle)")
ax[2].bar(x, os_, width=0.27, color=C["adj"], label="Rate: observed, shrunk")
ax[2].bar(x + 0.27, lv, width=0.27, color="white", edgecolor=C["icpt"], hatch="...", label="Level: observed, shrunk")
ax[2].axhline(0, color="0.5", lw=0.6)
ax[2].set_xticks(x, BINLAB, rotation=30); ax[2].set_xlabel("Traces per learner"); ax[2].set_ylabel("Gain (millinats per trial)")
ax[2].set_ylim(top=max(list(po) + list(lv)) * 1.35)
ax[2].legend(fontsize=8, loc="upper left"); ax[2].set_title("c", loc="left", fontweight="bold")
save(fig, "Fig4.png")

# ---------- Fig 4: scheduling consequences ----------
PL = pd.read_csv(os.path.join(RESULTS, "per_learner_theory_hlr.csv.gz"))
v = PL["bin"].values >= 0
fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
bins_ = np.linspace(-6, 6, 97)
ax[0].hist((PL["u_naive"] - PL["u_pop"])[v], bins=bins_, color="white", edgecolor=C["hlr"], hatch="//", label="Naive estimate")
ax[0].hist((PL["u_eb"] - PL["u_pop"])[v], bins=bins_, color=C["hlr"], alpha=0.8, label="Shrunk estimate")
ax[0].set_xlabel("Interval change, log2(personal / population)"); ax[0].set_ylabel("Learners")
ax[0].legend(fontsize=8); ax[0].set_title("a", loc="left", fontweight="bold")
S = J("RESULTS_v3_sim_hlr.json")["hlr_world"]
pols = ["population", "naive", "shrunk", "oracle"]
for j, rs_ in enumerate((0.85, 0.90)):
    vals = [S["r%.2f_%s" % (rs_, p)]["mean_abs_dev"] for p in pols]
    ax[1].bar(np.arange(4) + (j - 0.5) * 0.38, vals, width=0.38, color=[C["hlr"], "white"][j], edgecolor=C["hlr"],
              hatch=[None, "..."][j], label="Target recall %.2f" % rs_)
ax[1].set_xticks(np.arange(4), [p.capitalize() for p in pols]); ax[1].set_ylabel("Mean |achieved recall - target|")
ax[1].legend(fontsize=8); ax[1].set_title("b  (semi-synthetic)", loc="left", fontweight="bold")
save(fig, "Fig5.png")

# ---------- Fig 5: semi-synthetic validation ----------
G_ = J("RESULTS_v3_sim_grid.json")["grid"]; P_ = J("RESULTS_v3_sim_probe_m1.json")
fig, ax = plt.subplots(1, 3, figsize=(W, 2.5))
tt = np.array([g_["tau2_true_sample"] for g_ in G_]); te = np.array([g_["tau2_est"] for g_ in G_])
ax[0].plot(tt, te, "o", color=C["raw"], ms=4)
mx = max(tt.max(), te.max()) * 1.2
ax[0].plot([0, mx], [0, mx], color="0.5", ls="--", lw=0.8)
ax[0].set_xlabel("True τ²"); ax[0].set_ylabel("Estimated τ²"); ax[0].set_title("a", loc="left", fontweight="bold")
for g_ in G_:
    if g_["multiplier"] <= 0:
        continue
    rm = [b["mean_rho_model"] for b in g_["bins"][:-1]]; rt = [b["rho_true"] for b in g_["bins"][:-1]]
    ax[1].plot(rt, rm, "o", color=plt.cm.viridis(np.log2(g_["multiplier"] * 2) / 5.5), ms=3.5)
ax[1].plot([0, 1], [0, 1], color="0.5", ls="--", lw=0.8)
ax[1].set_xlabel("True reliability (squared corr.)"); ax[1].set_ylabel("Model-based reliability"); ax[1].set_title("b", loc="left", fontweight="bold")
G_ = [g_ for g_ in G_ if g_["multiplier"] > 0]
for pol, mk in (("naive", "o"), ("shrunk", "s"), ("oracle", "^")):
    px = [g_["personalization"]["predicted_" + pol] * 1000 for g_ in G_]; py = [g_["personalization"]["observed_" + pol] * 1000 for g_ in G_]
    ax[2].plot(px, py, mk, mfc="none" if pol == "naive" else C["raw"], mec=C["raw"], ms=4, label=pol.capitalize())
lo = min(min([g_["personalization"]["predicted_naive"] for g_ in G_]), min([g_["personalization"]["observed_naive"] for g_ in G_])) * 1000
hi = max([g_["personalization"]["predicted_oracle"] for g_ in G_] + [g_["personalization"]["observed_oracle"] for g_ in G_]) * 1000
ax[2].plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8)
ax[2].set_xlabel("Predicted gain (millinats per trial)"); ax[2].set_ylabel("Observed gain (millinats per trial)"); ax[2].legend(fontsize=8)
ax[2].set_title("c", loc="left", fontweight="bold")
save(fig, "Fig6.png")

# ---------- Fig 6: learners versus items ----------
fig, ax = plt.subplots(1, 2, figsize=(W, 2.8))
for src, key, lab, mk in ((raw, "raw", "Learners", "o"), (items, "item", "Lexemes", "s")):
    if key == "raw":
        r_rate = [b["mean_rho"] for b in src["bins"]]; r_lvl = [b["mean_rho_intercept"] for b in src["bins"]]
    else:
        r_rate = [b["rate"]["mean_rho"] for b in src["bins"]]; r_lvl = [b["intercept"]["mean_rho"] for b in src["bins"]]
    ax[0].plot(x, r_rate, marker=mk, color=C[key], lw=1.2, ms=4, label="%s: rate" % lab)
    ax[0].plot(x, r_lvl, marker=mk, color=C[key], lw=1.2, ms=4, ls="--", mfc="none", label="%s: level" % lab)
ax[0].set_xticks(x, BINLAB, rotation=30); ax[0].set_ylim(0, 1.3); ax[0].set_yticks(np.arange(0, 1.01, 0.2))
ax[0].set_xlabel("Traces per learner or per lexeme")
ax[0].set_ylabel("Mean model-based reliability"); ax[0].legend(fontsize=8, loc="upper left", ncol=2); ax[0].set_title("a", loc="left", fontweight="bold")
li = [r_ for r_ in items["latent_split_oddeven"] if r_["bin"] == "all"][0]
ll_ = [r_ for r_ in raw["latent_split"]["oddeven"] if r_["bin"] == "all"][0]
labs = ["Learners: rate", "Lexemes: rate", "Learners: level", "Lexemes: level"]
src_ = [(ll_, "latent_corr_rate", C["raw"]), (li, "latent_corr_rate", C["item"]), (ll_, "latent_corr_intercept", C["raw"]), (li, "latent_corr_intercept", C["item"])]
for j, (row_, k_, col) in enumerate(src_):
    v = row_[k_]; lo_, hi_ = row_[k_ + "_ci"]
    ax[1].bar(j, v, color=col if "rate" in k_ else "white", edgecolor=col, hatch=None if "rate" in k_ else "//")
    ax[1].errorbar(j, v, yerr=[[max(v - lo_, 0)], [max(hi_ - v, 0)]], color="k", lw=0.8, capsize=2)
ax[1].set_xticks(range(4), labs, rotation=20, fontsize=8); ax[1].set_ylim(min(0, min(r_[0][r_[1]] for r_ in src_)) - 0.05, 1.05)
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_ylabel("Latent odd/even correlation"); ax[1].set_title("b", loc="left", fontweight="bold")
save(fig, "Fig3.png")
print("figures written to", FIG)
