# -*- coding: utf-8 -*-
# Figures for paper 17 (greyscale-safe, 300 dpi). Consumes RESULTS.json.
import json, numpy as np, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE=r"17 - Forgetting Curve Minimum Evidence - 07.12.2026"
os.makedirs(BASE+"/draft/figures", exist_ok=True)
R=json.load(open(BASE+"/results/RESULTS.json"))
plt.rcParams.update({"font.size":9,"figure.dpi":300})

# Fig 1: reliability-horizon curve + bootstrap CI + PER-BIN permutation null band + reference bars
rc=R["H1_reliability_horizon"]["curve"]
labs=[c["bin"] for c in rc]; rel=[c["reliability"] for c in rc]
lo=[c["ci"][0] for c in rc]; hi=[c["ci"][1] for c in rc]; nullb=[c["null"] for c in rc]
x=np.arange(len(labs))
fig,ax=plt.subplots(figsize=(6.0,3.6))
ax.errorbar(x, rel, yerr=[np.array(rel)-np.array(lo), np.array(hi)-np.array(rel)], fmt="-o", color="0.1", ms=5, lw=1.4, capsize=3, label="split-half reliability (95% CI)")
ax.plot(x, nullb, "-s", color="0.5", ms=4, lw=1.1, label="within-user permutation null (per bin)")
ax.fill_between(x, 0, nullb, color="0.85", zorder=0)
ax.axhline(0.70, color="k", ls=":", lw=0.9); ax.text(len(labs)-1, 0.71, "individual-decision bar (0.70)", fontsize=6.5, ha="right")
ax.axhline(0.50, color="k", ls=":", lw=0.6); ax.text(0, 0.51, "group bar (0.50)", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel("Review-history length (number of reviews)")
ax.set_ylabel("Split-half reliability of user forgetting rate"); ax.set_ylim(0,0.9)
ax.set_title("User forgetting-rate reliability is low and flat across review history:\ngenuine (above its null) but never reaches 0.70 or even 0.50")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(BASE+"/draft/figures/fig1_reliability_horizon.png"); plt.close(fig)

# Fig 2: personalization gain per bin - naive vs partial-pooled (held-out log-loss vs population default)
pb=R["H3_personalization"]["per_bin"]
labs2=[p["bin"] for p in pb]
gn=[p["personalized"]["median_gain"] for p in pb]; ge=[p["partial_pooled"]["median_gain"] for p in pb]
nlo=[p["personalized"]["ci"][0] for p in pb]; nhi=[p["personalized"]["ci"][1] for p in pb]
elo=[p["partial_pooled"]["ci"][0] for p in pb]; ehi=[p["partial_pooled"]["ci"][1] for p in pb]
x=np.arange(len(labs2)); w=0.38
fig,ax=plt.subplots(figsize=(6.2,3.6))
ax.bar(x-w/2, gn, w, color="0.35", edgecolor="k", lw=0.5, label="naive per-learner rate")
ax.bar(x+w/2, ge, w, color="0.72", edgecolor="k", lw=0.5, label="partial-pooled (empirical Bayes)")
ax.errorbar(x-w/2, gn, yerr=[np.array(gn)-np.array(nlo), np.array(nhi)-np.array(gn)], fmt="none", ecolor="k", lw=0.7, capsize=2)
ax.errorbar(x+w/2, ge, yerr=[np.array(ge)-np.array(elo), np.array(ehi)-np.array(ge)], fmt="none", ecolor="k", lw=0.7, capsize=2)
ax.axhline(0, color="k", lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(labs2); ax.set_xlabel("Review-history length (number of reviews)")
ax.set_ylabel("Held-out log-loss gain\n(personalized minus population default)")
ax.set_title("Personalizing the forgetting rate never beats the population default:\nneither the naive nor the partial-pooled rate, at any history length")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(BASE+"/draft/figures/fig2_personalization_gain.png"); plt.close(fig)
print("figures written:", os.listdir(BASE+"/draft/figures"))
