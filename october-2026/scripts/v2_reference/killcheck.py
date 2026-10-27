# -*- coding: utf-8 -*-
# KILL-CHECK + mandatory 10k-user pilot (bank note): does the Duolingo HLR data support per-user
# forgetting-rate estimation, is the decay-slope sign predominantly negative (forgetting exists), and
# is the split-half reliability of the user-level forgetting rate measurable and history-length-dependent
# (the minimum-evidence story)? GO/NO-GO for paper 17.
import pandas as pd, numpy as np, gzip, os
BASE=r"17 - Forgetting Curve Minimum Evidence - 07.12.2026"
F=os.path.join(BASE,"data","duolingo_hlr.csv.gz")
SEED=20260712; rng=np.random.default_rng(SEED)

# --- pass 1: per-user review counts (cheap columns only) to sample 10k eligible users ---
print("pass1: counting reviews per user...", flush=True)
counts={}
for ch in pd.read_csv(F, usecols=["user_id"], chunksize=2_000_000):
    vc=ch["user_id"].value_counts()
    for u,c in vc.items(): counts[u]=counts.get(u,0)+int(c)
cser=pd.Series(counts); print(f"  users={len(cser):,} traces={cser.sum():,} median/user={cser.median():.0f}", flush=True)
elig=cser[cser>=20].index.to_numpy()   # need >=20 reviews to talk about a forgetting curve
samp=set(rng.choice(elig, size=min(10000,len(elig)), replace=False).tolist())
print(f"  eligible(>=20)={len(elig):,}; sampled {len(samp):,} users", flush=True)

# --- pass 2: load sampled users' rows (recall, lag, history) ---
print("pass2: loading sampled rows...", flush=True)
parts=[]
for ch in pd.read_csv(F, usecols=["p_recall","delta","user_id","history_seen","lexeme_id"], chunksize=2_000_000):
    parts.append(ch[ch["user_id"].isin(samp)])
df=pd.concat(parts, ignore_index=True); del parts
df=df[df["delta"]>0].copy()
df["log_lag"]=np.log(df["delta"].astype("float64"))
df["recalled"]=df["p_recall"].astype("float64")   # session recall fraction in [0,1]
print(f"  loaded {len(df):,} rows for {df['user_id'].nunique():,} users", flush=True)

# --- per-user decay slope (recall ~ log_lag) + slope sign distribution ---
def slope(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<5 or np.std(x)<1e-9: return np.nan
    b=np.polyfit(x,y,1)[0]; return b
rows=[]
for u,g in df.groupby("user_id", sort=False):
    s=slope(g["log_lag"], g["recalled"])
    rows.append((u, len(g), s))
U=pd.DataFrame(rows, columns=["user","n","slope"]).dropna()
neg=float((U["slope"]<0).mean())
print(f"\nSLOPE SIGN: {neg*100:.1f}% of users have NEGATIVE recall-vs-log-lag slope (forgetting)", flush=True)
print(f"  slope median={U['slope'].median():.4f}  mean={U['slope'].mean():.4f}", flush=True)

# --- split-half reliability of the user slope, stratified by history length ---
def half_slopes(g):
    g=g.sort_values("history_seen"); idx=np.arange(len(g))
    a=g.iloc[idx%2==0]; b=g.iloc[idx%2==1]
    return slope(a["log_lag"],a["recalled"]), slope(b["log_lag"],b["recalled"])
recs=[]
for u,g in df.groupby("user_id", sort=False):
    sa,sb=half_slopes(g)
    if sa==sa and sb==sb: recs.append((u,len(g),sa,sb))
H=pd.DataFrame(recs, columns=["user","n","sa","sb"])
# reliability by history-length bin (Spearman-Brown corrected split-half)
from scipy.stats import pearsonr
def sb_rel(a,b):
    if len(a)<10: return np.nan
    r=pearsonr(a,b)[0]; return 2*r/(1+r) if (1+r)!=0 else np.nan
bins=[(20,40),(40,80),(80,160),(160,10**9)]
print("\nSPLIT-HALF RELIABILITY of the user forgetting slope by history length:", flush=True)
for lo,hi in bins:
    sub=H[(H["n"]>=lo)&(H["n"]<hi)]
    print(f"  {lo}-{hi if hi<10**9 else '+':>4} reviews: n_users={len(sub):>5}  reliability={sb_rel(sub['sa'].values,sub['sb'].values):.3f}", flush=True)

overall_rel=sb_rel(H["sa"].values,H["sb"].values)
import json
out=dict(users_total=int(len(cser)), traces_total=int(cser.sum()), median_reviews=float(cser.median()),
         eligible_ge20=int(len(elig)), sampled=len(samp), rows_loaded=int(len(df)),
         pct_negative_slope=round(neg,4), slope_median=round(float(U["slope"].median()),5),
         overall_split_half_reliability=round(float(overall_rel),4))
json.dump(out, open(os.path.join(BASE,"results","killcheck.json"),"w"), indent=1)
GO = (neg>0.6) and (overall_rel==overall_rel)
print("\n=== KILL-CHECK:", "GO" if GO else "INSPECT", "===")
print(json.dumps(out, indent=1))
