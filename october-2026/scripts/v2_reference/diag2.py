# -*- coding: utf-8 -*-
# Diagnostic 2: does USER-CENTERING log-lag + NOT penalizing the intercept remove the spurious
# permutation-null reliability (stats reviewer M1)? Compare old vs fixed estimator on 6k users.
import pandas as pd, numpy as np, os
from scipy.stats import pearsonr
BASE=r"17 - Forgetting Curve Minimum Evidence - 07.12.2026"
F=os.path.join(BASE,"data","duolingo_hlr.csv.gz"); rng=np.random.default_rng(1)
counts={}
for ch in pd.read_csv(F, usecols=["user_id"], chunksize=3_000_000):
    for u,c in ch["user_id"].value_counts().items(): counts[u]=counts.get(u,0)+int(c)
cser=pd.Series(counts); elig=cser[cser>=20].index.to_numpy()
samp=set(rng.choice(elig,6000,replace=False).tolist())
parts=[]
for ch in pd.read_csv(F, usecols=["delta","user_id","session_seen","session_correct","timestamp"], chunksize=3_000_000):
    parts.append(ch[ch["user_id"].isin(samp)])
df=pd.concat(parts,ignore_index=True); df=df[df["delta"]>0].copy()
df["ll"]=np.log(df["delta"].astype(float))
df["ss"]=df["session_seen"].astype(float).clip(lower=1); df["sc"]=df["session_correct"].astype(float).clip(lower=0)
df=df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
def wl(z,sc,ss,ridge_slope=1e-2,pen_intercept=False,it=30):
    if len(z)<5 or np.std(z)<1e-6: return np.nan
    X=np.column_stack([np.ones(len(z)),z]); b=np.zeros(2)
    P=np.diag([ridge_slope if pen_intercept else 0.0, ridge_slope])
    for _ in range(it):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); W=ss*p*(1-p)+1e-9
        try: step=np.linalg.solve((X*W[:,None]).T@X+P, X.T@(sc-ss*p)-P@b)
        except: break
        b+=step
        if np.max(np.abs(step))<1e-7: break
    return b[1]
def sb(a,b):
    a=np.array(a); b=np.array(b); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<20 or np.std(a)<1e-9 or np.std(b)<1e-9: return np.nan
    r=pearsonr(a,b)[0]; return 2*r/(1+r)
# per-user arrays with USER-CENTERED log-lag
G=[]
for u,g in df.groupby("user_id",sort=False):
    ll=g["ll"].values; G.append((u, ll-ll.mean(), g["sc"].values, g["ss"].values))
for tag,(pen,rs) in {"OLD(global-center-equiv,pen-intercept)":(True,1e-2),"FIXED(user-center,no-intercept-pen)":(False,1e-2)}.items():
    o1=[];o2=[];n1=[];n2=[]
    for u,z,sc,ss in G:
        oe=np.arange(len(z))%2
        o1.append(wl(z[oe==0],sc[oe==0],ss[oe==0],rs,pen)); o2.append(wl(z[oe==1],sc[oe==1],ss[oe==1],rs,pen))
        z1=rng.permutation(z[oe==0]); z2=rng.permutation(z[oe==1])
        n1.append(wl(z1,sc[oe==0],ss[oe==0],rs,pen)); n2.append(wl(z2,sc[oe==1],ss[oe==1],rs,pen))
    print(f"{tag}: observed SB={sb(o1,o2):.3f}  independent-perm null={sb(n1,n2):.3f}", flush=True)
