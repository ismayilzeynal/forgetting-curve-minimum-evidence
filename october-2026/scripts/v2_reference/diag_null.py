# -*- coding: utf-8 -*-
# Diagnostic: is the 0.43 permutation-null reliability a real heteroscedasticity artifact or a coded
# leak? Compare (A) shared-permutation null (current), (B) independent-permutation-per-half null,
# (C) permute recall instead of lag, (D) fully independent random pairing per half. On 6k users.
import pandas as pd, numpy as np, os, math
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
df["z"]=np.log(df["delta"].astype(float))-np.log(df["delta"].astype(float)).mean()
df["ss"]=df["session_seen"].astype(float).clip(lower=1); df["sc"]=df["session_correct"].astype(float).clip(lower=0)
df=df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
def wl(z,sc,ss,ridge=1e-2,it=30):
    if len(z)<5 or np.std(z)<1e-6: return np.nan
    X=np.column_stack([np.ones(len(z)),z]); b=np.zeros(2)
    for _ in range(it):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); W=ss*p*(1-p)+1e-9
        try: step=np.linalg.solve((X*W[:,None]).T@X+ridge*np.eye(2), X.T@(sc-ss*p)-ridge*b)
        except: break
        b+=step
        if np.max(np.abs(step))<1e-7: break
    return b[1]
def sb(a,b):
    a=np.array(a); b=np.array(b); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<20 or np.std(a)<1e-9 or np.std(b)<1e-9: return np.nan
    r=pearsonr(a,b)[0]; return 2*r/(1+r)
G=[(u,g["z"].values,g["sc"].values,g["ss"].values) for u,g in df.groupby("user_id",sort=False)]
# observed
o1=[];o2=[]
for u,z,sc,ss in G:
    oe=np.arange(len(z))%2; o1.append(wl(z[oe==0],sc[oe==0],ss[oe==0])); o2.append(wl(z[oe==1],sc[oe==1],ss[oe==1]))
print("OBSERVED split-half SB:", round(sb(o1,o2),3))
def nullrel(mode):
    a=[];b=[]
    for u,z,sc,ss in G:
        n=len(z); oe=np.arange(n)%2
        if mode=="A":   zz=rng.permutation(z); z1,z2=zz[oe==0],zz[oe==1]; c1,c2=sc[oe==0],sc[oe==1]; s1,s2=ss[oe==0],ss[oe==1]
        elif mode=="B": z1=rng.permutation(z[oe==0]); z2=rng.permutation(z[oe==1]); c1,c2=sc[oe==0],sc[oe==1]; s1,s2=ss[oe==0],ss[oe==1]
        elif mode=="Crecall": z1,z2=z[oe==0],z[oe==1]; c1=rng.permutation(sc[oe==0]); c2=rng.permutation(sc[oe==1]); s1,s2=ss[oe==0],ss[oe==1]
        a.append(wl(z1,c1,s1)); b.append(wl(z2,c2,s2))
    return round(sb(a,b),3)
print("NULL A (shared perm, current):", nullrel("A"))
print("NULL B (independent perm/half):", nullrel("B"))
print("NULL C (permute recall/half):", nullrel("Crecall"))
