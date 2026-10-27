# -*- coding: utf-8 -*-
# Round-2 fixes, single data load:
# (1) estimator_comparison.json - location-INVARIANT (within-user center, free intercept) vs
#     location-NON-INVARIANT (global center, penalized intercept) estimator, per-bin observed
#     split-half reliability + overall permutation null, on the same seeded 40k sample. Deposits
#     the flagship methodological numbers (was prose-only "about 0.56 / 0.43").
# (2) Corrected personalization -> merged into RESULTS.json H3:
#     - population slope fit on TRAINING portions only (genuinely out-of-sample);
#     - empirical-Bayes shrinkage with reliability weight w = per-bin split-half reliability (Kelley),
#       reporting the realized median shrinkage weight;
#     - population baseline held-out log-loss per trial (so gains are judgeable);
#     - full-personalization robustness (personalize BOTH intercept and slope) vs population.
import pandas as pd, numpy as np, os, json, math
from scipy.stats import pearsonr
BASE=r"17 - Forgetting Curve Minimum Evidence - 07.12.2026"
F=os.path.join(BASE,"data","duolingo_hlr.csv.gz")
SEED=20260712; rng=np.random.default_rng(SEED)
MIN_REVIEWS=20; SAMPLE=40000; N_BOOT=1000
BINS=[(20,30),(30,50),(50,80),(80,130),(130,250),(250,10**9)]; BINLAB=["20-29","30-49","50-79","80-129","130-249","250+"]

counts={}
for ch in pd.read_csv(F, usecols=["user_id"], chunksize=3_000_000):
    for u,c in ch["user_id"].value_counts().items(): counts[u]=counts.get(u,0)+int(c)
cser=pd.Series(counts); elig=cser[cser>=MIN_REVIEWS].index.to_numpy()
samp=set(rng.choice(elig, size=min(SAMPLE,len(elig)), replace=False).tolist())
parts=[]
for ch in pd.read_csv(F, usecols=["delta","user_id","session_seen","session_correct","timestamp"], chunksize=3_000_000):
    parts.append(ch[ch["user_id"].isin(samp)])
df=pd.concat(parts, ignore_index=True); del parts
df=df[df["delta"]>0].copy()
df["ll"]=np.log(df["delta"].astype("float64"))
df["ss"]=df["session_seen"].astype("float64").clip(lower=1); df["sc"]=df["session_correct"].astype("float64").clip(lower=0)
df=df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
gmean=df["ll"].mean()
print(f"{len(df):,} rows / {df['user_id'].nunique():,} users; global mean log-lag={gmean:.3f}", flush=True)

def wl(z, sc, ss, ridge_slope=1e-2, pen_intercept=False, it=30):
    if len(z)<5 or np.std(z)<1e-6: return (np.nan,np.nan)
    X=np.column_stack([np.ones(len(z)), z]); b=np.zeros(2)
    P=np.diag([ridge_slope if pen_intercept else 0.0, ridge_slope])
    for _ in range(it):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); W=ss*p*(1-p)+1e-9
        try: step=np.linalg.solve((X*W[:,None]).T@X+P, X.T@(sc-ss*p)-P@b)
        except np.linalg.LinAlgError: break
        b+=step
        if np.max(np.abs(step))<1e-7: break
    return (b[0], b[1])
def sb(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<20 or np.std(a)<1e-9 or np.std(b)<1e-9: return np.nan
    r=pearsonr(a,b)[0]; return 2*r/(1+r) if (1+r)>1e-9 else np.nan

# build per-user arrays
U=[]
for u,g in df.groupby("user_id", sort=False):
    ll=g["ll"].values
    U.append(dict(n=len(g), ll=ll, zc=ll-ll.mean(), zg=ll-gmean, sc=g["sc"].values, ss=g["ss"].values))
del df

# ---------- (1) estimator comparison ----------
def halves(estkey):
    pen = (estkey=="noninv"); zsel = "zg" if pen else "zc"
    bo=[];be=[];nbo=[];nbe=[];ns=[]
    for d in U:
        n=d["n"]; oe=np.arange(n)%2; z=d[zsel]; sc=d["sc"]; ss=d["ss"]; ns.append(n)
        bo.append(wl(z[oe==0],sc[oe==0],ss[oe==0],pen_intercept=pen)[1])
        be.append(wl(z[oe==1],sc[oe==1],ss[oe==1],pen_intercept=pen)[1])
        z1=rng.permutation(z[oe==0]); z2=rng.permutation(z[oe==1])
        nbo.append(wl(z1,sc[oe==0],ss[oe==0],pen_intercept=pen)[1]); nbe.append(wl(z2,sc[oe==1],ss[oe==1],pen_intercept=pen)[1])
    return np.array(bo),np.array(be),np.array(nbo),np.array(nbe),np.array(ns)
comp={}
for estkey,lab in [("inv","location-invariant (within-user center, free intercept)"),("noninv","location-non-invariant (global center, penalized intercept)")]:
    bo,be,nbo,nbe,ns=halves(estkey)
    per=[]
    for (lo,hi),bl in zip(BINS,BINLAB):
        m=(ns>=lo)&(ns<hi)
        per.append(dict(bin=bl, reliability=round(float(sb(bo[m],be[m])),3), null=round(float(sb(nbo[m],nbe[m])),3)))
    comp[estkey]=dict(label=lab, overall_reliability=round(float(sb(bo,be)),3), overall_null=round(float(sb(nbo,nbe)),3), per_bin=per)
    print(f"{estkey}: overall rel={comp[estkey]['overall_reliability']} null={comp[estkey]['overall_null']}", flush=True)
json.dump(dict(seed=SEED, sample=len(U), note="Same 40k-learner sample as the main analysis. Non-invariant = global-mean-centered log-lag with a ridge penalty on the intercept; invariant = within-learner-centered log-lag with a free intercept. Both use ridge 1e-2 on the slope.", estimators=comp),
          open(os.path.join(BASE,"results","estimator_comparison.json"),"w"), indent=1)

# ---------- (2) corrected personalization ----------
R=json.load(open(os.path.join(BASE,"results","RESULTS.json")))
relbybin={c["bin"]:c["reliability"] for c in R["H1_reliability_horizon"]["curve"]}
# per-user training/held-out with within-user-centered z; POP fit on TRAINING rows only
tr_z=[];tr_sc=[];tr_ss=[]
for d in U:
    k=int(math.floor(d["n"]*0.8)); d["k"]=k
    d["a_tr"],d["b_tr"]=wl(d["zc"][:k],d["sc"][:k],d["ss"][:k])
    tr_z.append(d["zc"][:k]); tr_sc.append(d["sc"][:k]); tr_ss.append(d["ss"][:k])
POP_A,POP_B=wl(np.concatenate(tr_z),np.concatenate(tr_sc),np.concatenate(tr_ss))
print(f"POP (train-only) a={POP_A:.4f} b={POP_B:.4f}", flush=True)
# reliability-weighted EB per bin (Kelley): b_eb = w*b_tr + (1-w)*POP_B, w = bin reliability
def binlab(n):
    for (lo,hi),bl in zip(BINS,BINLAB):
        if lo<=n<hi: return bl
    return None
wreal=[]
for d in U:
    bl=binlab(d["n"]); w=max(0.0,min(1.0, relbybin.get(bl,0.0)))
    d["w"]=w
    d["b_eb"]=w*d["b_tr"]+(1-w)*POP_B if np.isfinite(d["b_tr"]) else POP_B
    if np.isfinite(d["b_tr"]): wreal.append(w)
def heldout_ll(getpred, lo, hi):
    out=[]
    for d in U:
        if not (lo<=d["n"]<hi) or not np.isfinite(d["a_tr"]): continue
        k=d["k"]; z=d["zc"][k:]; sc=d["sc"][k:]; ss=d["ss"][k:]
        if len(z)<1: continue
        a,b=getpred(d); eta=np.clip(a+b*z,-30,30); p=np.clip(1/(1+np.exp(-eta)),1e-6,1-1e-6)
        ll=float((-(sc*np.log(p)+(ss-sc)*np.log(1-p))).sum()); nn=float(ss.sum())
        if nn>0: out.append((ll,nn))
    return out
def summ_gain(base, cand):
    g=np.array([ (bb-cc)/nn for (bb,nb),(cc,nn) in zip(base,cand) if nn>0 ])
    g=g[np.isfinite(g)]
    if len(g)<20: return dict(n=int(len(g)),median_gain=None,ci=[None,None],favors=False)
    med=float(np.median(g)); bs=[float(np.median(rng.choice(g,len(g),replace=True))) for _ in range(N_BOOT)]
    ci=[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)]
    return dict(n=int(len(g)),median_gain=round(med,4),ci=ci,favors=bool(ci[0] is not None and ci[0]>0))
pers=[]
for (lo,hi),bl in zip(BINS,BINLAB):
    base=heldout_ll(lambda d:(d["a_tr"],POP_B), lo,hi)
    pp  =heldout_ll(lambda d:(d["a_tr"],d["b_tr"]), lo,hi)
    eb  =heldout_ll(lambda d:(d["a_tr"],d["b_eb"]), lo,hi)
    full=heldout_ll(lambda d:(d["a_tr"],d["b_tr"]), lo,hi)  # slope personalized; intercept already per-user
    fullab=heldout_ll(lambda d:(d["a_tr"],d["b_tr"]), lo,hi)
    base_ll_per=float(np.sum([x[0] for x in base])/np.sum([x[1] for x in base])) if base else None
    pers.append(dict(bin=bl, base_logloss_per_trial=round(base_ll_per,4) if base_ll_per else None,
                     personalized=summ_gain(base,pp), partial_pooled=summ_gain(base,eb)))
    print(f"  {bl:<8} base_ll={pers[-1]['base_logloss_per_trial']} naive={pers[-1]['personalized']['median_gain']} EB={pers[-1]['partial_pooled']['median_gain']}", flush=True)
# full personalization (both intercept AND slope per-user) vs full population, as robustness
fullrows=[]
for (lo,hi),bl in zip(BINS,BINLAB):
    basefull=heldout_ll(lambda d:(POP_A,POP_B), lo,hi)
    candfull=heldout_ll(lambda d:(d["a_tr"],d["b_tr"]), lo,hi)
    fullrows.append(dict(bin=bl, gain=summ_gain(basefull,candfull)))
    print(f"  FULL {bl:<8} both-personalized vs both-population gain={fullrows[-1]['gain']['median_gain']} ci={fullrows[-1]['gain']['ci']}", flush=True)
def horiz(key):
    for p in pers:
        if p[key]["favors"]: return p["lo"] if "lo" in p else None
    return None
R["H3_personalization"]=dict(
    per_bin=pers,
    horizon_personalized=next((BINS[i][0] for i,p in enumerate(pers) if p["personalized"]["favors"]), None),
    horizon_partial_pooled=next((BINS[i][0] for i,p in enumerate(pers) if p["partial_pooled"]["favors"]), None),
    eb_median_shrinkage_weight=round(float(np.median(wreal)),3) if wreal else None,
    eb_note="Empirical-Bayes slope b_eb = w*b_personalized + (1-w)*b_population with w set to the bin's split-half reliability (Kelley regression to the mean).",
    pop_fit="training portions only (out-of-sample for each learner's held-out tail)",
    full_personalization_robustness=fullrows)
json.dump(R, open(os.path.join(BASE,"results","RESULTS.json"),"w"), indent=1)
print("\nEB median shrinkage weight:", R["H3_personalization"]["eb_median_shrinkage_weight"])
print("horizons naive/EB:", R["H3_personalization"]["horizon_personalized"], R["H3_personalization"]["horizon_partial_pooled"])
print("wrote estimator_comparison.json + updated RESULTS.json H3")
