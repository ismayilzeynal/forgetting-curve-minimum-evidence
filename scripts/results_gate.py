# -*- coding: utf-8 -*-
# CONFIRMATORY analysis for paper 17 (frozen plan + round-1 estimator fix). Per-user forgetting rate =
# slope of a weighted binomial GLM on WITHIN-USER-CENTERED log-lag (fixes the global-centering artifact
# that inflated both the reliability and its permutation null), with NO penalty on the intercept.
# Reports: reliability-horizon curve + PER-BIN permutation null; empirical-Bayes partial-pooling; and a
# SLOPE-ISOLATED, out-of-sample personalization test (personalized/partial-pooled/population, same
# per-user intercept). Writes results/RESULTS.json.
import pandas as pd, numpy as np, os, json, math
from scipy.stats import pearsonr, spearmanr
BASE=r"17 - Forgetting Curve Minimum Evidence - 07.12.2026"
F=os.path.join(BASE,"data","duolingo_hlr.csv.gz")
SEED=20260712; rng=np.random.default_rng(SEED)
MIN_REVIEWS=20; SAMPLE=40000; REL_IND=0.70; REL_GRP=0.50; N_BOOT=1000
BINS=[(20,30),(30,50),(50,80),(80,130),(130,250),(250,10**9)]; BINLAB=["20-29","30-49","50-79","80-129","130-249","250+"]

print("pass1", flush=True)
counts={}
for ch in pd.read_csv(F, usecols=["user_id"], chunksize=3_000_000):
    for u,c in ch["user_id"].value_counts().items(): counts[u]=counts.get(u,0)+int(c)
cser=pd.Series(counts); elig=cser[cser>=MIN_REVIEWS].index.to_numpy()
samp=set(rng.choice(elig, size=min(SAMPLE,len(elig)), replace=False).tolist())
print("pass2", flush=True)
parts=[]
for ch in pd.read_csv(F, usecols=["delta","user_id","session_seen","session_correct","timestamp"], chunksize=3_000_000):
    parts.append(ch[ch["user_id"].isin(samp)])
df=pd.concat(parts, ignore_index=True); del parts
df=df[df["delta"]>0].copy()
df["ll"]=np.log(df["delta"].astype("float64"))
df["ss"]=df["session_seen"].astype("float64").clip(lower=1); df["sc"]=df["session_correct"].astype("float64").clip(lower=0)
df=df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
print(f"  {len(df):,} rows / {df['user_id'].nunique():,} users", flush=True)

def wl(z, sc, ss, ridge_slope=1e-2, it=30):
    # z should be user-centered; intercept is NOT penalized
    if len(z)<5 or np.std(z)<1e-6: return (np.nan,np.nan)
    X=np.column_stack([np.ones(len(z)), z]); b=np.zeros(2); P=np.diag([0.0, ridge_slope])
    for _ in range(it):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); W=ss*p*(1-p)+1e-9
        try: step=np.linalg.solve((X*W[:,None]).T@X+P, X.T@(sc-ss*p)-P@b)
        except np.linalg.LinAlgError: break
        b+=step
        if np.max(np.abs(step))<1e-7: break
    return (b[0], b[1])

print("precompute (user-centered)...", flush=True)
UARR=[]
for u,g in df.groupby("user_id", sort=False):
    ll=g["ll"].values; z=ll-ll.mean()
    UARR.append(dict(user=u, n=len(g), z=z, sc=g["sc"].values, ss=g["ss"].values))
# population slope: pooled GLM with user-centered z (so it is a within-user pooled slope)
allz=np.concatenate([d["z"] for d in UARR]); allsc=np.concatenate([d["sc"] for d in UARR]); allss=np.concatenate([d["ss"] for d in UARR])
POP_A,POP_B=wl(allz, allsc, allss)
print(f"  {len(UARR):,} users; pooled within-user slope={POP_B:.4f}", flush=True)
for d in UARR:
    n=d["n"]; z=d["z"]; sc=d["sc"]; ss=d["ss"]
    d["b_full"]=wl(z,sc,ss)[1]
    oe=np.arange(n)%2
    d["b_o"]=wl(z[oe==0],sc[oe==0],ss[oe==0])[1]; d["b_e"]=wl(z[oe==1],sc[oe==1],ss[oe==1])[1]
    k=int(math.floor(n*0.8)); d["k"]=k; d["a_tr"],d["b_tr"]=wl(z[:k],sc[:k],ss[:k])
UD=pd.DataFrame([{k:d[k] for k in ("user","n","b_full","b_o","b_e","a_tr","b_tr")} for d in UARR])
neg=float((UD["b_full"]<0).mean()); print(f"forgetting: {neg*100:.1f}% negative; median b={UD['b_full'].median():.4f}", flush=True)

def sb(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<20 or np.std(a)<1e-9 or np.std(b)<1e-9: return np.nan
    r=pearsonr(a,b)[0]; return 2*r/(1+r) if (1+r)>1e-9 else np.nan
def boot_sb(a,b,nb=N_BOOT):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<20: return [None,None]
    vals=[sb(a[idx],b[idx]) for idx in (rng.integers(0,len(a),len(a)) for _ in range(nb))]; vals=[v for v in vals if v==v]
    return [round(float(np.percentile(vals,2.5)),3), round(float(np.percentile(vals,97.5)),3)] if vals else [None,None]

# per-bin reliability + PER-BIN permutation null (independent within-half lag permutation)
def perm_null_bin(users_bin):
    n1=[];n2=[]
    for d in users_bin:
        n=d["n"]; oe=np.arange(n)%2; z=d["z"]; sc=d["sc"]; ss=d["ss"]
        z1=rng.permutation(z[oe==0]); z2=rng.permutation(z[oe==1])
        n1.append(wl(z1,sc[oe==0],ss[oe==0])[1]); n2.append(wl(z2,sc[oe==1],ss[oe==1])[1])
    return sb(n1,n2)
rel_curve=[]
for (lo,hi),lab in zip(BINS,BINLAB):
    ub=[d for d in UARR if lo<=d["n"]<hi]
    sub=UD[(UD["n"]>=lo)&(UD["n"]<hi)]
    rel=sb(sub["b_o"].values, sub["b_e"].values); ci=boot_sb(sub["b_o"].values, sub["b_e"].values)
    nulls=[perm_null_bin(ub if len(ub)<=6000 else [ub[i] for i in rng.choice(len(ub),6000,replace=False)]) for _ in range(8)]
    nullv=round(float(np.nanmedian(nulls)),3)
    rel_curve.append(dict(bin=lab,lo=lo,n_users=int(len(sub)),reliability=round(float(rel),3) if rel==rel else None, ci=ci, null=nullv,
                          net=round(float(rel)-nullv,3) if rel==rel else None))
    print(f"  rel {lab:<8} n={len(sub):>5} SB={rel:.3f} CI{ci} null={nullv} net={round(float(rel)-nullv,3) if rel==rel else None}", flush=True)
rels=[c["reliability"] for c in rel_curve if c["reliability"] is not None]; mids=[BINS[i][0] for i,c in enumerate(rel_curve) if c["reliability"] is not None]
rho_trend=float(spearmanr(mids,rels)[0]) if len(rels)>=3 else float("nan")
def horizon(thr):
    for c in rel_curve:
        if c["reliability"] is not None and c["reliability"]>=thr: return c["lo"]
    return None
H_ind=horizon(REL_IND); H_grp=horizon(REL_GRP)
obs_overall=round(float(sb(UD["b_o"].values,UD["b_e"].values)),3)

# empirical-Bayes shrinkage: b_eb = w*b_tr + (1-w)*b_pop, w = tau2/(tau2+se2). tau2 = var(b_full)-mean(se2 proxy)
btr=UD["b_tr"].dropna().values; tau2=max(1e-6, float(np.var(btr))-float(np.median(np.abs(btr-np.median(btr)))**2))
# per-user se proxy ~ scale/sqrt(n_train); use residual-based simple proxy
def se_proxy(d):
    return 1.0/max(1.0, math.sqrt(max(1,d["k"])) )  # coarse: shorter train -> larger se
for d in UARR:
    if np.isfinite(d["b_tr"]):
        se2=se_proxy(d)**2; w=tau2/(tau2+se2); d["b_eb"]=w*d["b_tr"]+(1-w)*POP_B
    else: d["b_eb"]=POP_B

# SLOPE-ISOLATED, out-of-sample personalization: held-out log-loss with per-user intercept a_tr and
# slope in {b_tr (personalized), b_eb (partial-pooled), POP_B (population)} - only the slope differs.
def ll_bin(getslope, lo, hi):
    tot_ll=0.0; tot_n=0.0; per=[]
    for d in UARR:
        if not (lo<=d["n"]<hi) or not np.isfinite(d["a_tr"]): continue
        k=d["k"]; z=d["z"][k:]; sc=d["sc"][k:]; ss=d["ss"][k:]
        if len(z)<1: continue
        b=getslope(d); eta=np.clip(d["a_tr"]+b*z,-30,30); p=np.clip(1/(1+np.exp(-eta)),1e-6,1-1e-6)
        ll=float((-(sc*np.log(p)+(ss-sc)*np.log(1-p))).sum()); n=float(ss.sum())
        per.append((ll,n))
    return per
pers=[]
for (lo,hi),lab in zip(BINS,BINLAB):
    base=ll_bin(lambda d:POP_B, lo,hi); pp=ll_bin(lambda d:d["b_tr"], lo,hi); eb=ll_bin(lambda d:d["b_eb"], lo,hi)
    def gains(cand):
        g=[(bb-cc)/nn for (bb,nb),(cc,nn) in zip(base,cand) if nn>0]  # base_ll - cand_ll per trial (>0 => cand better)
        return np.array([x for x in g if np.isfinite(x)])
    def summ(cand):
        g=gains(cand)
        if len(g)<20: return dict(n=int(len(g)),median_gain=None,ci=[None,None],favors=False)
        med=float(np.median(g)); bs=[float(np.median(rng.choice(g,len(g),replace=True))) for _ in range(N_BOOT)]
        ci=[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)]
        return dict(n=int(len(g)),median_gain=round(med,4),ci=ci,favors=bool(ci[0] is not None and ci[0]>0))
    pers.append(dict(bin=lab,lo=lo, personalized=summ(pp), partial_pooled=summ(eb)))
    pp_s=pers[-1]["personalized"]; eb_s=pers[-1]["partial_pooled"]
    print(f"  pers {lab:<8} full n={pp_s['n']} gain={pp_s['median_gain']} CI{pp_s['ci']} | EB gain={eb_s['median_gain']} CI{eb_s['ci']}", flush=True)
def horizon_pers(key):
    for p in pers:
        if p[key]["favors"]: return p["lo"]
    return None

R=dict(
  config=dict(min_reviews=MIN_REVIEWS, sample=len(samp), rows=int(len(df)), rel_ind=REL_IND, rel_grp=REL_GRP, bins=BINLAB, seed=SEED,
              estimator="within-user-centered log-lag, weighted binomial, no intercept penalty"),
  data=dict(users_total=int(len(cser)), traces_total=int(cser.sum()), median_reviews=float(cser.median()), eligible_ge20=int(len(elig))),
  forgetting=dict(pct_negative_slope=round(neg,4), median_slope=round(float(UD["b_full"].median()),5), pooled_within_user_slope=round(POP_B,5)),
  H1_reliability_horizon=dict(curve=rel_curve, monotone_rho=round(rho_trend,3), horizon_ind=H_ind, horizon_grp=H_grp,
                              overall_reliability=obs_overall, overall_null=round(float(np.nanmedian([c["null"] for c in rel_curve])),3)),
  H3_personalization=dict(per_bin=pers, horizon_personalized=horizon_pers("personalized"), horizon_partial_pooled=horizon_pers("partial_pooled")),
)
json.dump(R, open(os.path.join(BASE,"results","RESULTS.json"),"w"), indent=1)
print("\n=== SUMMARY ===")
print("rel horizon 0.70:",H_ind," 0.50:",H_grp," monotone rho:",round(rho_trend,3)," overall:",obs_overall)
print("null (median across bins):", R["H1_reliability_horizon"]["overall_null"])
print("personalized horizon:",horizon_pers("personalized")," partial-pooled horizon:",horizon_pers("partial_pooled"))
