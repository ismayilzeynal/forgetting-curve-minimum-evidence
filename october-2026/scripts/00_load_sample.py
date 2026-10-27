# -*- coding: utf-8 -*-
# Draw the same seeded 40,000-learner sample as the v2 confirmatory run (identical pass-1 code,
# so the rng draw is identical) and cache every column the v3 analyses need.
# Input: the public Dataverse file settles.acl16.learning_traces.13m.csv.gz
# (doi:10.7910/DVN/N8XJME, sha256 26cf9b33f2e9b2e8aa9a8f262a017b30699ca2f625c56a12ce699d4ea7076873).
# Usage: python 00_load_sample.py <csv.gz> <cache_dir>
import sys, os, json
import numpy as np, pandas as pd

F, CACHE = sys.argv[1], sys.argv[2]
SEED = 20260712; MIN_REVIEWS = 20; SAMPLE = 40000
rng = np.random.default_rng(SEED)

counts = {}
for ch in pd.read_csv(F, usecols=["user_id"], chunksize=3_000_000):
    for u, c in ch["user_id"].value_counts().items():
        counts[u] = counts.get(u, 0) + int(c)
cser = pd.Series(counts)
elig = cser[cser >= MIN_REVIEWS].index.to_numpy()
samp = set(rng.choice(elig, size=min(SAMPLE, len(elig)), replace=False).tolist())

cols = ["p_recall", "timestamp", "delta", "user_id", "learning_language", "ui_language", "lexeme_id",
        "history_seen", "history_correct", "session_seen", "session_correct"]
parts = []
nlex_all = set()
for ch in pd.read_csv(F, usecols=cols, chunksize=3_000_000):
    nlex_all.update(ch["lexeme_id"].unique().tolist())
    parts.append(ch[ch["user_id"].isin(samp)])
df = pd.concat(parts, ignore_index=True); del parts
n_raw = len(df)
df = df[df["delta"] > 0].copy()
# same ordering rule as v2 (stable sort keeps file order within equal timestamps)
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
df.to_parquet(os.path.join(CACHE, "sample40k.parquet"), index=False)
meta = dict(seed=SEED, sample=len(samp), users_total=int(len(cser)), traces_total=int(cser.sum()),
            median_reviews=float(cser.median()), eligible_ge20=int(len(elig)), lexemes_total=len(nlex_all),
            rows_sample_raw=int(n_raw), rows_sample_delta_pos=int(len(df)),
            users_in_rows=int(df["user_id"].nunique()), lexemes_in_sample=int(df["lexeme_id"].nunique()))
json.dump(meta, open(os.path.join(CACHE, "sample_meta.json"), "w"), indent=1)
print(meta)
