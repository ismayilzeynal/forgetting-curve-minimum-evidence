# -*- coding: utf-8 -*-
# Resolve every number in the manuscript sources from the result files. A token has the form
# {{alias:path|fmt}}: alias names a results JSON (table ALIASES), path is a dotted key path in
# which a segment "name[label]" selects the list element whose "bin" (or "label", or
# "multiplier") equals label and "name[3]" selects by index; fmt is a Python format spec (".3f"),
# "int" (thousands separators), "pct0"/"pct1" (value x 100 with 0 or 1 decimals), "abs.3f"
# (absolute value), "neg.3f" (sign flipped) or "milli.2f" (value x 1000, used for gains in
# millinats). Negative numbers print with a minus sign (U+2212).
# Keys and selectors may contain dots (grid[0.0], r0.90_population): the path is split only on
# dots outside brackets, and a key that is not found is retried by joining it with the following
# segments (longest first is not needed: the first join that exists is unique in these files).
# Usage: python 11_resolve.py <in.md> <out.md> <report.tsv>
# Fails on any unresolved token, so no number reaches the manuscript without a source.
import sys, os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
_local = os.path.join(os.path.dirname(HERE), "results")
_v3 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "v3-cbb-2026-10", "results")
RES = os.environ.get("CBB_RESULTS") or (_local if os.path.isdir(_local) else _v3)
ALIASES = {
    "meta": "sample_meta.json",
    "v2rep": "v2_reproduction.json",
    "pop": "population_models.json",
    "mom": "RESULTS_v3_main.json",
    "momitems": "RESULTS_v3_items.json",
    "raw": "RESULTS_v3_hier_raw.json",
    "adj": "RESULTS_v3_hier_adj.json",
    "hlr": "RESULTS_v3_hier_hlr.json",
    "items": "RESULTS_v3_hier_items.json",
    "thraw": "RESULTS_v3_theory_raw.json",
    "thadj": "RESULTS_v3_theory_adj.json",
    "thhlr": "RESULTS_v3_theory_hlr.json",
    "simgrid": "RESULTS_v3_sim_grid.json",
    "simprobe": "RESULTS_v3_sim_probe_m1.json",
    "simhlr": "RESULTS_v3_sim_hlr.json",
    "der": "RESULTS_v3_derived.json",
    "sep": "RESULTS_v3_separation.json",
    "rnull": "RESULTS_v3_rank_null.json",
    "der4": "RESULTS_v4_derived.json",
    "jul": "RESULTS.json",
    "kc": "killcheck.json",
}
# v4 revision: der4 is written by 16_derived_v4.py from the frozen files into results_v4/
RES4 = os.path.join(os.path.dirname(HERE), "results_v4")
# r2 revision: jul is the result file of the confirmatory run of 12 July 2026, as published in the
# public repository (commit 0d0f0da1, results/RESULTS.json; local copy in the paper folder's repo/)
RESJUL = os.environ.get("CBB_JULY_RESULTS") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "repo", "results")
# r4 revision: kc is the output of the July feasibility check (killcheck.py, 10,000 learners) that the
# plan's first paragraph reports on; it lies in the paper folder's results/
RESKC = os.environ.get("CBB_KILLCHECK_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
TOKEN = re.compile(r"\{\{([a-z0-9]+):([^|}]+)(?:\|([^}]+))?\}\}")
_cache = {}


def load(alias):
    if alias not in _cache:
        folder = RES4 if alias == "der4" else RESJUL if alias == "jul" else RESKC if alias == "kc" else RES
        _cache[alias] = json.load(open(os.path.join(folder, ALIASES[alias]), encoding="utf-8"))
    return _cache[alias]


def nozero(s):
    # v4 revision: a negative value that rounds to zero prints without its sign
    return s[1:] if s.startswith("−") and re.fullmatch(r"−0(\.0+)?%?", s) else s


def segments(path):
    out, buf, depth = [], "", 0
    for ch in path:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    out.append(buf)
    return out


def select(obj, name, sel):
    obj = obj[name]
    if re.fullmatch(r"-?\d+", sel):
        return obj[int(sel)]
    hits = [x for x in obj if isinstance(x, dict) and (x.get("bin") == sel or x.get("label") == sel
                                                       or str(x.get("multiplier")) == sel)]
    if len(hits) != 1:
        raise KeyError("selector %s[%s] matched %d" % (name, sel, len(hits)))
    return hits[0]


def walk(obj, path):
    ss = segments(path)
    i = 0
    while i < len(ss):
        seg = ss[i]
        m = re.fullmatch(r"([A-Za-z0-9_.]+)\[([^\]]+)\]", seg)
        if m:
            obj = select(obj, m.group(1), m.group(2))
            i += 1
            continue
        if isinstance(obj, list):
            obj = obj[int(seg)]
            i += 1
            continue
        if seg in obj:
            obj = obj[seg]
            i += 1
            continue
        for j in range(i + 1, len(ss)):
            key = ".".join(ss[i:j + 1])
            mm = re.fullmatch(r"([A-Za-z0-9_.]+)\[([^\]]+)\]", key)
            if mm and mm.group(1) in obj:
                obj = select(obj, mm.group(1), mm.group(2))
                i = j + 1
                break
            if key in obj:
                obj = obj[key]
                i = j + 1
                break
        else:
            raise KeyError("%s (in %s)" % (seg, path))
    return obj


def fmt(v, f):
    if v is None:
        raise ValueError("null value")
    if not f:
        return str(v)
    if f == "int":
        out = "{:,}".format(int(round(float(v))))
        return out.replace("-", "−") if out.startswith("-") else out
    if f.startswith("pct"):
        out = ("{:.%df}" % int(f[3:])).format(100 * float(v))
        return out.replace("-", "−") if out.startswith("-") else out
    if f.startswith("abs"):
        return format(abs(float(v)), f[3:])
    if f.startswith("neg"):
        v, f = -float(v), f[3:]
    if f.startswith("milli"):
        v, f = 1000 * float(v), f[5:]
    out = format(float(v), f)
    return out.replace("-", "−") if out.startswith("-") else out


def on_boundary(val, f, s):
    # r2 revision (recompute-r2-01): a stored value that is already rounded and whose dropped part at the
    # printed precision is exactly 5 cannot be rounded correctly; such a cell must be printed at the
    # stored precision or resolved from an unrounded source (results_v4, "boundary")
    from decimal import Decimal
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return False
    p = s.replace(",", "").replace("−", "").replace("%", "")
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", p):
        return False
    nd = len(p.split(".")[1]) if "." in p else 0
    scale = 1000 if (f or "").startswith("milli") else 100 if (f or "").startswith("pct") else 1
    q = (abs(Decimal(repr(float(val)))) * scale).scaleb(nd)
    return q - int(q) == Decimal("0.5")


def main(src, dst, report):
    text = open(src, encoding="utf-8").read()
    rows = []

    def rep(m):
        alias, path, f = m.group(1), m.group(2), m.group(3)
        val = walk(load(alias), path)
        s = nozero(fmt(val, f))
        if on_boundary(val, f, s):
            raise SystemExit("stored value on a rounding boundary: %s = %r printed %s" % (m.group(0), val, s))
        rows.append((m.group(0), s, ALIASES[alias], path, f or "", repr(val)))
        return s
    out = TOKEN.sub(rep, text)
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        raise SystemExit("unresolved tokens: %s" % left[:10])
    open(dst, "w", encoding="utf-8", newline="").write(out)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("token\tprinted\tfile\tkey\tfmt\traw_value\n")
        for r_ in rows:
            fh.write("\t".join(r_) + "\n")
    print("resolved", len(rows), "tokens in", os.path.basename(src))
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:4])
