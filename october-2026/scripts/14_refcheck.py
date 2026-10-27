# -*- coding: utf-8 -*-
# Reference integrity check: every DOI in the reference list is looked up on Crossref (DataCite
# for the dataset) and compared with the entry: all author surnames, year, title words (with
# subtitle), volume, issue and pages. Non-DOI entries are listed for manual URL checks.
# Usage: python 14_refcheck.py <references.md> <out.json>
import sys, re, json, time, urllib.request, unicodedata

UA = {"User-Agent": "refcheck (mailto:ismayil.zeynalov.bakhish@bsu.edu.az)"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


lines = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip() and not l.startswith("#")]
out = []
for l in lines:
    m = re.search(r"https://doi\.org/(\S+)", l)
    rec = dict(entry=l[:90])
    if not m:
        rec["status"] = "no DOI (check URL manually)"
        out.append(rec); continue
    doi = m.group(1).rstrip(".")
    try:
        if doi.lower().startswith(("10.7910/", "10.17605/")):     # DataCite: Dataverse, OSF
            a = get("https://api.datacite.org/dois/" + doi)["data"]["attributes"]
            title = a["titles"][0]["title"]; fams = [c["name"].split(",")[0] for c in a["creators"]]
            year = a["publicationYear"]; vol = iss = page = None
        else:
            msg = get("https://api.crossref.org/works/" + doi)["message"]
            title = (msg.get("title") or [""])[0] + " " + " ".join(msg.get("subtitle") or [])
            fams = [a_.get("family", "") for a_ in msg.get("author", [])]
            dp = (msg.get("published-print") or msg.get("journal-issue", {}).get("published-print") or msg.get("issued"))
            year = dp["date-parts"][0][0]
            vol, iss, page = msg.get("volume"), msg.get("issue"), msg.get("page")
        problems = []
        ne = norm(l)
        for f in fams:
            if f and norm(f).strip() and norm(f).split()[-1] not in ne:
                problems.append("author %s missing" % f)
        if str(year) not in l:
            problems.append("year %s not in entry" % year)
        tw = [w for w in norm(title).split() if len(w) > 3]
        miss = [w for w in tw if w not in ne]
        if miss:
            problems.append("title words missing: %s" % miss[:6])
        if vol and ("*%s*" % vol) not in l:
            problems.append("volume %s" % vol)
        if iss and ("(%s)" % iss) not in l:
            problems.append("issue %s" % iss)
        if page and page.replace("\u2013", "-") not in l:
            problems.append("pages %s" % page)
        rec.update(doi=doi, crossref_title=title.strip(), year=year, authors=fams, status="ok" if not problems else "CHECK", problems=problems)
    except Exception as e:
        rec.update(doi=doi, status="lookup failed: %s" % e)
    out.append(rec)
    print(rec["status"], "|", rec.get("doi"), "|", "; ".join(rec.get("problems", [])))
    time.sleep(0.2)
# v4: dash characters from Crossref titles and page ranges are written as plain hyphens
txt = json.dumps(out, indent=1, ensure_ascii=False).replace(chr(0x2013), "-").replace(chr(0x2014), "-")
open(sys.argv[2], "w", encoding="utf-8").write(txt)
print("entries", len(out), "ok", sum(r_["status"] == "ok" for r_ in out))
