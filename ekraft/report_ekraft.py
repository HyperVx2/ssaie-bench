#!/usr/bin/env python3
"""Summary tables for the EKraft run. Standard library only.

  python3 ekraft/report_ekraft.py                 # reads results/, writes analysis_out/ekraft_report.md
  python3 ekraft/report_ekraft.py results out_dir

Works with partial results: sections without data are marked "not run yet".
"""
import collections, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "analysis_out")
BASE = "qwen3:4b"
try:
    EK = json.load(open(os.path.join(HERE, "model.json")))["ollama_name"]
except Exception:
    EK = "ekraft-v1-plan:4b"


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def pct(k, n):
    if not n:
        return "-"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * p:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


def last_ok(rs, keyf):
    by = {}
    for r in rs:
        k = keyf(r)
        if r.get("status", "ok") == "ok" or k not in by:
            by[k] = r
    return by


md = ["# EKraft v1 test results", "",
      f"EKraft model: `{EK}`. Baseline: the untuned `{BASE}` (same 4B size and family).",
      "Brackets are 95% Wilson confidence intervals. p-values are exact McNemar tests on paired items.", ""]
rep = {"ekraft_model": EK, "base_model": BASE}

# ---------------------------------------------------------------- Task A
md += ["## Task A: answering the 200 MMLU-Pro CS questions (letter-only)", ""]
a = last_ok([r for r in rows(os.path.join(R, "laptop", "constrained.jsonl")) if r.get("task") == "A"],
            lambda r: (r["model"], r["id"]))
res = {m: {i: r.get("correct", False) for (mm, i), r in a.items() if mm == m and r.get("status") == "ok"}
       for m in (EK, BASE)}
if res[EK]:
    md += ["| Model | Correct |", "|---|---|"]
    for m in (BASE, EK):
        md.append(f"| `{m}` | {pct(sum(res[m].values()), len(res[m]))} |")
    common = set(res[EK]) & set(res[BASE])
    b = sum(res[BASE][i] and not res[EK][i] for i in common)
    c = sum(res[EK][i] and not res[BASE][i] for i in common)
    md += ["", f"Paired on {len(common)} questions: base right / EKraft wrong = {b}, EKraft right / base wrong = {c}, "
               f"p = {mcnemar(b, c):.3f}.",
           "EKraft was trained to write items, not to answer them; this checks whether fine-tuning damaged "
           "its general CS knowledge.", ""]
    rep["task_a"] = {m: [sum(v.values()), len(v)] for m, v in res.items()} | {"mcnemar_p": mcnemar(b, c)}
else:
    md += ["Not run yet (laptop: `bash ekraft/run_ekraft.sh laptop`).", ""]

# ---------------------------------------------------------------- speed
md += ["## Speed and memory on the target machines", ""]
spd = []
for tier in ("vps", "homelab", "pi"):
    rs = last_ok(rows(os.path.join(R, tier, "speed.jsonl")), lambda r: (r["model"], r["task"], r["id"]))
    for m in (BASE, EK):
        mine = [r for (mm, t, i), r in rs.items() if mm == m]
        if not mine:
            continue
        load = [r for r in mine if r["task"] == "load"]
        ok = [r for r in mine if r["task"] in ("A", "B") and r.get("status") == "ok" and r.get("gen_tps")]
        loaded = "yes" if load and load[-1].get("status") == "ok" else "NO"
        tps = st.median(r["gen_tps"] for r in ok) if ok else None
        bw = [r["wall_s"] for r in ok if r["task"] == "B"]
        rss = max((r.get("peak_runner_rss") or 0 for r in ok), default=0)
        spd.append(f"| {tier} | `{m}` | {loaded} | {tps:.1f} |" if tps else f"| {tier} | `{m}` | {loaded} | - |")
        spd[-1] += f" {st.median(bw):.0f} s |" if bw else " - |"
        spd[-1] += f" {rss / 2**30:.2f} GB |" if rss else " - |"
if spd:
    md += ["| Machine | Model | Loads | Median tokens/s | Median time per item | Peak model RAM |",
           "|---|---|---|---|---|---|"] + spd + [""]
else:
    md += ["Not run yet (servers: `bash ekraft/run_ekraft.sh vps` / `homelab` / `pi`).", ""]

# ---------------------------------------------------------------- item writing
items = last_ok(rows(os.path.join(R, "laptop", "ekraft_items.jsonl")), lambda r: (r["set"], r["model"], r["id"]))
jud = {}
for j in rows(os.path.join(R, "frontier", "ekraft_judgements.jsonl")):
    if j.get("judged") or j.get("reason") or j["uid"] not in jud:
        jud[j["uid"]] = j
old_j = {j["uid"]: j for j in rows(os.path.join(R, "frontier", "judgements.jsonl"))}


def score(j, fmt_ok):
    """usable = format OK and the judge says the key is the single correct answer and clarity >= 3."""
    if not fmt_ok or not j or not j.get("judged"):
        return {"fmt": fmt_ok, "judged": bool(j and j.get("judged")), "key": False, "clear": False,
                "distr": False, "aligned": False, "usable": False}
    key = j.get("key_correct") is True
    clear = (j.get("clarity") or 0) >= 3
    return {"fmt": True, "judged": True, "key": key, "clear": clear, "distr": (j.get("distractors") or 0) >= 3,
            "aligned": j.get("aligned") is True, "usable": key and clear}


def table(groups):
    out = ["| Writer | Items | Format OK | Key correct | Clear (>=3) | Good distractors (>=3) | On topic/level | **Usable** |",
           "|---|---|---|---|---|---|---|---|"]
    for name, sc in groups:
        n = len(sc)
        f = lambda k: pct(sum(s[k] for s in sc), n)  # noqa: E731
        out.append(f"| {name} | {n} | {f('fmt')} | {f('key')} | {f('clear')} | {f('distr')} | {f('aligned')} | **{f('usable')}** |")
    return out


md += ["## Item writing, judged blind by Gemini (Round 1 rubric)", "",
       "**Usable** = right format, the judge agrees the marked key is the single correct answer, and clarity is "
       "at least 3 of 4. Items that fail the format checks count as not usable.", ""]
if not items:
    md += ["Not run yet (laptop: `bash ekraft/run_ekraft.sh laptop`).", ""]
else:
    njud = sum(1 for (s, m, i) in items if f"{m}|{s}|{i}" in jud)
    if njud < len(items):
        md += [f"**Judging incomplete: {njud} of {len(items)} items judged.** Run `python3 ekraft/judge_ekraft.py`.", ""]
    sc = collections.defaultdict(dict)
    for (s, m, i), r in items.items():
        sc[(s, m)][i] = score(jud.get(f"{m}|{s}|{i}"), (r.get("checks") or {}).get("passes_all", False))

    # A) EKraft's own subjects
    md += ["### A. EKraft's subjects (231 held-out prompts: SE1 and NET1)", ""]
    ek, bs = sc.get(("ekraft", EK), {}), sc.get(("ekraft", BASE), {})
    md += table([(f"`{BASE}` (untuned)", list(bs.values())), (f"`{EK}`", list(ek.values()))]) + [""]
    common = set(ek) & set(bs)
    if common:
        b = sum(bs[i]["usable"] and not ek[i]["usable"] for i in common)
        c = sum(ek[i]["usable"] and not bs[i]["usable"] for i in common)
        md += [f"Paired on {len(common)} prompts: only base usable = {b}, only EKraft usable = {c}, p = {mcnemar(b, c):.3f}.", ""]
        rep["ekraft_set"] = {"base_usable": [sum(x["usable"] for x in bs.values()), len(bs)],
                             "ekraft_usable": [sum(x["usable"] for x in ek.values()), len(ek)],
                             "mcnemar_p": mcnemar(b, c)}
    meta = {i: r for (s, m, i), r in items.items() if s == "ekraft"}
    for label, field in (("Subject", "subject"), ("Bloom level", "bloom")):
        vals = sorted({meta[i][field] for i in meta})
        md += [f"Usable by {label.lower()}:", "", f"| {label} | `{BASE}` | `{EK}` |", "|---|---|---|"]
        for v in vals:
            ids = [i for i in meta if meta[i][field] == v]
            md.append(f"| {v} | {pct(sum(bs[i]['usable'] for i in ids if i in bs), sum(1 for i in ids if i in bs))} | "
                      f"{pct(sum(ek[i]['usable'] for i in ids if i in ek), sum(1 for i in ids if i in ek))} |")
        md.append("")

    # B) the 20 SSAIE topics, against Round 1
    md += ["### B. The 20 SSAIE Task B topics (general CS, mostly outside EKraft's training subjects)", "",
           "Same topics and greedy settings as Round 1; Round 1 rows reuse the Round 1 judgements.", ""]
    lapq = last_ok([r for r in rows(os.path.join(R, "laptop", "quality.jsonl")) if r.get("task") == "B"],
                   lambda r: (r["model"], r["id"]))
    groups = []
    r1 = [score(old_j.get(f"{BASE}|{i}"), (r.get("checks") or {}).get("passes_all", False))
          for (m, i), r in lapq.items() if m == BASE]
    if r1:
        groups.append((f"`{BASE}` (Round 1)", r1))
    gem = [score(j, True) for u, j in old_j.items() if u.startswith("gemini|")]
    if gem:
        groups.append(("Gemini (Round 1)", gem))
    groups.append((f"`{EK}`", list(sc.get(("ssaie", EK), {}).values())))
    md += table(groups) + [""]

    notes = [(f"{j['model']} {j['set']} {j['id']}", j.get("note")) for j in jud.values()
             if j.get("judged") and j["model"] == EK and not (j.get("key_correct") is True)]
    if notes:
        md += ["Judge notes on EKraft items with a wrong or disputed key (first 15):", ""]
        md += [f"- {k}: {n}" for k, n in notes[:15]] + [""]

md += ["## Caveats", "",
       "- The judge is one LLM (Gemini), not a teacher. Treat these numbers as independent but secondary evidence; "
       "a blind teacher rating is still needed.",
       "- EKraft v1 was trained on synthetic items written by Claude; it is a baseline for a later version trained "
       "on real, human-written items.",
       "- EKraft items were sampled (temperature 0.7, seed 42) on set A and greedy on set B; one run each.", ""]

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "ekraft_report.md"), "w", encoding="utf-8").write("\n".join(md))
json.dump(rep, open(os.path.join(OUT, "ekraft_report.json"), "w"), indent=2)
print("\n".join(md))
print(f"\nSaved: {os.path.join(OUT, 'ekraft_report.md')}")
