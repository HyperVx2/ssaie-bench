#!/usr/bin/env python3
"""Analysis for the SEA-SSAIE 2027 benchmark. Usage: python3 analyze.py <results_dir> <out_dir>"""
import json, math, os, sys, statistics as st, collections, random

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_ORDER = ["gemma3:1b", "llama3.2:1b", "qwen3.5:0.8b", "qwen3:1.7b", "qwen3.5:2b", "llama3.2:3b", "qwen2.5:3b", "granite4:3b", "ministral-3:3b", "phi4-mini:3.8b", "qwen3:4b", "qwen3.5:4b", "gemma3:4b", "qwen3:8b"]


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def latest(rs):
    """Keep the last 'ok' row per (model, task, id); fall back to the last row if none ok."""
    by = {}
    for r in rs:
        k = (r["model"], r["task"], r["id"])
        if k not in by or r.get("status") == "ok" or by[k].get("status") != "ok":
            if by.get(k, {}).get("status") == "ok" and r.get("status") != "ok":
                continue
            by[k] = r
    return by


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def iqr(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 4:
        return None
    q = st.quantiles(xs, n=4)
    return q[2] - q[0]


laptop_file = os.path.join(R, "laptop", "quality.jsonl")
if not os.path.exists(laptop_file):
    laptop_file = os.path.join(R, "laptop", "quality_final.jsonl")
lap = latest(rows(laptop_file) or rows(os.path.join(R, "laptop", "quality_partial.jsonl")))
srv = {t: latest(rows(os.path.join(R, t, "speed.jsonl"))) for t in ("vps", "homelab", "pi")}
gem_a = {r["id"]: r for r in rows(os.path.join(R, "frontier", "answers.jsonl"))}
gem_b = {r["id"]: r for r in rows(os.path.join(R, "frontier", "generated.jsonl"))}
judg = {r["uid"]: r for r in rows(os.path.join(R, "frontier", "judgements.jsonl"))}
gem_model = json.load(open(os.path.join(R, "frontier", "model.json")))["model"]
items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(R, "..", "data", "items.jsonl"))} \
    if os.path.exists(os.path.join(R, "..", "data", "items.jsonl")) else {}
models = [m for m in MODEL_ORDER if any(k[0] == m for k in lap)]
con = {}
for t in ("laptop", "vps", "homelab"):
    for k, r in latest(rows(os.path.join(R, t, "constrained.jsonl"))).items():
        if k[1] == "A" and r.get("status") == "ok" and k not in con:
            con[k] = r
report = {"gemini_model": gem_model}

# ------------------------------------------------------------------ RQ1 accuracy
acc = []
item_ids = sorted({k[2] for k in lap if k[1] == "A"})
for m in models + ["gemini"]:
    if m == "gemini":
        A = [gem_a[i] for i in item_ids if i in gem_a]
    else:
        A = [lap[(m, "A", i)] for i in item_ids if (m, "A", i) in lap and lap[(m, "A", i)].get("status") == "ok"]
    n = len(A)
    k = sum(1 for r in A if r.get("correct"))
    p, lo, hi = wilson(k, n)
    orig = [r for r in A if r.get("origin") == "original_mmlu"]
    new = [r for r in A if r.get("origin") == "new_in_mmlu_pro"]
    acc.append({"model": m, "n": n, "correct": k, "acc": p, "lo": lo, "hi": hi,
                "parse_rate": sum(1 for r in A if r.get("parsed")) / n if n else None,
                "acc_original": sum(r["correct"] for r in orig) / len(orig) if orig else None,
                "acc_new": sum(r["correct"] for r in new) / len(new) if new else None,
                "n_original": len(orig), "n_new": len(new)})
report["accuracy"] = acc

acc_c = []
for m in models:
    A = [con[(m, "A", i)] for i in item_ids if (m, "A", i) in con]
    n = len(A); k = sum(1 for r in A if r.get("correct"))
    p, lo, hi = wilson(k, n)
    orig = [r for r in A if r.get("origin") == "original_mmlu"]; new = [r for r in A if r.get("origin") == "new_in_mmlu_pro"]
    ff = next(a for a in acc if a["model"] == m)
    acc_c.append({"model": m, "n": n, "correct": k, "acc": p, "lo": lo, "hi": hi,
                  "valid_rate": sum(1 for r in A if r.get("parsed")) / n if n else None,
                  "acc_original": sum(r["correct"] for r in orig) / len(orig) if orig else None,
                  "acc_new": sum(r["correct"] for r in new) / len(new) if new else None,
                  "freeform_acc": ff["acc"], "freeform_parse": ff["parse_rate"],
                  "machine": sorted({r.get("tier") for r in A})})
report["accuracy_constrained"] = acc_c
g = next(a for a in acc if a["model"] == "gemini")
report["gemini_acc"] = g
mc = []
for a in acc_c:
    b = c = 0
    for i in item_ids:
        x = con.get((a["model"], "A", i), {}).get("correct", False)
        y = gem_a.get(i, {}).get("correct", False)
        b += x and not y; c += y and not x
    mc.append({"model": a["model"], "local_only": b, "gemini_only": c, "p": mcnemar_exact(b, c)})
report["mcnemar_constrained"] = mc
local = [a for a in acc if a["model"] != "gemini" and a["n"] == 200]
best = max(local, key=lambda a: a["acc"]) if local else None
if best:
    b = c = 0
    for i in item_ids:
        x = lap.get((best["model"], "A", i), {}).get("correct", False)
        y = gem_a.get(i, {}).get("correct", False)
        b += x and not y
        c += y and not x
    report["mcnemar_best_vs_gemini"] = {"model": best["model"], "local_only": b, "gemini_only": c,
                                        "p": mcnemar_exact(b, c)}

# ------------------------------------------------------------------ RQ2 item writing
def b_rows(m):
    if m == "gemini":
        return [dict(r, model="gemini") for r in gem_b.values()]
    return [lap[k] for k in lap if k[0] == m and k[1] == "B" and lap[k].get("status") == "ok"]


gen = []
for m in models + ["gemini"]:
    B = b_rows(m)
    n = len(B)
    det = sum(1 for r in B if r["checks"]["passes_all"])
    rep = sum(1 for r in B if r.get("failure") == "repeat_limit")
    J = [judg.get(f"{m}|{r['id']}") for r in B]
    jud = [j for j in J if j and j.get("judged")]
    usable = sum(1 for j in jud if j.get("key_correct") is True and (j.get("clarity") or 0) >= 3
                 and (j.get("distractors") or 0) >= 3)
    gen.append({"model": m, "n": n, "det_pass": det, "repeat_fail": rep, "judged": len(jud),
                "key_correct": sum(1 for j in jud if j.get("key_correct") is True),
                "aligned": sum(1 for j in jud if j.get("aligned") is True),
                "clarity_mean": st.mean([j["clarity"] for j in jud if isinstance(j.get("clarity"), (int, float))]) if jud else None,
                "distractors_mean": st.mean([j["distractors"] for j in jud if isinstance(j.get("distractors"), (int, float))]) if jud else None,
                "usable": usable, "usable_rate": usable / n if n else None,
                "judge_complete": len(jud) == det})
report["generation"] = gen

# ------------------------------------------------------------------ RQ3 speed / memory
def speed_rows(tier, m):
    if tier == "laptop":
        src = lap
        A = [src[k] for k in src if k[0] == m and k[1] == "A" and src[k].get("status") == "ok"
             and src[k].get("speed_subset") and "wall_s" in src[k]]
        B = [src[k] for k in src if k[0] == m and k[1] == "B" and src[k].get("status") == "ok"
             and k[2] in ("B01", "B02", "B03", "B04", "B05") and "wall_s" in src[k]]
    else:
        src = srv[tier]
        A = [src[k] for k in src if k[0] == m and k[1] == "A" and src[k].get("status") == "ok" and "wall_s" in src[k]]
        B = [src[k] for k in src if k[0] == m and k[1] == "B" and src[k].get("status") == "ok" and "wall_s" in src[k]]
    L = [src[k] for k in src if k[0] == m and k[1] == "load" and src[k].get("status") == "ok"]
    loadfail = [src[k] for k in src if k[0] == m and k[1] == "load" and src[k].get("status") != "ok"]
    return A, B, L, loadfail


speed = []
for tier in ("homelab", "vps", "laptop", "pi"):
    for m in models:
        A, B, L, LF = speed_rows(tier, m)
        if not (A or B or L or LF):
            continue
        speed.append({
            "tier": tier, "model": m, "quant": (A or B or L or [{}])[0].get("quantization"),
            "loaded": bool(L), "load_s": min((r["load_s"] for r in L), default=None),
            "a_n": len(A), "a_wall_med": med([r["wall_s"] for r in A]), "a_wall_iqr": iqr([r["wall_s"] for r in A]),
            "prompt_tps_med": med([r["prompt_tps"] for r in A]),
            "b_n": len(B), "b_wall_med": med([r["wall_s"] for r in B]),
            "gen_tps_med": med([r["gen_tps"] for r in B]), "gen_tps_iqr": iqr([r["gen_tps"] for r in B]),
            "peak_rss_gb": max([r.get("peak_runner_rss") or 0 for r in A + B], default=0) / 1e9,
            "min_avail_gb": min([r.get("min_mem_available") or 1e18 for r in A + B], default=0) / 1e9,
            "errors": 0})
report["speed"] = speed

# agreement across machines on the 20-item subset (determinism check)
agree = []
subset = [i for i in item_ids if lap.get((models[0], "A", i), {}).get("speed_subset")] if models else []
for m in models:
    for tier in ("vps", "homelab", "pi"):
        same = tot = 0
        for i in subset:
            a = lap.get((m, "A", i), {})
            b = srv[tier].get((m, "A", i), {})
            if a.get("status") == "ok" and b.get("status") == "ok":
                tot += 1
                same += a.get("pred") == b.get("pred")
        if tot:
            agree.append({"model": m, "tier": tier, "same": same, "n": tot})
report["agreement"] = agree

# concurrency
conc = []
for tier in ("homelab", "vps", "laptop", "pi"):
    for r in rows(os.path.join(R, tier, "concurrency.jsonl")):
        reqs = [q for q in r["requests"] if q and "wall_s" in q]
        conc.append({"tier": tier, "n": r["concurrency"], "round": r["round"],
                     "agg_tps": r["aggregate_gen_tps"], "max_latency": max(q["wall_s"] for q in reqs) if reqs else None,
                     "mean_req_tps": st.mean([q["gen_tps"] for q in reqs if q.get("gen_tps")]) if reqs else None,
                     "errors": sum(1 for q in r["requests"] if q and "error" in q)})
report["concurrency"] = conc

json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------ console summary
def f(x, d=2):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


print(f"Gemini model: {gem_model}")
print("\nRQ1 accuracy (200 items)")
for a in acc:
    print(f"  {a['model']:15s} n={a['n']:3d} acc={f(a['acc'],3)} [{f(a['lo'],3)},{f(a['hi'],3)}] parse={f(a['parse_rate'],2)} "
          f"orig={f(a['acc_original'],3)} new={f(a['acc_new'],3)}")
print("  McNemar:", report.get("mcnemar_best_vs_gemini"))
print("\nRQ1 CONSTRAINED accuracy")
for a in acc_c:
    print(f"  {a['model']:15s} n={a['n']} acc={f(a['acc'],3)} [{f(a['lo'],3)},{f(a['hi'],3)}] valid={f(a['valid_rate'],2)} "
          f"orig={f(a['acc_original'],3)} new={f(a['acc_new'],3)} | freeform acc={f(a['freeform_acc'],3)} parse={f(a['freeform_parse'],2)} {a['machine']}")
for x in mc: print("  McNemar vs gemini:", x)
print("\nRQ2 generation")
for g in gen:
    print(f"  {g['model']:15s} n={g['n']} det_pass={g['det_pass']} repeat={g['repeat_fail']} judged={g['judged']} "
          f"key_ok={g['key_correct']} aligned={g['aligned']} clar={f(g['clarity_mean'])} dist={f(g['distractors_mean'])} usable={g['usable']}")
print("\nRQ3 speed")
for s in speed:
    print(f"  {s['tier']:8s} {s['model']:15s} {s['quant']} load={f(s['load_s'],1)}s A_med={f(s['a_wall_med'],2)}s "
          f"ptps={f(s['prompt_tps_med'],1)} B_med={f(s['b_wall_med'],1)}s gtps={f(s['gen_tps_med'],1)} rss={f(s['peak_rss_gb'],2)}GB "
          f"minavail={f(s['min_avail_gb'],2)}GB nA={s['a_n']} nB={s['b_n']}")
print("\nAgreement:", [(a['model'], a['tier'], f"{a['same']}/{a['n']}") for a in agree])
print("\nConcurrency:")
for c in conc:
    print(f"  {c['tier']:8s} x{c['n']} r{c['round']} agg={f(c['agg_tps'],1)} tok/s maxlat={f(c['max_latency'],1)}s errors={c['errors']}")
