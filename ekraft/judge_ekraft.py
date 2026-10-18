#!/usr/bin/env python3
"""Gemini judging of the EKraft item-writing results (same rubric and judge model as Round 1).

Run it on the machine that has the Gemini key (~/.ssaie/gemini.key), after copying
results/laptop/ekraft_items.jsonl there:

  python3 ekraft/judge_ekraft.py

The judge sees only the topic, Bloom level, question, options (reshuffled) and marked key. It never
sees which model wrote an item, and items from EKraft and qwen3:4b are mixed in every batch.
Resumes where it stopped; if the free daily quota runs out, re-run it later.
Output: results/frontier/ekraft_judgements.jsonl
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from bench import CFG, check_b, log  # noqa: E402
from frontier import (OUT, RUBRIC, DailyQuota, append, batches, call, done_ids, key,  # noqa: E402
                      now, parse_json, pick_model, shuffled)


def main():
    src = os.path.join(ROOT, "results", "laptop", "ekraft_items.jsonl")
    if not os.path.exists(src):
        sys.exit(f"{src} not found. Copy it from the laptop first (see ekraft/README.md, step 3).")
    k = key()
    model = pick_model(k)                 # reuses results/frontier/model.json, i.e. the Round 1 judge
    path = os.path.join(OUT, "ekraft_judgements.jsonl")
    done = set()                          # judged, or failed the format checks; missed items are retried
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            j = json.loads(line)
            if j.get("judged") or j.get("reason"):
                done.add(j["uid"])

    latest = {}
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") == "ok":
            latest[(r["set"], r["model"], r["id"])] = r
    pending = []
    for (pset, m, iid), r in latest.items():
        uid = f"{m}|{pset}|{iid}"
        if uid in done:
            continue
        resp = r.get("response")
        if not check_b(resp)["passes_all"]:
            append(path, {"uid": uid, "set": pset, "model": m, "id": iid, "judged": False,
                          "reason": "failed deterministic checks", "ts": now()})
            continue
        stem, opts, key_l, order = shuffled(uid, resp)
        topic = r["topic"] if pset == "ssaie" else f"{r['topic']} ({r['subject']})"
        pending.append({"uid": uid, "set": pset, "model": m, "id": iid, "topic": topic, "bloom": r["bloom"],
                        "stem": stem, "options": opts, "key": key_l, "order": order})
    random.Random(2027).shuffle(pending)  # mix models within every batch
    log(f"Judge: {model}; {len(pending)} items to judge -> {path}")

    try:
        for b in batches(pending, CFG["gemini_batch_size"]):
            blocks = [f"UID: {x['id']}-{i}\nTopic: {x['topic']}\nBloom level: {x['bloom']}\n"
                      f"Question: {x['stem']}\n" + "\n".join(f"{L}. {v}" for L, v in x["options"].items())
                      + f"\nMarked key: {x['key']}" for i, x in enumerate(b)]
            text, usage, wall, ver = call(model, RUBRIC + "\n\n" + "\n\n---\n\n".join(blocks), k)
            try:
                arr = parse_json(text)
            except Exception:
                arr = []
            by_uid = {str(a.get("uid")): a for a in arr if isinstance(a, dict)}
            # match by UID; fall back to position only when the judge dropped every UID but kept the order
            by_pos = not by_uid and len(arr) == len(b)
            for i, x in enumerate(b):
                a = by_uid.get(f"{x['id']}-{i}") or (arr[i] if by_pos and isinstance(arr[i], dict) else None)
                row = {"uid": x["uid"], "set": x["set"], "model": x["model"], "id": x["id"],
                       "judged": a is not None, "judge_model": model, "judge_version": ver,
                       "shown_key": x["key"], "order": x["order"], "ts": now()}
                if a:
                    row.update({k2: a.get(k2) for k2 in ("judge_answer", "key_correct", "clarity",
                                                           "distractors", "aligned", "note")})
                append(path, row)
            log(f"Judged {len(b)} items ({len(done_ids(path, 'uid'))} total)")
    except DailyQuota:
        log("Free daily quota used up. Progress is saved; re-run the same command tomorrow.")
        sys.exit(2)
    log("EKraft judging complete. Next: python3 ekraft/report_ekraft.py")


if __name__ == "__main__":
    main()
