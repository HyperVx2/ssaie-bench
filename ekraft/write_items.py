#!/usr/bin/env python3
"""EKraft item writing (laptop). Standard library only; reuses bench.py.

Two prompt sets:
  ssaie   the 20 Task B topics from data/topics.json, written by EKraft with the Round 1 settings
          (temperature 0, seed 42), so they compare directly with the Round 1 qwen3:4b and Gemini items.
  ekraft  231 held-out EKraft prompts (ekraft/prompts_ekraft.jsonl: Software Engineering 1 and Data
          Communication and Networking 1). Written by EKraft AND by the plain qwen3:4b, both at
          temperature 0.7, top_p 0.9, seed 42 (how EKraft was evaluated during development).

EKraft gets the prompt format it was trained on and answers as plain text; qwen3:4b gets the Round 1
JSON prompt. Every EKraft answer is converted to the same JSON shape, so bench.check_b() and
frontier.py's judge treat both models the same way.

  python3 ekraft/write_items.py --model ekraft-v1-plan:4b            # everything (resumes)
  python3 ekraft/write_items.py --model ekraft-v1-plan:4b --n 50     # quick partial run

Output: results/laptop/ekraft_items.jsonl
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import bench  # noqa: E402
from bench import CFG, api_post, check_b, log, now, prompt_b, timing_fields  # noqa: E402

ITEM = re.compile(r"^(?P<stem>.+?)\n\s*A[.)]\s*(?P<A>.+?)\n\s*B[.)]\s*(?P<B>.+?)\n\s*C[.)]\s*(?P<C>.+?)"
                  r"\n\s*D[.)]\s*(?P<D>.+?)\n\s*Answer:\s*\**(?P<key>[ABCD])\b", re.S)


def native_prompt(subject, topic, bloom):
    return ("Write one 4-option multiple-choice exam item.\n"
            f"Subject: {subject}\nTopic: {topic}\nBloom level: {bloom}")


def native_to_json(raw):
    """'Plan: ...\\n<stem>\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer: X' -> bench JSON string (or None)."""
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    plan = ""
    if t.startswith("Plan:"):
        first, _, t = t.partition("\n")
        plan, t = first[5:].strip(), t.strip()
    m = ITEM.match(t)
    if not m:
        return None
    return json.dumps({"stem": m["stem"].strip(), "options": {k: m[k].strip() for k in "ABCD"},
                       "key": m["key"], "rationale": plan}, ensure_ascii=False)


def ensure_model(name):
    try:
        api_post("/api/show", {"model": name}, timeout=60)
    except Exception:
        if name.startswith("ekraft"):
            sys.exit(f"{name} is not installed. Run: bash ekraft/run_ekraft.sh laptop")
        bench.pull(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="EKraft Ollama tag, e.g. ekraft-v1-plan:4b")
    ap.add_argument("--base", default="qwen3:4b", help="untuned model that writes the same EKraft prompts")
    ap.add_argument("--n", type=int, default=None, help="only the first N EKraft prompts")
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()

    topics = json.load(open(os.path.join(ROOT, "data", "topics.json")))
    prompts = [json.loads(l) for l in open(os.path.join(HERE, "prompts_ekraft.jsonl"), encoding="utf-8")]
    prompts = prompts[:a.n] if a.n else prompts

    sampled = dict(CFG["options"], temperature=a.temp, top_p=0.9, num_predict=CFG["task_b_num_predict"])
    greedy = dict(CFG["options"], num_predict=CFG["task_b_num_predict"])        # Round 1 settings
    jobs = []   # (set, model, style, id, subject, topic, bloom, options)
    for t in topics:
        jobs.append(("ssaie", a.model, "native", t["id"], "Computer Science", t["topic"], t["bloom"], greedy))
    for p in prompts:
        for m, style in ((a.model, "native"), (a.base, "bench_json")):
            jobs.append(("ekraft", m, style, p["id"], p["subject"], p["topic"], p["bloom"], sampled))

    outdir = os.path.join(ROOT, "results", "laptop")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ekraft_items.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add((r["set"], r["model"], r["id"]))
    todo = [j for j in jobs if (j[0], j[1], j[3]) not in done]
    log(f"{len(jobs)} items in total, {len(todo)} to do -> {path}")
    if not todo:
        return

    server = bench.Server(outdir)
    server.start(num_parallel=1)
    try:
        # group by model so each model loads once (MAX_LOADED_MODELS=1)
        for model in (a.model, a.base):
            mine = [j for j in todo if j[1] == model]
            if not mine:
                continue
            ensure_model(model)
            meta = bench.model_meta(model)
            think_ok = "thinking" in (meta.get("capabilities") or [])
            t0 = time.time()
            for k, (pset, m, style, iid, subj, topic, bloom, opts) in enumerate(mine, 1):
                if style == "native":
                    payload = {"model": m, "prompt": native_prompt(subj, topic, bloom), "stream": False,
                               "keep_alive": "30m", "options": opts}
                else:
                    payload = {"model": m, "prompt": prompt_b({"topic": f"{topic} (course: {subj})", "bloom": bloom}),
                               "stream": False, "keep_alive": "30m", "options": opts, "format": "json"}
                if think_ok:
                    payload["think"] = False
                row = {"set": pset, "model": m, "id": iid, "subject": subj, "topic": topic, "bloom": bloom,
                       "prompt_style": style, "temperature": opts["temperature"], "ts": now()}
                try:
                    s = time.time()
                    r = api_post("/api/generate", payload)
                    r["_wall_s"] = time.time() - s
                    raw = r.get("response") or ""
                    resp = native_to_json(raw) if style == "native" else raw
                    row.update(status="ok", raw=raw, response=resp or raw,
                               parsed=resp is not None, checks=check_b(resp), **timing_fields(r))
                except Exception as e:
                    err = str(e)
                    if "repeat limit" in err or "prediction aborted" in err:
                        row.update(status="ok", raw=None, response=None, parsed=False,
                                   checks=check_b(None), failure="repeat_limit", error=err[:300])
                    else:
                        row.update(status="error", error=err[:500])
                        log(f"{m} {pset} {iid}: error {err[:150]}")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if k % 10 == 0 or k == len(mine):
                    el = time.time() - t0
                    log(f"{m}: {k}/{len(mine)} done, ~{el / k * (len(mine) - k) / 60:.0f} min left for this model")
    finally:
        server.stop()
    log(f"Item writing done: {path}")


if __name__ == "__main__":
    main()
