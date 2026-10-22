#!/usr/bin/env python3
"""Second, independent judge for the Round 1 item-writing results (standard library only).

Why: Gemini wrote 20 items and also judged them (20/20 usable). A judge from another
vendor re-scores Gemini's 20 items, plus qwen3:4b's 20 items on the same topics as a
control, so a lower score can be told apart from a stricter judge.

Same rubric text, same option shuffle (seeded by the same uid) and the same batch
layout as frontier.py's Gemini judging. Only the judge model changes.

Key: ~/.ssaie/<provider>.key (never printed or logged). Providers (free tiers):
  mistral  https://console.mistral.ai  (Experiment plan: phone verification, no card;
           requests may be used for training; the items are not sensitive)
  groq     https://console.groq.com    (fallback; openai/gpt-oss-120b)

  python3 judge2.py mistral            # judge the 40 items, resumes if stopped
  python3 judge2.py mistral --report   # summary only
  JUDGE2_MODEL=mistral-medium-latest python3 judge2.py mistral   # override the model

Output: results/frontier/judge2_<provider>.jsonl and analysis_out/judge2_<provider>.md
"""
import json, os, re, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import check_b, log  # noqa: E402
from frontier import RUBRIC, shuffled, collect_b_items, batches, parse_json  # noqa: E402

PROVIDERS = {
    "mistral": {"url": os.environ.get("JUDGE2_URL", "https://api.mistral.ai/v1/chat/completions"),
                "models": ["mistral-large-latest", "mistral-medium-latest"]},
    "groq": {"url": os.environ.get("JUDGE2_URL", "https://api.groq.com/openai/v1/chat/completions"),
             "models": ["openai/gpt-oss-120b"]},
}
SOURCES = ("gemini", "qwen3:4b")          # Gemini's own items + the control
BATCH = 20
OUT = os.path.join(HERE, "results", "frontier")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key(provider):
    p = os.path.expanduser(f"~/.ssaie/{provider}.key")
    if not os.path.exists(p):
        sys.exit(f"Missing {p}. Put the API key in that file (one line), then: chmod 600 {p}")
    return open(p).read().strip()


def call(url, model, prompt, k, retries=8):
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    delay = 15
    for attempt in range(retries):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {k}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            text = data["choices"][0]["message"]["content"] or ""
            return text, data.get("usage", {}), round(time.time() - t0, 2), data.get("model", model)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = int(float(ra)) + 1 if ra and ra.replace(".", "").isdigit() else delay
                log(f"HTTP {e.code}; waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg[:400]}")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as e:
            log(f"Request problem ({e}); waiting {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("Gave up after retries")


def parse_array(text):
    try:
        v = parse_json(text)
        if isinstance(v, dict):            # some models wrap the array in an object
            v = next((x for x in v.values() if isinstance(x, list)), [])
        return v if isinstance(v, list) else []
    except Exception:
        s, e = text.find("["), text.rfind("]")
        try:
            return json.loads(text[s:e + 1]) if s >= 0 < e else []
        except Exception:
            return []


def pick_model(provider, k):
    meta = os.path.join(OUT, f"judge2_{provider}_model.json")
    if os.environ.get("JUDGE2_MODEL"):
        return os.environ["JUDGE2_MODEL"]
    if os.path.exists(meta):
        return json.load(open(meta))["model"]
    for m in PROVIDERS[provider]["models"]:
        try:
            _, _, _, ver = call(PROVIDERS[provider]["url"], m, 'Reply with the JSON {"ok": true}', k, retries=2)
            json.dump({"model": m, "model_version": ver, "chosen_at": now()}, open(meta, "w"), indent=2)
            log(f"Using {m} ({ver})")
            return m
        except Exception as e:
            log(f"{m}: not usable ({str(e)[:160]})")
    sys.exit("No model usable on this key right now.")


def done_uids(path):
    s = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("judged") or r.get("reason"):
                s.add(r["uid"])
    return s


def judge(provider):
    k = key(provider)
    model = pick_model(provider, k)
    path = os.path.join(OUT, f"judge2_{provider}.jsonl")
    done = done_uids(path)
    pending = []
    for (m, iid, topic, bloom, resp) in collect_b_items():
        uid = f"{m}|{iid}"
        if m not in SOURCES or uid in done:
            continue
        if not check_b(resp)["passes_all"]:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"uid": uid, "model": m, "id": iid, "judged": False,
                                    "reason": "failed deterministic checks", "ts": now()}) + "\n")
            continue
        stem, opts, key_l, order = shuffled(uid, resp)
        pending.append({"uid": uid, "model": m, "id": iid, "topic": topic, "bloom": bloom,
                        "stem": stem, "options": opts, "key": key_l, "order": order})
    # mix sources inside every batch so the judge cannot tell them apart by position
    pending.sort(key=lambda x: (x["id"], x["model"]))
    for b in batches(pending, BATCH):
        blocks = [f"UID: {x['uid'].split('|')[1]}-{i}\nTopic: {x['topic']}\nBloom level: {x['bloom']}\n"
                  f"Question: {x['stem']}\n" + "\n".join(f"{L}. {v}" for L, v in x["options"].items())
                  + f"\nMarked key: {x['key']}" for i, x in enumerate(b)]
        text, usage, wall, ver = call(PROVIDERS[provider]["url"], model,
                                      RUBRIC + "\n\n" + "\n\n---\n\n".join(blocks), k)
        arr = parse_array(text)
        by_uid = {str(a.get("uid")): a for a in arr if isinstance(a, dict)}
        with open(path, "a", encoding="utf-8") as f:
            for i, x in enumerate(b):
                a = by_uid.get(f"{x['uid'].split('|')[1]}-{i}")
                row = {"uid": x["uid"], "model": x["model"], "id": x["id"], "judged": a is not None,
                       "judge_model": model, "judge_version": ver, "shown_key": x["key"],
                       "order": x["order"], "batch_wall_s": wall, "usage": usage, "ts": now()}
                if a:
                    row.update({k2: a.get(k2) for k2 in ("judge_answer", "key_correct", "clarity",
                                                           "distractors", "aligned", "note")})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_ok = sum(1 for i, x in enumerate(b) if by_uid.get(f"{x['uid'].split('|')[1]}-{i}"))
        log(f"Judged batch of {len(b)} ({n_ok} parsed)")
    missing = [x["uid"] for x in pending if x["uid"] not in done_uids(path)]
    if missing:
        log(f"{len(missing)} items not parsed; run the same command again to retry them")


def usable(j):
    return bool(j and j.get("judged") and j.get("key_correct") is True
                and int(j.get("clarity") or 0) >= 3 and int(j.get("distractors") or 0) >= 3)


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def report(provider):
    def load(p):
        d = {}
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                if r.get("judged") or r.get("reason") or r["uid"] not in d:
                    d[r["uid"]] = r        # last useful row wins (retries)
        return d
    g = load(os.path.join(OUT, "judgements.jsonl"))
    j2 = load(os.path.join(OUT, f"judge2_{provider}.jsonl"))
    meta = os.path.join(OUT, f"judge2_{provider}_model.json")
    model = json.load(open(meta))["model"] if os.path.exists(meta) else os.environ.get("JUDGE2_MODEL", "?")
    lines = [f"# Second judge ({provider}: {model}) vs Gemini, Round 1 items", "",
             "Usable = key correct, clarity >= 3 and distractors >= 3 (paper definition).", "",
             "| Items written by | n | Gemini usable | Second judge usable | Agree on usable | Cohen's kappa | Key-correct agree |",
             "|---|---|---|---|---|---|---|"]
    allpairs = []
    for m in SOURCES:
        uids = sorted(u for u in g if g[u]["model"] == m)
        both = [u for u in uids if u in j2 and (j2[u].get("judged") or j2[u].get("reason"))]
        pairs = [(usable(g[u]), usable(j2[u])) for u in both]
        allpairs += pairs
        kc = [(bool(g[u].get("key_correct")), bool(j2[u].get("key_correct"))) for u in both
              if g[u].get("judged") and j2[u].get("judged")]
        lines.append(f"| {m} | {len(both)}/{len(uids)} | {sum(a for a, _ in pairs)} | {sum(b for _, b in pairs)} | "
                     f"{sum(a == b for a, b in pairs)}/{len(pairs)} | {kappa(pairs):.2f} | "
                     f"{sum(a == b for a, b in kc)}/{len(kc)} |")
    lines += ["", f"All {len(allpairs)} items: agree on usable {sum(a == b for a, b in allpairs)}/{len(allpairs)}, "
              f"kappa {kappa(allpairs):.2f}.", "", "## Items where the judges disagree", ""]
    for u in sorted(j2):
        if u in g and usable(g[u]) != usable(j2[u]):
            r = j2[u]
            lines.append(f"- {u}: Gemini {'usable' if usable(g[u]) else 'not usable'}, second judge "
                         f"{'usable' if usable(r) else 'not usable'} (key_correct={r.get('key_correct')}, "
                         f"clarity={r.get('clarity')}, distractors={r.get('distractors')}): {r.get('note') or r.get('reason')}")
    os.makedirs(os.path.join(HERE, "analysis_out"), exist_ok=True)
    out = os.path.join(HERE, "analysis_out", f"judge2_{provider}.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in PROVIDERS:
        sys.exit(__doc__)
    provider = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if "--report" not in sys.argv:
        judge(provider)
    report(provider)


if __name__ == "__main__":
    main()
