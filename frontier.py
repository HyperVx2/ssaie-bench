#!/usr/bin/env python3
"""Frontier baseline and judge using the Gemini API free tier (standard library only).

Key: ~/.ssaie/gemini.key (never printed or logged).

  python3 frontier.py answer     # Gemini answers the 200 Task A items (batches of 20)
  python3 frontier.py generate   # Gemini writes the 20 Task B items
  python3 frontier.py judge      # Gemini scores every Task B item (needs results/laptop/quality.jsonl)

Every step resumes where it stopped. If the free daily quota runs out, the
script saves progress and tells you to re-run it later.
"""
import json, os, random, re, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import CFG, LETTERS, check_b, prompt_b, log  # noqa: E402

OUT = os.path.join(HERE, "results", "frontier")
API = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


class DailyQuota(Exception):
    pass


def key():
    p = os.path.expanduser("~/.ssaie/gemini.key")
    if not os.path.exists(p):
        sys.exit("Missing ~/.ssaie/gemini.key")
    return open(p).read().strip()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def call(model, prompt, k, retries=8):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    delay = 20
    for attempt in range(retries):
        req = urllib.request.Request(API.format(m=model), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": k})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return text, data.get("usageMetadata", {}), round(time.time() - t0, 2), data.get("modelVersion")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 429:
                if re.search(r"PerDay|per day|daily", msg, re.I):
                    raise DailyQuota(msg[:300])
                m = re.search(r'"retryDelay":\s*"(\d+)', msg)
                wait = int(m.group(1)) + 2 if m else delay
                log(f"Rate limited; waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                delay = min(delay * 2, 300)
                continue
            if e.code in (500, 502, 503, 504):
                log(f"Server error {e.code}; waiting {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg[:400]}")
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as e:
            log(f"Request problem ({e}); waiting {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("Gave up after retries")


def pick_model(k):
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("GEMINI_MODEL"):          # manual override, e.g. if one model's quota is gone
        log(f"Using {os.environ['GEMINI_MODEL']} (override)")
        return os.environ["GEMINI_MODEL"]
    p = os.path.join(OUT, "model.json")
    if os.path.exists(p):
        return json.load(open(p))["model"]
    for m in CFG["gemini_model_preference"]:
        try:
            text, usage, wall, ver = call(m, 'Return the JSON {"ok": true}', k, retries=2)
            json.dump({"model": m, "model_version": ver, "chosen_at": now()}, open(p, "w"), indent=2)
            log(f"Using {m} ({ver})")
            return m
        except DailyQuota:
            log(f"{m}: daily quota exhausted, trying next")
        except Exception as e:
            log(f"{m}: not usable ({str(e)[:120]})")
    sys.exit("No Gemini model usable on this key right now.")


def done_ids(path, field="id"):
    s = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                s.add(json.loads(line)[field])
            except Exception:
                pass
    return s


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json(text):
    s = text.strip()
    s = re.sub(r"^```(json)?|```$", "", s, flags=re.M).strip()
    return json.loads(s)


def batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ----------------------------------------------------------------- Task A
def do_answer(k, model):
    items = [json.loads(l) for l in open(os.path.join(HERE, "data", "items.jsonl"), encoding="utf-8")]
    path = os.path.join(OUT, "answers.jsonl")
    done = done_ids(path)
    todo = [i for i in items if i["id"] not in done]
    for b in batches(todo, CFG["gemini_batch_size"]):
        blocks = []
        for it in b:
            opts = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(it["options"]))
            blocks.append(f"ID: {it['id']}\nQuestion: {it['question']}\nOptions:\n{opts}")
        prompt = ("Answer each multiple-choice computer science question below. Return only a JSON array "
                  "of objects {\"id\": <ID>, \"answer\": <single option letter>} in the same order.\n\n"
                  + "\n\n---\n\n".join(blocks))
        text, usage, wall, ver = call(model, prompt, k)
        try:
            got = {str(x["id"]): str(x["answer"]).strip().upper()[:1] for x in parse_json(text)}
        except Exception:
            got = {}
        for it in b:
            pred = got.get(it["id"])
            pred = pred if pred and pred in LETTERS[:len(it["options"])] else None
            append(path, {"id": it["id"], "model": model, "model_version": ver, "pred": pred,
                          "gold": it["answer"], "correct": pred == it["answer"], "parsed": pred is not None,
                          "origin": it["origin"], "batch_wall_s": wall, "batch_size": len(b),
                          "usage": usage, "ts": now()})
        log(f"Answered {len(b)} items ({len(done_ids(path))}/{len(items)})")


# ----------------------------------------------------------------- Task B
def do_generate(k, model):
    topics = json.load(open(os.path.join(HERE, "data", "topics.json")))
    path = os.path.join(OUT, "generated.jsonl")
    done = done_ids(path)
    todo = [t for t in topics if t["id"] not in done]
    for b in batches(todo, 10):
        tasks = "\n".join(f"- ID {t['id']}: topic \"{t['topic']}\", Bloom level \"{t['bloom']}\"" for t in b)
        prompt = ("You are a computer science instructor writing an exam for first- or second-year "
                  "undergraduates. For EACH task below write ONE multiple-choice question with exactly four "
                  "options and exactly one correct answer. Do not use \"all of the above\" or \"none of the "
                  "above\". Return only a JSON array; each element has keys \"id\" (the task ID), \"stem\", "
                  "\"options\" (object with keys \"A\",\"B\",\"C\",\"D\"), \"key\" (one of A-D) and "
                  "\"rationale\" (one sentence).\n\n" + tasks)
        text, usage, wall, ver = call(model, prompt, k)
        try:
            arr = {str(x.get("id")): x for x in parse_json(text) if isinstance(x, dict)}
        except Exception:
            arr = {}
        for t in b:
            obj = arr.get(t["id"])
            resp = json.dumps({kk: obj.get(kk) for kk in ("stem", "options", "key", "rationale")},
                              ensure_ascii=False) if obj else ""
            append(path, {"id": t["id"], "model": model, "model_version": ver, "topic": t["topic"],
                          "bloom": t["bloom"], "response": resp, "checks": check_b(resp),
                          "batch_wall_s": wall, "batch_size": len(b), "usage": usage, "ts": now()})
        log(f"Generated {len(b)} items")


# ----------------------------------------------------------------- judging
RUBRIC = """You are an expert computer science educator reviewing exam items written for first- and
second-year undergraduates. For EACH item below, judge it independently and return only a JSON array of
objects with these keys, in the same order:
  "uid": copy the item's UID exactly,
  "judge_answer": the option letter (A-D) you believe is correct,
  "key_correct": true if the marked key is the single correct answer, else false,
  "clarity": integer 1-4 (1 = confusing or ambiguous, 4 = perfectly clear),
  "distractors": integer 1-4 (1 = implausible or obviously wrong, 4 = all plausible to a weak student),
  "aligned": true if the item tests the stated topic at the stated Bloom level, else false,
  "note": at most 15 words.
Judge content only; ignore which source might have written an item."""


def collect_b_items():
    items = []
    lap = os.path.join(HERE, "results", "laptop", "quality.jsonl")
    if os.path.exists(lap):
        for line in open(lap, encoding="utf-8"):
            r = json.loads(line)
            if r.get("task") == "B" and r.get("status") == "ok":
                items.append((r["model"], r["id"], r["topic"], r["bloom"], r.get("response")))
    else:
        log("results/laptop/quality.jsonl not found; judging only Gemini's own items")
    gen = os.path.join(OUT, "generated.jsonl")
    if os.path.exists(gen):
        for line in open(gen, encoding="utf-8"):
            r = json.loads(line)
            items.append(("gemini", r["id"], r["topic"], r["bloom"], r.get("response")))
    return items


def shuffled(uid, text):
    s = text.strip()
    obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    opts = obj["options"]
    if isinstance(opts, list):
        opts = {LETTERS[i]: v for i, v in enumerate(opts)}
    key_letter = str(obj["key"]).strip().upper()[:1]
    order = list("ABCD")
    random.Random(uid).shuffle(order)            # new position i holds old option order[i]
    new_opts = {LETTERS[i]: opts[o] for i, o in enumerate(order)}
    new_key = LETTERS[order.index(key_letter)]
    return obj["stem"], new_opts, new_key, order


def do_judge(k, model):
    path = os.path.join(OUT, "judgements.jsonl")
    done = done_ids(path, "uid")
    pending = []
    for (m, iid, topic, bloom, resp) in collect_b_items():
        uid = f"{m}|{iid}"
        if uid in done:
            continue
        if not check_b(resp)["passes_all"]:
            append(path, {"uid": uid, "model": m, "id": iid, "judged": False,
                          "reason": "failed deterministic checks", "ts": now()})
            continue
        stem, opts, key_l, order = shuffled(uid, resp)
        pending.append({"uid": uid, "model": m, "id": iid, "topic": topic, "bloom": bloom,
                        "stem": stem, "options": opts, "key": key_l, "order": order})
    for b in batches(pending, CFG["gemini_batch_size"]):
        blocks = [f"UID: {x['uid'].split('|')[1]}-{i}\nTopic: {x['topic']}\nBloom level: {x['bloom']}\n"
                  f"Question: {x['stem']}\n" + "\n".join(f"{L}. {v}" for L, v in x["options"].items())
                  + f"\nMarked key: {x['key']}" for i, x in enumerate(b)]
        text, usage, wall, ver = call(model, RUBRIC + "\n\n" + "\n\n---\n\n".join(blocks), k)
        try:
            arr = parse_json(text)
        except Exception:
            arr = []
        by_uid = {str(a.get("uid")): a for a in arr if isinstance(a, dict)}
        for i, x in enumerate(b):
            a = by_uid.get(f"{x['uid'].split('|')[1]}-{i}") or (arr[i] if i < len(arr) and isinstance(arr[i], dict) else None)
            row = {"uid": x["uid"], "model": x["model"], "id": x["id"], "judged": a is not None,
                   "judge_model": model, "judge_version": ver, "shown_key": x["key"], "order": x["order"],
                   "ts": now()}
            if a:
                row.update({k2: a.get(k2) for k2 in ("judge_answer", "key_correct", "clarity",
                                                       "distractors", "aligned", "note")})
            append(path, row)
        log(f"Judged {len(b)} items ({len(done_ids(path, 'uid'))} total)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("answer", "generate", "judge", "all"):
        sys.exit(__doc__)
    k = key()
    model = pick_model(k)
    steps = ["answer", "generate", "judge"] if sys.argv[1] == "all" else [sys.argv[1]]
    try:
        for s in steps:
            log(f"--- {s} ---")
            {"answer": do_answer, "generate": do_generate, "judge": do_judge}[s](k, model)
    except DailyQuota:
        log("Free daily quota used up. Progress is saved; re-run the same command tomorrow.")
        sys.exit(2)
    log("Frontier step(s) complete.")


if __name__ == "__main__":
    main()
