#!/usr/bin/env python3
"""SEA-SSAIE 2027 benchmark harness (standard library only).

Runs small open models through Ollama and logs answers, timings and memory.

  python3 bench.py --tier laptop           # quality run: 200 items + 20 prompts, all models, + concurrency
  python3 bench.py --tier vps              # speed run: 20 items + 5 prompts, server models, + concurrency
  python3 bench.py --tier homelab          # same as vps

Safe to stop (Ctrl+C) and re-run: finished requests are skipped.
"""
import argparse, json, os, platform, re, shutil, signal, subprocess, sys, threading, time
import urllib.request, urllib.error
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
LETTERS = "ABCDEFGHIJ"
PORT = CFG["ollama_port"]
BASE = f"http://127.0.0.1:{PORT}"
MODELS_DIR = os.path.expanduser("~/ssaie-models")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- Ollama server
class Server:
    def __init__(self, logdir):
        self.proc, self.logdir = None, logdir

    def start(self, num_parallel=1):
        self.stop()
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{PORT}", OLLAMA_MODELS=MODELS_DIR,
                   OLLAMA_NUM_PARALLEL=str(num_parallel), OLLAMA_MAX_LOADED_MODELS="1",
                   OLLAMA_KEEP_ALIVE="30m")
        os.makedirs(MODELS_DIR, exist_ok=True)
        logf = open(os.path.join(self.logdir, "ollama_server.log"), "a")
        self.proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=logf, stderr=logf)
        global SERVER_PID
        SERVER_PID = self.proc.pid
        for _ in range(60):
            try:
                v = api_get("/api/version", timeout=2)
                log(f"Ollama {v.get('version')} up on port {PORT} (parallel={num_parallel})")
                return v.get("version")
            except Exception:
                time.sleep(1)
        sys.exit("Ollama server did not start; see results/<tier>/ollama_server.log")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def api_get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def api_post(path, payload, timeout=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or CFG["request_timeout_s"]) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def pull(model):
    log(f"Pulling {model} ...")
    req = urllib.request.Request(BASE + "/api/pull", data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = -10
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            m = json.loads(line)
            if "error" in m:
                raise RuntimeError(m["error"])
            if m.get("total") and m.get("completed") is not None:
                pct = int(100 * m["completed"] / m["total"])
                if pct >= last + 10:
                    last = pct
                    log(f"  {model}: {pct}%")
    log(f"  {model}: done")


# ----------------------------------------------------------------- memory sampling
def read_meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) * 1024
    return d


SERVER_PID = None


def runner_rss():
    """Resident memory (bytes) of every process spawned by our Ollama server (the model runners)."""
    if not SERVER_PID:
        return 0
    parent = {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                parent[int(pid)] = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
            except Exception:
                pass
    total = 0
    for pid in parent:
        p, depth = pid, 0
        while p in parent and p != SERVER_PID and depth < 10:
            p, depth = parent[p], depth + 1
        if p == SERVER_PID and pid != SERVER_PID:
            try:
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
            except Exception:
                pass
    return total


class MemSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.reset()

    def reset(self):
        try:
            self.min_avail, self.max_rss = read_meminfo().get("MemAvailable"), runner_rss()
        except Exception:
            self.min_avail, self.max_rss = None, 0

    def run(self):
        while not self.stop_flag.is_set():
            try:
                a = read_meminfo().get("MemAvailable", 0)
                self.min_avail = a if self.min_avail is None else min(self.min_avail, a)
                self.max_rss = max(self.max_rss, runner_rss())
            except Exception:
                pass
            time.sleep(0.5)


# ----------------------------------------------------------------- prompts and parsing
def prompt_a(item):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return ("The following is a multiple-choice question about computer science. "
            "Reply with only the letter of the correct option.\n\n"
            f"Question: {item['question']}\n\nOptions:\n{opts}\n\nAnswer:")


def prompt_b(t):
    return ("You are a computer science instructor writing an exam. Write ONE multiple-choice "
            f"question on the topic \"{t['topic']}\" at the \"{t['bloom']}\" level of Bloom's taxonomy, "
            "for first- or second-year undergraduate students. Give exactly four options with exactly "
            "one correct answer. Do not use \"all of the above\" or \"none of the above\".\n"
            "Return only JSON with these keys: \"stem\" (string), \"options\" (object with keys "
            "\"A\", \"B\", \"C\", \"D\"), \"key\" (one of \"A\",\"B\",\"C\",\"D\"), "
            "\"rationale\" (one sentence).")


def parse_letter(text, n):
    allowed = LETTERS[:n]
    t = re.sub(r"[*_`#]", "", text or "").strip()
    patterns = [(r"answer(?: is|:)?\s*\(?([A-J])\)?(?![a-z])", re.I),   # "answer is C", "Answer: (C)"
                (r"option\s*\(?([A-J])\)?(?![a-z])", re.I),            # "option C"
                (r"^\(?([A-J])\)?(?:[.):]|\s*$)", 0),                  # "C", "C.", "(C)", "C) ..."
                (r"^([B-HJ])\s", 0),                                   # "B because" (not A/I: English words)
                (r"\(([A-J])\)", 0)]                                   # "... (C) ..."
    for p, fl in patterns:
        m = re.search(p, t, fl)
        if m and m.group(1).upper() in allowed:
            return m.group(1).upper()
    return None


def check_b(text):
    """Deterministic checks for a generated item (also used again in analysis)."""
    res = {"json_ok": False, "four_options": False, "distinct": False, "key_ok": False,
           "no_aota": False, "stem_ok": False}
    try:
        s = (text or "").strip()
        s = s[s.find("{"): s.rfind("}") + 1]
        obj = json.loads(s)
        res["json_ok"] = True
        opts = obj.get("options", {})
        if isinstance(opts, list):
            opts = {LETTERS[i]: v for i, v in enumerate(opts)}
        vals = [str(opts.get(k, "")).strip() for k in "ABCD"]
        res["four_options"] = set(opts.keys()) == set("ABCD") and all(vals)
        res["distinct"] = len({v.lower() for v in vals}) == 4
        res["key_ok"] = str(obj.get("key", "")).strip().upper()[:1] in "ABCD" and bool(obj.get("key"))
        res["no_aota"] = not any(re.search(r"(all|none) of the above", v, re.I) for v in vals)
        res["stem_ok"] = len(str(obj.get("stem", "")).strip()) >= 15
    except Exception:
        pass
    res["passes_all"] = all(res.values())
    return res


# ----------------------------------------------------------------- core runner
def generate(model, prompt, num_predict, think_ok, fmt=None, keep_alive="30m"):
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
               "options": dict(CFG["options"], num_predict=num_predict)}
    if think_ok:
        payload["think"] = False
    if fmt:
        payload["format"] = fmt
    t0 = time.time()
    r = api_post("/api/generate", payload)
    r["_wall_s"] = time.time() - t0
    return r


def timing_fields(r):
    ns = 1e9
    pe, pd = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)
    ec, ed = r.get("eval_count", 0), r.get("eval_duration", 0)
    return {"wall_s": round(r["_wall_s"], 3), "total_s": round(r.get("total_duration", 0) / ns, 3),
            "load_s": round(r.get("load_duration", 0) / ns, 3),
            "prompt_tokens": pe, "prompt_s": round(pd / ns, 3),
            "prompt_tps": round(pe / (pd / ns), 2) if pd else None,
            "gen_tokens": ec, "gen_s": round(ed / ns, 3),
            "gen_tps": round(ec / (ed / ns), 2) if ed else None}


class Results:
    def __init__(self, path):
        self.path, self.done = path, set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok" or r.get("final"):
                        self.done.add((r["model"], r["task"], r["id"]))
                except Exception:
                    pass
        self.lock = threading.Lock()

    def add(self, row):
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_meta(model):
    try:
        s = api_post("/api/show", {"model": model}, timeout=60)
        d = s.get("details", {})
        return {"quantization": d.get("quantization_level"), "parameter_size": d.get("parameter_size"),
                "family": d.get("family"), "format": d.get("format"),
                "capabilities": s.get("capabilities", [])}
    except Exception as e:
        return {"error": str(e)}


def run_model(model, items, topics, res, tier, mode, sampler, constrained=False):
    meta = model_meta(model)
    think_ok = "thinking" in (meta.get("capabilities") or [])
    base = {"tier": tier, "mode": mode, "model": model, "host": platform.node(),
            "quantization": meta.get("quantization"), "parameter_size": meta.get("parameter_size")}
    todo_a = [it for it in items if (model, "A", it["id"]) not in res.done]
    todo_b = [] if constrained else [t for t in topics if (model, "B", t["id"]) not in res.done]
    if not todo_a and not todo_b:
        log(f"{model}: already complete, skipping")
        return
    # cold load (previous model is unloaded because MAX_LOADED_MODELS=1)
    sampler.reset()
    avail_before = read_meminfo().get("MemAvailable")
    try:
        r = generate(model, "Hello", 1, think_ok)
        res.add(dict(base, task="load", id="load", status="ok", ts=now(),
                     mem_available_before=avail_before, **timing_fields(r)))
        log(f"{model}: loaded in {r.get('load_duration', 0)/1e9:.1f}s "
            f"({meta.get('parameter_size')}, {meta.get('quantization')})")
    except Exception as e:
        res.add(dict(base, task="load", id="load", status="error", error=str(e), ts=now(),
                     mem_available_before=avail_before, final=True))
        log(f"{model}: FAILED TO LOAD -> {str(e)[:200]}")
        return
    fails = 0
    jobs = [("A", it) for it in todo_a] + [("B", t) for t in todo_b]
    t_start = time.time()
    for k, (task, x) in enumerate(jobs, 1):
        sampler.reset()
        try:
            if task == "A":
                if constrained:
                    # Grammar-constrained decoding: the model can only emit {"answer": "<valid letter>"}.
                    schema = {"type": "object", "required": ["answer"],
                              "properties": {"answer": {"type": "string",
                                                        "enum": list(LETTERS[:len(x["options"])])}}}
                    r = generate(model, prompt_a(x), 16, think_ok, fmt=schema)
                    try:
                        letter = json.loads(r.get("response") or "{}").get("answer")
                    except Exception:
                        letter = None
                    letter = letter if isinstance(letter, str) and letter in LETTERS[:len(x["options"])] else None
                else:
                    r = generate(model, prompt_a(x), CFG["task_a_num_predict"], think_ok)
                    letter = parse_letter(r.get("response"), len(x["options"]))
                extra = {"pred": letter, "gold": x["answer"], "correct": letter == x["answer"],
                         "parsed": letter is not None, "origin": x["origin"],
                         "speed_subset": x["speed_subset"]}
            else:
                r = generate(model, prompt_b(x), CFG["task_b_num_predict"], think_ok, fmt="json")
                extra = {"topic": x["topic"], "bloom": x["bloom"], "checks": check_b(r.get("response"))}
            res.add(dict(base, task=task, id=x["id"], status="ok", ts=now(),
                         response=r.get("response"), peak_runner_rss=sampler.max_rss,
                         min_mem_available=sampler.min_avail, **timing_fields(r), **extra))
            fails = 0
        except Exception as e:
            if "repeat limit" in str(e) or "prediction aborted" in str(e):
                # The model itself degenerated into repetition: a model outcome, not an infrastructure
                # error. It is deterministic at temperature 0, so record it as final and move on.
                if task == "A":
                    extra = {"pred": None, "gold": x["answer"], "correct": False, "parsed": False,
                             "origin": x["origin"], "speed_subset": x["speed_subset"]}
                else:
                    extra = {"topic": x["topic"], "bloom": x["bloom"], "checks": check_b(None)}
                res.add(dict(base, task=task, id=x["id"], status="ok", final=True, ts=now(),
                             response=None, failure="repeat_limit", error=str(e)[:300], **extra))
                log(f"{model} {task} {x['id']}: model output degenerated (repeat limit) - recorded as failed item")
                continue
            fails += 1
            res.add(dict(base, task=task, id=x["id"], status="error", error=str(e)[:500], ts=now()))
            log(f"{model} {task} {x['id']}: error {str(e)[:150]}")
            if fails >= 3:
                log(f"{model}: 3 consecutive errors, moving to next model")
                return
        if k % 10 == 0 or k == len(jobs):
            el = time.time() - t_start
            log(f"{model}: {k}/{len(jobs)} done, ~{el/k*(len(jobs)-k)/60:.0f} min left for this model")


def run_concurrency(model, topics, res, tier, server):
    server.start(num_parallel=max(CFG["concurrency_levels"]))
    meta = model_meta(model)
    think_ok = "thinking" in (meta.get("capabilities") or [])
    try:
        generate(model, "Hello", 1, think_ok)
    except Exception as e:
        log(f"Concurrency: {model} failed to load: {e}")
        return
    for n in CFG["concurrency_levels"]:
        for rnd in range(CFG["concurrency_rounds"]):
            rid = f"C{n}-r{rnd}"
            if (model, "C", rid) in res.done:
                continue
            out = [None] * n

            def worker(i):
                try:
                    r = generate(model, prompt_b(topics[(rnd * n + i) % len(topics)]),
                                 CFG["concurrency_num_predict"], think_ok)
                    out[i] = timing_fields(r)
                except Exception as e:
                    out[i] = {"error": str(e)[:300]}
            t0 = time.time()
            th = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            [t.start() for t in th]
            [t.join() for t in th]
            wall = time.time() - t0
            toks = sum(o.get("gen_tokens", 0) for o in out if o)
            res.add({"tier": tier, "mode": "concurrency", "model": model, "task": "C", "id": rid,
                     "status": "ok", "ts": now(), "concurrency": n, "round": rnd,
                     "round_wall_s": round(wall, 3), "aggregate_gen_tps": round(toks / wall, 2),
                     "requests": out, "host": platform.node()})
            log(f"Concurrency {n}x round {rnd}: {toks/wall:.1f} tok/s total, "
                f"max latency {max(o.get('wall_s', 0) for o in out):.1f}s")


def sysinfo(tier, version):
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    mi = read_meminfo()
    return {"tier": tier, "ts": now(), "host": platform.node(), "kernel": platform.release(),
            "cpu_model": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip(),
            "nproc": os.cpu_count(), "flags_avx2": "avx2" in sh("grep -m1 flags /proc/cpuinfo"),
            "mem_total": mi.get("MemTotal"), "mem_available": mi.get("MemAvailable"),
            "swap_total": mi.get("SwapTotal"), "ollama_version": version,
            "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2").strip('"'),
            "virt": sh("systemd-detect-virt 2>/dev/null"), "python": platform.python_version()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["laptop", "vps", "homelab", "pi"])
    ap.add_argument("--models", nargs="*", help="override model list")
    ap.add_argument("--skip-concurrency", action="store_true")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--constrained", action="store_true",
                    help="re-run Task A on all 200 items with letter-only constrained output")
    a = ap.parse_args()

    if not shutil.which("ollama"):
        sys.exit("ollama not found; run setup.sh first")
    items_path = os.path.join(HERE, "data", "items.jsonl")
    if not os.path.exists(items_path):
        sys.exit("data/items.jsonl missing; run prepare_items.py on the laptop first and copy the kit")
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    topics = json.load(open(os.path.join(HERE, "data", "topics.json")))

    mode = "quality" if a.tier == "laptop" else "speed"
    if a.constrained:
        mode = "constrained"
        a.skip_concurrency = True
    if mode == "speed":
        items = [i for i in items if i["speed_subset"]]
        topics = topics[:CFG["speed_task_b_prompts"]]
    models = a.models or (CFG["server_models"] + (CFG["laptop_only_models"] if a.tier == "laptop" else []))
    if a.constrained and not a.models:
        sys.exit("--constrained needs --models (split the models across machines)")

    outdir = os.path.join(HERE, "results", a.tier)
    os.makedirs(outdir, exist_ok=True)
    server = Server(outdir)
    version = server.start(num_parallel=1)
    json.dump(sysinfo(a.tier, version), open(os.path.join(outdir, "sysinfo.json"), "w"), indent=2)

    if not a.skip_pull:
        for m in models:
            try:
                pull(m)
            except Exception as e:
                log(f"Pull failed for {m}: {e}")

    res = Results(os.path.join(outdir, f"{mode}.jsonl"))
    sampler = MemSampler()
    sampler.start()
    log(f"Tier={a.tier} mode={mode}: {len(items)} items + {len(topics)} prompts x {len(models)} models")
    try:
        for m in models:
            run_model(m, items, topics, res, a.tier, mode, sampler, constrained=a.constrained)
        if not a.skip_concurrency:
            run_concurrency(CFG["concurrency_model"], json.load(open(os.path.join(HERE, "data", "topics.json"))),
                            Results(os.path.join(outdir, "concurrency.jsonl")), a.tier, server)
    finally:
        server.stop()
    log(f"ALL DONE. Results are in {outdir}")


if __name__ == "__main__":
    main()
