#!/usr/bin/env python3
"""Build the fixed item set from MMLU-Pro (computer science, test split).

Writes data/items.jsonl (200 items, seed 2027). The first 20 items form the
speed subset used on every machine. Run once, on the laptop; copy the kit
(with data/items.jsonl) to the servers afterwards so every machine uses the
identical file.
"""
import json, os, random, sys, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "items.jsonl")
PARQUET_URL = ("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
               "data/test-00000-of-00001.parquet")
N_ITEMS, SEED = 200, 2027
LETTERS = "ABCDEFGHIJ"


def load_parquet():
    import pyarrow.parquet as pq  # installed by setup.sh into .venv
    path = os.path.join(HERE, "data", "mmlu_pro_test.parquet")
    if not os.path.exists(path):
        print("Downloading MMLU-Pro test split (~4 MB)...")
        urllib.request.urlretrieve(PARQUET_URL, path)
    rows = pq.read_table(path).to_pylist()
    return [r for r in rows if r["category"] == "computer science"]


def load_datasets_server():
    print("pyarrow unavailable; using the Hugging Face datasets-server API...")
    rows, offset = [], 0
    where = urllib.parse.quote("\"category\"='computer science'")
    while True:
        url = ("https://datasets-server.huggingface.co/filter?dataset=TIGER-Lab%2FMMLU-Pro"
               f"&config=default&split=test&where={where}&offset={offset}&length=100")
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        batch = [x["row"] for x in data["rows"]]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= data.get("num_rows_total", 0):
            break
    return rows


def main():
    try:
        rows = load_parquet()
    except ImportError:
        rows = load_datasets_server()
    rows = sorted(rows, key=lambda r: int(r["question_id"]))
    print(f"Computer science items found: {len(rows)} (expected 410)")
    if len(rows) < N_ITEMS:
        sys.exit("Too few items; aborting.")
    sample = random.Random(SEED).sample(rows, N_ITEMS)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            opts = list(r["options"])
            f.write(json.dumps({
                "id": f"A{i+1:03d}",
                "question_id": int(r["question_id"]),
                "question": r["question"],
                "options": opts,
                "answer": r["answer"],
                "answer_index": int(r["answer_index"]),
                "src": r["src"],
                "origin": "original_mmlu" if str(r["src"]).startswith("ori_mmlu") else "new_in_mmlu_pro",
                "speed_subset": i < 20,
            }, ensure_ascii=False) + "\n")
    n_orig = sum(1 for r in sample if str(r["src"]).startswith("ori_mmlu"))
    print(f"Wrote {OUT}: {N_ITEMS} items ({n_orig} original MMLU, {N_ITEMS-n_orig} new). Seed {SEED}.")


if __name__ == "__main__":
    main()
