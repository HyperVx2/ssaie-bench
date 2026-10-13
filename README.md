# ssaie-bench

Benchmark code, data and results for the SEA-SSAIE 2027 short paper
**"Can a Provincial College Run Its Own Exam AI? Benchmarking Small Language Models on Already-Owned, CPU-Only Hardware"**
(Chad Aga D. Abinguna, MSIT, Graduate School, Samar State University).

Nine open models (0.8B–8B, via [Ollama](https://ollama.com)) were run on CPU-only machines the author already owned or rented (a 2015 i5-6500 server, a cloud VPS, a laptop, and a Raspberry Pi 4 for speed only),
against a free-tier Gemini Flash baseline, on:

- **Task A – answering:** 200 computer-science items from the MMLU-Pro test split (seed 2027);
- **Task B – writing:** one 4-option MCQ for each of 20 CS topic × Bloom-level prompts;
- **Task C – deployment:** latency, throughput, free memory, and 1/2/4-user concurrency.

## Headline results

| | Accuracy (constrained) | Usable items /20 |
|---|---|---|
| Gemini 3 Flash (cloud, free tier) | 89.0% | 20 (self-judged) |
| Best model in a 4 GB server (qwen3:4b) | 44.0% | 15 |
| Best laptop model (qwen3:8b) | 52.0% | 18 |
| 1B-class models | 13–22% | 0–5 |

The 2015 on-premise server (i5-6500) answered items 1.6× faster than the rented VPS; a Raspberry Pi 4 ran 1–4B models 12–16× slower than that server. A blind human rating of 40 generated
items agreed with the model judge on 90% of key-correctness decisions (Cohen's κ = 0.78) and 95% of usable/unusable decisions (κ = 0.90).
Full numbers: [`analysis/report.json`](analysis/report.json).

## Layout

| Path | Contents |
|---|---|
| `bench.py`, `frontier.py`, `prepare_items.py`, `setup.sh`, `config.json` | Benchmark harness (Python standard library only) |
| `RUNNING.md` | Step-by-step instructions for reproducing the runs |
| `data/` | The 200 sampled MMLU-Pro items and the 20 item-writing prompts |
| `results/<tier>/` | Raw per-request results (JSONL) and system info for each machine; `results/frontier/` holds Gemini answers, items and judgements |
| `human_rating/ratings.jsonl` | The 40 blind human ratings next to the model judge's scores |
| `analysis/` | Analysis and figure scripts, `report.json`, figures |

## Reproduce the analysis

```bash
python3 analysis/analyze.py results analysis_out
```

## Data and licences

Code: MIT (see `LICENSE`). `data/items.jsonl` is a subset of MMLU-Pro
(TIGER-Lab, [MIT licence](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro); Wang et al., arXiv:2406.01574).
Model outputs are provided for research reproducibility. No personal or student data was used.

## Use of generative AI

Anthropic's Claude helped write the harness, analysis scripts and paper draft; the author designed the study, ran all experiments,
rated the items and verified the results.
