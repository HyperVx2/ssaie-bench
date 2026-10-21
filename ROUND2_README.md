# ssaie-bench: Round 2 (testing newer small models)

Goal: find a model that fits a 4 GB server and beats **qwen3:4b (44%)** on the same 200 MMLU-Pro CS items, using the same `--constrained` (letter-only) setup as before. A model that cannot load in 4 GB, or scores below 44%, is still a valid result.

## What is in this package

| File | Purpose |
|---|---|
| `round2.sh` | One command per machine: sets up Ollama, downloads the models, runs the tests, bundles the results |
| `ROUND2_README.md` | This file |

The model list in `analysis/analyze.py` is also updated so the new tags appear in the tables. `config.json` does not need editing.

## Models and why each is tested

| Model | Download | Why |
|---|---|---|
| `qwen3.5:4b` | 3.3-4.0 GB | Same size and family as the qwen3:4b baseline; the fairest head-to-head. Tight on 4 GB |
| `qwen3.5:2b` | 2.7-3.1 GB | Newer generation between qwen3.5:0.8b (13-22%) and qwen3:1.7b (34%) |
| `qwen2.5:3b` | 1.9 GB | Older generation at 3B; shows whether newer actually means better. Qwen license, not Apache |
| `granite4:3b` (optional) | 2.1 GB | Different vendor (IBM, Apache 2.0); checks the result is not Qwen-specific |
| `ministral-3:3b` (optional) | 3.0 GB | Mistral family, not yet in the study (Apache 2.0); text only is used |

Sizes are from each model's Ollama page, checked 15 Oct 2026. The default list in `round2.sh` is the first three. To add the optional ones:

```bash
MODELS="qwen3.5:2b qwen2.5:3b qwen3.5:4b granite4:3b ministral-3:3b" bash round2.sh vps
```

Not included: DeepSeek-R1 distills (math-tuned reasoning models, likely to fail the letter-only format), Gemma 4 E2B (4.6-7.5 GB, breaks the 4 GB rule), SmolLM3 (no Ollama tag confirmed), and the SEA-LION models (Hugging Face only, GGUF untested).

## Before you start

- Update the kit: `git pull`.
- Free disk: about 10 GB per machine for the models (they download to `~/ssaie-models`).
- Homelab: use the LXC with 4 cores and 4 GB RAM, as in Round 1, and pause other heavy VMs during the run so timings are not skewed.
- Laptop: sleep = Never and lid close = Do nothing while plugged in; keep the WSL window open.
- Run inside `tmux` so a dropped connection does not stop it. The script can be stopped with Ctrl+C and re-run; it skips finished work.

## Steps

On each machine, inside the `ssaie-bench` folder:

```bash
tmux new -s round2 'bash round2.sh vps'        # on the VPS
tmux new -s round2 'bash round2.sh homelab'    # on the homelab LXC
tmux new -s round2 'bash round2.sh laptop'     # on the laptop (WSL)
tmux new -s round2 'bash round2.sh pi'         # on the Raspberry Pi 4 (64-bit OS only; speed run, no accuracy test)
```

What the script does:

1. Installs Ollama if missing (`setup.sh vps`; it does not re-download the dataset, since `data/items.jsonl` is already in the repo).
2. Accuracy: all 200 questions, letter-only (`--constrained`).
3. Servers: a speed run (20 questions plus 5 item-writing prompts). Laptop: the quality run (200 questions plus 20 item-writing prompts).
4. Packs the results into `results_round2_<tier>_<date>.tgz` and prints its path.

Send back the `.tgz` files.

Rough time per model (an estimate from Round 1, not a measurement): about 15-30 minutes for the accuracy run on a server, and about 30 minutes for the laptop quality run.

## What to expect in the log

- `FAILED TO LOAD`: the model did not fit in memory. This is a result, not an error; leave it.
- `token repeat limit reached` errors: the model got stuck repeating itself. Also a result.
- If the run dies or the machine restarts, run the same command again. It resumes.

## After the runs

Compare each model with qwen3:4b (44%) using the analysis script:

```bash
python3 analysis/analyze.py results analysis_out
```

New models appear in the main tables only if they are in the laptop quality results, which `round2.sh laptop` produces. Gemini judging (`python3 frontier.py judge` on the VPS) is only needed for models that beat 44%.

## Known limits

- `round2.sh` was syntax-checked and the patched analysis was checked to give identical output on the existing results. The full script has not been run end-to-end, because it needs Ollama and the real models. Report any failure.
- The new tags were checked against the Ollama library pages, not by pulling them.
- Check each model's license card before the college deploys one. The Ollama pages for Qwen3.5 do not state a license.
