# EKraft v1: test kit for the SEA-SSAIE machines

EKraft is a Qwen3-4B fine-tuned (QLoRA) to **write** 4-option multiple-choice exam items for Software Engineering 1 and Data Communication and Networking 1. This folder tests it on the same machines and with the same tools as Rounds 1 and 2, and compares it with the untuned `qwen3:4b`.

You only run commands. Everything else (download, install into Ollama, tests, bundling the results) is automatic.

## What gets tested

| Machine | What runs | Compared with | Approx. time |
|---|---|---|---|
| Laptop (WSL) | **Task A**: the 200 MMLU-Pro questions, letter-only. **Item writing**: 20 SSAIE topics + 231 EKraft prompts; `qwen3:4b` also writes the 231 | `qwen3:4b` 44% on Task A; `qwen3:4b` items on the same prompts | 3-4 h |
| VPS (4 GB) | Speed run: 20 questions + 5 item-writing prompts | Round 1 `qwen3:4b` speed on the same machine | 15-30 min |
| Homelab LXC (4 GB) | Same speed run | Same | 15-30 min |
| Raspberry Pi 4 | Same speed run (optional) | Same | 30-60 min |
| VPS (Gemini key) | Gemini judges every item, blind | Round 1 Gemini judgements | 10-20 min |

Times are estimates from Round 1, not measurements. The model is a 2.5 GB download per machine (about 5 GB free disk needed during install; 2.5 GB after).

## Before you start

- In the `ssaie-bench` folder: `git pull` (or check out this PR's branch: `git fetch origin pull/<PR number>/head:ekraft-v1 && git checkout ekraft-v1`).
- Laptop: plugged in, sleep = Never, lid close = Do nothing, WSL window open.
- Homelab: the 4-core / 4 GB LXC from Round 1. Pause other heavy VMs during the run.
- Every command below can be stopped with Ctrl+C and run again; it continues where it stopped.

## Step 1. Laptop (WSL)

```bash
cd "/path/to/your/project/ssaie-bench"
tmux new -s ekraft 'bash ekraft/run_ekraft.sh laptop'
```

Detach with `Ctrl+B` then `D`; come back with `tmux attach -t ekraft`.

Near the start it prints one sample item written by EKraft. It should look like:

```
Plan: <one line with the key fact>
<question>
A. ...
B. ...
C. ...
D. ...
Answer: <letter>
```

## Step 2. Servers (can run at the same time as Step 1)

Copy the updated kit to each server the same way as in `RUNNING.md` Step 3 (or `git pull` there), then:

```bash
# VPS
cd ~/ssaie-bench && tmux new -s ekraft 'bash ekraft/run_ekraft.sh vps'
# Homelab LXC
cd /root/ssaie-bench && tmux new -s ekraft 'bash ekraft/run_ekraft.sh homelab'
# Raspberry Pi 4 (optional)
cd ~/ssaie-bench && tmux new -s ekraft 'bash ekraft/run_ekraft.sh pi'
```

## Step 3. Gemini judging (VPS, after the laptop finishes)

```bash
# from WSL: send the laptop's item file to the VPS
scp "/path/to/your/project/ssaie-bench/results/laptop/ekraft_items.jsonl" <user>@<VPS>:~/ssaie-bench/results/laptop/
# on the VPS:
cd ~/ssaie-bench && python3 ekraft/judge_ekraft.py
```

If it says the free daily quota is used up, run the same command later; it resumes. It uses the same Gemini model as the Round 1 judging (read from `results/frontier/model.json`).

## Step 4. Send the results back

Each machine prints the path of a `results_ekraft_<machine>_<date>.tgz` file at the end. Send those, plus `results/frontier/ekraft_judgements.jsonl` from the VPS, to Jay.

To see the summary yourself (any machine with all results copied in):

```bash
python3 ekraft/report_ekraft.py
```

It writes `analysis_out/ekraft_report.md` and works with partial results.

## If the download link stops working

The model may be switched to private while a terms-of-use question is checked (see "About the model").
In that case Jay sends the file directly. Save it in this folder as `ekraft/ekraft-v1-4b-plan.Q4_K_M.gguf`
and run the same command; the script uses the local copy and still checks its sha256.

## What to expect in the logs

| Message | Meaning |
|---|---|
| `already installed, skipping download` | The model is in Ollama from an earlier run. Fine. |
| `sha256 mismatch` | The download was damaged. The script deleted it; run the same command again. |
| `FAILED TO LOAD` | The model did not fit in memory. That is a result, not an error; leave it. |
| `repeat limit` | The model got stuck repeating itself on one item. Recorded as a failed item. |
| `Pull failed for ...` | Should not appear for EKraft (it is installed locally). For `qwen3:4b` it means no internet. |

## Files in this folder

| File | Purpose |
|---|---|
| `run_ekraft.sh` | One command per machine: Ollama setup, model download and install, tests, results bundle |
| `model.json` | Download link, checksum and Ollama name of the model |
| `Modelfile.template` | Ollama settings for EKraft (chat format it was trained with, thinking turned off) |
| `prompts_ekraft.jsonl` | 231 held-out EKraft prompts (subject, topic, Bloom level only; no answers) |
| `write_items.py` | Item writing on the laptop; output `results/laptop/ekraft_items.jsonl` |
| `judge_ekraft.py` | Blind Gemini judging with the Round 1 rubric; output `results/frontier/ekraft_judgements.jsonl` |
| `report_ekraft.py` | Summary tables; output `analysis_out/ekraft_report.md` |

Nothing outside this folder is changed. `bench.py`, `frontier.py` and `config.json` are reused as they are.

## How the comparison is kept fair

- Task A uses `bench.py --constrained`, exactly as for the 44% `qwen3:4b` baseline, on the same laptop.
- Speed uses the same `bench.py` speed run as Round 1.
- On the 231 EKraft prompts, EKraft and `qwen3:4b` get the same topics, Bloom levels and sampling (temperature 0.7, top_p 0.9, seed 42). Each gets the prompt format it works best with: EKraft its training format, `qwen3:4b` the Round 1 JSON prompt.
- The judge never sees which model wrote an item, options are reshuffled, and both models' items are mixed in every batch.

## About the model

- Base: Qwen3-4B (Apache 2.0). Fine-tuned with QLoRA (rank 16, 2 epochs) on 1,685 **synthetic** items written by Claude for the two subjects, in a "plan-first" format (a one-line plan with the key fact, then the item). Exported to GGUF Q4_K_M (2.5 GB).
- Development result (50 prompts, one AI rater, not independent): 64% usable. This kit provides the independent check.
- This is a baseline. A second version trained on real, human-written items is planned.
- **Disclosure:** the training items were generated by Claude (Anthropic) and not reviewed by a teacher, and the
  development score came from the same model family. Whether Anthropic's terms allow training a model on
  Claude outputs for research is still being checked; until then, please do not redistribute the model or
  the training data, and treat any results as provisional.
