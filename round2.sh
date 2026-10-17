#!/usr/bin/env bash
# Round 2 of the SEA-SSAIE benchmark: test newer small models with one command.
#
# Usage (from inside the ssaie-bench folder, ideally inside tmux):
#   bash round2.sh vps        # on the VPS
#   bash round2.sh homelab    # on the homelab container (keep it at 4 cores / 4 GB RAM)
#   bash round2.sh laptop     # on the laptop (WSL)
#   bash round2.sh pi         # on the Raspberry Pi 4 (64-bit OS): speed only, no accuracy run
#
# Override the model list if needed:
#   MODELS="qwen3.5:2b qwen2.5:3b" bash round2.sh vps
#
# It is safe to stop (Ctrl+C) and re-run: finished work is skipped.
# When it ends it prints the path of a results bundle (.tgz) to send back.
set -u
cd "$(dirname "$0")"

TIER="${1:-}"
case "$TIER" in
  laptop|vps|homelab|pi) ;;
  *) echo "Usage: bash round2.sh laptop|vps|homelab|pi"; exit 1 ;;
esac

# Edit this list to change what gets tested. Tags checked on ollama.com/library, 15 Oct 2026.
MODELS="${MODELS:-qwen3.5:2b qwen2.5:3b qwen3.5:4b}"

mkdir -p results
LOG="results/round2_${TIER}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "Round 2 on tier=$TIER, host=$(hostname)"
say "Models: $MODELS"
free -h | tee -a "$LOG" || true

# 1) Packages + Ollama. 'setup.sh vps' only installs packages/Ollama; the 200 items are already in data/.
#    (Do NOT use 'setup.sh laptop' here: it re-downloads the dataset for no reason.)
say "Step 1: setup (installs Ollama if missing)"
bash setup.sh vps 2>&1 | tee -a "$LOG"

if [ ! -f data/items.jsonl ] || [ "$(wc -l < data/items.jsonl)" -ne 200 ]; then
  say "ERROR: data/items.jsonl should have 200 lines. Run 'git pull' and try again."
  exit 1
fi

# 2) Accuracy: all 200 questions, letter-only (this is how the 44% baseline was measured).
#    bench.py downloads each model itself (into ~/ssaie-models). A model that cannot load is logged
#    as FAILED TO LOAD, which is a valid result.
if [ "$TIER" = "pi" ]; then
  say "Step 2: skipped on the Pi (speed only, as in Round 1)"
else
  say "Step 2: constrained accuracy run (downloads models on first use)"
  python3 bench.py --tier "$TIER" --constrained --models $MODELS 2>&1 | tee -a "$LOG"
fi

# 3) Speed (servers, Pi) or item-writing quality (laptop).
if [ "$TIER" = "laptop" ]; then
  say "Step 3: quality run (200 questions + 20 item-writing prompts per model)"
  python3 bench.py --tier laptop --skip-concurrency --models $MODELS 2>&1 | tee -a "$LOG"
elif [ "$TIER" = "pi" ]; then
  say "Step 3: Pi speed run (20 questions + 5 item-writing prompts + concurrency, same as Round 1)"
  [ "$(uname -m)" = "aarch64" ] || say "WARNING: the Pi needs a 64-bit OS (uname -m should print aarch64)."
  python3 bench.py --tier pi --models $MODELS 2>&1 | tee -a "$LOG"
else
  say "Step 3: speed run (20 questions + 5 item-writing prompts per model)"
  python3 bench.py --tier "$TIER" --skip-concurrency --models $MODELS 2>&1 | tee -a "$LOG"
fi

# 4) Bundle the results so they can be sent back as one file.
OUTFILE="results_round2_${TIER}_$(date +%Y%m%d).tgz"
tar czf "$OUTFILE" "results/$TIER" "$LOG" 2>/dev/null
say "DONE. Send this file back: $(pwd)/$OUTFILE"
