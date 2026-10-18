#!/usr/bin/env bash
# EKraft test run: one command per machine.
#
# Usage (from inside the ssaie-bench folder, ideally inside tmux):
#   bash ekraft/run_ekraft.sh laptop    # Task A (200 questions) + item writing (251 prompts)
#   bash ekraft/run_ekraft.sh vps       # speed run on the 4 GB server
#   bash ekraft/run_ekraft.sh homelab   # speed run on the 4 GB LXC
#   bash ekraft/run_ekraft.sh pi        # speed run on the Raspberry Pi 4 (64-bit OS)
#
# Safe to stop (Ctrl+C) and re-run: finished work is skipped.
# When it ends it prints the path of a results bundle (.tgz) to send back.
set -u
cd "$(dirname "$0")/.."

TIER="${1:-}"
case "$TIER" in
  laptop|vps|homelab|pi) ;;
  *) echo "Usage: bash ekraft/run_ekraft.sh laptop|vps|homelab|pi"; exit 1 ;;
esac

mkdir -p results
LOG="results/ekraft_${TIER}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
field() { python3 -c "import json,sys; print(json.load(open('ekraft/model.json'))[sys.argv[1]])" "$1"; }

TAG="$(field ollama_name)"
FILE="$(field file)"
URL="$(field url)"
SHA="$(field sha256)"
MDIR="$HOME/ssaie-models/ekraft"
PORT="$(python3 -c "import json; print(json.load(open('config.json'))['ollama_port'])")"
export OLLAMA_HOST="127.0.0.1:${PORT}"

say "EKraft run on tier=$TIER, host=$(hostname), model=$TAG"
free -h | tee -a "$LOG" || true

# 1) Packages + Ollama (same as Round 2; does not touch data/items.jsonl).
say "Step 1: setup (installs Ollama if missing)"
bash setup.sh vps 2>&1 | tee -a "$LOG"

# 2) Get the model into the benchmark's own model store (~/ssaie-models), the same store bench.py uses.
say "Step 2: install the EKraft model into Ollama"
OLLAMA_MODELS="$HOME/ssaie-models" ollama serve >> "results/ekraft_${TIER}_ollama.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 1; done

if ollama show "$TAG" >/dev/null 2>&1; then
  say "  $TAG is already installed, skipping download"
else
  mkdir -p "$MDIR"
  if [ -f "ekraft/$FILE" ]; then
    say "  using the copy in ekraft/$FILE"
    cp "ekraft/$FILE" "$MDIR/$FILE"
  elif [ ! -f "$MDIR/$FILE" ]; then
    case "$URL" in http*) ;; *) say "ERROR: ekraft/model.json has no download link yet. Ask Jay."; exit 1 ;; esac
    say "  downloading $FILE (2.5 GB; resumes if interrupted)"
    AUTH=()
    [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
    curl -L --fail --retry 5 -C - "${AUTH[@]}" -o "$MDIR/$FILE.part" "$URL" || {
      say "ERROR: download failed. Check the internet connection and run the same command again (it resumes)."
      exit 1; }
    mv "$MDIR/$FILE.part" "$MDIR/$FILE"
  fi
  say "  checking the file (sha256)"
  GOT="$(sha256sum "$MDIR/$FILE" | cut -d' ' -f1)"
  if [ "$GOT" != "$SHA" ]; then
    say "ERROR: the downloaded file is damaged (sha256 mismatch). It was deleted; run the same command again."
    rm -f "$MDIR/$FILE"; exit 1
  fi
  sed "s|@@GGUF@@|$MDIR/$FILE|" ekraft/Modelfile.template > "$MDIR/Modelfile"
  ollama create "$TAG" -f "$MDIR/Modelfile" 2>&1 | tail -3 | tee -a "$LOG"
  ollama show "$TAG" >/dev/null 2>&1 || { say "ERROR: 'ollama create' failed; see the lines above."; exit 1; }
  rm -f "$MDIR/$FILE"        # Ollama keeps its own copy; free 2.5 GB of disk
fi

say "  sanity check (one item, may take a minute on a CPU):"
curl -s "http://$OLLAMA_HOST/api/generate" -d "{\"model\": \"$TAG\", \"stream\": false, \"prompt\": \"Write one 4-option multiple-choice exam item.\nSubject: Data Communication and Networking 1\nTopic: Topologies\nBloom level: Remember\", \"options\": {\"temperature\": 0, \"num_predict\": 200}}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('response','(no response)'))" | tee -a "$LOG"

kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; trap - EXIT
sleep 2                       # bench.py starts its own Ollama on the same port

# 3) The tests.
if [ "$TIER" = "laptop" ]; then
  say "Step 3a: Task A, all 200 questions, letter-only (same setup as the qwen3:4b 44% baseline)"
  python3 bench.py --tier laptop --constrained --skip-pull --models "$TAG" 2>&1 | tee -a "$LOG"
  say "Step 3b: item writing (20 SSAIE topics + 231 EKraft prompts; qwen3:4b writes the same 231 for comparison)"
  python3 ekraft/write_items.py --model "$TAG" 2>&1 | tee -a "$LOG"
else
  [ "$TIER" = "pi" ] && [ "$(uname -m)" != "aarch64" ] && say "WARNING: the Pi needs a 64-bit OS (uname -m should print aarch64)."
  say "Step 3: speed run (20 questions + 5 item-writing prompts, same as Round 1)"
  python3 bench.py --tier "$TIER" --skip-concurrency --skip-pull --models "$TAG" 2>&1 | tee -a "$LOG"
fi

# 4) Bundle the results so they can be sent back as one file.
OUTFILE="results_ekraft_${TIER}_$(date +%Y%m%d).tgz"
tar czf "$OUTFILE" "results/$TIER" "$LOG" 2>/dev/null
say "DONE. Send this file back: $(pwd)/$OUTFILE"
