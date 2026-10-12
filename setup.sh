#!/usr/bin/env bash
# One-time setup on each machine. Usage:  bash setup.sh laptop | vps | homelab
set -e
cd "$(dirname "$0")"
TIER="$1"
[ -z "$TIER" ] && { echo "Usage: bash setup.sh laptop|vps|homelab"; exit 1; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== Installing packages"
$SUDO apt-get update -y -qq
$SUDO apt-get install -y -qq curl python3 python3-venv zstd tmux util-linux >/dev/null

if ! command -v ollama >/dev/null; then
  echo "== Installing Ollama (official installer from ollama.com)"
  curl -fsSL https://ollama.com/install.sh | sh
fi
# The benchmark starts its own Ollama on port 11500; stop the background service so it can't interfere.
if systemctl is-active --quiet ollama 2>/dev/null; then
  $SUDO systemctl disable --now ollama || true
fi
ollama --version || true

if [ "$TIER" = "laptop" ]; then
  echo "== Preparing the 200-item set (laptop only)"
  python3 -m venv ~/.ssaie-venv
  ~/.ssaie-venv/bin/pip install -q pyarrow
  ~/.ssaie-venv/bin/python prepare_items.py
fi
mkdir -p results
echo "== Setup done for tier: $TIER"
