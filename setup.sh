#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
error() { echo -e "${RED}[setup]${NC} $*" >&2; }

# ── Node version check ────────────────────────────────────────────────────────
REQUIRED_NODE=18
NODE_VER=$(node -v 2>/dev/null | sed 's/v//' | cut -d. -f1)
if [[ -z "$NODE_VER" ]]; then
  error "Node.js not found. Install Node >= ${REQUIRED_NODE}: https://nodejs.org"
  exit 1
fi
if (( NODE_VER < REQUIRED_NODE )); then
  error "Node ${NODE_VER} detected — need >= ${REQUIRED_NODE}."
  exit 1
fi
info "Node $(node -v)  npm $(npm -v)"

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing dependencies…"
npm install

# ── Model placeholder ─────────────────────────────────────────────────────────
MODEL_DIR="public/model"
if [[ ! -d "$MODEL_DIR" ]]; then
  mkdir -p "$MODEL_DIR"
fi

if compgen -G "${MODEL_DIR}/*.json" > /dev/null 2>&1; then
  info "Model files found in ${MODEL_DIR}/ ✓"
else
  warn "No model files found in ${MODEL_DIR}/."
  warn "Drop your model.json + *.bin weight shards there before running the app."
fi

echo ""
info "Setup complete. Run ./start.sh to start the dev server."
