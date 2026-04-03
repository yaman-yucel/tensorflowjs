#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[start]${NC} $*"; }
warn() { echo -e "${YELLOW}[start]${NC} $*"; }

# ── Preflight checks ──────────────────────────────────────────────────────────
if [[ ! -d "node_modules" ]]; then
  warn "node_modules missing — running setup first…"
  bash setup.sh
fi

MODEL_DIR="public/model"
if ! compgen -G "${MODEL_DIR}/*.json" > /dev/null 2>&1; then
  warn "No model.json found in ${MODEL_DIR}/."
  warn "The app will show an error screen until you add the model files."
fi

# ── Detect LAN IP for mobile testing ─────────────────────────────────────────
LAN_IP=""
if command -v ip &>/dev/null; then
  LAN_IP=$(ip route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1); exit}')
elif command -v ifconfig &>/dev/null; then
  LAN_IP=$(ifconfig | awk '/inet /{print $2}' | grep -v '127.0.0.1' | head -1)
fi

echo ""
info "Starting dev server…"
if [[ -n "$LAN_IP" ]]; then
  echo -e "  ${CYAN}Local   ${NC}→  http://localhost:5173"
  echo -e "  ${CYAN}Mobile  ${NC}→  http://${LAN_IP}:5173"
  echo ""
  warn "Camera requires HTTPS on non-localhost origins."
  warn "For mobile testing use one of:"
  warn "  1) Same LAN + Chrome flags: chrome://flags/#unsafely-treat-insecure-origin-as-secure"
  warn "  2) ngrok: ngrok http 5173  (gives a public HTTPS URL)"
  warn "  3) Deploy with nginx.conf for full HTTPS."
fi
echo ""

npx vite --host
