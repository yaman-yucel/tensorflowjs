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

# ── Self-signed TLS certificate ──────────────────────────────────────────────
CERT_DIR="certs"
CERT_FILE="${CERT_DIR}/cert.pem"
KEY_FILE="${CERT_DIR}/key.pem"

if [[ -f "$CERT_FILE" && -f "$KEY_FILE" ]]; then
  info "TLS certificates already exist in ${CERT_DIR}/ ✓"
else
  if ! command -v openssl &>/dev/null; then
    error "openssl not found — cannot generate certificates."
    error "Install openssl and re-run setup."
    exit 1
  fi

  mkdir -p "$CERT_DIR"

  # Detect LAN IP for SAN so mobile devices accept the cert
  LAN_IP=""
  if command -v ip &>/dev/null; then
    LAN_IP=$(ip route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1); exit}')
  elif command -v ifconfig &>/dev/null; then
    LAN_IP=$(ifconfig | awk '/inet /{print $2}' | grep -v '127.0.0.1' | head -1)
  fi

  SAN="DNS:localhost,IP:127.0.0.1"
  [[ -n "$LAN_IP" ]] && SAN="${SAN},IP:${LAN_IP}"

  info "Generating self-signed certificate (SAN: ${SAN})…"
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY_FILE" \
    -out    "$CERT_FILE" \
    -days   365 \
    -subj   "/CN=localhost" \
    -addext "subjectAltName=${SAN}" \
    2>/dev/null

  info "Certificate → ${CERT_FILE}  (valid 365 days)"
  info "Key         → ${KEY_FILE}"
  warn "Trust the cert once in your browser / system keychain to silence warnings:"
  warn "  macOS : open ${CERT_FILE}  (Keychain Access → Always Trust)"
  warn "  Linux : sudo cp ${CERT_FILE} /usr/local/share/ca-certificates/doc-scanner.crt && sudo update-ca-certificates"
  warn "  Android: copy ${CERT_FILE} to phone → Settings → Security → Install certificate"
fi

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
