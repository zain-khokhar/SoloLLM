#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "[solollm-backend] Repo root: ${REPO_ROOT}"

if command -v apt-get >/dev/null 2>&1; then
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    nginx \
    tesseract-ocr
fi

if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  "${PYTHON_BIN}" -m venv "${REPO_ROOT}/.venv"
fi

"${REPO_ROOT}/.venv/bin/pip" install --upgrade pip wheel
"${REPO_ROOT}/.venv/bin/pip" install -r "${REPO_ROOT}/requirements.txt"

if [[ -f "${REPO_ROOT}/requirements-optional.txt" ]]; then
  echo "[solollm-backend] Optional dependencies available in requirements-optional.txt"
fi

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "[solollm-backend] Created ${REPO_ROOT}/.env from template"
fi

mkdir -p "${REPO_ROOT}/data"

cat <<'EOF'

[solollm-backend] Bootstrap complete.

Next steps:
1. Edit .env with your production values.
2. Install and start Ollama separately on the server.
3. Start the backend with:
   .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --proxy-headers
4. Put Nginx in front of the backend using deploy/nginx/api.your-domain.com.conf.

EOF