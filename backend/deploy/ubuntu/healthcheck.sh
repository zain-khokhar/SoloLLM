#!/usr/bin/env bash
set -euo pipefail

TARGET_URL="${1:-http://127.0.0.1:8000/api/health}"

echo "[solollm-backend] Checking ${TARGET_URL}"
curl --fail --silent --show-error "${TARGET_URL}"
echo
echo "[solollm-backend] Health check passed"