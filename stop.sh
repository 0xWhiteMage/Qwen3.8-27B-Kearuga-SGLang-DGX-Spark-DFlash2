#!/usr/bin/env bash
set -euo pipefail

# Stop the Qwen3.8-27B SGLang container started by any launcher.
CONTAINER="qwen3.8-27b-sglang"
PID_FILE=".sglang.pid"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "Stopping ${CONTAINER}..."
  docker stop "${CONTAINER}" >/dev/null
  echo "Stopped ${CONTAINER}."
else
  echo "${CONTAINER} is not running"
fi

if [[ -f "${PID_FILE}" ]]; then
  rm -f "${PID_FILE}"
fi

if [[ "${1:-}" == "--clean" || "${1:-}" == "--purge" ]]; then
  echo "Cleaning up local Triton cache..."
  rm -rf .cache/triton/*
  echo "Cache cleaned."
fi

echo "Logs remain available via: docker logs ${CONTAINER} or .sglang.log"
