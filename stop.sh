#!/usr/bin/env bash
set -euo pipefail

# Stop the Qwen3.8-27B SGLang container started by either launcher.
CONTAINER="qwen3.8-27b-sglang"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "Stopping ${CONTAINER}..."
  docker stop "${CONTAINER}" >/dev/null
  echo "Stopped ${CONTAINER}."
else
  echo "${CONTAINER} is not running"
fi

echo "Logs remain available via: docker logs ${CONTAINER}"
