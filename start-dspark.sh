#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B with native upstream DSpark on SGLang (DGX Spark / GB10, aarch64)
# Zero-overlay alternative: runs directly on the base day-0 SGLang image.
#
# Prereqs:
#   1. Ensure .env is present (use .env.sample as template).
#   2. Run directly: ./start-dspark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CPUSET="${CPUSET:-5-9,15-19}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1048576}"

PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"

TARGET_MODEL="${TARGET_MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
TARGET_REV="${TARGET_REV:-554ebba9b5f1b79dc11246341960360e6ef05ef4}"
DSPARK_MODEL="${DSPARK_MODEL:-RadixArk/Qwen3.8-27B-DSpark}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
DSPARK_IMAGE="${DSPARK_IMAGE:-lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6}"

PRIORITY_SCHEDULING="${PRIORITY_SCHEDULING:-1}"
DEFAULT_PRIORITY_VALUE="${DEFAULT_PRIORITY_VALUE:-0}"
PRIORITY_PREEMPTION_THRESHOLD="${PRIORITY_PREEMPTION_THRESHOLD:-10}"

MAMBA_SLOTS_PER_REQ=5
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))
CONTAINER_NAME="qwen3.8-27b-sglang"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"
export HF_TOKEN="${HF_TOKEN:-}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

PRIORITY_ARGS=()
if [[ "${PRIORITY_SCHEDULING}" == "1" ]]; then
  PRIORITY_ARGS=(
    --enable-priority-scheduling
    --default-priority-value "${DEFAULT_PRIORITY_VALUE}"
    --priority-scheduling-preemption-threshold "${PRIORITY_PREEMPTION_THRESHOLD}"
  )
fi
PIN_ARGS=()
[[ -n "${CPUSET}" ]] && PIN_ARGS=(--cpuset-cpus "${CPUSET}")

echo "Starting ${TARGET_MODEL} @ ${TARGET_REV:0:8} with native DSpark ${DSPARK_MODEL} (block size ${DSPARK_BLOCK_SIZE})"
echo "Listening on ${HOST}:${PORT}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (DSpark native)
EOF

docker run -d   --name "${CONTAINER_NAME}"   --network host   --ipc host   --privileged   --cap-add IPC_LOCK   --ulimit memlock=-1:-1   --ulimit stack=67108864   --gpus all   --shm-size 32g   "${PIN_ARGS[@]}"   -e HF_HOME=/root/.cache/huggingface   -e TRITON_CACHE_DIR=/root/.triton   -e HF_TOKEN="${HF_TOKEN:-}"   -e FLASHINFER_CUDA_ARCH_LIST="12.1f"   -e CUTE_DSL_ARCH="sm_120a"   -e PYTHONUNBUFFERED=1   -v "${HF_HOME}:/root/.cache/huggingface"   -v "${TRITON_CACHE_DIR}:/root/.triton"   "${DSPARK_IMAGE}"   python3 -m sglang.launch_server   --model-path "${TARGET_MODEL}"   --revision "${TARGET_REV}"   --served-model-name "${SERVED_MODEL_NAME}"   --trust-remote-code   --mem-fraction-static "${MEM_FRACTION}"   --attention-backend flashinfer   --chunked-prefill-size "${CHUNKED_PREFILL}"   --max-prefill-tokens "${CHUNKED_PREFILL}"   --disable-prefill-cuda-graph   --kv-cache-dtype fp8_e4m3   --mamba-ssm-dtype bfloat16   --mamba-full-memory-ratio 4.21   --mamba-radix-cache-strategy extra_buffer   --max-mamba-cache-size "${MAMBA_CACHE_SIZE}"   --max-running-requests "${MAX_CONCURRENT_REQUESTS}"   --max-total-tokens "${MAX_TOTAL_TOKENS}"   --context-length "${CONTEXT_LENGTH}"   --speculative-algorithm DSPARK   --speculative-draft-model-path "${DSPARK_MODEL}"   --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"   --speculative-draft-attention-backend flashinfer   --reasoning-parser qwen3   --tool-call-parser qwen3_coder   --sampling-defaults model   --enable-metrics   --enable-cache-report   --cuda-graph-max-bs-decode 4   --sleep-on-idle   "${PRIORITY_ARGS[@]}"   --host "${HOST}"   --port "${PORT}"   >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
max_probes=120
probe=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1 || curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  probe=$((probe + 1))
  if (( probe > max_probes )); then
    echo "Timeout waiting for SGLang readiness after ${max_probes} probes."
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  sleep 5
done

echo "SGLang is ready (native DSpark active)"
