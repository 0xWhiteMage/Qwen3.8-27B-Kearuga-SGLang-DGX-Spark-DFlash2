#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B with EAGLE speculative decoding on SGLang
# High-concurrency profile for 8–32 parallel agent seats.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load optional .env overrides
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    value="${value#\"}" ; value="${value%\"}"
    value="${value#\'}" ; value="${value%\'}"
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

EAGLE_IMAGE="${EAGLE_IMAGE:-lmsysorg/sglang:v0.4.3.post2-cu124-arm64}"
TARGET_MODEL="${TARGET_MODEL:-0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4}"
TARGET_REV="${TARGET_REV:-8ea86bdcdd34c84b3d25b69fb7fcc8fc48d0cdd0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
MEM_FRACTION="${MEM_FRACTION:-0.78}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-32}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1048576}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-32}"
CONTINUOUS_DECODE_STEPS="${CONTINUOUS_DECODE_STEPS:-1}"
CPUSET="${CPUSET:-${CPU_AFFINITY:-}}"
HEALTH_TIMEOUT_SECS="${HEALTH_TIMEOUT_SECS:-900}"

SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
SPEC_ATTENTION_MODE="${SPEC_ATTENTION_MODE:-flashinfer}"

PRIORITY_SCHEDULING="${PRIORITY_SCHEDULING:-1}"
DEFAULT_PRIORITY_VALUE="${DEFAULT_PRIORITY_VALUE:-0}"
PRIORITY_PREEMPTION_THRESHOLD="${PRIORITY_PREEMPTION_THRESHOLD:-10}"

if (( CONTEXT_LENGTH != 262144 )); then
  echo "CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported by this qualified profile (use 262144)"
  exit 1
fi

MAMBA_SLOTS_PER_REQ=5
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))

if (( MAX_TOTAL_TOKENS < CONTEXT_LENGTH )); then
  echo "MAX_TOTAL_TOKENS (${MAX_TOTAL_TOKENS}) must be >= CONTEXT_LENGTH (${CONTEXT_LENGTH})"
  exit 1
fi

case "${PRIORITY_SCHEDULING}" in
  0|1) : ;;
  *) echo "PRIORITY_SCHEDULING must be 0 or 1, got '${PRIORITY_SCHEDULING}'"; exit 1 ;;
esac

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
    echo "Log: ${LOG_FILE}"
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

# Auto bind-mount local host model paths if specified as absolute paths
MODEL_MOUNT_ARGS=()
if [[ "${TARGET_MODEL}" == /* ]]; then
  MODEL_MOUNT_ARGS+=(-v "${TARGET_MODEL}:${TARGET_MODEL}")
fi

# Conditional revisions
REVISION_ARGS=()
[[ -n "${TARGET_REV:-}" ]] && REVISION_ARGS=(--revision "${TARGET_REV}")

echo "Starting EAGLE ${SPEC_STEPS}/${SPEC_TOPK}/${SPEC_DRAFT} high-concurrency profile"
echo "Target: ${TARGET_MODEL} @ ${TARGET_REV:0:8}"
echo "Seats: ${MAX_CONCURRENT_REQUESTS}; context: ${CONTEXT_LENGTH}; shared tokens: ${MAX_TOTAL_TOKENS}"
echo "Mamba pool: ${MAMBA_CACHE_SIZE}; graph max batch: ${CUDA_GRAPH_MAX_BS}"
echo "Priority: ${PRIORITY_SCHEDULING} (default=${DEFAULT_PRIORITY_VALUE}, threshold=${PRIORITY_PREEMPTION_THRESHOLD})"
echo "Image: ${EAGLE_IMAGE}"
echo "First boot captures EAGLE graphs through C${CUDA_GRAPH_MAX_BS} and can take several minutes."

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (EAGLE high-concurrency)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  --gpus all \
  --shm-size 32g \
  "${PIN_ARGS[@]}" \
  "${MODEL_MOUNT_ARGS[@]}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PYTHONUNBUFFERED=1 \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  "${EAGLE_IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  "${REVISION_ARGS[@]}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend flashinfer \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
  --max-prefill-tokens "${CHUNKED_PREFILL}" \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${SPEC_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT}" \
  --speculative-attention-mode "${SPEC_ATTENTION_MODE}" \
  --enable-linear-replayssm-spec \
  --enable-torch-compile \
  --torch-compile-max-bs "${TORCH_COMPILE_MAX_BS}" \
  --num-continuous-decode-steps "${CONTINUOUS_DECODE_STEPS}" \
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --enable-cache-report \
  --sleep-on-idle \
  "${PRIORITY_ARGS[@]}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" | grep --line-buffered -v "Enabled fused SiLU+mul+FP4-quant for dense MLP down_proj input" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
start_time="$(date +%s)"
max_wait="${HEALTH_TIMEOUT_SECS}"

while true; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo
    echo "ERROR: container exited unexpectedly during startup."
    echo "Review ${LOG_FILE} for details."
    exit 1
  fi

  if curl -s -f -o /dev/null "${READY_URL}"; then
    if curl -s -f -o /dev/null "${HEALTH_URL}"; then
      echo
      echo "Qwen3.8-27B with EAGLE is ready at http://${HOST}:${PORT}"
      exit 0
    fi
  fi

  now="$(date +%s)"
  elapsed=$(( now - start_time ))
  if (( elapsed > max_wait )); then
    echo
    echo "ERROR: server failed to become ready within ${max_wait}s."
    echo "Review ${LOG_FILE} for details."
    exit 1
  fi

  sleep 2
done
