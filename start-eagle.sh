#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B EAGLE high-concurrency profile for one DGX Spark / GB10.
# Measured priority: C8, C16 and C32. One speculative head per process.

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

CONTEXT_LENGTH="${EAGLE_CONTEXT_LENGTH:-262144}"
MAX_CONCURRENT_REQUESTS="${EAGLE_MAX_CONCURRENT_REQUESTS:-32}"
MAX_TOTAL_TOKENS="${EAGLE_MAX_TOTAL_TOKENS:-1048576}"
MEM_FRACTION="${EAGLE_MEM_FRACTION:-0.90}"
CPUSET="${EAGLE_CPUSET:-5-9,15-19}"
CHUNKED_PREFILL="${EAGLE_CHUNKED_PREFILL:-8192}"

SPEC_STEPS="${EAGLE_SPEC_STEPS:-3}"
SPEC_TOPK="${EAGLE_SPEC_TOPK:-1}"
SPEC_DRAFT="${EAGLE_SPEC_DRAFT:-4}"
CONTINUOUS_DECODE_STEPS="${EAGLE_CONTINUOUS_DECODE_STEPS:-1}"
TORCH_COMPILE_MAX_BS="${EAGLE_TORCH_COMPILE_MAX_BS:-4}"
CUDA_GRAPH_MAX_BS="${EAGLE_CUDA_GRAPH_MAX_BS:-32}"
SPEC_ATTENTION_MODE="${EAGLE_SPEC_ATTENTION_MODE:-decode}"

PRIORITY_SCHEDULING="${PRIORITY_SCHEDULING:-1}"
DEFAULT_PRIORITY_VALUE="${DEFAULT_PRIORITY_VALUE:-0}"
PRIORITY_PREEMPTION_THRESHOLD="${PRIORITY_PREEMPTION_THRESHOLD:-10}"

EAGLE_IMAGE="${EAGLE_IMAGE:-lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6}"
TARGET_MODEL="${TARGET_MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
TARGET_REV="${TARGET_REV:-554ebba9b5f1b79dc11246341960360e6ef05ef4}"

if (( CONTEXT_LENGTH != 262144 )); then
  echo "EAGLE_CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported by this qualified profile (use 262144)"
  exit 1
fi
if (( MAX_CONCURRENT_REQUESTS < 8 )); then
  echo "EAGLE_MAX_CONCURRENT_REQUESTS must be >= 8 for the high-concurrency profile"
  exit 1
fi
if (( MAX_TOTAL_TOKENS < CONTEXT_LENGTH )); then
  echo "EAGLE_MAX_TOTAL_TOKENS (${MAX_TOTAL_TOKENS}) must be >= EAGLE_CONTEXT_LENGTH (${CONTEXT_LENGTH})"
  exit 1
fi
if [[ "${SPEC_TOPK}" != "1" ]] || (( SPEC_DRAFT != SPEC_STEPS + 1 )); then
  echo "Qualified EAGLE geometry requires topk=1 and draft=steps+1"
  exit 1
fi
case "${PRIORITY_SCHEDULING}" in
  0|1) : ;;
  *) echo "PRIORITY_SCHEDULING must be 0 or 1"; exit 1 ;;
esac

MAMBA_SLOTS_PER_REQ=4
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS_PER_REQ ))
SERVED_MODEL_NAME="qwen3.8-27b-sglang"
CONTAINER_NAME="qwen3.8-27b-sglang"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

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
  --gpus all \
  --shm-size 32g \
  "${PIN_ARGS[@]}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  "${EAGLE_IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  --revision "${TARGET_REV}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend flashinfer \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
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
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then echo "  still starting..."; fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready (EAGLE ${SPEC_STEPS}/${SPEC_TOPK}/${SPEC_DRAFT}, C8-C${MAX_CONCURRENT_REQUESTS})"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Anthropic-compatible: http://${HOST}:${PORT}/v1/messages"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"
