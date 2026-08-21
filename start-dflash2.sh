#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B with DFlash 2 on SGLang (DGX Spark / GB10, aarch64)
# Default launcher for this recipe.
#
# Prereqs:
#   1. Build the DFlash 2 overlay image once:
#      bash patch/build-dflash2-image.sh
#   2. Ensure .env is present (use .env.sample as template).
#
# Notes:
#   - Uses DFlash 2 speculative decoding (z-lab/Qwen3.8-27B-DFlash2 @ 50307d4).
#   - Requires the DFlash 2 overlay image: lmsysorg/sglang:qwen38-27b-dflash2
#   - Do NOT enable torch.compile or continuous-decode-steps > 1 on DFlash 2
#     (measured to crawl to 3–6 tok/s on this build).
#   - One speculative head per process; switching heads requires a restart.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load optional .env overrides
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

# Defaults
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CPUSET="${CPUSET:-${CPU_AFFINITY:-5-9,15-19}}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-4}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1048576}"

# Server network & model naming
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"

# DFlash 2 specific
DFLASH_IMAGE="${DFLASH_IMAGE:-lmsysorg/sglang:qwen38-27b-dflash2}"
TARGET_MODEL="${TARGET_MODEL:-0xWhiteMage/Qwen3.8-27B-Kearuga-NVFP4}"
TARGET_REV="${TARGET_REV:-554ebba9b5f1b79dc11246341960360e6ef05ef4}"
DFLASH_MODEL="${DFLASH_MODEL:-0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3}"
DFLASH_REV="${DFLASH_REV:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"
DFLASH_DRAFT_TOKENS="${DFLASH_DRAFT_TOKENS:-8}"
DFLASH_DRAFT_WINDOW_SIZE="${DFLASH_DRAFT_WINDOW_SIZE:-2048}"
PRIORITY_SCHEDULING="${PRIORITY_SCHEDULING:-1}"
DEFAULT_PRIORITY_VALUE="${DEFAULT_PRIORITY_VALUE:-0}"
PRIORITY_PREEMPTION_THRESHOLD="${PRIORITY_PREEMPTION_THRESHOLD:-10}"

# This qualified DFlash profile is native-context only. MAX_TOTAL_TOKENS is the
# shared KV pool; it does not change the per-request context limit.
if (( CONTEXT_LENGTH != 262144 )); then
  echo "CONTEXT_LENGTH '${CONTEXT_LENGTH}' unsupported by this qualified profile (use 262144)"
  exit 1
fi

# Qwen3.8 DFlash uses five GDN state slots per admitted request.
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

# Optional. Export HF_TOKEN before launch for authenticated Hugging Face pulls.
export HF_TOKEN="${HF_TOKEN:-}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting ${TARGET_MODEL} @ ${TARGET_REV:0:8} with DFlash 2 ${DFLASH_MODEL} @ ${DFLASH_REV:0:8}"
echo "Per-request context: ${CONTEXT_LENGTH} tokens (native; YaRN off)"
echo "Max concurrent requests: ${MAX_CONCURRENT_REQUESTS} (mamba pool ${MAMBA_CACHE_SIZE} slots)"
echo "KV token pool: ${MAX_TOTAL_TOKENS}; DFlash draft window: ${DFLASH_DRAFT_WINDOW_SIZE}"
echo "Priority scheduling: ${PRIORITY_SCHEDULING} (default=${DEFAULT_PRIORITY_VALUE}, preemption threshold=${PRIORITY_PREEMPTION_THRESHOLD})"
echo "Spec decode: DFLASH draft=${DFLASH_DRAFT_TOKENS} rev=${DFLASH_REV:0:8}"
echo "Image: ${DFLASH_IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (DFlash 2)
EOF

PIN_ARGS=()
[[ -n "${CPUSET}" ]] && PIN_ARGS=(--cpuset-cpus "${CPUSET}")
PREFILL_GRAPH_ARGS=(--disable-prefill-cuda-graph)
PRIORITY_ARGS=()
if [[ "${PRIORITY_SCHEDULING}" == "1" ]]; then
  PRIORITY_ARGS=(
    --enable-priority-scheduling
    --default-priority-value "${DEFAULT_PRIORITY_VALUE}"
    --priority-scheduling-preemption-threshold "${PRIORITY_PREEMPTION_THRESHOLD}"
  )
fi

docker run -d   --name "${CONTAINER_NAME}"   --network host   --ipc host   --privileged   --cap-add IPC_LOCK   --ulimit memlock=-1:-1   --ulimit stack=67108864   --gpus all   --shm-size 32g   "${PIN_ARGS[@]}"   -e HF_HOME=/root/.cache/huggingface   -e TRITON_CACHE_DIR=/root/.triton   -e HF_TOKEN="${HF_TOKEN:-}"   -e FLASHINFER_CUDA_ARCH_LIST="12.1f"   -e CUTE_DSL_ARCH="sm_120a"   -e PYTHONUNBUFFERED=1   -v "${HF_HOME}:/root/.cache/huggingface"   -v "${TRITON_CACHE_DIR}:/root/.triton"   "${DFLASH_IMAGE}"   python3 -m sglang.launch_server   --model-path "${TARGET_MODEL}"   --revision "${TARGET_REV}"   --served-model-name "${SERVED_MODEL_NAME}"   --trust-remote-code   --mem-fraction-static "${MEM_FRACTION}"   --attention-backend flashinfer   --chunked-prefill-size "${CHUNKED_PREFILL}"   --max-prefill-tokens "${CHUNKED_PREFILL}"   "${PREFILL_GRAPH_ARGS[@]}"   --kv-cache-dtype fp8_e4m3   --mamba-ssm-dtype bfloat16   --mamba-full-memory-ratio 4.21   --mamba-radix-cache-strategy extra_buffer   --max-mamba-cache-size "${MAMBA_CACHE_SIZE}"   --max-running-requests "${MAX_CONCURRENT_REQUESTS}"   --max-total-tokens "${MAX_TOTAL_TOKENS}"   --context-length "${CONTEXT_LENGTH}"   --speculative-algorithm DFLASH   --speculative-draft-model-path "${DFLASH_MODEL}"   --speculative-draft-model-revision "${DFLASH_REV}"   --speculative-num-draft-tokens "${DFLASH_DRAFT_TOKENS}"   --speculative-draft-window-size "${DFLASH_DRAFT_WINDOW_SIZE}"   --reasoning-parser qwen3   --tool-call-parser qwen3_coder   --sampling-defaults model   --enable-metrics   --enable-cache-report   --cuda-graph-max-bs-decode 4   --sleep-on-idle   "${PRIORITY_ARGS[@]}"   --host "${HOST}"   --port "${PORT}"   >/dev/null

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
    echo "Timeout waiting for SGLang readiness after ${max_probes} probes (10 minutes)."
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting... (probe ${probe}/${max_probes})"
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "SGLang is ready (DFlash 2 active)"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Anthropic-compatible: http://${HOST}:${PORT}/v1/messages"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Thinking: ON by default (disable per request: chat_template_kwargs {\"enable_thinking\": false})"
echo "SGLang is ready and responding; shell is now free."
