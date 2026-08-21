#!/usr/bin/env bash
set -euo pipefail

# Build the exact DFlash 2 overlay qualified by this recipe.
#
# The mutable lmsysorg/sglang:qwen38-27b tag is intentionally not used: it
# currently resolves to a different image. The pinned SM121/Qwen3.8 base plus
# these six checksummed files is byte-equivalent to the qualified live stack.

if [[ $# -gt 0 ]]; then
  echo "usage: $0" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
overlay="${script_dir}/overlay-dflash2"
base_image="lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6"
tag="lmsysorg/sglang:qwen38-27b-dflash2"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

[[ -f "${overlay}/MANIFEST.sha256" ]] || {
  echo "missing ${overlay}/MANIFEST.sha256" >&2
  exit 1
}
(cd "${overlay}" && sha256sum -c MANIFEST.sha256) || {
  echo "overlay checksum mismatch" >&2
  exit 1
}

files=(
  kernels/ops/speculative/dflash.py
  srt/models/dflash.py
  srt/model_executor/model_runner_components/spec_aux_hidden_state.py
  srt/speculative/dflash_utils.py
  srt/speculative/dflash_worker_v2.py
  srt/speculative/dflash2_compat.py
)

mkdir -p "${stage}/overlay/sglang"
cp -r "${overlay}/sglang/." "${stage}/overlay/sglang/"
if [[ -d "${overlay}/test" ]]; then
  mkdir -p "${stage}/overlay/test"
  cp -r "${overlay}/test/." "${stage}/overlay/test/"
fi

{
  echo "FROM ${base_image}"
  for rel in "${files[@]}"; do
    echo "COPY overlay/sglang/${rel} /sgl-workspace/sglang/python/sglang/${rel}"
  done
  if [[ -d "${overlay}/test" ]]; then
    echo "COPY overlay/test /sgl-workspace/sglang/test"
  fi
  echo 'LABEL org.opencontainers.image.description="Pinned Qwen3.8 SM121 base + qualified DFlash 2 selector-capture overlay"'
  echo 'LABEL io.0xwhitemage.dflash2.upstream="sgl-project/sglang#35371,#35496"'
} > "${stage}/Dockerfile"

export DOCKER_BUILDKIT=1
docker build -t "${tag}" -f "${stage}/Dockerfile" "${stage}"
echo "built ${tag} from ${base_image} with ${#files[@]} verified files"
