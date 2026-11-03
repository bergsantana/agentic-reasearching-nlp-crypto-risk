#!/usr/bin/env bash
# Step 0 GPU check: the Blackwell card is visible inside Docker on WSL2, and vLLM loads a small
# model on it. If vLLM fails here, step 6 falls back to llama.cpp (see tech/build-plan.md risks).
#
# Usage: scripts/gpu_check.sh            # both checks
#        SKIP_VLLM=1 scripts/gpu_check.sh # container check only
set -euo pipefail

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-base-ubuntu24.04}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.33.0}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
NAME=cryptonlp-gpu-check

echo "== 1. Host driver"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

echo "== 2. GPU inside a container ($CUDA_IMAGE)"
if ! docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi; then
  cat <<'MSG'
FAILED: Docker can't see the GPU. Install the NVIDIA Container Toolkit in this WSL distro:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker      # or: sudo service docker restart
Then run this script again.
MSG
  exit 1
fi

[[ "${SKIP_VLLM:-0}" == 1 ]] && { echo "== vLLM check skipped"; exit 0; }

echo "== 3. vLLM ($VLLM_IMAGE) loading $VLLM_MODEL"
docker rm -f "$NAME" >/dev/null 2>&1 || true
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" --gpus all --ipc=host -p "127.0.0.1:$PORT:8000" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$VLLM_IMAGE" "$VLLM_MODEL" --max-model-len 2048 --gpu-memory-utilization 0.5 >/dev/null

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    echo "FAILED: vLLM exited during startup. Last log lines:"; docker logs --tail 40 "$NAME"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "FAILED: vLLM not ready after 10 min"; docker logs --tail 40 "$NAME"; exit 1; }

curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$VLLM_MODEL\", \"prompt\": \"Bitcoin is\", \"max_tokens\": 16, \"temperature\": 0}" \
  | python3 -c 'import json,sys; print("completion:", json.load(sys.stdin)["choices"][0]["text"].strip())'
docker logs "$NAME" 2>&1 | grep -iE "vllm.*version|compute capability|model weights took|KV cache" | head -6 || true
echo "OK: vLLM serves $VLLM_MODEL on the GPU with $VLLM_IMAGE. Pin this image in docker-compose.yml in step 6."
