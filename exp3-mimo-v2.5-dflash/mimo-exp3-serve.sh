#!/usr/bin/env bash
# Experiment 3: MiMo-V2.5-NVFP4 (310B MoE) TP=2 across r0 (.4 head) + r3 (.3 worker)
# Image: ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0 (vLLM 0.24, native dflash)
# Networking/env cloned from proven ~/dspark-nvfp4-serve.sh (same .4/.3 pair).
# usage: mimo-exp3-serve.sh <rank 0|1> [none|dflash|mtp]
set -uo pipefail
RANK="${1:?usage: $0 <node_rank 0=head/1=worker> [none|dflash|mtp]}"
SPECMODE="${2:-none}"
MASTER=10.100.10.4; IF=enp1s0f1np1; HCA=rocep1s0f1
SELF=$(ip -4 addr show $IF 2>/dev/null|awk '/inet /{print $2}'|cut -d/ -f1); SELF=${SELF:-$MASTER}
HEADLESS=""; [ "$RANK" != "0" ] && HEADLESS="--headless"
IMG=ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0

MODEL_HOST=$(ls -d "$HOME"/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/*/ | head -1)
MODEL_CTR="/hf/hub/${MODEL_HOST#"$HOME"/.cache/huggingface/hub/}"
DRAFT_HOST="$HOME/models/mimo-dflash-draft"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.84}"   # HARD CEILING 0.84 for this TP=2
K="${K:-8}"

SPEC=""
case "$SPECMODE" in
  dflash) SPEC="{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":$K}" ;;
  mtp)    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":1}" ;;
  none)   ;;
  *) echo "bad specmode $SPECMODE"; exit 1 ;;
esac

# watchdog + gpu clear before every launch
setsid "$HOME/fastkill.sh" </dev/null >/dev/null 2>&1 &
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true
docker rm -f mimo-exp3 >/dev/null 2>&1 || true

docker run --gpus all -d --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/hf" \
  -v "$DRAFT_HOST:/draft:ro" \
  -v "$HOME/mimo-exp3-patches/mimo_v2.py:/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2.py:ro" \
  -v "$HOME/mimo-exp3-patches/modelopt.py:/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/modelopt.py:ro" \
  -v "$HOME/mimo-exp3-patches/mimo_v2_omni.py:/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2_omni.py:ro" \
  -v "$HOME/mimo-exp3-patches/mimo_v2_mtp.py:/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/mimo_v2_mtp.py:ro" \
  --name mimo-exp3 \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT=/hf/vllm-cache \
  -e VLLM_HOST_IP=$SELF -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_NET=IB -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm "$IMG" serve "$MODEL_CTR" \
    --served-model-name mimo-v25 --host 0.0.0.0 --port 8000 \
    --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --enforce-eager --kv-cache-dtype fp8 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    ${SPEC:+--speculative-config "$SPEC"} \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$RANK" --master-addr $MASTER --master-port 25000 $HEADLESS
echo "launched mimo-exp3 rank=$RANK spec=$SPECMODE rc=$?"
