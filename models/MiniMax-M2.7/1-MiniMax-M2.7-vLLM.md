# MiniMax-M2.7 vLLM Deployment Guide

This guide serves [MiniMax-M2.7](https://github.com/MiniMax-AI/MiniMax-M2.7) through an OpenAI-compatible vLLM endpoint. The model weights are available from [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M2.7).

For the latest compatibility and hardware requirements, consult the [official vLLM deployment guide](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/vllm_deploy_guide.md). The official guide currently requires Linux, Python 3.9 through 3.12, a GPU with compute capability 7.0 or newer, and about 220 GB of memory for the weights.

## Install vLLM

Create an isolated environment and install the latest compatible vLLM release:

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

## Start the server

For a four-GPU deployment:

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.7 --trust-remote-code \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

For an eight-GPU deployment with expert parallelism:

```bash
SAFETENSORS_FAST_GPU=1 vllm serve \
    MiniMaxAI/MiniMax-M2.7 --trust-remote-code \
    --enable_expert_parallel --tensor-parallel-size 8 \
    --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think
```

The server listens on `http://localhost:8000/v1` by default.

## Verify the endpoint

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2.7",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain mixture-of-experts models briefly."}
        ]
    }'
```

If vLLM reports that the model is unsupported, upgrade vLLM. If CUDA graph capture causes an illegal memory access, add `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` to the server command, as recommended by the official guide.
