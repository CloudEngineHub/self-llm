# MiniMax-M2.7 SGLang Deployment Guide

This guide serves [MiniMax-M2.7](https://github.com/MiniMax-AI/MiniMax-M2.7) through an OpenAI-compatible SGLang endpoint. The model weights are available from [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M2.7).

For the latest compatibility and hardware requirements, consult the [official SGLang deployment guide](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/sglang_deploy_guide.md). The official guide currently requires Linux, Python 3.9 through 3.12, a GPU with compute capability 7.0 or newer, and about 220 GB of memory for the weights.

## Install SGLang

Create an isolated environment and install SGLang:

```bash
uv venv
source .venv/bin/activate
uv pip install sglang
```

Use SGLang 0.5.4.post1 or newer for MiniMax-M2 family support.

## Start the server

For a four-GPU deployment:

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.7 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

For an eight-GPU deployment with expert parallelism:

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.7 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

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
