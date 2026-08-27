# MiniMax-M2.7 Transformers Deployment Guide

This guide runs [MiniMax-M2.7](https://github.com/MiniMax-AI/MiniMax-M2.7) directly with Transformers. The model weights are available from [Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M2.7).

For the latest compatibility and hardware requirements, consult the [official Transformers deployment guide](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/transformers_deploy_guide.md). The official guide currently requires Linux, Python 3.9 through 3.12, Transformers 4.57.1, a GPU with compute capability 7.0 or newer, and about 220 GB of memory for the weights.

## Install dependencies

Create an isolated environment and install the tested Transformers release:

```bash
uv venv
source .venv/bin/activate
uv pip install transformers==4.57.1 torch accelerate --torch-backend=auto
```

## Run inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "MiniMaxAI/MiniMax-M2.7"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Explain mixture-of-experts models briefly.",
            }
        ],
    }
]

model_inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

with torch.inference_mode():
    generated_ids = model.generate(model_inputs, max_new_tokens=256)

new_tokens = generated_ids[:, model_inputs.shape[1]:]
print(tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

Keep `trust_remote_code=True`; the model depends on its repository implementation. If downloading from Hugging Face is unavailable on your network, configure an approved proxy or mirror before running the script.
