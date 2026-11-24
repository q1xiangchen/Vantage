# Installation

Vantage runs in two kinds of Python environment:

* the **pipeline environment** — the pipeline, the benchmarks, the G3T 3D
  reconstruction and view synthesis by reprojection (PyTorch, CUDA 12.4);
* one or two **vLLM environments** — the servers for the open-weight VLMs.

| vLLM env | Models |
|---|---|
| `requirements/vllm-0.11.txt` | Qwen3-VL-4B/8B, InternVL3-8B |
| `requirements/vllm-0.19.txt` | Qwen3.6-27B, Gemma-4-31B |

API models (GPT-5.4) need only the pipeline environment and an `OPENAI_API_KEY`.

## Pipeline environment

```bash
uv venv envs/vantage_pipeline --python 3.11
uv pip install --python envs/vantage_pipeline/bin/python \
    torch==2.5.1 torchvision==0.20.1 --torch-backend=cu124
uv pip install --python envs/vantage_pipeline/bin/python -r requirements/vantage_pipeline.txt
```

### G3T (3D reconstruction)

The scene is reconstructed in 3D with [G3T](https://github.com/g3t-paper/g3t),
cloned into `tools/third_party/`. Its dependencies are already covered by
`requirements/vantage_pipeline.txt`.

```bash
mkdir -p tools/third_party
git clone https://github.com/g3t-paper/g3t.git tools/third_party/g3t
git -C tools/third_party/g3t checkout 193ce19574b73f0778a475ee54aadeb848e86b88
```

The weights download from Hugging Face (`thatbrguy/g3t`) on first use. To
load a local checkpoint instead, set `VANTAGE_G3T_CKPT=/path/to/g3t.pt`.

## vLLM environments

```bash
# Qwen3-VL / InternVL3
uv venv envs/vllm-0.11 --python 3.11
uv pip install --python envs/vllm-0.11/bin/python \
    -r requirements/vllm-0.11.txt --torch-backend=cu128

# Qwen3.6-27B / Gemma-4-31B
uv venv envs/vllm-0.19 --python 3.11
uv pip install --python envs/vllm-0.19/bin/python \
    -r requirements/vllm-0.19.txt --torch-backend=cu128
```