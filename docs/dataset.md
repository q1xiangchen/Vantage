# Evaluation Datasets

Vantage is evaluated on five spatial-reasoning benchmarks. Each is read from
`data/<benchmark>/` (paths below are relative to the repository root); the question-type subsets used in the paper are listed in
`config/benchmarks/<benchmark>.json` (edit `question_type` there, or pass
`--question_type`, to evaluate a different subset).

| Benchmark | Subset used in the paper |
|---|---|
| MindCube (tiny) | `among`, `around`, `rotation` |
| MMSI-Bench | Positional Relationship (6 types) + Attribute (2 types) |
| BLINK | `multiview_reasoning` (val, 133 samples) |
| OmniSpatial | `Perspective_Taking` |
| SPINBench | `dynamic_rotation`, `dynamic_translation` |

## MMSI-Bench

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/RunsenXu/MMSI-Bench/tree/main).

The resulting layout:

```
Vantage
├── ...
├── data
│   ├── ...
│   ├── mmsi
│   │   ├── MMSI_Bench.parquet
│   │   └── images # After the first run of the code, it will be automatically created.
│   │       ├── 0_0.jpg
│   │       ├── ...
│   ├── ...
├── ...
```

## MindCube

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/MLL-Lab/MindCube).

Unzip `data.zip` and move its contents into `data/mindcube`:

```bash
unzip data.zip
mkdir -p data/mindcube
mv raw other_all_image data/mindcube
```

The resulting layout:

```
Vantage
├── ...
├── data
│   ├── ...
│   ├── mindcube
│   │   ├── raw
│   │   │   ├── MindCube.jsonl
│   │   │   ├── MindCube_train.jsonl
│   │   │   └── MindCube_tinybench.jsonl
│   │   └── other_all_image
│   │       ├── around
│   │       ├── among
│   │       └── rotation
│   ├── ...
├── ...
```

## BLINK (Multi-view Reasoning)

Only the `Multi-view_Reasoning` subset (val split, 133 samples) is used. The
BLINK paper's `prompt` field is fed to the model verbatim; test-set answers
are hidden on HF, so we score on val only.

Download the val parquet from 🤗 [Hugging Face](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)
and extract its images (run with the pipeline environment's python):

```bash
python - <<'PY'
import json, os
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

parquet = hf_hub_download(
    repo_id="BLINK-Benchmark/BLINK",
    filename="Multi-view_Reasoning/val-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir="data/blink",
)
img_dir = "data/blink/images"; os.makedirs(img_dir, exist_ok=True)
meta = []
for row in pq.read_table(parquet).to_pylist():
    idx = row["idx"]; paths = []
    for i in (1, 2):
        p = f"{img_dir}/{idx}_{i}.jpg"
        with open(p, "wb") as f: f.write(row[f"image_{i}"]["bytes"])
        paths.append(f"images/{idx}_{i}.jpg")
    meta.append({"idx": idx, "question": row["question"], "prompt": row["prompt"],
                 "choices": list(row["choices"]), "answer": row["answer"],
                 "sub_task": row["sub_task"], "image_paths": paths})
with open("data/blink/multiview_reasoning_val.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
PY
```

The resulting layout:

```
Vantage
├── ...
├── data
│   ├── ...
│   ├── blink
│   │   ├── Multi-view_Reasoning
│   │   │   └── val-00000-of-00001.parquet
│   │   ├── images
│   │   │   ├── val_Multi-view_Reasoning_1_1.jpg
│   │   │   ├── val_Multi-view_Reasoning_1_2.jpg
│   │   │   └── ...
│   │   └── multiview_reasoning_val.json
│   ├── ...
├── ...
```

## OmniSpatial

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/qizekun/OmniSpatial).

Unzip `OmniSpatial-test.zip` into `data/omnispatial`:

```bash
unzip OmniSpatial-test.zip
mv OmniSpatial-test data/omnispatial
```

The resulting layout:

```
Vantage
├── ...
├── data
│   ├── ...
│   ├── omnispatial
│   │   ├── data.json
│   │   ├── Complex_Logic
│   │   │   ├── 1.png
│   │   │   ├── ...
│   │   ├── Dynamic_Reasoning
│   │   │   ├── 1.png
│   │   │   ├── ...
│   │   ├── ...
│   ├── ...
├── ...
```

## SPINBench

Download the dataset from 🤗 [Hugging Face](https://huggingface.co/datasets/YuyouZhang/SpinBench)
and unzip the images.

```bash
mkdir -p data/spinbench
# put test.jsonl and images.zip under data/spinbench, then
cd data/spinbench && unzip images.zip
```

The resulting layout:

```
Vantage
├── ...
├── data
│   ├── ...
│   ├── spinbench
│   │   ├── test.jsonl
│   │   └── images
│   │       ├── cars_rotation_02c87884b4.jpg
│   │       ├── ...
│   ├── ...
├── ...
```
