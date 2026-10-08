# Usage

## Quick start

```bash
# Vantage on MindCube with Qwen3-VL-8B (launches vLLM, runs, scores, shuts down)
VLLM_PYTHON=envs/vllm-0.11/bin/python VANTAGE_PIPELINE_PYTHON=envs/vantage_pipeline/bin/python \
    bash scripts/run.sh qwen3-vl-8b mindcube vantage

# The bare-VQA baseline for the same model / benchmark
VLLM_PYTHON=envs/vllm-0.11/bin/python VANTAGE_PIPELINE_PYTHON=envs/vantage_pipeline/bin/python \
    bash scripts/run.sh qwen3-vl-8b mindcube baseline

# An API model (no vLLM server)
OPENAI_API_KEY=... VANTAGE_PIPELINE_PYTHON=envs/vantage_pipeline/bin/python \
    bash scripts/run.sh gpt-5.4 mmsi vantage
```

Results land in `results/<mode>/<benchmark>_<model>/` (`<mode>`: `vantage` or `baseline`):

| File | Content |
|---|---|
| `config.json` | the fully resolved run configuration |
| `predictions.jsonl` | one line per sample: raw answer, route, question type, correctness |
| `results.csv` | per-sample scoring details |
| `results_summary.json` | overall and per-question-type accuracy |
| `sessions.zip` | per-sample traces (`session-<id>/trace.jsonl`) and synthesized views |

Add `--resume` to continue an interrupted run, `--max_samples N` for a smoke
test, and `--keep_sessions` to keep `sessions/` unzipped.

## Configuration

A run's configuration is assembled, lowest to highest precedence, from:

1. the defaults in [`workflow/config.py`](../workflow/config.py),
2. `config/benchmarks/<benchmark>.json` — the question-type subset and the
   default concurrency,
3. `config/models/<model>.json` — the model id, sampling parameters, output-token
   budget, answer-prompt wording and (under `serve`) the vLLM launch settings,
4. command-line arguments (`python -m entrypoints.agent --help`).

To add a model, drop a new JSON into `config/models/`; to evaluate another subset,
edit `question_type` in `config/benchmarks/` or pass `--question_type`. Override
the concurrency with `--concurrency N`.

## Running without `run.sh`

1. **Launch vLLM** (in the vLLM environment; it stays in the foreground and
   registers its port in `logs/serve.json`, or `$VANTAGE_SERVE_FILE` if set):

   ```bash
   envs/vllm-0.11/bin/python -m entrypoints.launch_vllm \
       --model Qwen/Qwen3-VL-8B-Instruct --tp 1 --max_model_len 32768 --no_async_scheduling
   ```

   The serve flags come from the `serve` block of `config/models/<model>.json`
   (Qwen3.6-27B and Gemma-4-31B use `--tp 2`, from `envs/vllm-0.19`).

2. **Run the Vantage pipeline** (in the pipeline environment, once the server is up):

   ```bash
   envs/vantage_pipeline/bin/python -m entrypoints.agent \
       --benchmark mmsi --model qwen3-vl-8b --mode vantage  # or baseline mode
   ```

## Pipeline

`vantage` mode runs on the same frozen VLM throughout:

**Stage 1 — planning**
1. **Question analysis** (`workflow/nodes/s1_analyst.py`) — analyzes the question
   and the input views: which viewpoint the question is posed from and what
   evidence is missing in this view.
2. **View planning** (`workflow/nodes/s2_pose_planner.py`) — plans the view to
   seek as a relative camera move (`pose` + `magnitude`) from a reference
   view (one of the input views), and writes `reasoning guidance` relating the new
   view to the question. The move is grounded to a 6-DoF pose by
   `tools/apis/pose_utils.py`.

**Stage 2 — view synthesis and final VQA** (`workflow/nodes/s4_solver.py`)
1. **View synthesis** (`tools/apis/view_synthesizer.py`) — reconstructs the scene in
   3D with G3T and renders the planned view by reprojecting the point cloud.
2. **Final VQA** — answers the original question over the original views + the
   synthesized view, with the question analysis and the reasoning guidance as
   context.

Any failure degrades gracefully: a failed question analysis falls back to a bare
VQA; a failed view plan or view synthesis answers from the question analysis. Every sample's trace
records which path it took (`route`) and why (`*_error`).