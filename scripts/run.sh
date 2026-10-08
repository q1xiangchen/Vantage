#!/bin/bash
# Run one (model, benchmark, mode) evaluation end to end:
#   1. launch a vLLM server for the model (skipped for API models, e.g. gpt-5.4),
#   2. wait until it answers /health,
#   3. run the agent over the benchmark and score it,
#   4. shut the server down.
#
# Usage:
#   bash scripts/run.sh <model> <benchmark> [baseline|vantage] [extra agent args...]
#   e.g. bash scripts/run.sh qwen3-vl-8b mindcube vantage --resume
#
# <model> is a preset in config/models/, <benchmark> one of config/benchmarks/.
# The vLLM server and the pipeline may live in different Python environments:
#   VLLM_PYTHON              python with vLLM installed          (default: python)
#   VANTAGE_PIPELINE_PYTHON  python with the pipeline + G3T deps (default: python)
# Other knobs (optional):
#   GPU_MEM_UTIL  vLLM --gpu-memory-utilization (default 0.85: leaves room for the
#                 G3T reconstruction, which shares the GPU and is invisible to vLLM)
#   OUTPUT_ROOT   results root (default: results)
set -euo pipefail

MODEL_KEY="${1:?usage: run.sh <model> <benchmark> [baseline|vantage] [agent args...]}"
BENCH="${2:?usage: run.sh <model> <benchmark> [baseline|vantage] [agent args...]}"
MODE="${3:-vantage}"
shift $(( $# < 3 ? $# : 3 ))

cd "$(dirname "$0")/.."
VLLM_PYTHON="${VLLM_PYTHON:-python}"
VANTAGE_PIPELINE_PYTHON="${VANTAGE_PIPELINE_PYTHON:-python}"
MODEL_CFG="config/models/${MODEL_KEY}.json"
BENCH_CFG="config/benchmarks/${BENCH}.json"
[ -f "$MODEL_CFG" ] || { echo "[run] unknown model preset: $MODEL_CFG" >&2; exit 1; }
[ -f "$BENCH_CFG" ] || { echo "[run] unknown benchmark: $BENCH_CFG" >&2; exit 1; }

cfg() {  # cfg <file> <python expression over the parsed json `c`>
    "$VANTAGE_PIPELINE_PYTHON" -c "import json,sys; c=json.load(open(sys.argv[1])); v=$2; print('' if v is None else v)" "$1"
}
MODEL=$(cfg "$MODEL_CFG" "c['model']")
BASE_URL=$(cfg "$MODEL_CFG" "c.get('base_url', 'vllm')")
CONCURRENCY=$(cfg "$BENCH_CFG" "c.get('concurrency', 16)")

AGENT_ARGS=(--benchmark "$BENCH" --model "$MODEL_KEY" --mode "$MODE"
            --output_root "${OUTPUT_ROOT:-results}")

if [ "$BASE_URL" = "vllm" ]; then
    TP=$(cfg "$MODEL_CFG" "c['serve']['tp']")
    MAX_MODEL_LEN=$(cfg "$MODEL_CFG" "c['serve']['max_model_len']")
    NO_ASYNC=$(cfg "$MODEL_CFG" "c['serve'].get('no_async_scheduling', False)")

    mkdir -p logs
    RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
    # Per-run registry so concurrent runs on one node never share an endpoint.
    export VANTAGE_SERVE_FILE="$PWD/logs/serve_${RUN_ID}.json"
    export VANTAGE_SERVE_LOG="$PWD/logs/serve_${RUN_ID}.log"

    LAUNCH_ARGS=(--model "$MODEL" --tp "$TP" --max_num_seqs "$CONCURRENCY"
                 --max_model_len "$MAX_MODEL_LEN"
                 --gpu_memory_utilization "${GPU_MEM_UTIL:-0.85}")
    [ "$NO_ASYNC" = "True" ] && LAUNCH_ARGS+=(--no_async_scheduling)

    "$VLLM_PYTHON" -m entrypoints.launch_vllm "${LAUNCH_ARGS[@]}" &
    VLLM_PID=$!

    cleanup() {
        set +e
        if [ -f "$VANTAGE_SERVE_FILE" ]; then
            SRV_PID=$(cfg "$VANTAGE_SERVE_FILE" "next(iter(c.get('$MODEL', {}).values()), {}).get('pid')")
            [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
        fi
        kill "$VLLM_PID" 2>/dev/null
    }
    trap cleanup EXIT

    echo "[run] waiting for vLLM ($MODEL, tp=$TP); log: $VANTAGE_SERVE_LOG"
    PORT=""
    for _ in $(seq 1 60); do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "[run] vLLM exited; see $VANTAGE_SERVE_LOG" >&2; exit 1; }
        if [ -f "$VANTAGE_SERVE_FILE" ]; then
            PORT=$(cfg "$VANTAGE_SERVE_FILE" "next(iter(c.get('$MODEL', {}).values()), {}).get('port')")
            if [ -n "$PORT" ] && curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
                echo "[run] vLLM ready on port $PORT"
                break
            fi
        fi
        sleep 30
    done
    curl -sf "http://127.0.0.1:${PORT:-0}/health" >/dev/null \
        || { echo "[run] vLLM did not become ready in time" >&2; exit 1; }
fi

"$VANTAGE_PIPELINE_PYTHON" -m entrypoints.agent "${AGENT_ARGS[@]}" "$@"
