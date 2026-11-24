import argparse
import asyncio
import json
import os
import shutil
import traceback
from typing import Dict

import torch
from tqdm.asyncio import tqdm

from evals import (
    BaseBenchmark,
    BaseBenchmarkSample,
    BENCHMARK_REGISTRY,
    BenchmarkFactory
)
from workflow.config import (
    BENCHMARK_CONFIG_DIR,
    MODEL_CONFIG_DIR,
    MODES,
    MODE_VANTAGE,
    AgentConfig,
)
from workflow.state import AgentState
from evals.run_utils import select_sample_ids, completed_prediction
from tools.utils.llm_invoke import is_truncation_error
from workflow.utils.archive_utils import archive_sessions
from workflow.workflow import AgentWorkflow


def _model_keys():
    return sorted(f[:-5] for f in os.listdir(MODEL_CONFIG_DIR) if f.endswith('.json'))


def _sample_id(value):
    """Preserve string IDs and convert integer-like IDs to JSON-native integers."""
    return value if isinstance(value, str) else int(value)


parser = argparse.ArgumentParser(description='Run Vantage (or the bare-VQA baseline) on a benchmark.')
parser.add_argument('--benchmark', type=str, required=True,
                    choices=[b for b in BENCHMARK_REGISTRY if b != 'none'])
parser.add_argument('--model', type=str, required=True, choices=_model_keys(),
                    help='Model configuration: config/models/<model>.json.')
parser.add_argument('--mode', type=str, default=MODE_VANTAGE, choices=list(MODES))
parser.add_argument('--omit_camera_context', action='store_true', help='Omit camera relations while retaining the Vantage pipeline.')
parser.add_argument('--sample_ids_file', default=None, help='Fixed JSON sample-ID list.')
parser.add_argument('--question_type', nargs='+', default=None,
                    help='Question-type subset; overrides config/benchmarks/<benchmark>.json.')
parser.add_argument('--concurrency', type=int, default=None,
                    help='Samples in flight; default from the benchmark config.')
parser.add_argument('--run_tag', type=str, default=None,
                    help='Suffix appended to the run directory name.')
parser.add_argument('--output_root', type=str, default='results',
                    help='Run directories go to <output_root>/<mode>/<run_name>.')
parser.add_argument('--work_dir', type=str, default=None,
                    help='Explicit run directory (overrides --output_root).')
parser.add_argument('--base_url', type=str, default=None,
                    help="Override the model endpoint ('vllm' or an OpenAI-compatible URL).")
parser.add_argument('--api_key', type=str, default=None,
                    help='API key for an OpenAI-compatible endpoint (default: OPENAI_API_KEY).')
parser.add_argument('--proxy', type=str, default=None,
                    help='HTTP(S) proxy for the model endpoint.')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Evaluate at most this many new samples (smoke tests).')
parser.add_argument('--resume', action='store_true',
                    help='Skip completed and truncated samples; retry only execution failures.')
parser.add_argument('--keep_sessions', action='store_true',
                    help='Do not pack sessions/ into sessions.zip after evaluation.')


async def worker(
    workflow: AgentWorkflow,
    benchmark: BaseBenchmark,
    sample: BaseBenchmarkSample,
    predictions: Dict,
    prediction_file: str,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock
):
    async with semaphore:
        sample_id = _sample_id(sample.sample_id)

        route = None
        status, error = 'complete', None
        try:
            session_dir = workflow.logger.get_session_dir(str(sample_id))
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)

            bbox2d = None if (not hasattr(sample, 'bbox')) or sample.bbox is None \
                else torch.tensor(sample.bbox)
            final_state: AgentState = await workflow.arun(
                instruction=sample.question,
                images=sample.images,
                geometry_image_indices=getattr(sample, 'geometry_image_indices', None),
                bbox2d=bbox2d,
                answer=sample.answer,
                session_id=str(sample_id),
                answer_format=benchmark.data_specific_prompt,
            )
            _, summary = workflow.get_final_answer(final_state)
            record = final_state.get('workspace', {}).get('trace_record') or {}
            route = record.get('route')
            error = (record.get('vqa_answer') or {}).get('error')
            if error and is_truncation_error(error):
                status = 'truncated'
        except Exception as e:
            print(f'[Error] {str(e)}')
            print(traceback.format_exc())
            summary = ''
            error = f'{type(e).__name__}: {e}'
            status = 'truncated' if is_truncation_error(error) else 'failed'

        async with lock:
            predictions[sample_id] = summary
            saved_jsonl = {'sample_id': sample_id, 'content': summary, 'route': route,
                           'status': status, 'error': error}
            with open(prediction_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(saved_jsonl) + '\n')


def build_config(args: argparse.Namespace) -> AgentConfig:
    """defaults < config/benchmarks/<benchmark>.json < config/models/<model>.json < CLI."""
    config = AgentConfig()
    config.update_from_json(os.path.join(BENCHMARK_CONFIG_DIR, f'{args.benchmark}.json'))
    with open(os.path.join(MODEL_CONFIG_DIR, f'{args.model}.json')) as f:
        model_config = json.load(f)
    model_config.pop('serve', None)   # vLLM launch settings, read by scripts/run.sh
    config.update(model_config)
    cli = {k: v for k, v in vars(args).items() if k != 'model'}
    if args.question_type is not None and args.sample_ids_file is None:
        config.sample_ids_file = None
    config.update(cli)
    return config


async def main():
    args = parser.parse_args()
    config = build_config(args)
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max_samples must be positive')
    if config.concurrency < 1:
        parser.error('--concurrency must be positive')
    if config.mode == MODE_VANTAGE and not torch.cuda.is_available():
        parser.error('vantage mode requires a CUDA GPU for G3T')

    if config.work_dir is None:
        config.work_dir = os.path.join(args.output_root, config.mode, config.run_name())
    os.makedirs(config.work_dir, exist_ok=True)

    # Ray Serve replicas reload the resolved config from this file.
    config_path = os.path.join(config.work_dir, 'config.json')
    os.environ['AGENT_CONFIG_FILE'] = os.path.abspath(config_path)
    if config.api_key and not os.environ.get('OPENAI_API_KEY'):
        # config.json never stores the key; hand it to the replicas via the env.
        os.environ['OPENAI_API_KEY'] = config.api_key

    # 1. Initialize Benchmark
    benchmark: BaseBenchmark = BenchmarkFactory.create_benchmark(
        benchmark_name=config.benchmark,
        question_type=config.question_type,
    )
    select_sample_ids(benchmark, config.sample_ids_file)
    cohort = [_sample_id(sid) for sid in benchmark.sample_ids()]
    if not cohort:
        parser.error('the benchmark selection contains no samples')
    cohort_path = os.path.join(config.work_dir, 'sample_ids.json')
    prediction_file = os.path.join(config.work_dir, 'predictions.jsonl')
    resolved_config = config.to_json()
    if args.resume and os.path.exists(prediction_file):
        try:
            with open(config_path, encoding='utf-8') as f:
                previous_config = json.load(f)
            with open(cohort_path, encoding='utf-8') as f:
                previous_cohort = json.load(f)
        except (OSError, ValueError) as exc:
            parser.error(f'cannot verify the existing run for --resume: {exc}')
        runtime_fields = {'work_dir', 'concurrency', 'run_tag', 'enable_logging'}
        changed = [key for key, value in resolved_config.items()
                   if key not in runtime_fields and previous_config.get(key) != value]
        if changed or previous_cohort != cohort:
            reason = ', '.join(changed) or 'sample IDs'
            parser.error(f'cannot resume with different {reason}; choose another run directory')

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(resolved_config, f, indent=4)
    with open(cohort_path, 'w', encoding='utf-8') as f:
        json.dump(cohort, f)
    benchmark.apply_prompt_style(config.answer_prompt_style)

    predictions, done = {}, set()
    if args.resume and os.path.exists(prediction_file):
        kept_records = []
        with open(prediction_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        for line in lines:
            record = json.loads(line)
            content = record['content']
            if completed_prediction(record):
                predictions[record['sample_id']] = content
                done.add(record['sample_id'])
                kept_records.append(record)
        with open(prediction_file, 'w') as f:
            f.writelines([json.dumps(rec) + '\n' for rec in kept_records])
        print(f'Resuming benchmarking. Found {len(done)} completed samples.')
    else:
        with open(prediction_file, 'w') as f:
            pass

    # Select work before launching model and reconstruction tools.
    pending = [sid for sid in cohort if sid not in done]
    if args.max_samples is not None:
        pending = pending[:args.max_samples]
    pending = set(pending)
    workflow = AgentWorkflow() if pending else None

    # 3. Dispatch Benchmark Samples
    concurrency = min(config.concurrency, len(benchmark))
    print(f'Executing tasks with concurrency={concurrency}')
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    tasks = []
    for sample in benchmark:
        if sample.sample_id not in pending:
            continue
        tasks.append(asyncio.create_task(
            worker(
                workflow=workflow,
                benchmark=benchmark,
                sample=sample,
                predictions=predictions,
                prediction_file=prediction_file,
                semaphore=semaphore,
                lock=write_lock,
            )
        ))

    # 4. Start Benchmarking
    print('Starting inference loop...')
    if tasks:
        try:
            await tqdm.gather(*tasks, desc=f'Evaluating {benchmark.__class__.__name__}')
        finally:
            workflow.shutdown()

    # 5. Evaluate Predictions
    print('Evaluating predictions...')
    if args.max_samples is not None:
        scored_ids = benchmark.sample_ids()
        benchmark.data = benchmark.data[scored_ids.isin(done | pending)].reset_index(drop=True)
    predictions = {sid: predictions.get(sid, '') for sid in benchmark.sample_ids()}
    benchmark.evaluate(predictions, output_dir=config.work_dir)
    print(f'Evaluation finished. Results saved to: {os.path.abspath(config.work_dir)}')

    # 6. Pack sessions/ into one zip (keeps the inode count low).
    if not args.keep_sessions:
        print('Packing sessions/ into sessions.zip ...')
        zip_path = archive_sessions(config.work_dir)
        if zip_path:
            print(f'Sessions archived to: {zip_path}')


if __name__ == '__main__':
    asyncio.run(main())
