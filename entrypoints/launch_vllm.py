import argparse
import datetime
import json
import os
import random
import socket
import subprocess
import sys
from typing import Dict, List, Tuple
import uuid

import pynvml


try:
    import fcntl
    def lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:
    import msvcrt
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class FileLock:

    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'a+')  # 'a+' creates the file if missing
        self.file.seek(0)
        lock_file(self.file)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            unlock_file(self.file)
            self.file.close()
            self.file = None


class LogRedirector:

    def __init__(self, log_file_handle):
        self.log_file_handle = log_file_handle
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.log_file_handle
        sys.stderr = self.log_file_handle
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


def get_local_ip() -> str:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = '127.0.0.1'
        print(f'[Launcher] Cannot get local ip, error msg: {e}')
    finally:
        if s:
            s.close()
    return ip


def find_free_port(min_port=30001, max_port=65535) -> int:
    if min_port > max_port:
        raise ValueError('min_port must be less than or equal to max_port')

    ports_to_try = list(range(min_port, max_port + 1))
    random.shuffle(ports_to_try)

    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    raise IOError(f"No available port found within the range [{min_port}, {max_port}]")


def find_free_gpus(num_gpus: int) -> List[int]:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_gpus = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if not procs:
                free_gpus.append(i)
        except pynvml.NVMLError as e:
            print(f'[Launcher] Could not query processes for GPU {i}: {e}')
    
    pynvml.nvmlShutdown()

    if len(free_gpus) < num_gpus:
        raise ValueError(
            f'Not enough free GPUs. Found {len(free_gpus)}, but need {num_gpus}. '
            f'Available GPUs: {free_gpus}'
        )

    return free_gpus[:num_gpus]


def get_current_time() -> str:
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y/%m/%d %H:%M:%S")
    return formatted_time


def get_launcher(args) -> List[str]:
    if args.port is None:
        args.port = find_free_port()
        print(f'[Launcher] No port specified. Found and using free port: {args.port}')

    vllm_args = [
        'vllm.entrypoints.openai.api_server', 
        '--model', args.model,
        '--tensor-parallel-size', str(args.tp),
        '--max-model-len', str(args.max_model_len),
        '--max-num-seqs', str(args.max_num_seqs),
        '--dtype', 'auto',
        '--host', '0.0.0.0',
        '--port', str(args.port),
        '--trust-remote-code',
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
    ]

    if args.served_model_name:
        vllm_args.extend(['--served-model-name', args.served_model_name])

    # Workaround for vLLM issue #45198: with TP>1, custom all-reduce paths can
    # trigger Triton JIT compilation (e.g. _zero_kv_blocks_kernel) during
    # inference and deadlock the workers. Disabling it falls back to NCCL.
    if args.tp > 1:
        vllm_args.append('--disable-custom-all-reduce')

    # InternVL (e.g. OpenGVLab/InternVL3-8B). vLLM 0.11.0's internvl VIDEO
    # profiling crashes at engine start: dummy video frames are int64 and PIL
    # Image.fromarray dies with KeyError ((1,1,3),'<i8'). We never send video, so
    # disabling that modality skips the profiling path entirely.
    if 'internvl' in args.model.lower():
        vllm_args.extend(['--limit-mm-per-prompt.video', '0'])

    if 'qwen3-vl' in args.model.lower():
        vllm_args.extend([
            '--limit-mm-per-prompt.video', '0',
            '--distributed-executor-backend', 'mp',
            # mm processor cache disabled: suspected cause of device-side
            # asserts in the mm encoder under high concurrency. Costs a few %
            # CPU-side reprocessing per image.
            '--mm-processor-cache-gb', '0',
        ])
        # Async scheduling crashes vLLM 0.11.0 mid-run on multi-image inputs
        # (mm-encoder gather assert) and at max_model_len=65536; the model configs
        # therefore serve Qwen3-VL with --no_async_scheduling (~20-25% slower).
        if not args.no_async_scheduling:
            vllm_args.append('--async-scheduling')
        # Data-parallel ViT encoding only makes sense across multiple TP ranks.
        if args.tp > 1:
            vllm_args.extend(['--mm-encoder-tp-mode', 'data'])

    # Gemma 4 instruct (e.g. gemma-4-31B-it). Requires vllm>=0.19 (gemma4 arch).
    # Even with thinking disabled (no <|think|> in the system prompt) the model
    # emits an empty `<|channel>thought\n<channel|>` block before the answer;
    # the gemma4 reasoning parser strips it into `reasoning_content` so the
    # answer parsers downstream see clean content.
    if 'gemma-4' in args.model.lower():
        vllm_args.extend(['--reasoning-parser', 'gemma4'])

    # Qwen3.6 dense VLM (e.g. Qwen3.6-27B). Requires vllm>=0.19. It always emits <think> blocks, so enable the qwen3
    # reasoning parser per the HF model card — vLLM then surfaces them as
    # `reasoning_content`, which cot_reasoner.py reads directly. All flags below
    # are verified present in vllm 0.19.1's config schema.
    if 'qwen3.6' in args.model.lower():
        vllm_args.extend([
            '--reasoning-parser', 'qwen3',
            '--async-scheduling',
            '--mm-processor-cache-gb', '50',
        ])
        # GDN prefill kernel. vLLM's `auto` prefers FlashInfer on Hopper (SM90);
        # default to triton, which needs no flashinfer install (a vLLM build
        # without flashinfer crashes every request under `auto` on SM90). Pass
        # --gdn_prefill_backend auto to let vLLM pick.
        if args.gdn_prefill_backend != 'auto':
            vllm_args.extend(['--gdn-prefill-backend', args.gdn_prefill_backend])
        # Data-parallel ViT encoding only helps across multiple TP ranks.
        if args.tp > 1:
            vllm_args.extend(['--mm-encoder-tp-mode', 'data'])

    print(f'[Launcher] {" ".join(vllm_args)}')

    launcher = [sys.executable, '-m'] + vllm_args
    return launcher


def prepare_envs(num_gpus: int) -> Tuple[Dict[str, str], List[int]]:
    env = os.environ.copy()

    # set visible gpus
    try:
        selected_gpus = find_free_gpus(num_gpus)
        print(f'[Launcher] Found {len(selected_gpus)} free GPUs: {selected_gpus}')
    except Exception as e:
        print(f'[Launcher] Error finding free GPUs: {e}')
        raise
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))
    return env, selected_gpus


def setup_record(
    serve_file: str, 
    lock_file: str, 
    args: argparse.Namespace, 
    uid: str, 
    pid: str,
    gpus: List[int],
) -> None:
    with FileLock(lock_file):
        if os.path.exists(serve_file):
            with open(serve_file, 'r', encoding='utf-8') as f:
                serve_dict = json.load(f)
        else:
            serve_dict = {}
        
        model_key = args.model if args.served_model_name is None \
            else args.served_model_name

        if model_key not in serve_dict:
            serve_dict[model_key] = {}

        serve_dict[model_key][uid] = {
            'pid': pid,
            'ip': get_local_ip(),
            'port': str(args.port),
            'tp': str(args.tp),
            'gpus': gpus,
            'max_model_len': str(args.max_model_len),
            'max_num_seqs': str(args.max_num_seqs),
            'create_time': get_current_time(),
        }

        with open(serve_file, 'w', encoding='utf-8') as f:
            json.dump(serve_dict, f, indent=2, ensure_ascii=False)


def cleanup_record(
    serve_file: str, 
    lock_file: str, 
    model: str, 
    uid: str
) -> None:
    with FileLock(lock_file):
        try:
            with open(serve_file, 'r+', encoding='utf-8') as f:
                serve_dict = json.load(f)

                if model in serve_dict and uid in serve_dict[model]:
                    del serve_dict[model][uid]
                    if not serve_dict[model]: 
                        del serve_dict[model]

                    f.seek(0)
                    f.truncate()
                    json.dump(serve_dict, f, indent=2, ensure_ascii=False)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f'[Launcher] Cleanup skipped, file might be missing, empty or entry not found: {e}')
            pass


def launch_vllm_server(args: argparse.Namespace):
    uid = str(uuid.uuid4())

    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # The caller (e.g. scripts/run.sh) can pin the log path via VANTAGE_SERVE_LOG;
    # otherwise fall back to a per-launch uuid name.
    log_file = os.environ.get('VANTAGE_SERVE_LOG') or os.path.join(log_dir, f'serve_{uid}.log')

    # Coordination registry. Defaults to logs/serve.json but can be overridden
    # via VANTAGE_SERVE_FILE so two runs sharing one node (and this filesystem) each
    # get their own registry and never cross-register under the same model key.
    serve_file = os.environ.get('VANTAGE_SERVE_FILE') or os.path.join(log_dir, 'serve.json')
    lock_file = serve_file + '.lock'

    with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
        with LogRedirector(f):
            print(f'--- Launcher Log for Service UID: {uid} ---')
            launcher = get_launcher(args)
            envs, selected_gpus = prepare_envs(args.tp)

            process = None
            model_key = args.model if args.served_model_name is None \
                else args.served_model_name
            try:
                # 1. Launch the subprocess
                process = subprocess.Popen(
                    launcher,
                    stdout=f,
                    stderr=f,
                    env=envs,
                )
                pid = process.pid
                print(f'[Launcher] vLLM server (PID: {pid}) for model "{model_key}" started.')

                # 2. Register the service
                setup_record(serve_file, lock_file, args, uid, str(pid), selected_gpus)

                # 3. Wait for the process to complete
                process.wait()

            finally:
                if process:
                    print(f'[Launcher] vLLM server (PID: {pid}) has terminated. Cleaning up record.')
                    cleanup_record(serve_file, lock_file, model_key, uid)
                else:
                    print(f'[Launcher] Process failed to launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('vLLM Model Launcher')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--served_model_name', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=32768)
    parser.add_argument('--max_num_seqs', type=int, default=16)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95,
                        help='Fraction of GPU memory vLLM may reserve. Lower this '
                             '(e.g. 0.6) when sharing a single GPU with other models '
                             'such as the view-synthesis model.')
    parser.add_argument('--gdn_prefill_backend', type=str, default='triton',
                        choices=['auto', 'triton', 'flashinfer', 'cutedsl'],
                        help='GDN prefill kernel for Qwen3.6 (hybrid GDN arch). '
                             'Default triton; auto defers to vLLM.')
    parser.add_argument('--no_async_scheduling', action='store_true',
                        help='Serve Qwen3-VL without --async-scheduling (see '
                             'get_launcher).')
    args = parser.parse_args()

    launch_vllm_server(args)
