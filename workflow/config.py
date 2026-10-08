from argparse import Namespace
from dataclasses import asdict, dataclass, fields
import json
import os
from typing import Any, Dict, List, Optional


# Run modes.
#   baseline : bare direct VQA over the original views (no planning, no view synthesis).
#   vantage  : stage 1 (s1_analyst question analysis -> s2_pose_planner view planning)
#              -> stage 2 (s4_solver view synthesis of the planned view -> final VQA).
MODE_BASELINE = 'baseline'
MODE_VANTAGE = 'vantage'
MODES = (MODE_BASELINE, MODE_VANTAGE)

# Which stage-1 outputs the final VQA sees as context: every question-analysis
# field, and only the view plan's `reasoning guidance`.
ANALYSIS_CONTEXT_FIELDS: Optional[List[str]] = None   # None = all fields
PLANNING_CONTEXT_FIELDS: List[str] = ['reasoning guidance']

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
MODEL_CONFIG_DIR = os.path.join(_REPO_ROOT, 'config', 'models')
BENCHMARK_CONFIG_DIR = os.path.join(_REPO_ROOT, 'config', 'benchmarks')


@dataclass
class AgentConfig:
    """Global configuration for a Vantage run.

    Built from (lowest to highest precedence) the dataclass defaults, the
    per-benchmark JSON (config/benchmarks/<bench>.json), the per-model JSON
    (config/models/<model>.json) and CLI arguments. The resolved config is dumped
    to `<work_dir>/config.json`; Ray Serve replicas reload it from there via the
    `AGENT_CONFIG_FILE` env var.
    """

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    benchmark: str = 'mindcube'
    """One of evals.BENCHMARK_REGISTRY."""

    question_type: Optional[List[str]] = None
    """Question-type subset to evaluate. None = every sample of the benchmark."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model: str = ''
    """Model identifier sent to the backend (HF repo id for vLLM, e.g.
    'Qwen/Qwen3-VL-8B-Instruct'; API model name otherwise, e.g. 'gpt-5.4')."""

    base_url: str = 'vllm'
    """'vllm' = the local vLLM server(s) registered by entrypoints.launch_vllm;
    anything else = an OpenAI-compatible endpoint URL."""

    api_key: str = ''
    """API key for an OpenAI-compatible endpoint. Leave empty to read
    OPENAI_API_KEY from the environment (never written to config.json)."""

    proxy: str = ''
    """Optional HTTP(S) proxy for the model endpoint."""

    sampling: Optional[Dict[str, Any]] = None
    """Sampling parameters sent with every request (temperature, top_p, top_k, ...).
    None/{} = send none (backend defaults; required for OpenAI reasoning models)."""

    max_tokens: int = 16384
    """Output-token budget per request."""

    answer_prompt_style: str = 'default'
    """Benchmark answer-prompt wording for the final VQA:
      - 'default':  the suite's own MCQ answer suffix.
      - 'official': each benchmark's official wording."""

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    mode: str = MODE_VANTAGE
    """One of MODES."""

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    run_tag: str = ''
    """Free-form suffix appended to run_name()."""

    work_dir: Optional[str] = None
    """Run output directory. None = results/<mode>/<run_name()>."""

    concurrency: int = 1
    """Number of samples evaluated in parallel."""

    enable_logging: bool = True
    """Write the per-sample trace (sessions/session-<id>/trace.jsonl)."""

    def update(self, data: Dict[str, Any]) -> None:
        names = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in names and value is not None:
                setattr(self, key, value)
        self._validate()

    def update_from_json(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            self.update(json.load(f))

    def update_from_args(self, args: Namespace) -> None:
        self.update(vars(args))

    def _validate(self) -> None:
        from evals import BENCHMARK_REGISTRY
        if self.benchmark not in BENCHMARK_REGISTRY:
            raise ValueError(
                f'Benchmark "{self.benchmark}" not supported. Available benchmarks: '
                f'{list(BENCHMARK_REGISTRY.keys())}')
        if self.mode not in MODES:
            raise ValueError(f'mode="{self.mode}" is invalid. Expected one of {MODES}.')

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data['api_key'] = ''   # never persist a key
        return data

    def run_name(self) -> str:
        """Leaf folder name for a run: '<benchmark>_<model>[_<run_tag>]'."""
        parts = [self.benchmark, self.model.replace('/', '--')]
        if self.run_tag:
            parts.append(self.run_tag)
        return '_'.join(parts)


global_config = None


def get_config() -> AgentConfig:
    global global_config
    if global_config is None:
        global_config = AgentConfig()
        config_file = os.getenv('AGENT_CONFIG_FILE')
        if config_file:
            global_config.update_from_json(config_file)
    return global_config
