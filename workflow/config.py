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
    """Store model, benchmark, and pipeline settings for a run."""

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
    """Model identifier passed to the inference backend."""

    base_url: str = 'vllm'
    """Inference endpoint URL or the local vLLM registry."""

    api_key: str = ''
    """API key override for an OpenAI-compatible endpoint."""

    proxy: str = ''
    """Optional HTTP(S) proxy for the model endpoint."""

    sampling: Optional[Dict[str, Any]] = None
    """Sampling parameters sent to the model with each request."""

    max_tokens: int = 16384
    """Output-token budget per request."""

    answer_prompt_style: str = 'default'
    """Answer instructions selected from the default or official wording."""

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    mode: str = MODE_VANTAGE
    """One of MODES."""

    omit_camera_context: bool = False
    """Hide pose text at all stages while retaining b1 shared reconstruction."""

    sample_ids_file: Optional[str] = None
    """Optional JSON list of fixed benchmark sample IDs."""

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
        if self.omit_camera_context and self.mode != MODE_VANTAGE:
            raise ValueError('omit_camera_context requires vantage mode.')

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
