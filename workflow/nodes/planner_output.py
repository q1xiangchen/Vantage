"""Shared plan contract for the pipeline nodes.

`PlannerOutput` is the single value that flows through the workspace key
`planner_output`: s1_analyst (question analysis) creates it, s2_pose_planner (view
planning) fills the route/viewpoint, and s4_solver (final VQA) reads it to answer. The
`ROUTE_*` constants record the per-sample path (distinct from the run `mode` in config).
The small parsing helpers are shared by s1 and s2.
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple

from tools.apis import AgentContext
from tools.apis.pose_utils import ViewpointSpec
from workflow.utils.parse_utils import parse_json_str


# A successful view plan: s4 synthesizes the planned view and answers over it.
ROUTE_SYNTHESIS = 'synthesis'

# Fallback routes set by s1/s2 when their call fails (or, for s2, the view plan is
# inconsistent: missing reasoning guidance, ungroundable pose). s4 then answers from
# the question analysis, or by bare VQA if s1 failed too; the route on the trace
# record shows which stage degraded.
ROUTE_S1_ERROR_SKIP = 's1_error_skip'
ROUTE_S2_ERROR_SKIP = 's2_error_skip'


@dataclass
class PlannerOutput(AgentContext):
    """The accumulated plan, written incrementally by s1_analyst -> s2_pose_planner and
    consumed by s4_solver.

    Attributes:
        analysis (Optional[str]): The question analysis (JSON-serialized parsed
            form, or raw). None for baseline or when s1 produced nothing.
        route (Optional[str]): ROUTE_SYNTHESIS, or a ROUTE_*_ERROR_SKIP fallback.
            None until s2 runs.
        viewpoint (Optional[ViewpointSpec]): For 'synthesis', the grounded planned view.
        reasoning (Optional[str]): The s2 pose-planner justification.
        failed (bool): True when an upstream planning call failed; s4 then degrades.
        analysis_raw / analysis_parsed / planning_raw / planning_parsed: the question
            analysis and view planning model outputs, carried so s4 can emit the consolidated trace record.
        analysis_error / planning_error (Optional[str]): WHY a step degraded, carried onto
            the trace record.
    """
    analysis: Optional[str] = None
    route: Optional[str] = None
    viewpoint: Optional[ViewpointSpec] = None
    reasoning: Optional[str] = None
    failed: bool = False
    analysis_raw: Optional[str] = None
    analysis_parsed: Optional[Dict[str, Any]] = None
    planning_raw: Optional[str] = None
    planning_parsed: Optional[Dict[str, Any]] = None
    analysis_error: Optional[str] = None
    planning_error: Optional[str] = None

    def to_message_content(self) -> str:
        if self.route == ROUTE_SYNTHESIS and self.viewpoint is not None:
            return (
                f'route: synthesis — view = "{self.viewpoint.magnitude}" '
                f'"{self.viewpoint.pose_family}" from Image {self.viewpoint.reference_view}'
            )
        if self.route is not None:
            return f'route: {self.route}'
        return 'analysis only'


# ----- shared parsing -------------------------------------------------------------
async def parse_planner_output(result: Any) -> Tuple[Dict[str, Any], str]:
    """Parse a reasoner result into (parsed_json, raw_content). Raises on unparseable
    JSON (via `parse_json_str` -> `json.loads`), which is what makes `invoke_with_retry`
    retry with the error fed back."""
    raw_content = result.content
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    parsed, _ = parse_json_str(raw_content)
    return parsed, raw_content


def analysis_str(parsed: Optional[Dict[str, Any]], raw: str) -> str:
    return json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else raw


def format_stage_fields(
    parsed: Optional[Dict[str, Any]],
    raw: Optional[str] = None,
    fields: Optional[List[str]] = None,
) -> str:
    """Format a step's parsed model output as a readable 'key: value' context block for
    the s4 VQA prompt, picking which attributes to expose.

    `fields=None` -> ALL parsed keys (order preserved); a list -> only those keys (order
    preserved, unknown/empty keys skipped); `[]` -> empty string (drop the stage). Falls
    back to `raw` only when there is no parsed dict AND no field filter is given. Nested dict/list values
    are JSON-encoded inline."""
    if not isinstance(parsed, dict) or not parsed:
        return (raw or '').strip() if fields is None else ''
    keys = list(parsed.keys()) if fields is None else [k for k in fields if k in parsed]
    lines = []
    for k in keys:
        v = parsed.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f'{k}: {v}')
    return '\n'.join(lines).strip()
