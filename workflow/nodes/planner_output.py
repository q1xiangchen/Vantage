"""Represent and parse outputs from the planning stages."""

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
    """Store intermediate analysis and the grounded view plan."""
    analysis: Optional[str] = None
    route: Optional[str] = None
    viewpoint: Optional[ViewpointSpec] = None
    reasoning: Optional[str] = None
    failed: bool = False
    reference_view_repair: Optional[Dict[str, Any]] = None
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
    """Parse the model’s planning JSON and retain its raw response."""
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
    """Format selected intermediate fields for the next prompt."""
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
