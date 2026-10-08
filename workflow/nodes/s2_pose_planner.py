"""s2_pose_planner: stage-1 view planning — plan the view to seek from the question
analysis and ground it to a 6-DoF pose.

Planning only — never answers. Updates the workspace `PlannerOutput` in place with the
route and the grounded view; the reasoning guidance rides in `planning_parsed`.
"""

import functools
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage
from PIL import Image
from ray.serve.handle import DeploymentHandle

from tools.apis.pose_utils import MAGNITUDES, ground_viewpoint
from workflow.logging import AgentLogger
from workflow.nodes.planner_output import (
    PlannerOutput,
    ROUTE_S2_ERROR_SKIP,
    ROUTE_SYNTHESIS,
    parse_planner_output,
)
from workflow.nodes.s1_analyst import images_in
from workflow.prompts.s2_pose_planner import build_pose_planner_prompt
from workflow.state import AgentState
from tools.utils.llm_invoke import invoke_with_retry


class S2PosePlanner:
    """Stage-1 view planning: plan the view to seek and ground it."""

    def __init__(self, reasoner: DeploymentHandle, logger: AgentLogger):
        self.reasoner = reasoner
        self.logger = logger

    async def run(self, state: AgentState) -> Dict[str, Any]:
        workspace = state.get('workspace', {})
        po: PlannerOutput = workspace['planner_output']
        images = images_in(workspace)
        instruction = workspace['instruction'].text

        try:
            s2_parsed, s2_raw = await self._call_pose_planner(
                instruction, po.analysis or '', images)
        except Exception as e:  # noqa: BLE001 - degrade to answer-from-analysis
            po.route = ROUTE_S2_ERROR_SKIP
            po.planning_error = f'{type(e).__name__}: {e}'
            msg = AIMessage(f'[s2_pose_planner] failed -> answer from analysis: {e}')
            return {'workspace': {**workspace, 'planner_output': po}, 'messages': [msg]}

        po.planning_raw = s2_raw
        po.planning_parsed = s2_parsed
        self._apply_view_plan(po, s2_parsed, num_views=len(images))
        msg = AIMessage(f'[s2_pose_planner] {po.to_message_content()}')
        return {'workspace': {**workspace, 'planner_output': po}, 'messages': [msg]}

    async def _call_pose_planner(
        self, instruction: str, analysis: str, images: List[Image.Image],
    ) -> Tuple[Dict[str, Any], str]:
        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=images if images else None,
        )
        prompter = functools.partial(build_pose_planner_prompt, instruction, analysis)
        return await invoke_with_retry(
            invoker=invoker, prompter=prompter,
            parser=parse_planner_output, max_retries=3,
        )

    def _apply_view_plan(
        self, po: PlannerOutput, parsed: Dict[str, Any], num_views: int,
    ) -> None:
        """Fill `po` with the grounded view plan. Falls back to ROUTE_S2_ERROR_SKIP if
        the parsed fields are inconsistent (e.g. no reasoning guidance)."""
        reasoning = parsed.get('reasoning')
        try:
            pose_family = str(parsed.get('pose') or '').strip()
            magnitude = str(parsed.get('magnitude') or '').strip() or MAGNITUDES[-1]
            reference_view = self._parse_reference_view(parsed, num_views)
            guidance = str(parsed.get('reasoning guidance') or '').strip()
            if not guidance:
                raise ValueError("a view plan requires a non-empty 'reasoning guidance'.")
            viewpoint = ground_viewpoint(
                pose_family, magnitude, reference_view=reference_view,
                reasoning=reasoning or '', num_views=num_views or None)
        except Exception as e:  # noqa: BLE001 - inconsistent plan -> skip (answer from analysis)
            po.route = ROUTE_S2_ERROR_SKIP
            po.reasoning = reasoning
            po.planning_error = f'inconsistent view plan - {type(e).__name__}: {e}'
            return
        po.route = ROUTE_SYNTHESIS
        po.viewpoint = viewpoint
        po.reasoning = reasoning

    @staticmethod
    def _parse_reference_view(parsed: Dict[str, Any], num_views: int = None) -> int:
        """Parse the planner's 1-indexed reference view. Range validation is left to
        ground_viewpoint.

        With exactly one input view the field has a single legal value, so a
        digitless answer (e.g. a compass bearing) is coerced to 1 rather than
        failing the plan."""
        m = re.search(r'\d+', str(parsed.get('reference_view')))
        if m is None:
            if num_views == 1:
                return 1
            raise ValueError(
                f'reference_view={parsed.get("reference_view")!r} is invalid. '
                'Expected a 1-indexed integer.')
        return int(m.group())
