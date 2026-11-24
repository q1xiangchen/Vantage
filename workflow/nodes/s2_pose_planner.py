"""Plan a camera move to reveal missing evidence."""

import functools
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage
from PIL import Image
from ray.serve.handle import DeploymentHandle

from tools.apis.pose_utils import MAGNITUDES, ground_viewpoint
from workflow.utils.reference_view import resolve_reference_view_text
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
                instruction, po.analysis or '', images,
                geometry_facts=(workspace.get('pose_facts') or {}).get('text', ''))
        except Exception as e:  # noqa: BLE001 - degrade to answer-from-analysis
            po.route = ROUTE_S2_ERROR_SKIP
            po.planning_error = f'{type(e).__name__}: {e}'
            msg = AIMessage(f'[s2_pose_planner] failed -> answer from analysis: {e}')
            return {'workspace': {**workspace, 'planner_output': po}, 'messages': [msg]}

        po.planning_raw = s2_raw
        po.planning_parsed = s2_parsed
        self._apply_view_plan(po, s2_parsed, num_views=len(images),
            allowed_reference_views=[i+1 for i in workspace['geometry_image_indices']] if 'geometry_image_indices' in workspace else None)
        msg = AIMessage(f'[s2_pose_planner] {po.to_message_content()}')
        return {'workspace': {**workspace, 'planner_output': po}, 'messages': [msg]}

    async def _call_pose_planner(
        self, instruction: str, analysis: str, images: List[Image.Image],
        geometry_facts: str = '',
    ) -> Tuple[Dict[str, Any], str]:
        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=images if images else None,
        )
        prompter = functools.partial(
            build_pose_planner_prompt, instruction, analysis, geometry_facts=geometry_facts)
        return await invoke_with_retry(
            invoker=invoker, prompter=prompter,
            parser=parse_planner_output, max_retries=3,
        )

    def _apply_view_plan(
        self, po: PlannerOutput, parsed: Dict[str, Any], num_views: int, allowed_reference_views=None,
    ) -> None:
        """Ground a valid view plan or route an invalid plan to answer fallback."""
        reasoning = parsed.get('reasoning')
        try:
            pose_family = str(parsed.get('pose') or '').strip()
            magnitude = str(parsed.get('magnitude') or '').strip() or MAGNITUDES[-1]
            reference_view = self._resolve_reference_view(po, parsed, num_views)
            if allowed_reference_views is not None and reference_view not in allowed_reference_views:
                raise ValueError('Answer-candidate image cannot be the reconstruction reference')
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

    def _resolve_reference_view(self, po, parsed, num_views):
        """Preserve valid stage-2 indices; only failed resolution consults stage 1."""
        def resolve(value):
            try:
                ref = self._parse_reference_view({'reference_view': value}, num_views)
                if 1 <= ref <= num_views:
                    return ref, 'strict'
            except ValueError:
                pass
            ref = resolve_reference_view_text(value, num_views)
            return ref, 'text'

        ref, how = resolve(parsed.get('reference_view'))
        if ref is not None:
            if how != 'strict':
                po.reference_view_repair = {'source': 's2', 'resolution': how, 'resolved': ref}
            return ref
        analysis = po.analysis_parsed or {}
        value = analysis.get('reference_view', analysis.get('reference_viewpoint'))
        ref, how = resolve(value)
        if ref is None:
            raise ValueError('Neither stage 2 nor stage 1 names a valid reference view')
        po.reference_view_repair = {'source': 's1', 'resolution': how,
                                   's2_value': parsed.get('reference_view'),
                                   's1_value': value, 'resolved': ref}
        return ref

    @staticmethod
    def _parse_reference_view(parsed: Dict[str, Any], num_views: int = None) -> int:
        """Read the planner’s 1-indexed reference-view number."""
        m = re.search(r'\d+', str(parsed.get('reference_view')))
        if m is None:
            if num_views == 1:
                return 1
            raise ValueError(
                f'reference_view={parsed.get("reference_view")!r} is invalid. '
                'Expected a 1-indexed integer.')
        return int(m.group())
