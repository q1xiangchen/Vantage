"""Synthesize the planned view and answer the question."""

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from PIL import Image
from ray.serve.handle import DeploymentHandle

from tools.apis import FinalAnswer
from tools.apis.pose_utils import ORBIT_FAMILIES, ELEVATION_FAMILIES
from tools.utils.llm_invoke import is_truncation_error
from workflow.config import (
    ANALYSIS_CONTEXT_FIELDS,
    PLANNING_CONTEXT_FIELDS,
    get_config,
)
from workflow.logging import AgentLogger, build_trace_record
from workflow.nodes.planner_output import (
    PlannerOutput,
    ROUTE_SYNTHESIS,
    format_stage_fields,
)
from workflow.prompts.s4_solver import (
    build_followup_answer_prompt,
    build_synthesis_answer_prompt,
    build_vqa_prompt,
)
from workflow.state import AgentState
from workflow.utils.pose_guide import forward_key


class S4Solver:
    """Stage 2: view synthesis + final VQA (see module docstring)."""

    def __init__(
        self,
        reasoner: DeploymentHandle,
        logger: AgentLogger,
        view_synthesizer: Optional[DeploymentHandle] = None,
    ) -> None:
        self.reasoner = reasoner                  # final VQA
        self.logger = logger
        self.view_synthesizer = view_synthesizer  # view synthesis; None = disabled

    async def run(self, state: AgentState) -> Dict[str, Any]:
        workspace = state.get('workspace', {})
        session_id = state['session_id']
        instruction: str = workspace['instruction'].text
        answer_format: str = workspace['instruction'].answer_format
        images = self._images(workspace)
        po: PlannerOutput = workspace.get('planner_output') or PlannerOutput()

        log_msgs: List[AIMessage] = []
        new_view: Optional[Image.Image] = None
        synthesis_error: Optional[str] = None
        synthesis_diag: Dict[str, Any] = {}

        analysis_ctx = format_stage_fields(
            po.analysis_parsed, po.analysis_raw, ANALYSIS_CONTEXT_FIELDS)
        guidance_ctx = format_stage_fields(
            po.planning_parsed, po.planning_raw, PLANNING_CONTEXT_FIELDS)

        facts = workspace.get('pose_facts') or {}
        geometry_facts = facts.get('text') or ''

        if po.route == ROUTE_SYNTHESIS and po.viewpoint is not None:
            new_view, synthesis_error, synthesis_diag = await self._synthesize(
                po.viewpoint, images, log_msgs, forward_key(session_id),
                geometry_indices=workspace.get('geometry_image_indices'))

        answer_images = images
        if new_view is not None:
            # (c) answer over the originals + the synthesized view.
            spec = po.viewpoint
            answer_images = images + [new_view]
            prompt = build_synthesis_answer_prompt(
                instruction=instruction,
                pose_family=spec.pose_family,
                magnitude=spec.magnitude,
                reference_view=spec.reference_view,
                num_original=len(images),
                analysis_context=analysis_ctx,
                guidance_context=guidance_ctx,
                answer_format=answer_format, geometry_facts=geometry_facts,
            )
        elif po.analysis is not None:
            # (b) answer the original question with the available stage context.
            if po.route == ROUTE_SYNTHESIS:
                log_msgs.append(AIMessage(
                    f'[s4_solver] view synthesis unavailable/failed -> answering from '
                    f'analysis ({synthesis_error})'))
            prompt = build_followup_answer_prompt(
                instruction, analysis_ctx, guidance_ctx, answer_format, geometry_facts)
        else:
            # (a) bare direct VQA.
            prompt = build_vqa_prompt(instruction, answer_format, geometry_facts)

        # A max_tokens cut-off has no answer to salvage and is not retried: degrade to
        # an empty answer and carry the reason on the trace record.
        vqa_error: Optional[str] = None
        try:
            answer_text, answer_thinking = await self._run(prompt, answer_images)
        except RuntimeError as e:
            if not is_truncation_error(str(e)):
                raise
            answer_text, answer_thinking = '', None
            vqa_error = f'{type(e).__name__}: {e}'
            log_msgs.append(AIMessage(f'[s4_solver] VQA truncated -> empty answer: {e}'))
        log_msgs.append(AIMessage(f'[s4_solver] answer: {answer_text}'))

        record = self._build_record(
            po, instruction, answer_text, prompt, new_view, answer_thinking,
            synthesis_error=synthesis_error, synthesis_diag=synthesis_diag, vqa_error=vqa_error)
        if 'geometry_image_indices' in workspace:
            record['geometry_image_indices'] = workspace['geometry_image_indices']
        if 'pose_facts' in workspace:
            record['pose_facts'] = facts
        if po.reference_view_repair:
            record['reference_view_repair'] = po.reference_view_repair
        if forward_key(session_id) and self.view_synthesizer is not None:
            try:
                await self.view_synthesizer.release_forward.remote(forward_key(session_id))
            except Exception as e:
                # Cleanup failure must not discard an already sampled VQA answer.
                record['pose_cache_release_error'] = f'{type(e).__name__}: {e}'
        return self._finish(
            workspace, session_id, answer_text, record, log_msgs, synth_image=new_view)

    # ----- record -------------------------------------------------------------
    def _build_record(
        self,
        po: PlannerOutput,
        instruction: str,
        answer_text: str,
        prompt: str,
        new_view: Optional[Image.Image],
        answer_thinking: Optional[str] = None,
        synthesis_error: Optional[str] = None,
        synthesis_diag: Optional[Dict[str, Any]] = None,
        vqa_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = po.viewpoint
        return build_trace_record(
            mode=get_config().mode,
            route=po.route or 'fallback',
            origin_question=instruction,
            analysis_raw=po.analysis_raw,
            analysis_parsed=po.analysis_parsed,
            planning_raw=po.planning_raw,
            planning_parsed=po.planning_parsed,
            viewpoint={
                'pose': spec.pose_family,
                'magnitude': spec.magnitude,
                'reference_view': spec.reference_view,
                'dof': spec.dof,
                'target_pose': spec.target_pose,
            } if spec is not None else None,
            synthesized=new_view is not None,
            synthesis_error=synthesis_error,
            analysis_error=po.analysis_error,
            planning_error=po.planning_error,
            answer=answer_text,
            answer_prompt=prompt,
            answer_thinking=answer_thinking,
            synthesis_diag=synthesis_diag or {},
            vqa_error=vqa_error,
        )

    # ----- helpers ------------------------------------------------------------
    @staticmethod
    def _images(workspace: Dict[str, Any]) -> List[Image.Image]:
        if 'input_images' in workspace:
            return list(workspace['input_images'].images)
        return []

    async def _synthesize(
        self,
        spec,
        images: List[Image.Image],
        log_msgs: List[AIMessage],
        shared_key: str = '',
        geometry_indices=None,
    ) -> tuple[Optional[Image.Image], Optional[str], Dict[str, Any]]:
        """Render the planned view and return the image or failure details."""
        if self.view_synthesizer is None:
            return None, 'view synthesizer unavailable (not deployed)', {}
        if spec is None:
            return None, 'no grounded viewpoint spec', {}
        if not images:
            return None, 'no input images', {}
        from workflow.utils.geometry_inputs import geometry_view
        try:
            images, spec = geometry_view(images, spec, geometry_indices)
        except ValueError as exc:
            return None, str(exc), {}
        anchored_images = images if shared_key else self._anchor_first(images, spec.reference_view)
        if spec.pose_family in ORBIT_FAMILIES:
            pivot_mode = 'centroid'
        elif spec.pose_family in ELEVATION_FAMILIES:
            pivot_mode = 'elevation'
        else:
            pivot_mode = 'ahead'
        synthesized = await self.view_synthesizer.synthesize.remote(
            input_images=anchored_images, target_pose=spec.target_pose,
            pivot_mode=pivot_mode,
            anchor_index=spec.reference_view - 1 if shared_key else 0,
            forward_key=shared_key,
        )
        if synthesized.err is not None:
            return None, synthesized.err['msg'], {}
        if synthesized.result is None:
            return None, 'view synthesis returned no image', {}
        r = synthesized.result
        log_msgs.append(AIMessage(
            f'[s4_solver] synthesized 1 view (reference_view={spec.reference_view})'))
        return r.image, None, getattr(r, 'diag', {}) or {}

    @staticmethod
    def _anchor_first(
        images: List[Image.Image], reference_view: int,
    ) -> List[Image.Image]:
        """Move the selected reference image to the front."""
        idx = reference_view - 1
        if idx <= 0 or idx >= len(images):
            return images
        return [images[idx]] + images[:idx] + images[idx + 1:]

    async def _run(
        self, prompt: str, images: List[Image.Image], max_retries: int = 3,
    ) -> tuple[str, Optional[str]]:
        """Send a prepared prompt and image set to the vision-language model."""
        err_msg = None
        for i in range(max_retries + 1):
            out = await self.reasoner.cot_reason.remote(
                prompt=prompt, input_images=images if images else None,
            )
            if out.err is None:
                return out.result.content, out.result.reasoning_content
            err_msg = out.err['msg']
            if is_truncation_error(err_msg):
                raise RuntimeError(err_msg)
            if i < max_retries:
                await asyncio.sleep(0.5)
        raise RuntimeError(err_msg)

    def _finish(
        self,
        workspace: Dict[str, Any],
        session_id: str,
        answer_text: str,
        record: Dict[str, Any],
        log_msgs: List[AIMessage],
        synth_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        final_answer = FinalAnswer(
            result=answer_text, natural_language_summary=answer_text)
        if get_config().enable_logging:
            self.logger.log_trace(session_id, record, synth_image=synth_image)
        updated_workspace = {
            **workspace, 'final_answer': final_answer, 'trace_record': record}
        return {'workspace': updated_workspace, 'messages': log_msgs}
