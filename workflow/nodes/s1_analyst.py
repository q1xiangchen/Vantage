"""s1_analyst: stage-1 question analysis.

Planning only — never answers. Writes a `PlannerOutput` into the workspace; s4_solver
reads it. In baseline mode it emits an empty plan so s4 does a bare direct VQA.
"""

import functools
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from PIL import Image
from ray.serve.handle import DeploymentHandle

from workflow.config import MODE_BASELINE, get_config
from workflow.logging import AgentLogger
from workflow.nodes.planner_output import (
    PlannerOutput,
    ROUTE_S1_ERROR_SKIP,
    analysis_str,
    parse_planner_output,
)
from workflow.prompts.s1_analyst import build_s1_analyst_prompt
from workflow.state import AgentState
from tools.utils.llm_invoke import invoke_with_retry


def images_in(workspace: Dict[str, Any]) -> List[Image.Image]:
    """The input views from the workspace (shared by s1 and s2)."""
    if 'input_images' in workspace:
        return list(workspace['input_images'].images)
    return []


class S1Analyst:
    """Stage-1 question analysis."""

    def __init__(self, reasoner: DeploymentHandle, logger: AgentLogger):
        self.reasoner = reasoner
        self.logger = logger

    async def run(self, state: AgentState) -> Dict[str, Any]:
        workspace = state.get('workspace', {})

        if get_config().mode == MODE_BASELINE:
            return self._emit(
                workspace, PlannerOutput(),
                '[s1_analyst] baseline mode -> bare VQA (analysis skipped)')

        images = images_in(workspace)
        try:
            parsed, raw = await self._call_analyst(workspace, images)
        except Exception as e:  # noqa: BLE001 - degrade to bare VQA on any failure
            return self._emit(
                workspace,
                PlannerOutput(failed=True, route=ROUTE_S1_ERROR_SKIP,
                              analysis_error=f'{type(e).__name__}: {e}'),
                f'[s1_analyst] analysis failed -> bare VQA: {e}')
        po = PlannerOutput(
            analysis=analysis_str(parsed, raw), analysis_raw=raw, analysis_parsed=parsed)
        return self._emit(workspace, po, '[s1_analyst] produced question analysis')

    @staticmethod
    def _emit(
        workspace: Dict[str, Any], po: PlannerOutput, log: Optional[str],
    ) -> Dict[str, Any]:
        msgs = [AIMessage(log)] if log else []
        return {'workspace': {**workspace, 'planner_output': po}, 'messages': msgs}

    async def _call_analyst(
        self, workspace: Dict[str, Any], images: List[Image.Image],
    ) -> Tuple[Dict[str, Any], str]:
        instruction = workspace['instruction'].text
        invoker = functools.partial(
            self.reasoner.cot_reason.remote,
            input_images=images if images else None,
        )
        prompter = functools.partial(build_s1_analyst_prompt, instruction)
        return await invoke_with_retry(
            invoker=invoker, prompter=prompter,
            parser=parse_planner_output, max_retries=3,
        )
