"""Prompts for the s4_solver node (stage-2 Final VQA).

Three answer prompts, one per s4 shape:
  - build_vqa_prompt            : bare direct VQA (baseline / s1 failure)
  - build_followup_answer_prompt: answer the original q with the stage-1 context
                                  (view-plan or view-synthesis failure)
  - build_synthesis_answer_prompt: answer over the original views + the synthesized view,
                                  with the question analysis and reasoning guidance.
"""

from typing import Optional


# --- Final VQA ----------------------------------------------------------------
# The question is answered directly by the frozen VLM, followed by the benchmark's
# answer-format rule (data_specific_prompt).
def build_vqa_prompt(question: str, answer_format: str = '') -> str:
    if answer_format:
        return f'{question}\n\n{answer_format}'
    return question


# --- Answer over the synthesized view -------------------------------------------
# We hand the model BOTH the original input views AND the synthesized view (labeled),
# plus the stage-1 context blocks: the question analysis and the view plan's
# reasoning guidance (see *_CONTEXT_FIELDS in workflow/config.py). The model then
# answers the ORIGINAL question over everything. The synthesized view is supporting
# evidence, not ground truth (it may carry synthesis artifacts), so the model is told
# to cross-check it against the originals.
SYNTHESIS_ANSWER_PROMPT = """
You are answering a spatial question about a scene. You are given the original
input view(s) plus ONE additional synthesized view, rendered from a new camera
viewpoint to expose evidence that was hard to read in the originals.

**[Images]**
- Image 1{originals_range}: the original input view(s).
- Image {synth_index}: a SYNTHESIZED informative view, produced by moving the camera
  ("{magnitude}" "{pose_family}") starting from input Image {reference_view}. It may
  contain rendering artifacts — treat it as supporting evidence, not ground truth,
  and cross-check it against the original views.
{analysis_block}
{planning_block}
Use all the images together to answer the original question below.

**[Question]**
{instruction}

{answer_format}
""".strip()


def _section(title: str, body: Optional[str]) -> str:
    """Format an optional labeled section, or '' when the body is empty."""
    body = (body or '').strip()
    if not body:
        return ''
    return f'\n**[{title}]**\n{body}\n'


def build_synthesis_answer_prompt(
    instruction: str,
    pose_family: str,
    magnitude: str,
    reference_view: int,
    num_original: int,
    analysis_context: Optional[str] = None,
    guidance_context: Optional[str] = None,
    answer_format: str = '',
) -> str:
    """Build the answer prompt over the originals + the synthesized view, which is
    appended LAST, so its 1-indexed position is `num_original + 1`.
    `analysis_context` (question analysis) and `guidance_context` (reasoning
    guidance) are omitted when empty."""
    synth_index = num_original + 1
    originals_range = f'-{num_original}' if num_original > 1 else ''
    return SYNTHESIS_ANSWER_PROMPT.format(
        originals_range=originals_range,
        synth_index=synth_index,
        magnitude=magnitude,
        pose_family=pose_family,
        reference_view=reference_view,
        analysis_block=_section('Prior viewpoint analysis', analysis_context),
        planning_block=_section('Viewpoint planning', guidance_context),
        instruction=instruction,
        answer_format=answer_format or '',
    ).strip()


# --- Answer-from-analysis (view-plan skip / failure) ---------------------------
# Follow-up "now answer" turn: the model first produced the structured question
# analysis (prompts/s1_analyst.py); we feed that analysis back as its own prior work and
# ask it to commit to an answer over the SAME input views. For the stateless reasoner
# this single call is equivalent to a real 2-turn exchange.
FOLLOWUP_ANSWER_PROMPT = """
You previously analyzed this question and produced the following structured
viewpoint analysis:
{analysis}
{planning_block}
Now, using that analysis together with the input images, answer the original question.

**[Question]**
{instruction}

{answer_format}
""".strip()


def build_followup_answer_prompt(
    instruction: str,
    analysis_context: str,
    guidance_context: Optional[str] = None,
    answer_format: str = '',
) -> str:
    """Answer-from-analysis prompt. `analysis_context` is the question analysis fed
    back as prior work; `guidance_context` is the optional reasoning guidance, present
    only when view planning produced output."""
    return FOLLOWUP_ANSWER_PROMPT.format(
        instruction=instruction,
        analysis=analysis_context,
        planning_block=_section('Viewpoint planning', guidance_context),
        answer_format=answer_format or '',
    ).strip()
