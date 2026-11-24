"""Build prompts for final visual question answering."""

from typing import Optional

GEOMETRY_TITLE = 'Camera geometry of the input images (from 3D reconstruction)'


# --- Final VQA ----------------------------------------------------------------
# The question is answered directly by the frozen VLM, followed by the benchmark's
# answer-format rule (data_specific_prompt).
def build_vqa_prompt(question: str, answer_format: str = '', geometry_facts: str = '') -> str:
    if geometry_facts.strip():
        question = _section(GEOMETRY_TITLE, geometry_facts).strip() + '\n\n' + question
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
    geometry_facts: str = '',
) -> str:
    """Build final VQA over the input views and the appended synthesized view."""
    synth_index = num_original + 1
    originals_range = f'-{num_original}' if num_original > 1 else ''
    template = SYNTHESIS_ANSWER_PROMPT
    if geometry_facts.strip():
        template = template.replace('{analysis_block}', '{geometry_block}{analysis_block}', 1)
    return template.format(
        geometry_block=_section(GEOMETRY_TITLE, geometry_facts),
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
    geometry_facts: str = '',
) -> str:
    """Build fallback VQA from the input views and available planning context."""
    template = FOLLOWUP_ANSWER_PROMPT
    if geometry_facts.strip():
        template = template.replace('{planning_block}', '{planning_block}{geometry_block}', 1)
    return template.format(
        geometry_block=_section(GEOMETRY_TITLE, geometry_facts),
        instruction=instruction,
        analysis=analysis_context,
        planning_block=_section('Viewpoint planning', guidance_context),
        answer_format=answer_format or '',
    ).strip()
