"""Add shared reconstruction-derived camera relations to the planning prompts."""
from typing import Optional, Sequence, Tuple

from workflow.prompts.s4_solver import GEOMETRY_TITLE

QUALIFIER = ' beyond the camera geometry provided below'

S1_PERMISSION = ('You may rely on these relations between the input images when '
                 'choosing the reference viewpoint and resolving the camera actions in '
                 'the question.')
S2_PERMISSION = ('You may rely on these relations between the input images when '
                 'choosing the reference view and the camera action.')

_QUESTION_ANCHOR = '**[Question]**'


def _geometry_section(permission: str) -> str:
    # `{geometry_facts}` stays a str.format placeholder: the facts text is substituted
    # as a format ARGUMENT, so braces inside it can never break the template.
    return (f'**[{GEOMETRY_TITLE}]**\n{{geometry_facts}}\n{permission}\n\n'
            f'{_QUESTION_ANCHOR}')


# s1: one "DO NOT" bullet forbids cross-view camera relations.
S1_REWRITES: Tuple[Tuple[str, str], ...] = (
    ('- determine relative camera positions between views\n',
     f'- determine relative camera positions between views{QUALIFIER}\n'),
    (_QUESTION_ANCHOR, _geometry_section(S1_PERMISSION)),
)

# s2: the CORE MISSION sentence, the `reasoning` field definition and one "DO NOT"
# bullet forbid cross-view relations. (The `reasoning` anchor is the clause shared by
# the stage-1-fed and the stage-1-free (s2_answer) variants of the field.)
S2_REWRITES: Tuple[Tuple[str, str], ...] = (
    ('never infer relative camera poses or shared layout between input views.',
     f'never infer relative camera poses or shared layout between input views{QUALIFIER}.'),
    ('do not relate input views to each other here.',
     f'do not relate input views to each other here{QUALIFIER}.'),
    ('- infer relative camera positions or any spatial relations between input views\n',
     f'- infer relative camera positions or any spatial relations between input '
     f'views{QUALIFIER}\n'),
    (_QUESTION_ANCHOR, _geometry_section(S2_PERMISSION)),
)


def add_geometry_facts(template: str, rewrites: Sequence[Tuple[str, str]]) -> str:
    """Insert camera relations into the template using exact-match anchors."""
    for old, new in rewrites:
        n = template.count(old)
        if n != 1:
            raise ValueError(
                f'pose_facts_stages=all: anchor {old[:60]!r} occurs {n}x in the '
                'planning template (expected 1) - geometry_facts rewrites out of sync.')
        template = template.replace(old, new)
    return template


def has_facts(geometry_facts: Optional[str]) -> bool:
    return bool((geometry_facts or '').strip())
