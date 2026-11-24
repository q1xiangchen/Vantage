"""Resolve textual references to an unambiguous input view."""
import re
from typing import Any, Optional

_ORDINALS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
    'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5, '6th': 6, '7th': 7, '8th': 8,
    '9th': 9, '10th': 10,
}
_NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10,
}
_VIEW_NOUN = r'(?:image|img|view|viewpoint|photo|photograph|picture|frame|figure)'
_ORD_RE = re.compile(
    r'\b(' + '|'.join(_ORDINALS) + r')[\s_-]+(?:input[\s_-]+)?' + _VIEW_NOUN + r'\b')
_NOUN_NUM_RE = re.compile(
    r'\b' + _VIEW_NOUN + r'[\s_]*(?:#|no\.?|number)?[\s_]*(\d+|'
    + '|'.join(_NUMBER_WORDS) + r')\b')
# The on-image stamp is `input_images.images[k]`, 1-indexed (tools/apis/cot_reasoner.py).
_STAMP_RE = re.compile(r'images\s*\[\s*(\d+)\s*\]')


def resolve_reference_view_text(value: Any, num_views: Optional[int]) -> Optional[int]:
    """Resolve an unambiguous textual reference to a 1-indexed view."""
    text = str(value if value is not None else '').strip().lower()
    if not text:
        return None
    found = set()
    for m in _ORD_RE.finditer(text):
        found.add(_ORDINALS[m.group(1)])
    for m in _NOUN_NUM_RE.finditer(text):
        tok = m.group(1)
        found.add(int(tok) if tok.isdigit() else _NUMBER_WORDS[tok])
    for m in _STAMP_RE.finditer(text):
        found.add(int(m.group(1)))   # images[0] -> 0 -> invalid -> no repair
    if len(found) != 1:
        return None
    k = found.pop()
    if k < 1 or (num_views is not None and k > num_views):
        return None
    return k
