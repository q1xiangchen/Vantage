"""Extract explicit final choices from multiple-choice responses."""
import re
import unicodedata
from typing import NamedTuple, Optional

# Choice-token helpers adapted from lmms-eval. See licenses/lmms-eval.txt.
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c26b6b2713c3ec155962ff990f956bf78e0efe45/lmms_eval/tasks/_task_utils/mcq_extract.py
_OPTION_PREFIX = re.compile(r"(?:option|choice)\b\s*[:：]?\s*", re.IGNORECASE)

_LATEX_WRAPPER = re.compile(r"\\(?:text|mathrm|mathbf|textbf)\s*\{")

_ALTERNATIVE = re.compile(r"(?:and\s*/\s*or\b|and\b|or\b|[/&,+|]|或(?:者)?|和|또는|か|と)\s*", re.IGNORECASE)

_WRAPPERS = (("**", "**"), ("__", "__"), ("`", "`"), ("*", "*"), ("_", "_"), ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("(", ")"), ("（", "）"), ("[", "]"), ("{", "}"), ("$", "$"))

_PUNCTUATION = " \t\r\n.!?,;:。！？，；："

_EMPHASIS_CLOSERS = ("**", "__", "*", "_", "`")

class _Choice(NamedTuple):
    letter: str
    tail: str
    wrapped: bool
    uppercase: bool

def _word_character(character: str) -> bool:
    """Include Unicode combining marks and joins in word/identifier boundaries."""
    return character.isalnum() or character == "_" or unicodedata.category(character) in {"Mn", "Mc", "Me", "Cf"}

def _starts_with_alternative(text: str) -> bool:
    """Return whether ``text`` opens with a connector to another choice."""
    body = text.lstrip(" \t")
    connector = _ALTERNATIVE.match(body)
    if connector is None:
        return False
    return body.startswith("/") or _read_choice(body[connector.end() :]) is not None

def _read_choice(fragment: str, grammatical_suffix: str = "") -> Optional[_Choice]:
    """Read one adjacent ASCII choice, with paired presentation wrappers."""
    text = fragment.lstrip().lstrip(":：=,，;；").lstrip()
    closers = []
    option_prefix_used = False
    while text:
        wrapper = _LATEX_WRAPPER.match(text)
        if wrapper:
            closers.append("}")
            text = text[wrapper.end() :].lstrip()
            continue
        prefix = _OPTION_PREFIX.match(text) if not option_prefix_used else None
        if prefix:
            option_prefix_used = True
            text = text[prefix.end() :].lstrip()
            continue
        pair = next((pair for pair in _WRAPPERS if text.startswith(pair[0])), None)
        if pair is None:
            break
        closers.append(pair[1])
        text = text[len(pair[0]) :].lstrip()

    if not text or not text[0].isascii() or not text[0].isalpha():
        return None
    letter, tail = text[0], text[1:]
    # Python's Unicode word boundary also excludes letters inside Chinese text,
    # identifiers and accented words; ASCII-only regex boundaries would not.
    if grammatical_suffix and tail.startswith(grammatical_suffix):
        tail = tail[len(grammatical_suffix) :]
    if tail and _word_character(tail[0]) and not (closers and tail.startswith(closers[-1])):
        return None
    punctuation = ""
    labelled = False
    for closer in reversed(closers):
        separated = bool(tail[:1].isspace())
        tail = tail.lstrip()
        # Punctuation may be inside a presentation wrapper: ``**B.**``.
        inner_punctuation = ".:：!?。！？" + (")" if closer != ")" else "")
        punctuation_end = len(tail) - len(tail.lstrip(inner_punctuation))
        punctuation += tail[:punctuation_end]
        labelled = labelled or any(mark in tail[:punctuation_end] for mark in ".:：)")
        remainder = tail[punctuation_end:]
        separated = separated or bool(remainder[:1].isspace())
        tail = remainder.lstrip()
        if not tail.startswith(closer):
            # Emphasis may wrap a punctuated label with its option text:
            # ``**B. dog**``, ``**(C) Red**``. Whitespace is required, so
            # ``**B.dog**`` stays a non-label. An alternative glued to the
            # label (``**B. or C**``, ``**(A) or (B)**``) still abstains.
            line, separator, rest = tail.partition("\n")
            closer_at = line.find(closer)
            skipped = line[:closer_at]
            if closer not in _EMPHASIS_CLOSERS or not labelled or not separated or closer_at <= 0 or _starts_with_alternative(skipped):
                return None
            tail = line[closer_at:] + separator + rest
        tail = tail[len(closer) :]
        labelled = labelled or closer == ")"
    tail = punctuation + tail
    if tail and _word_character(tail[0]):
        return None
    if tail and tail[0] in ".:-/'’" and len(tail) > 1 and (tail[1].isalnum() or tail[1] in "/_"):
        # ``B.com``, ``C:/path``, ``A-list`` and ``B's`` are not labels.
        # A slash followed by a single choice is handled as an alternative below.
        if tail[0] != "/":
            return None
    return _Choice(letter.upper(), tail, bool(closers), letter.isupper())

def _has_alternative(choice: _Choice) -> bool:
    """Reject an adjacent second selection, not letters in an explanation."""
    tail = choice.tail.lstrip(" \t")
    label_punctuation = tail[:1] in (".", ":", ")", "：")
    if label_punctuation:
        tail = tail[1:].lstrip(" \t")
    connector = _ALTERNATIVE.match(tail)
    if connector:
        return tail.startswith("/") or _read_choice(tail[connector.end() :]) is not None
    # A compact sequence ``B C`` is ambiguous. A separate later implicit line
    # does not invalidate an earlier explicit declaration.
    if label_punctuation or tail.startswith(("\n", "\r")):
        return False
    other = _read_choice(tail)
    return other is not None and not other.tail.strip(_PUNCTUATION)

def _closing_brace(text: str, start: int) -> Optional[int]:
    """Find a box's closing brace, allowing balanced presentation wrappers."""
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_answer(response, letters='ABCDEFGHYN'):
    if not isinstance(response, str) or not response.strip():
        return ''
    # Do not search private reasoning or generated programs for choice letters.
    text = re.sub(r'<think\b[^>]*>.*?(?:</think\s*>|\Z)', '', response, flags=re.I | re.S)
    def fence(m):
        body = m.group().split('\n', 1)[1].rsplit(m.group()[:3], 1)[0].strip()
        return '\n' + body + '\n' if re.fullmatch(r'[A-Z]', body) and body in letters else ''
    text = re.sub(r'```[^\n]*\n.*?(?:```|\Z)|~~~[^\n]*\n.*?(?:~~~|\Z)', fence, text, flags=re.S)
    # Markdown emphasis does not change answer semantics.
    text = text.replace('**', '').replace('__', '')
    # Ignore non-option inline code, retaining only single-letter answer wrappers.
    text = re.sub(r'`{1,2}([^`\n]+)`{1,2}', lambda m: m.group() if re.fullmatch(r'\s*[A-Z]\s*', m[1]) else '', text)
    candidates = []

    def choice(body):
        c = _read_choice(body)
        if not c or c.letter not in letters or _has_alternative(c):
            return ''
        # Ordinary prose such as "a chair" is not an answer label.
        tail = c.tail
        if not c.wrapped and tail.strip() and tail[:1].isspace() and tail.lstrip()[:1].isalpha():
            return ''
        return c.letter

    # Answer containers establish a scope, including an invalid/unfinished final
    # container: do not silently recover an earlier answer from such a response.
    for m in re.finditer(r'\\boxed\s*\{|<answer\b[^>]*>', text, re.I):
        if m.group().startswith('\\'):
            end = _closing_brace(text, m.end())
        else:
            close = re.search(r'</answer\s*>', text[m.end():], re.I)
            end = m.end() + close.start() if close else None
        candidates.append((m.start(), choice(text[m.end():end]) if end is not None else ''))
    # Angle-delimited option labels emitted under MindCube's answer format.
    for m in re.finditer(r'<([A-Z][.):]\s+[^<>\n]+)(?:>|(?=</))', text):
        candidates.append((m.start(), choice(m[1])))
    for m in re.finditer(r'<([A-Z])>', text):
        if m[1] in letters:
            candidates.append((m.start(), m[1]))
    # Explicit declarations across formats are ordered by their position.
    for m in re.finditer(r'\b(?:(?:the|my)\s+)?(?:(?:final|correct|best)\s+)?(?:answer|option|choice)\s*(?:is\b\s*:?|:)\s*|\bI\s+(?:choose|select|pick)\s+', text, re.I):
        candidates.append((m.start(), choice(text[m.end():].split('\n', 1)[0])))
    # Only exact one-letter wrappers; image identifiers and formula subscripts
    # are not candidates. Backtick spans with other contents are ignored.
    for pattern in (r'(?<!`)`{1,2}\s*([A-Z])\s*`{1,2}(?!`)',
                    r'(?<![\w\\_^])\{\s*([A-Z])\s*\}'):
        for m in re.finditer(pattern, text):
            if m[1] in letters:
                candidates.append((m.start(), m[1]))
    # An isolated answer paragraph may be followed by an explanation. Do not
    # interpret a multi-line list of options as an answer paragraph.
    offset = 0
    for block in re.split(r'(\n[ \t]*\n)', text):
        body = block.strip()
        c = _read_choice(body) if body and '\n' not in body else None
        if c and (not c.tail.strip(' .!') or c.wrapped or c.tail.startswith(('.', ')', ':'))):
            candidates.append((offset, choice(body)))
        offset += len(block)
    # A terminal standalone option or punctuated option label is final evidence.
    stripped = text.rstrip()
    last_start = stripped.rfind('\n') + 1
    if stripped:
        body = stripped[last_start:].strip()
        c = _read_choice(body)
        labels = re.findall(r'^\s*\(?([A-Z])[.):]\s', text.rsplit('\n\n', 1)[-1], re.M)
        if c and (not c.tail.strip(' .!') or ((c.wrapped or c.tail.startswith(('.', ')', ':'))) and len(set(labels)) <= 1)):
            candidates.append((last_start, choice(body)))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    # A response opening with an explicit option label may explain the choice;
    # a bare prefix of a word (Based/Answer) or an option list is never accepted.
    first = text.strip().split('\n', 1)[0]
    c = _read_choice(first)
    if c and not c.tail.strip(' .!'):
        return choice(first)
    c = _read_choice(text.strip())
    if c and c.tail.startswith(('.', ')', ':')):
        labels = re.findall(r'^\s*\(?([A-Z])[.):]\s', text.rsplit('\n\n', 1)[-1], re.M)
        if len(set(labels)) <= 1:
            return choice(text.strip())
    return ''
