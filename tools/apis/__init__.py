from .base import (
    AgentTool,
    AgentToolOutput,
    AgentContext,
    AGENT_CONTEXT_REGISTRY,
    InputBBoxes2D,
    InputImages,
    Instruction,
)
from .cot_reasoner import CoTReasoner, CoTReasonerOutput
from .final_answer import FinalAnswer
from .io import ImageLoader, ImageBase64Encoder
from .view_synthesizer import ViewSynthesizer, SynthesizedView


AGENT_TOOL_REGISTRY = {
    'CoTReasoner': CoTReasoner,
    'ImageLoader': ImageLoader,
    'ImageBase64Encoder': ImageBase64Encoder,
    'ViewSynthesizer': ViewSynthesizer,
}
