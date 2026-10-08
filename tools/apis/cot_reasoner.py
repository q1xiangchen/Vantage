import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
from ray import serve

from tools.apis.base import AgentTool, AgentToolOutput, AgentContext
from tools.apis.io import ImageBase64Encoder
from tools.llm_client import LLMClientFactory
from tools.utils.llm_invoke import TRUNCATION_MARKER
from tools.utils.mm_utils import add_label_to_image
from workflow.config import get_config

__ALL__ = ['CoTReasoner', 'CoTReasonerOutput']


# Sampling knobs that are NOT part of the OpenAI chat-completions schema. The `openai`
# SDK rejects them as unexpected kwargs, so they ride in `extra_body`, where vLLM
# picks them up. Everything else (temperature, top_p, presence_penalty, ...) is
# OpenAI-standard and stays a top-level kwarg.
NON_OPENAI_SAMPLING_PARAMS = ('top_k', 'min_p', 'repetition_penalty')


def split_sampling_kwargs(sampling: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return `sampling` with every non-OpenAI param moved under `extra_body`."""
    kwargs = dict(sampling or {})
    extra_body = dict(kwargs.pop('extra_body', None) or {})
    for name in NON_OPENAI_SAMPLING_PARAMS:
        if name in kwargs:
            value = kwargs.pop(name)
            if value is not None:
                extra_body[name] = value
    if extra_body:
        kwargs['extra_body'] = extra_body
    return kwargs


def uses_max_completion_tokens(model: str) -> bool:
    """OpenAI GPT / o-series models count output under `max_completion_tokens`."""
    model = model.lower()
    return 'gpt' in model or model.startswith(('o1', 'o3', 'o4'))


@dataclass
class CoTReasonerOutput(AgentContext):
    """
    content (str): The final, user-facing content or summary.
    reasoning_content (Optional[str]): The intermediate "thinking" steps of the model, often containing the plan, analysis, or breakdown of the problem. It can be `None` if the model does not produce a distinct thinking block.
    """
    content: str
    reasoning_content: Optional[str] = None


@serve.deployment
class CoTReasoner(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = None
    AUTOSCALING_MIN_REPLICAS = None
    AUTOSCALING_MAX_REPLICAS = None

    def __init__(self, image_encoder: ImageBase64Encoder) -> None:
        super().__init__()

        self.client, self.model = LLMClientFactory().create_client()
        self.image_encoder = image_encoder

        # Per-model request settings (config/models/<model>.json).
        config = get_config()
        self.sampling_kwargs = split_sampling_kwargs(config.sampling)
        token_key = 'max_completion_tokens' if uses_max_completion_tokens(self.model) \
            else 'max_tokens'
        self.token_kwargs = {token_key: config.max_tokens}

    @AgentTool.document_output_class(CoTReasonerOutput)
    async def cot_reason(
        self, 
        prompt: str, 
        input_images: Optional[Image.Image | List[Image.Image]] = None,
        other_images: Optional[Dict[str, Image.Image]] = None,
        add_label: bool = True,
    ) -> AgentToolOutput:
        """
        Acts as the agent's central reasoning and planning engine.

        This tool takes a user query or a complex task description and uses a large language model to perform Chain-of-Thought (CoT) reasoning. It can be used to break down problems, formulate plans for other tools, or generate final answers based on provided context.

        Args:
            prompt (str): The text prompt that requires reasoning. This can be a direct question, a task to be planned, or a request for code generation.
            image_sources (Optional[Image.Image | List[Image.Image]]): An optional single image or list of images to provide visual context for multimodal reasoning.
        """
        images_to_encode = []
        if input_images:
            if not isinstance(input_images, List):
                input_images = [input_images]
            for i, input_image in enumerate(input_images):
                # 1-indexed stamp so the on-image label matches the question's
                # natural-language "image k" numbering (and the planner's 1-indexed
                # reference_view).
                images_to_encode.append(
                    add_label_to_image(input_image, f'input_images.images[{i + 1}]')
                    if add_label else input_image
                )
        if other_images:
            for label, other_image in other_images.items():
                images_to_encode.append(
                    add_label_to_image(other_image, label) 
                    if add_label else other_image
                )

        base64_uris = []
        if images_to_encode:
            encode_result_refs = [
                self.image_encoder.encode_image.remote(image)
                for image in images_to_encode
            ]

            encode_results = await asyncio.gather(*encode_result_refs)
            for result in encode_results:
                if result.err:
                    return result
                base64_uris.append(result.result)

        image_parts = [
            {'type': 'image_url', 'image_url': {'url': uri}}
            for uri in base64_uris
        ]
        text_part = {'type': 'text', 'text': prompt}
        # Images before the text, as in the models' official chat layouts.
        content = image_parts + [text_part]

        try:
            messages = [{'role': 'user', 'content': content}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.sampling_kwargs,
                **self.token_kwargs,
            )

            choice = response.choices[0]
            output_message = choice.message
            content = output_message.content

            # The backend stopped at max_tokens: for a thinking model the thinking
            # block never closed and `content` is usually None, so there is no answer
            # to salvage. Fail with the marker, which tells the callers not to retry.
            finish_reason = getattr(choice, 'finish_reason', None)
            if finish_reason == 'length' or content is None:
                usage = getattr(response, 'usage', None)
                completion_tokens = getattr(usage, 'completion_tokens', None)
                return self.error(msg=(
                    f'{TRUNCATION_MARKER}: generation cut off at {self.token_kwargs} '
                    f'(finish_reason={finish_reason!r}, completion_tokens='
                    f'{completion_tokens}, content_is_none={content is None}) '
                    '- not retried'))

            if hasattr(output_message, 'reasoning_content'):
                reasoning_content = output_message.reasoning_content
            else:
                reasoning_content = None

            if reasoning_content is not None:
                reasoning_content = reasoning_content.replace('<think>', '').replace('</think>', '')
                cot_output = CoTReasonerOutput(
                    content=content,
                    reasoning_content=reasoning_content
                )
                return self.success(result=cot_output)

            thinking_part, sep, content_part = content.partition('</think>')
            if sep:
                reasoning_content = thinking_part.replace('<think>', '')
                cot_output = CoTReasonerOutput(
                    content=content_part,
                    reasoning_content=reasoning_content
                )
                return self.success(result=cot_output)

            cot_output = CoTReasonerOutput(
                content=content,
                reasoning_content=None
            )
            return self.success(result=cot_output)
         
        except Exception as e:
            import traceback
            cause = getattr(e, '__cause__', None)
            err_msg = f'An error occurred: {type(e).__name__}: {e}'
            if cause is not None:
                err_msg += f' | cause: {type(cause).__name__}: {cause}'
            traceback.print_exc()
            return self.error(msg=err_msg)
