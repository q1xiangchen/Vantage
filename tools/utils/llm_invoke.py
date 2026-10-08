import asyncio
from typing import Any, Callable, Optional


# Marker carried on the error message when a generation was cut off at max_tokens.
# The thinking block never closed, so there is no usable answer and re-issuing the
# identical request only burns GPU: callers fail out instead of retrying, and the
# marker lands on the per-sample trace record.
TRUNCATION_MARKER = 'TRUNCATED_MAX_TOKENS'


def is_truncation_error(msg: Optional[str]) -> bool:
    """True when `msg` is (or wraps) a max_tokens-truncation failure."""
    return bool(msg) and TRUNCATION_MARKER in str(msg)


async def invoke_with_retry(
    invoker: Callable,
    prompter: Callable,
    parser: Optional[Callable],
    max_retries: int = 3
) -> Any:
    """Call `invoker` with retries, feeding the previous error back into the prompt.

    A max_tokens truncation is the one failure that is not retried."""
    err_msg, response_content = None, None

    for i in range(max_retries + 1):
        try:
            prompt = prompter(err_msg=err_msg, response=response_content)
            output = await invoker(prompt=prompt)
            if output.err:
                raise RuntimeError(output.err['msg'])

            response_content = output.result.content
            if parser is not None:
                return await parser(output.result)
            else:
                return response_content

        except Exception as e:
            err_msg = str(e)
            if is_truncation_error(err_msg):
                raise RuntimeError(err_msg)
            if i < max_retries:
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(err_msg)
