import os
import random
from typing import List, Optional

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


def make_http_client(proxy: Optional[str]) -> Optional[httpx.AsyncClient]:
    """An httpx client routed through `proxy`, or None for a direct connection."""
    if not proxy:
        return None
    return httpx.AsyncClient(proxies={'http://': proxy, 'https://': proxy}, http2=True)


# Reasoning models (Qwen3.6-27B) can spend >10 min on a single non-streamed
# completion, blowing past the OpenAI SDK's 600s default read timeout and killing
# the whole sample. Override via VANTAGE_VLLM_TIMEOUT if you need to tune it.
_DEFAULT_READ_TIMEOUT = float(os.environ.get('VANTAGE_VLLM_TIMEOUT', '3600'))
_VLLM_TIMEOUT = httpx.Timeout(_DEFAULT_READ_TIMEOUT, connect=10.0)


class VLLMLBChatCompletions:
    """Chat completions spread uniformly at random over several vLLM endpoints."""

    def __init__(self, endpoints: List[str], api_key: str, proxy: str = ''):
        self.endpoints = endpoints
        self.api_key = api_key
        self.proxy = proxy

    async def create(self, **kwargs) -> ChatCompletion:
        base_url = random.choice(self.endpoints)
        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            http_client=make_http_client(self.proxy),
            timeout=_VLLM_TIMEOUT,
        ) as client:
            response = await client.chat.completions.create(**kwargs)
        return response


class VLLMLBChat:
    def __init__(self, endpoints: List[str], api_key: str, proxy: str = ''):
        self.completions = VLLMLBChatCompletions(endpoints, api_key, proxy)


class AsyncVLLMLBClient:
    def __init__(self, endpoints: List[str], api_key: str, proxy: str = ''):
        self.chat = VLLMLBChat(endpoints, api_key, proxy)
