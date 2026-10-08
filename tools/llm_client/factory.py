import json
import os
from typing import List, Tuple, Union

from openai import AsyncOpenAI

from tools.llm_client.vllm import AsyncVLLMLBClient, make_http_client
from workflow.config import get_config


AsyncClient = Union[AsyncOpenAI, AsyncVLLMLBClient]


class LLMClientFactory:

    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        # Must match launch_vllm.py: VANTAGE_SERVE_FILE points the agent at the same
        # per-run registry the launcher wrote to.
        self.serve_file = os.environ.get('VANTAGE_SERVE_FILE') or os.path.join(log_dir, 'serve.json')

    def get_all_vllm_endpoints(self, model: str) -> List[str]:
        if not os.path.exists(self.serve_file):
            raise FileNotFoundError(
                f'serve.json not found at "{self.serve_file}". Is the vLLM service running?')
        try:
            with open(self.serve_file, 'r', encoding='utf-8') as f:
                serve_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f'Could not parse serve.json at "{self.serve_file}".')

        model_instances = serve_data.get(model)
        if not model_instances:
            raise ValueError(f'No running service found for model "{model}" in serve.json.')
        return [f'http://{inst["ip"]}:{inst["port"]}/v1' for inst in model_instances.values()]

    def create_client(self) -> Tuple[AsyncClient, str]:
        config = get_config()
        model, base_url, proxy = config.model, config.base_url, config.proxy
        if not model:
            raise ValueError('"model" not set in config.')

        if base_url.lower().strip() == 'vllm':
            endpoints = self.get_all_vllm_endpoints(model)
            print(f'Found {len(endpoints)} vLLM instance(s) for {model}.')
            return AsyncVLLMLBClient(endpoints=endpoints, api_key='EMPTY', proxy=proxy), model

        # The key is never written to config.json, so Ray replicas (which reload the
        # config from there) fall back to OPENAI_API_KEY from the inherited env.
        api_key = config.api_key or os.environ.get('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError('No API key: set OPENAI_API_KEY or pass --api_key.')
        client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, http_client=make_http_client(proxy))
        return client, model
