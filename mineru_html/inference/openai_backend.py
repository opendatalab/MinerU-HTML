# inference/openai_backend.py
import asyncio

import nest_asyncio
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from mineru_html.inference.base_backend import InferenceBackend, ModelResponse


class DummyTokenizer:
    def __init__(self):
        self.chars_per_token = 4

    def encode(self, input_str: str) -> list[int]:
        return [0] * (len(input_str) // self.chars_per_token + 1)


class OpenaiAPIInferenceBackend(InferenceBackend):
    def __init__(
        self,
        base_url: str,
        sk: str,
        max_context_window: int,
        model: str,
        retry_times: int = 3,
        response_format: str = 'json',
    ):
        super().__init__(max_context_window, response_format)
        self.base_url = base_url
        self.sk = sk
        self.model = model
        self.retry_times = retry_times
        self.async_sdk = AsyncOpenAI(base_url=self.base_url, api_key=self.sk)
        self._tokenizer = None

    def build_pay_load(self, prompt: str) -> dict:
        pay_load = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            'temperature': 0,
            'max_tokens': 8192,
        }
        return pay_load

    def setup_llm(self):
        pass

    def get_tokenizer(self) -> DummyTokenizer:
        if self._tokenizer is None:
            self._tokenizer = DummyTokenizer()
        return self._tokenizer

    async def call_api_async(self, prompt: str) -> ModelResponse:
        payload = self.build_pay_load(prompt)
        response: ChatCompletion = await self.async_sdk.chat.completions.create(
            **payload
        )
        response_text = response.choices[0].message.content
        return ModelResponse(generated_text=response_text)

    async def call_api_async_with_retry(self, prompt: str) -> ModelResponse:
        for attempt in range(self.retry_times):
            try:
                return await self.call_api_async(prompt)
            except Exception as e:
                print(f'Error calling API: {e}')
                await asyncio.sleep(1)
        return ModelResponse(generated_text='')

    async def process_async(self, prompt_list: list[str]) -> list[ModelResponse]:
        tasks = [self.call_api_async_with_retry(item) for item in prompt_list]
        results = await asyncio.gather(*tasks)
        return results

    def generate(self, prompt_list: list[str], **kwargs) -> list[ModelResponse]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            nest_asyncio.apply()
            return loop.run_until_complete(self.process_async(prompt_list))
        else:
            return asyncio.run(self.process_async(prompt_list))
