# inference/vllm_backend.py
from typing import Any, Dict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from mineru_html.inference.base_backend import InferenceBackend, ModelResponse
from mineru_html.utils import get_all_item_ids


class VLLMInferenceBackend(InferenceBackend):
    def __init__(
        self,
        model_path: str,
        max_context_window: int,
        response_format: str,
        model_init_kwargs: Dict[str, Any] = {},
        model_gen_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(max_context_window, response_format)
        self.model_path = model_path
        self.model_init_kwargs = model_init_kwargs
        self.model_gen_kwargs = model_gen_kwargs
        self.tensor_parallel_size = model_init_kwargs.pop('tensor_parallel_size', 1)
        self.gpu_memory_utilization = model_init_kwargs.pop(
            'gpu_memory_utilization', 0.8
        )
        self.enforce_eager = model_init_kwargs.pop('enforce_eager', True)
        self.sample_paras = dict(
            top_k=model_gen_kwargs.pop('top_k', 1),
            top_p=model_gen_kwargs.pop('top_p', 0.95),
            temperature=model_gen_kwargs.pop('temperature', 0),
            max_tokens=model_gen_kwargs.pop('max_tokens', 16 * 1024),
        )
        self.gen_config = model_gen_kwargs
        self._llm = None
        self._tokenizer = None

    def setup_llm(self):
        if self._llm is None:
            self._llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                max_model_len=self.max_context_window,
                **self.model_init_kwargs,
            )

    def get_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def generate(self, prompt_list: list[str], **kwargs) -> list[ModelResponse]:
        tokenizer = self.get_tokenizer()
        prompt_with_template = []
        for prompt in prompt_list:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ]
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True,
            )
            prompt_with_template.append(chat_prompt)
        prompt_list = prompt_with_template

        if self.response_format == 'compact':
            sampling_params = []
            item_ids_list = kwargs.get('item_ids_list', [])
            if len(item_ids_list) == 0:
                item_ids_list = [get_all_item_ids(prompt) for prompt in prompt_list]
            for item_ids in item_ids_list:
                item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
                pattern = f'<answer>\\s*{item_pattern}\\s*</answer>'
                structured_outputs_params = StructuredOutputsParams(regex=pattern)
                sampling_params.append(
                    SamplingParams(
                        structured_outputs=structured_outputs_params,
                        **self.sample_paras,
                    )
                )
        else:
            sampling_params = SamplingParams(**self.sample_paras)
        model_output = self._llm.generate(
            prompt_list, sampling_params=sampling_params, **self.gen_config
        )
        response = []
        for res in model_output:
            response.append(
                ModelResponse(
                    prompt=res.prompt,
                    prompt_token_ids=res.prompt_token_ids,
                    generated_text=res.outputs[0].text,
                    generated_token_ids=res.outputs[0].token_ids,
                )
            )
        return response

    def cleanup(self):
        """Clean up vLLM engine resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
