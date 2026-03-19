# inference/transformers_backend.py
from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mineru_html.inference.base_backend import InferenceBackend, ModelResponse


class TransformersInferenceBackend(InferenceBackend):
    def __init__(
        self,
        model_path: str,
        max_context_window: int,
        tokenizer: Any,
        model_init_kwargs: Dict[str, Any] = {},
        model_gen_kwargs: Dict[str, Any] = {},
        response_format: str = 'compact',
    ):
        super().__init__(max_context_window, response_format)
        self.model_path = model_path
        self.model_init_kwargs = model_init_kwargs
        self.device_map = model_init_kwargs.pop('device_map', 'auto')
        self.dtype = model_init_kwargs.pop('dtype', 'auto')
        self.tokenizer = tokenizer
        self._llm = None

        self.gen_config = {
            'top_k': model_gen_kwargs.pop('top_k', 1),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'max_new_tokens': model_gen_kwargs.pop('max_new_tokens', 16 * 1024),
            'do_sample': model_gen_kwargs.pop('do_sample', False),
            'pad_token_id': tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'return_full_text': model_gen_kwargs.pop('return_full_text', False),
            **model_gen_kwargs,
        }

    def setup_llm(self):
        if self._llm is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map=self.device_map,
                dtype=self.dtype,
                **self.model_init_kwargs,
            )
            self._llm = pipeline(
                'text-generation',
                model=model,
                tokenizer=self.tokenizer,
                device_map=self.device_map,
            )

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def generate(self, prompt_list: list[str], **kwargs) -> list[ModelResponse]:
        model_output = self._llm(prompt_list, **self.gen_config)
        response = []
        for res in model_output:
            response.append(ModelResponse(generated_text=res[0]['generated_text']))
        return response
