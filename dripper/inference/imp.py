from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, override

from transformers import AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams


@dataclass
class ModelResponse:
    generated_text: str
    generated_token_ids: Optional[list[int]] = None
    prompt: Optional[str] = None
    prompt_token_ids: Optional[list[int]] = None


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        pass


class VLLMInferenceBackend(InferenceBackend):
    def __init__(self, model_path:str, model_init_kwargs: Dict[str, Any] = {}, model_gen_kwargs:Dict[str, Any] = {}):
        tensor_parallel_size = model_init_kwargs.pop('tensor_parallel_size', 1)
        self.sample_paras = SamplingParams(
            top_k=model_gen_kwargs.pop('top_k', 1),
            top_p=model_gen_kwargs.pop('top_p', 0.95),
            temperature=model_gen_kwargs.pop('temperature', 0),
            max_tokens=model_gen_kwargs.pop('max_tokens', 8 * 1024),
        )
        self.gen_config = model_gen_kwargs

        self._llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, **model_init_kwargs)

    @override
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        model_output = self._llm.generate(prompt_list, sampling_params=self.sample_paras, **self.gen_config)
        response = []
        for res in model_output:
            response.append(
                ModelResponse(
                    prompt=res.prompt,
                    prompt_token_ids=res.prompt_token_ids,
                    generated_text=res.outputs[0].text,
                    generated_token_ids=res.outputs[0].token_ids
                )
            )
        return response


class TransformersInferenceBackend(InferenceBackend):
    def __init__(self, model_path: str, tokenizer: Any, model_init_kwargs: Dict[str, Any] = {}, model_gen_kwargs: Dict[str, Any] = {}):
        device_map = model_init_kwargs.pop('device_map', 'auto')
        dtype = model_init_kwargs.pop('dtype', 'auto')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device_map,
            dtype=dtype,
            **model_init_kwargs
        )
        self._llm = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device_map=device_map
        )

        self.gen_config = {
            'top_k': model_gen_kwargs.pop('top_k', 1),
            'top_p': model_gen_kwargs.pop('top_p', 0.95),
            'temperature': model_gen_kwargs.pop('temperature', 0),
            'max_new_tokens': model_gen_kwargs.pop('max_new_tokens', 8 * 1024),
            'do_sample': model_gen_kwargs.pop('do_sample', False),
            'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'return_full_text': model_gen_kwargs.pop('return_full_text', False),
            **model_gen_kwargs
        }

    @override
    def generate(self, prompt_list: list[str]) -> list[ModelResponse]:
        model_output = self._llm(prompt_list,**self.gen_config)
        response = []
        for res in model_output:
            response.append(
                ModelResponse(
                    generated_text=res[0]['generated_text']
                )
            )
        return response
