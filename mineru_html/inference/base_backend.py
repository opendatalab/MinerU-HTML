from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from transformers import AutoTokenizer

from mineru_html.base import MinerUHTMLGenerateInput, MinerUHTMLGenerateOutput
from mineru_html.exceptions import MinerUHTMLError, MinerUHTMLInputTooLongError
from mineru_html.utils import build_dummy_response, get_all_item_ids


@dataclass
class ModelResponse:
    generated_text: str
    generated_token_ids: Optional[list[int]] = None
    prompt: Optional[str] = None
    prompt_token_ids: Optional[list[int]] = None


def get_attribute(obj, attr_name: str):
    """Safely get an attribute from an object.

    Args:
        obj: Object to get attribute from.
        attr_name: Name of the attribute.

    Returns:
        Attribute value if exists, None otherwise.
    """
    try:
        return object.__getattribute__(obj, attr_name)
    except AttributeError:
        return None


class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    This class defines the interface that all inference backends must implement.
    """

    def __init__(self, max_context_window: int, response_format: str = 'json'):
        """Initialize the inference backend.

        Args:
            max_context_window: Maximum context window size in tokens.
            response_format: Expected model output format: "json" or "compact".
        """
        self.max_context_window = max_context_window
        self.response_format = response_format

    def check_input_length(
        self, generate_input: MinerUHTMLGenerateInput
    ) -> Tuple[bool, list[int]]:
        """Check if input length exceeds context window.

        Args:
            generate_input: Input to check.

        Returns:
            Tuple of (is_valid, item_ids).
        """
        tokenizer = self.get_tokenizer()
        item_ids = get_all_item_ids(generate_input.full_prompt)
        dummy_response = build_dummy_response(item_ids, self.response_format)
        token_ids = tokenizer.encode(
            generate_input.full_prompt + '\n\n' + dummy_response
        )
        if len(token_ids) > self.max_context_window:
            return False, item_ids
        return True, item_ids

    def process(
        self, generate_input_map: Dict[str, MinerUHTMLGenerateInput]
    ) -> Tuple[Dict[str, MinerUHTMLGenerateOutput], Dict[str, MinerUHTMLError]]:
        """Process a batch of generate inputs.

        Args:
            generate_input_map: Dictionary mapping case_id to generate input.

        Returns:
            Tuple of (output_map, error_map).
        """
        if len(generate_input_map) == 0:
            return {}, {}
        valid_input_keys = []
        valid_input_str_list = []
        valid_input_item_ids_list = []
        error_map: Dict[str, MinerUHTMLError] = {}
        tokenizer = self.get_tokenizer()
        for k, v in generate_input_map.items():
            if callable(get_attribute(tokenizer, 'apply_chat_template')):
                msg = [{'role': 'user', 'content': v.full_prompt}]
                v.full_prompt = tokenizer.apply_chat_template(
                    msg, tokenize=False, enable_thinking=False
                )
            valid, item_ids = self.check_input_length(v)
            if valid:
                valid_input_keys.append(k)
                valid_input_str_list.append(v.full_prompt)
                valid_input_item_ids_list.append(item_ids)
            else:
                error_map[k] = MinerUHTMLInputTooLongError('Input too long', case_id=k)
        self.setup_llm()
        res_list = self.generate(
            valid_input_str_list, item_ids_list=valid_input_item_ids_list
        )
        output_map = {
            k: MinerUHTMLGenerateOutput(response=res.generated_text)
            for k, res in zip(valid_input_keys, res_list)
        }
        return output_map, error_map

    @abstractmethod
    def setup_llm(self):
        """Set up and load the LLM model."""
        pass

    @abstractmethod
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer for this backend.
        """
        pass

    @abstractmethod
    def generate(self, prompt_list: list[str], **kwargs) -> list[ModelResponse]:
        """Generate responses for a list of prompts.

        Args:
            prompt_list: List of prompt strings.
            **kwargs: Additional arguments for generation.
        """
        pass
