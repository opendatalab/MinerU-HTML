"""
Main API module for Dripper HTML extraction system.

This module provides the Dripper class, which implements the complete
HTML content extraction pipeline using large language models.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer
from vllm import LLM

from dripper.base import (DripperGenerateInput, DripperGenerateOutput,
                          DripperInput, DripperOutput, DripperProcessData,
                          TagType, check_and_find_max_item_id)
from dripper.exceptions import (DripperConfigError, DripperEnvError,
                                DripperLoadModelError, DripperPostprocessError,
                                DripperPreprocessError,
                                DripperResponseParseError, DripperTypeError)
from dripper.inference.inference import generate
from dripper.inference.logits import parse_llm_response
from dripper.inference.prompt import get_full_prompt
from dripper.process.map_to_main import extract_main_html
from dripper.process.simplify_html import simplify_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_vllm_environment(state_machine: str) -> None:
    """
    Check and ensure VLLM uses v0 API to avoid compatibility issues.

    This function verifies that the VLLM_USE_V1 environment variable is set
    to '0' when state machine is enabled, ensuring compatibility with the
    logits processing pipeline.

    Args:
        state_machine: State machine version string (None if disabled)

    Raises:
        DripperEnvError: If VLLM_USE_V1 is not set to '0' when required
    """
    if state_machine is None:
        return
    if os.environ.get('VLLM_USE_V1', '') != '0':
        raise DripperEnvError(
            'VLLM_USE_V1 environment variable is not set to 0.\n'
            'Please set it using "export VLLM_USE_V1=0"'
        )


class Dripper:
    """
    HTML main content extractor based on large language models.

    This class provides a complete HTML content extraction pipeline:
    1. Preprocessing: Simplify HTML structure and generate inference prompts
    2. Inference: Use LLM to identify main content regions
    3. Postprocessing: Parse model responses and extract final results

    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        model_path (str): Path to the model file
        tp (int): Tensor parallel size
        debug (bool): Whether debug mode is enabled
        raise_errors (bool): Whether to raise exceptions on errors
        use_fall_back (bool): Whether to use fallback extraction method
        state_machine (str): State machine version for logits processing
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Dripper instance.

        Args:
            config: Configuration dictionary that must contain 'model_path'

        Raises:
            DripperConfigError: When configuration is invalid
        """
        self.config = self._validate_config(config)
        self.max_sequence_length = 32 * 1024  # Maximum sequence length for the model
        self.model_path = config['model_path']
        self.tp = config.get('tp', 1)
        self.raise_errors = config.get('raise_errors', False)
        self.debug = config.get('debug', False)
        self.use_fall_back = config.get('use_fall_back', True)
        self.state_machine = config.get('state_machine', None)

        # Lazy-loaded attributes (initialized on first use)
        self._llm: Optional[LLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._trafilatura_settings = None
        self._trafilatura = None

    def get_trafilatura(self):
        """
        Get trafilatura extractor instance (lazy-loaded).

        Returns:
            Tuple of (trafilatura extractor function, extractor settings)
        """
        if self._trafilatura is None:
            from trafilatura.settings import Extractor

            self._trafilatura_settings = Extractor(
                output_format='html', comments=False
            )
            from trafilatura import extract

            self._trafilatura = extract
        return self._trafilatura, self._trafilatura_settings

    def fall_back_func(self, input_html: str, url: str) -> str:
        """
        Fallback extraction function using trafilatura.

        Used when the main LLM-based extraction fails or is unavailable.

        Args:
            input_html: Raw HTML content to extract from
            url: Optional URL where the HTML was obtained from

        Returns:
            Extracted main HTML content
        """
        t_extractor, t_settings = self.get_trafilatura()
        return t_extractor(input_html, url=url, options=t_settings)

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validated configuration dictionary (copy)

        Raises:
            DripperConfigError: When configuration is invalid
        """
        if not isinstance(config, dict):
            raise DripperConfigError('Configuration must be a dictionary')

        if 'model_path' not in config:
            raise DripperConfigError(
                'Configuration must contain "model_path" parameter'
            )

        if not os.path.exists(config['model_path']):
            logger.warning(f"Model path does not exist: {config['model_path']}")

        # Validate tensor parallel size
        tp = config.get('tp', 1)
        if not isinstance(tp, int) or tp < 1:
            raise DripperConfigError(
                'tp (tensor parallel size) must be a positive integer'
            )

        return config.copy()

    def get_tokenizer(self) -> AutoTokenizer:
        """
        Get tokenizer instance (lazy-loaded).

        Returns:
            AutoTokenizer instance

        Raises:
            DripperLoadModelError: When tokenizer loading fails
        """
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, use_fast=True
                )
            except Exception as e:
                raise DripperLoadModelError(
                    f'Tokenizer loading failed: {str(e)}'
                ) from e
        return self._tokenizer

    def get_llm(self) -> LLM:
        """
        Get LLM instance (lazy-loaded).

        Returns:
            VLLM LLM instance

        Raises:
            DripperLoadModelError: When model loading fails
        """
        if self._llm is None:
            check_vllm_environment(self.state_machine)
            try:
                logger.info(f'Loading model: {self.model_path}')
                self._llm = LLM(
                    model=self.model_path, tensor_parallel_size=self.tp
                )
                logger.info('Model loading completed')
            except Exception as e:
                raise DripperLoadModelError(
                    f'Model loading failed: {str(e)}'
                ) from e

        return self._llm

    def pre_process(
        self, raw_input: DripperInput
    ) -> Tuple[DripperGenerateInput, DripperProcessData]:
        """
        Preprocess raw input data.

        Simplifies the raw HTML into a format suitable for model processing
        and generates inference prompts.

        Args:
            raw_input: Raw input data

        Returns:
            Tuple containing generate input and process data

        Raises:
            DripperPreprocessError: When preprocessing fails
        """
        try:
            # Simplify HTML structure
            simplified_html, map_html = simplify_html(raw_input.raw_html)
            max_item_id = check_and_find_max_item_id(simplified_html)
            full_prompt = get_full_prompt(simplified_html)

            # Check if prompt length exceeds model's maximum sequence length
            tokenizer = self.get_tokenizer()
            prompt_length = len(tokenizer(full_prompt)['input_ids'])
            # Use max_item_id * 8 as approximate length of response
            if prompt_length + max_item_id * 8 >= self.max_sequence_length:
                raise DripperPreprocessError(
                    f'Preprocessing failed (case_id: {raw_input.case_id}): '
                    f'Generated prompt is too long (prompt_length: {prompt_length}, '
                    f'item_num: {max_item_id}), exceeds model maximum sequence length '
                    f'{self.max_sequence_length}.'
                )

            # Build generate input
            generate_input = DripperGenerateInput(
                alg_html=simplified_html,
                prompt=full_prompt,
                case_id=raw_input.case_id,
            )

            # Build process data
            process_data = DripperProcessData(
                raw_html=raw_input.raw_html,
                simpled_html=simplified_html,
                map_html=map_html,
                case_id=raw_input.case_id,
            )

            return generate_input, process_data

        except Exception as e:
            raise DripperPreprocessError(
                f'Preprocessing failed (case_id: {raw_input.case_id}): {str(e)}'
            ) from e

    def post_process(
        self,
        generate_output: DripperGenerateOutput,
        pre_process_data: DripperProcessData,
    ) -> DripperOutput:
        """
        Postprocess model output.

        Parses the model response and extracts the final main HTML content.

        Args:
            generate_output: Model-generated output
            pre_process_data: Data from preprocessing stage

        Returns:
            Final output result

        Raises:
            DripperResponseParseError: When response parsing fails
            DripperPostprocessError: When postprocessing fails
        """
        try:
            # Parse LLM response to get labels
            labels = parse_llm_response(generate_output.response)
            if not any(
                label == TagType.Main.value for label in labels.values()
            ):
                raise DripperResponseParseError(
                    f'Model response contains no main content labels, '
                    f'response: {labels}'
                )
            # Extract main HTML content based on labels
            main_html = extract_main_html(pre_process_data.map_html, labels)

            return DripperOutput(
                main_html=main_html, case_id=pre_process_data.case_id
            )
        except DripperResponseParseError as e:
            logger.error(
                f'Postprocessing failed (case_id: {pre_process_data.case_id}): {str(e)}'
            )
            raise e
        except Exception as e:
            raise DripperPostprocessError(
                f'Postprocessing failed (case_id: {pre_process_data.case_id}): {str(e)}'
            ) from e

    def _normalize_input(
        self,
        input_data: Union[DripperInput, List[DripperInput], str, List[str]],
    ) -> Dict[int, DripperInput]:
        """
        Normalize input data format.

        Converts various input formats (strings, DripperInput objects, or lists)
        into a standardized dictionary of DripperInput objects.

        Args:
            input_data: Input data in various formats

        Returns:
            Dictionary mapping indices to normalized DripperInput objects

        Raises:
            DripperTypeError: When input format is not supported
        """
        if isinstance(input_data, list):
            result = {}
            for idx, item in enumerate(input_data):
                if isinstance(item, str):
                    result[idx] = DripperInput(raw_html=item)
                elif isinstance(item, DripperInput):
                    result[idx] = item
                else:
                    raise DripperTypeError(
                        f'Unsupported input type: {type(item)}'
                    )
            return result

        elif isinstance(input_data, str):
            return {0: DripperInput(raw_html=input_data)}

        elif isinstance(input_data, DripperInput):
            return {0: input_data}

        else:
            raise DripperTypeError(
                f'Unsupported input type: {type(input_data)}'
            )

    def process(
        self,
        input_data: Union[DripperInput, List[DripperInput], str, List[str]],
    ) -> Union[List[DripperOutput], Tuple]:
        """
        Process input and return results.

        Complete processing pipeline:
        Input normalization → Preprocessing → Model inference → Postprocessing

        Args:
            input_data: Input data in various formats (string, DripperInput,
                       or lists of these)

        Returns:
            In normal mode: List of DripperOutput objects
            In debug mode: Tuple of (output_map, generate_inputs, process_datas)

        Raises:
            DripperError: When errors occur during processing (if raise_errors=True)
        """
        try:
            # Normalize input format
            input_map = self._normalize_input(input_data)
            logger.info(f'Starting to process {len(input_map)} inputs')

            # Preprocess all inputs
            generate_inputs = {}
            process_datas = {}

            for idx, raw_input in input_map.items():
                try:
                    generate_input, process_data = self.pre_process(raw_input)
                except Exception as e:
                    if self.raise_errors:
                        raise e
                    continue
                generate_inputs[idx] = generate_input
                process_datas[idx] = process_data

            # Get LLM instance and perform batch inference
            llm = self.get_llm()
            logger.info('Starting model inference')
            to_process_keys = sorted(generate_inputs.keys())
            generate_outputs = generate(
                llm,
                [generate_inputs[key] for key in to_process_keys],
                self.state_machine,
            )

            # Postprocess all outputs
            output_map = {}
            for idx, generate_output in zip(to_process_keys, generate_outputs):
                process_data = process_datas[idx]
                try:
                    output = self.post_process(generate_output, process_data)
                except Exception as e:
                    if self.raise_errors:
                        raise e
                    continue
                output_map[idx] = output

            # Handle cases that failed during preprocessing or postprocessing
            for idx in input_map.keys():
                if idx not in output_map:
                    if self.use_fall_back:
                        try:
                            output = self.fall_back_func(
                                input_map[idx].raw_html, input_map[idx].url
                            )
                            output_map[idx] = DripperOutput(
                                main_html=output,
                                case_id=input_map[idx].case_id,
                            )
                        except Exception as e:
                            if self.raise_errors:
                                raise e
                            output_map[idx] = DripperOutput(
                                main_html=None,
                                case_id=input_map[idx].case_id,
                            )
                    else:
                        output_map[idx] = DripperOutput(
                            main_html=None, case_id=input_map[idx].case_id
                        )

            logger.info(f'Processing completed, output {len(output_map)} results')

            # Return different formats based on debug mode
            if self.debug:
                # Debug mode: return output map, generate inputs, and process data
                return output_map, generate_inputs, process_datas
            else:
                # Normal mode: return only final output results
                sorted_keys = sorted(output_map.keys())
                output_list = [output_map[key] for key in sorted_keys]
                return output_list

        except Exception as e:
            logger.error(f'Error occurred during processing: {str(e)}')
            raise
