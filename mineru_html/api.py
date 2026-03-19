import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Literal, Tuple, TypeVar, Union

from mineru_html.base import MinerUHTMLBase, MinerUHTMLCase, MinerUHTMLInput
from mineru_html.exceptions import MinerUHTMLConfigError, MinerUHTMLTypeError
from mineru_html.inference.base_backend import InferenceBackend
from mineru_html.process import (build_prompt, convert2content,
                                 extract_main_html_fallback,
                                 extract_main_html_single,
                                 get_fallback_handler, parse_result,
                                 simplify_single_input)

ValidSingleInputType = Union[str, MinerUHTMLInput]
ValidInputType = Union[ValidSingleInputType, list[ValidSingleInputType]]


MinerUInType = TypeVar('MinerUInType', bound=MinerUHTMLBase)
MinerUOutType = TypeVar('MinerUOutType', bound=MinerUHTMLBase)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MinerUHTMLConfig:
    """Configuration for MinerUHTML extractor.

    Attributes:
        use_fall_back: Fallback method when main extraction fails.
            Options: "trafilatura", "bypass", or "empty".
        early_load: Whether to load the model early (default: True).
        prompt_version: Version of the prompt template (e.g."v2", "short_compact").
        response_format: Expected model output format. Only "json" or "compact" allowed.
        output_format: Expected output format.
    """

    use_fall_back: Literal['trafilatura', 'bypass', 'empty'] = 'trafilatura'
    early_load: bool = True
    prompt_version: str = 'short_compact'
    response_format: str = 'compact'
    output_format: str = 'mm_md'

    def valid(self) -> bool:
        if self.use_fall_back not in ['trafilatura', 'bypass', 'empty']:
            raise MinerUHTMLConfigError(
                'use_fall_back must be "trafilatura", "bypass" or "empty"'
            )
        if self.response_format not in ('json', 'compact'):
            raise MinerUHTMLConfigError(
                'response_format must be "json" or "compact"'
            )
        return True


def convert_input_to_map(
    input_data: ValidInputType,
) -> Tuple[Dict[str, MinerUHTMLCase], List[str]]:
    """Convert input data to a map of cases.

    Args:
        input_data: Input data, can be a single string, MinerUHTMLInput, or a list.

    Returns:
        A tuple containing:
            - Dictionary mapping case_id to MinerUHTMLCase
            - List of case_ids in input order
    """
    case_list: List[MinerUHTMLCase] = []

    input_data_list: List[ValidSingleInputType] = (
        input_data if isinstance(input_data, list) else [input_data]
    )

    for input_item in input_data_list:
        if isinstance(input_item, str):
            mineru_case = MinerUHTMLCase(
                input_data=MinerUHTMLInput(raw_html=input_item)
            )
        elif isinstance(input_item, MinerUHTMLInput):
            mineru_case = MinerUHTMLCase(input_data=input_item)
        else:
            mineru_case = MinerUHTMLCase(input_data=input_item)
            mineru_case.set_error(
                MinerUHTMLTypeError(f'Invalid input type: {type(input_item)}')
            )

        case_list.append(mineru_case)

    return {case.case_id: case for case in case_list}, [
        case.case_id for case in case_list
    ]


def map_function(
    input_case_map: Dict[str, MinerUHTMLCase],
    map_function: Callable[[MinerUHTMLCase], MinerUHTMLCase],
    apply_on_error: bool = False,
) -> Dict[str, MinerUHTMLCase]:
    """Apply a function to each case in the map.

    Args:
        input_case_map: Dictionary of cases to process.
        map_function: Function to apply to each case.
        apply_on_error: If True, only apply to cases with errors.
            If False (default), only apply to cases without errors.

    Returns:
        Dictionary of processed cases.
    """
    output_case_map = {}
    for k, case in input_case_map.items():
        try:
            if apply_on_error:
                is_apply = case.error is not None
            else:
                is_apply = case.error is None

            if is_apply:
                output_case = map_function(case)
            else:
                output_case = case

        except Exception as e:
            case.set_error(e)
            output_case = case

        output_case_map[k] = output_case

    return output_case_map


class MinerUHTMLGeneric:
    """Generic base class for MinerUHTML extractors."""

    def __init__(self, llm: InferenceBackend, config: MinerUHTMLConfig):
        """Initialize the extractor.

        Args:
            llm: Inference backend for LLM processing.
            config: Configuration for the extractor.
        """
        self.llm = llm
        self.config = config
        self.fallback_handler = get_fallback_handler(config.use_fall_back)
        if self.config.early_load:
            self.llm.setup_llm()

    def process(self, input_data: ValidInputType) -> List[MinerUHTMLCase]:
        """Process HTML input and extract main content.
        Args:
            input_data: Input HTML string(s) or MinerUHTMLInput object(s).

        Returns:
            List of MinerUHTMLCase objects containing extraction results.
        """
        case_map, input_keys = convert_input_to_map(input_data)

        case_map = map_function(case_map, simplify_single_input)

        generate_func = partial(build_prompt, prompt_version=self.config.prompt_version)
        case_map = map_function(case_map, generate_func)

        generate_input_map = {
            k: v.generate_input for k, v in case_map.items() if v.error is None
        }
        generate_output_map, error_map = self.llm.process(generate_input_map)
        for k, error in error_map.items():
            case_map[k].set_error(error)
        for k, generate_output in generate_output_map.items():
            case_map[k].generate_output = generate_output

        case_map = map_function(case_map, parse_result)

        case_map = map_function(case_map, extract_main_html_single)

        fallback_func = partial(
            extract_main_html_fallback, fallback_handler=self.fallback_handler
        )
        case_map = map_function(case_map, fallback_func, apply_on_error=True)

        convert_func = partial(convert2content, output_format=self.config.output_format)
        case_map = map_function(case_map, convert_func)

        return [case_map[k] for k in input_keys]
