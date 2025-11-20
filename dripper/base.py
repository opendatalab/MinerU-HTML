"""
Base data structures and utilities for Dripper HTML extraction.

This module defines core data classes and helper functions used throughout
the Dripper system for processing HTML content and managing extraction data.
"""

import re
from enum import Enum
from typing import Callable, Union

# HTML attribute and tag constants used for item identification and selection
ITEM_ID_ATTR = '_item_id'  # Attribute name for item IDs
TAIL_BLOCK_TAG = 'cc-alg-uc-text'  # Tag name for tail blocks
SELECT_ATTR = 'cc-select'  # Attribute name for selection markers
CLASS_ATTR = 'mark-selected'  # CSS class name for selected items


class TagType(Enum):
    """Enumeration for HTML tag types in the extraction process."""

    Main = 'main'  # Main content tag
    Other = 'other'  # Other/non-main content tag


def check_and_find_max_item_id(input_str: str) -> int:
    """
    Find and validate the maximum item ID in a string.

    Extracts all item IDs from the input string using the ITEM_ID_ATTR pattern,
    validates that they form a continuous sequence starting from 1, and returns
    the maximum ID value.

    Args:
        input_str: String containing HTML with item ID attributes

    Returns:
        Maximum item ID found, or 0 if no IDs are found

    Raises:
        ValueError: If IDs cannot be converted to integers, or if they don't
                    form a continuous sequence starting from 1
    """
    # Match all patterns like ITEM_ID_ATTR="XXX" and extract the XXX part
    pattern = ITEM_ID_ATTR + r'="(\d+)"'
    matches = re.findall(pattern, input_str)

    # Return 0 if no matches found
    if len(matches) == 0:
        return 0

    # Convert all matched strings to integers
    int_list = []
    for match in matches:
        try:
            int_list.append(int(match))
        except Exception:
            raise ValueError(f'error while convert match {match} to int')

    # Validate that IDs form a continuous sequence starting from 1
    target_value = 1
    for int_id in int_list:
        if int_id == target_value:
            target_value += 1
        else:
            raise ValueError(
                f'mistake find in int list, current target value is {target_value}, but find {int_id}'
                + '\n'
                + input_str
            )

    # Return the maximum ID if validation passes
    return int_list[-1]


class DripperGenerateInput:
    """
    Input data structure for LLM generation requests.

    Contains the algorithm-processed HTML, the full prompt (either as a string
    or generated from a callable), and optional case identification.
    Automatically extracts and validates the maximum item ID from the HTML.
    """

    def __init__(
        self,
        alg_html: str,
        prompt: Union[Callable[[str], str], str],
        case_id: str = None,
    ):
        """
        Initialize DripperGenerateInput.

        Args:
            alg_html: Algorithm-processed HTML string
            prompt: Either a callable that generates a prompt from HTML, or a string prompt
            case_id: Optional identifier for the case being processed

        Raises:
            ValueError: If prompt type is not supported
        """
        self.alg_html = alg_html

        # Generate full prompt from callable or use string directly
        if isinstance(prompt, Callable):
            self.full_prompt = prompt(alg_html)
        elif isinstance(prompt, str):
            self.full_prompt = prompt
        else:
            raise ValueError(f'Unsupported prompt type: {type(prompt)}')

        self.case_id = case_id
        # Extract and validate maximum item ID from HTML
        self.max_item_id = check_and_find_max_item_id(alg_html)

    @classmethod
    def from_dict(cls, data: dict) -> 'DripperGenerateInput':
        """
        Create DripperGenerateInput from a dictionary.

        Args:
            data: Dictionary containing 'alg_html', 'full_prompt', and optionally 'case_id'

        Returns:
            DripperGenerateInput instance
        """
        return cls(
            alg_html=data['alg_html'],
            prompt=data['full_prompt'],
            case_id=data.get('case_id', None),
        )

    def to_dict(self) -> dict:
        """
        Convert DripperGenerateInput to a dictionary.

        Returns:
            Dictionary representation of the input data
        """
        output_dict = {}
        if self.case_id is not None:
            output_dict['case_id'] = self.case_id
        output_dict['alg_html'] = self.alg_html
        output_dict['full_prompt'] = self.full_prompt
        return output_dict


class DripperGenerateOutput:
    """
    Output data structure for LLM generation responses.

    Contains the raw response from the LLM and optional case identification.
    """

    def __init__(self, response: str, case_id: str = None):
        """
        Initialize DripperGenerateOutput.

        Args:
            response: Raw response string from the LLM
            case_id: Optional identifier for the case being processed
        """
        self.case_id = case_id
        self.response = response

    @classmethod
    def from_dict(cls, data: dict) -> 'DripperGenerateOutput':
        """
        Create DripperGenerateOutput from a dictionary.

        Args:
            data: Dictionary containing 'response' and optionally 'case_id'

        Returns:
            DripperGenerateOutput instance
        """
        return cls(
            response=data['response'],
            case_id=data.get('case_id', None),
        )

    def to_dict(self) -> dict:
        """
        Convert DripperGenerateOutput to a dictionary.

        Returns:
            Dictionary representation of the output data
        """
        output_dict = {}
        if self.case_id is not None:
            output_dict['case_id'] = self.case_id
        output_dict['response'] = self.response
        return output_dict


class DripperProcessData:
    """
    Data structure for intermediate processing results.

    Contains the raw HTML, simplified HTML, and mapped HTML at different
    stages of the extraction pipeline.
    """

    def __init__(
        self, raw_html: str, simpled_html: str, map_html: str, case_id: str = None
    ):
        """
        Initialize DripperProcessData.

        Args:
            raw_html: Original raw HTML content
            simpled_html: Simplified HTML after preprocessing
            map_html: Mapped HTML with item identifiers
            case_id: Optional identifier for the case being processed
        """
        self.raw_html = raw_html
        self.simpled_html = simpled_html
        self.map_html = map_html
        self.case_id = case_id

    @classmethod
    def from_dict(cls, data: dict) -> 'DripperProcessData':
        """
        Create DripperProcessData from a dictionary.

        Args:
            data: Dictionary containing 'raw_html', 'simpled_html', 'map_html',
                  and optionally 'case_id'

        Returns:
            DripperProcessData instance
        """
        return cls(
            raw_html=data['raw_html'],
            simpled_html=data['simpled_html'],
            map_html=data['map_html'],
            case_id=data.get('case_id', None),
        )

    def to_dict(self) -> dict:
        """
        Convert DripperProcessData to a dictionary.

        Returns:
            Dictionary representation of the processing data
        """
        output_dict = {}
        if self.case_id is not None:
            output_dict['case_id'] = self.case_id
        output_dict['raw_html'] = self.raw_html
        output_dict['simpled_html'] = self.simpled_html
        output_dict['map_html'] = self.map_html
        return output_dict


class DripperInput:
    """
    Input data structure for the Dripper extraction API.

    Contains the raw HTML to be processed, optional URL, and optional case ID.
    """

    def __init__(self, raw_html: str, url: str = None, case_id: str = None):
        """
        Initialize DripperInput.

        Args:
            raw_html: Raw HTML content to extract main content from
            url: Optional URL where the HTML was obtained from
            case_id: Optional identifier for the case being processed
        """
        self.raw_html = raw_html
        self.url = url
        self.case_id = case_id

    @classmethod
    def from_dict(cls, data: dict) -> 'DripperInput':
        """
        Create DripperInput from a dictionary.

        Args:
            data: Dictionary containing 'raw_html' and optionally 'url' and 'case_id'

        Returns:
            DripperInput instance
        """
        return cls(
            raw_html=data['raw_html'],
            url=data.get('url', None),
            case_id=data.get('case_id', None),
        )

    def to_dict(self) -> dict:
        """
        Convert DripperInput to a dictionary.

        Returns:
            Dictionary representation of the input data
        """
        output_dict = {}
        if self.case_id is not None:
            output_dict['case_id'] = self.case_id
        output_dict['raw_html'] = self.raw_html
        if self.url is not None:
            output_dict['url'] = self.url
        return output_dict


class DripperOutput:
    """
    Output data structure for the Dripper extraction API.

    Contains the extracted main HTML content and optional case ID.
    """

    def __init__(self, main_html: str, case_id: str = None):
        """
        Initialize DripperOutput.

        Args:
            main_html: Extracted main HTML content
            case_id: Optional identifier for the case being processed
        """
        self.main_html = main_html
        self.case_id = case_id

    @classmethod
    def from_dict(cls, data: dict) -> 'DripperOutput':
        """
        Create DripperOutput from a dictionary.

        Args:
            data: Dictionary containing 'main_html' and optionally 'case_id'

        Returns:
            DripperOutput instance
        """
        return cls(
            main_html=data['main_html'],
            case_id=data.get('case_id', None),
        )

    def to_dict(self) -> dict:
        """
        Convert DripperOutput to a dictionary.

        Returns:
            Dictionary representation of the output data
        """
        output_dict = {}
        if self.case_id is not None:
            output_dict['case_id'] = self.case_id
        output_dict['main_html'] = self.main_html
        return output_dict
