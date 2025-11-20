"""
Logits processing and response parsing utilities.

This module provides functions for building token state machines and parsing
LLM responses into structured JSON format.
"""

import json

from transformers import AutoTokenizer

from dripper.exceptions import DripperLogitsError, DripperResponseParseError
from dripper.inference.logtis_processor.logits_v1 import \
    TokenStateMachine as TokenStateMachine_v1
from dripper.inference.logtis_processor.logits_v2 import \
    TokenStateMachine as TokenStateMachine_v2


def build_token_state_machine(
    max_count: int,
    tokenizer: AutoTokenizer,
    device: str = 'cuda',
    version: str = 'v1',
) -> TokenStateMachine_v1 | TokenStateMachine_v2:
    """
    Build a token state machine for structured generation.

    Creates a TokenStateMachine instance based on the specified version.
    v1 and v2 differ in their initialization logic (v2 includes Think logic).

    Args:
        max_count: Maximum number of items to process
        tokenizer: Tokenizer for encoding/decoding tokens
        device: Device to use for tensor operations ('cuda' or 'cpu')
        version: State machine version ('v1' or 'v2')

    Returns:
        TokenStateMachine instance (v1 or v2)

    Raises:
        DripperLogitsError: If version is invalid
    """
    if version == 'v1':
        return TokenStateMachine_v1(max_count, tokenizer, device)
    elif version == 'v2':
        return TokenStateMachine_v2(max_count, tokenizer, device)
    else:
        raise DripperLogitsError(f'Invalid version: {version}')


def find_brace_pair(response: str) -> str:
    """
    Extract JSON content by finding the first '{' and last '}' in response.

    Attempts to extract a valid JSON object from the response string by
    locating the outermost brace pair.

    Args:
        response: Raw response string from LLM

    Returns:
        Substring containing the JSON object (from first '{' to last '}')

    Raises:
        DripperResponseParseError: If no left brace is found
    """
    first_brace_index = response.find('{')
    if first_brace_index == -1:
        raise DripperResponseParseError('No left brace found.')
    last_brace_index = response.rfind('}')
    if last_brace_index == -1:
        # If no closing brace found, return from first brace to end
        return response[first_brace_index:]
    else:
        # Return substring from first '{' to last '}'
        return response[first_brace_index : last_brace_index + 1]


def parse_json_by_remove_last_chars(response: str) -> dict:
    """
    Parse JSON by progressively removing characters from the end.

    Attempts to parse JSON by trying progressively shorter prefixes of the
    response string, appending '}' to each attempt. This handles cases where
    the JSON might be truncated or have extra characters at the end.

    Args:
        response: Response string to parse

    Returns:
        Parsed JSON dictionary

    Raises:
        DripperResponseParseError: If no valid JSON prefix can be found
    """
    idx = len(response)
    while idx > 0:
        try:
            # Try parsing with current prefix plus closing brace
            return json.loads(response[:idx] + '}')
        except Exception:
            # If parsing fails, try shorter prefix
            idx -= 1
    raise DripperResponseParseError(
        'No valid prefix can be parsed as a json dict'
    )


def parse_llm_response(response: str) -> dict:
    """
    Parse LLM response into a dictionary.

    Attempts to extract and parse JSON from the LLM response using multiple
    strategies:
    1. Extract brace pair and parse directly
    2. If that fails, try progressive character removal

    Args:
        response: Raw response string from LLM

    Returns:
        Parsed dictionary with item IDs as keys and 'main'/'other' as values

    Raises:
        DripperResponseParseError: If response cannot be parsed as JSON
    """
    # First, try to extract JSON content by finding brace pair
    clean_response = find_brace_pair(response)

    try:
        # Try direct JSON parsing
        return json.loads(clean_response)
    except Exception as e:
        try:
            # If direct parsing fails, try progressive character removal
            return parse_json_by_remove_last_chars(clean_response)
        except Exception:
            raise DripperResponseParseError(
                f'Cannot parse JSON response, the raw response is {response}. '
                f'Error: {e}'
            )
