import json
import re

from mineru_html.base import MinerUHTMLCase, MinerUHTMLParseResult
from mineru_html.exceptions import (MinerUHTMLError,
                                    MinerUHTMLResponseParseError)


def find_brace_pair(response: str) -> str:
    """Extract JSON content between first '{' and last '}'.

    Args:
        response: Response string that may contain JSON.

    Returns:
        String containing JSON content between braces.
    """
    first_brace_index = response.find('{')
    if first_brace_index == -1:
        raise MinerUHTMLResponseParseError('No left brace found.')
    last_brace_index = response.rfind('}')
    if last_brace_index == -1:
        return response[first_brace_index:]
    else:
        return response[first_brace_index : last_brace_index + 1]


def parse_json_by_remove_last_chars(response: str) -> dict:
    """Parse JSON by progressively removing characters from the end.

    This is a fallback method when the response is not valid JSON but may
    contain a valid JSON prefix.

    Args:
        response: Response string to parse.

    Returns:
        Parsed dictionary.
    """
    idx = len(response)
    while idx > 0:
        if idx <= 1:
            raise MinerUHTMLResponseParseError(
                'No valid prefix can be parsed as a json dict'
            )
        try:
            return json.loads(response[:idx] + '}')
        except Exception:
            idx -= 1


def parse_llm_response(response: str) -> dict[str, str]:
    """Parse LLM response into a dictionary of item_id -> label.

    Supports both JSON format and compact format (e.g., "1main2other").

    Args:
        response: Raw response string from LLM.

    Returns:
        Dictionary mapping item_id (as string) to label ("main" or "other").
    """
    try:
        clean_response = find_brace_pair(response)
    except MinerUHTMLResponseParseError:
        pattern = r'(\d+)(main|other)'
        matches = [(m.group(1), m.group(2)) for m in re.finditer(pattern, response)]
        if len(matches) == 0:
            raise MinerUHTMLResponseParseError(
                f'Not valid response, the raw response is {response}'
            )
        pred_dict = {item_id: category for item_id, category in matches}
        return pred_dict
    else:
        try:
            return json.loads(clean_response)
        except Exception:
            try:
                return parse_json_by_remove_last_chars(clean_response)
            except Exception as e:
                raise MinerUHTMLResponseParseError(
                    f'Cannot parse JSON response, the raw response is {response}. Error: {e}'
                )


def parse_result(input_case: MinerUHTMLCase) -> MinerUHTMLCase:
    """
    Parse result from generate output

    Args:
        input_case: input case
    Returns:
        input case
    """
    try:
        item_label = parse_llm_response(input_case.generate_output.response)
        input_case.parse_result = MinerUHTMLParseResult(item_label=item_label)
        return input_case
    except Exception as e:
        if isinstance(e, MinerUHTMLError):
            e.set_case_id(input_case.case_id)
            raise e
        else:
            raise MinerUHTMLResponseParseError(
                f'Parse result failed: {str(e)}', case_id=input_case.case_id
            ) from e
