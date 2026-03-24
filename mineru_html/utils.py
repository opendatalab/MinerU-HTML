import re
from typing import List

from mineru_html.constants import ITEM_ID_ATTR


def get_all_item_ids(input_str: str) -> List[int]:
    """Extract all item IDs from HTML string.

    Args:
        input_str: HTML string containing elements with _item_id attributes.

    Returns:
        List of item IDs as integers.
    """
    pattern = f'\\s{ITEM_ID_ATTR}="(\\d+)"'
    matches = re.findall(pattern, input_str)
    if not matches:
        return []
    try:
        return [int(match) for match in matches]
    except ValueError as e:
        raise ValueError(f'Failed to convert item_id to int: {e}') from e


def build_dummy_response(item_ids: List[int], response_format: str) -> str:
    """Build a dummy response string for length estimation.

    Args:
        item_ids: List of item IDs to include in the response.
        response_format: Must be "json" or "compact". Compact uses e.g. "1other2other".

    Returns:
        Dummy response string in the specified format.
    """
    assert response_format == 'json' or response_format == 'compact'
    if response_format == 'json':
        if len(item_ids) == 0:
            return '{}'
        str_pieces = ['{']
        for item_idx in range(len(item_ids)):
            if item_idx == len(item_ids) - 1:
                str_pieces.append(f' "{item_ids[item_idx]}":"other"' + '}')
            else:
                str_pieces.append(f' "{item_ids[item_idx]}":"other",')
        return ''.join(str_pieces)
    elif response_format == 'compact':
        if len(item_ids) == 0:
            return ''
        return ''.join(f'{item_id}other' for item_id in item_ids)
