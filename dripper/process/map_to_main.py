"""
Map HTML to main content extraction utilities.

This module provides functions to extract main content from mapped HTML
using LLM response labels, removing non-main content elements.
"""

from typing import Callable

from lxml import html

from dripper.base import ITEM_ID_ATTR, TAIL_BLOCK_TAG, TagType
from dripper.process.html_utils import element_to_html, html_to_element


def remove_recursive_by_condition(
    root: html.HtmlElement, remove_condition: Callable[[html.HtmlElement], bool]
) -> html.HtmlElement:
    """
    Recursively remove elements from DOM based on a condition.

    Removes elements that satisfy the condition, and only processes children
    if the current element was not removed.

    Args:
        root: Root HTML element to process
        remove_condition: Function that returns True if element should be removed

    Returns:
        The root element (may be removed from its parent if condition matched)
    """
    current_removed = False
    if remove_condition(root):
        parent = root.getparent()
        if parent is not None:
            parent.remove(root)
            current_removed = True
    if not current_removed:
        for child in root.iterchildren():
            remove_recursive_by_condition(child, remove_condition)
    return root


def extract_main_html(map_html: str, response: dict) -> str:
    """
    Extract main content HTML using LLM response labels.

    Uses the LLM's response to identify which elements should be kept as main
    content, then extracts those elements and their ancestors/descendants from
    the mapped HTML.

    Args:
        map_html: Preprocessed HTML with item IDs (mapped HTML)
        response: LLM response dictionary mapping item IDs to tag types
                 (e.g., {'1': 'main', '2': 'other', ...})

    Returns:
        Extracted main content HTML string
    """
    root = html_to_element(map_html)

    # Collect all elements that should be kept (main content and their relations)
    elements_to_remained = set()
    for remained_id in response:
        if response[remained_id] == TagType.Main.value:
            # Find element with this item ID
            elem_list = root.xpath(f'//*[@{ITEM_ID_ATTR}="{remained_id}"]')
            if len(elem_list) > 0:
                elem = elem_list[0]
            else:
                continue

            # Add element itself and all its descendants
            for child in elem.iter():
                elements_to_remained.add(child)

            # Add all ancestors of the element
            for ancestor in elem.iterancestors():
                elements_to_remained.add(ancestor)

    # Remove all elements that are not in the set of elements to keep
    remove_recursive_by_condition(root, lambda x: x not in elements_to_remained)

    # Remove tail block tags (unwrap them, keeping their content)
    for tail_block in root.xpath(f'//{TAIL_BLOCK_TAG}'):
        tail_block.drop_tag()

    return element_to_html(root)
