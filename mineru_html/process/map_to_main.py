from typing import Callable, Literal

from lxml import html

from mineru_html.base import MinerUHTMLCase, MinerUHTMLOutput
from mineru_html.constants import ITEM_ID_ATTR, TAIL_BLOCK_TAG, TagType
from mineru_html.exceptions import (MinerUHTMLError, MinerUHTMLFallbackError,
                                    MinerUHTMLMapToMainError)
from mineru_html.process.html_utils import (decode_http_urls_only,
                                            element_to_html, html_to_element)


class BaseFallbackHandler:
    """Base class for fallback handlers."""

    def fallback_func(self, input_html: str) -> str:
        """Extract content using fallback method."""

        raise NotImplementedError('Subclasses must implement this method')


class BypassFallbackHandler(BaseFallbackHandler):
    """Fallback handler that returns the input HTML unchanged."""

    def fallback_func(self, input_html: str) -> str:
        """Return input HTML unchanged."""
        return input_html


class EmptyFallbackHandler(BaseFallbackHandler):
    """Fallback handler that returns empty string."""

    def fallback_func(self, input_html: str) -> str:
        return ''


class TrafilaturaFallbackHandler(BaseFallbackHandler):
    """Fallback handler using trafilatura library for content extraction."""
    def __init__(self):
        """Initialize trafilatura extractor."""
        try:
            from trafilatura import extract
            from trafilatura.settings import Extractor

            self.trafilatura_settings = Extractor(output_format='html', comments=False)
            self.trafilatura = extract
        except Exception as e:
            raise MinerUHTMLFallbackError(f'Failed to import trafilatura: {e}') from e

    def fallback_func(self, input_html: str) -> str:
        """Extract content using trafilatura."""
        result = self.trafilatura(input_html, options=self.trafilatura_settings)
        return result if result is not None else ''


def get_fallback_handler(
    fallback_type: Literal['trafilatura', 'bypass', 'empty'],
) -> BaseFallbackHandler:
    """Get a fallback handler instance.

    Args:
        fallback_type: Type of fallback handler to create.

    Returns:
        Fallback handler instance.
    """
    if fallback_type == 'trafilatura':
        return TrafilaturaFallbackHandler()
    elif fallback_type == 'bypass':
        return BypassFallbackHandler()
    elif fallback_type == 'empty':
        return EmptyFallbackHandler()
    else:
        raise MinerUHTMLFallbackError(f'Invalid fallback type: {fallback_type}')


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

    elements_to_remained = set()
    for remained_id in response:
        if response[remained_id] == TagType.Main.value:
            elem_list = root.xpath(f'//*[@{ITEM_ID_ATTR}="{remained_id}"]')
            if len(elem_list) > 0:
                elem = elem_list[0]
            else:
                continue
            for child in elem.iter():
                elements_to_remained.add(child)
            for ancestor in elem.iterancestors():
                elements_to_remained.add(ancestor)

    # Recall not selected br tags that are adjacent to main content
    last_element: html.HtmlElement | None = None
    for element in root.iter():
        if last_element is not None:
            if element.tag == 'br' and (
                last_element in elements_to_remained and not last_element.tag == 'br'
            ):
                elements_to_remained.add(element)
            if last_element.tag == 'br' and (
                element in elements_to_remained and not element.tag == 'br'
            ):
                elements_to_remained.add(last_element)
        last_element = element

    remove_recursive_by_condition(root, lambda x: x not in elements_to_remained)

    for tail_block in root.xpath(f'//{TAIL_BLOCK_TAG}'):
        tail_block.drop_tag()

    return decode_http_urls_only(element_to_html(root))


def extract_main_html_single(input_case: MinerUHTMLCase) -> MinerUHTMLCase:
    """
    Extract main content HTML using LLM response labels.
    Uses the LLM's response to identify which elements should be kept as main
    content, then extracts those elements and their ancestors/descendants from
    the mapped HTML.

    Args:
        input_case: Case containing process_data and parse_result.

    Returns:
        Case with output_data set to extracted main HTML.
    """
    try:
        main_html = extract_main_html(
            input_case.process_data.map_html, input_case.parse_result.item_label
        )
        input_case.output_data = MinerUHTMLOutput(main_html=main_html)
        return input_case
    except Exception as e:
        if isinstance(e, MinerUHTMLError):
            e.set_case_id(input_case.case_id)
            raise e
        else:
            raise MinerUHTMLMapToMainError(
                f'Extract main HTML with parse result {input_case.parse_result.item_label} failed: {str(e)}',
                case_id=input_case.case_id,
            ) from e


def extract_main_html_fallback(
    input_case: MinerUHTMLCase, fallback_handler: BaseFallbackHandler
) -> MinerUHTMLCase:
    """
    Extract main content HTML using fallback handler.

    Uses the fallback handler to extract main content HTML from the process data.
    """
    try:
        main_html = fallback_handler.fallback_func(input_case.input_data.raw_html)
        input_case.output_data = MinerUHTMLOutput(main_html=main_html)
        return input_case
    except Exception as e:
        if isinstance(e, MinerUHTMLError):
            e.set_case_id(input_case.case_id)
            raise e
        else:
            raise MinerUHTMLMapToMainError(
                f'Extract main HTML with fallback handler {fallback_handler.__class__.__name__} failed: {str(e)}',
                case_id=input_case.case_id,
            ) from e
