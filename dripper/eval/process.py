"""
HTML processing utilities for evaluation pipeline.

This module provides functions for preprocessing HTML, extracting labeled content,
and converting between HTML and text formats for evaluation purposes.
"""

import html_text
from lxml import html

from dripper.base import ITEM_ID_ATTR, SELECT_ATTR, TagType
from dripper.process.html_utils import element_to_html, html_to_element
from dripper.process.map_to_main import extract_main_html
from dripper.process.simplify_html import simplify_html


def prune_labeled_html(labeled_html: str) -> str:
    """
    Prune HTML DOM tree based on labeled selections.

    Only keeps elements that are selected in the labeled HTML and their ancestors.
    Also preserves adjacent `<br>` tags for proper text formatting.

    Args:
        labeled_html: HTML string with selection labels (SELECT_ATTR attributes)

    Returns:
        Pruned HTML string containing only selected elements and their ancestors
    """

    root = html_to_element(labeled_html)

    elements_to_remained: set[html.HtmlElement] = set()

    def walk_tree_to_add_elements(element: html.HtmlElement):
        """
        Recursively walk the tree to collect selected elements.

        If an element is selected, all its descendants are kept.
        Elements with 'display: none' are skipped.
        """
        style_attr = element.get('style', '')
        # Skip hidden elements
        if 'display: none' in style_attr or 'display:none' in style_attr:
            return
        if element.get(SELECT_ATTR) == 'true':
            # If this element is selected, keep all elements in its subtree
            for item in element.iter():
                elements_to_remained.add(item)
        else:
            # Check if any child elements are selected
            for item in element.iterchildren():
                walk_tree_to_add_elements(item)

    walk_tree_to_add_elements(root)

    # Add all ancestors of selected elements to maintain tree structure
    all_elements_to_remained = elements_to_remained.copy()
    for element in elements_to_remained:
        # Record all ancestors up to the root
        for ancestor in element.iterancestors():
            if ancestor not in all_elements_to_remained:
                all_elements_to_remained.add(ancestor)
            else:
                # If an ancestor is already in the set, we can break
                # (all higher ancestors are already added)
                break

    # Preserve adjacent <br> tags for proper text formatting
    # Add <br> tags that are adjacent to selected elements
    last_element: html.HtmlElement = None
    for element in root.iter():
        if last_element is not None:
            # If current element is <br> and previous is selected (non-<br>), keep <br>
            if (
                element.tag == 'br'
                and (
                    last_element in all_elements_to_remained
                    and not last_element.tag == 'br'
                )
            ):
                all_elements_to_remained.add(element)
            # If previous element is <br> and current is selected (non-<br>), keep <br>
            if (
                last_element.tag == 'br'
                and (
                    element in all_elements_to_remained
                    and not element.tag == 'br'
                )
            ):
                all_elements_to_remained.add(last_element)
        last_element = element

    # Remove all elements that are not in the keep set
    all_element_to_drop: list[html.HtmlElement] = []
    for element in root.iter():
        if element not in all_elements_to_remained:
            all_element_to_drop.append(element)
    for element in all_element_to_drop:
        if element.getparent() is not None:
            element.drop_tree()
    return element_to_html(root)


class HTML2TextWrapper:
    """
    Wrapper for html2text converter with configured settings.

    Converts HTML to Markdown text format, ignoring links and images.
    """

    def __init__(self):
        """Initialize HTML2Text converter with default settings."""
        import html2text

        self.converter = html2text.HTML2Text(bodywidth=0)
        self.converter.ignore_links = True
        self.converter.ignore_images = True

    def __call__(self, html_str: str, url: str = '') -> str:
        """
        Convert HTML string to Markdown text.

        Args:
            html_str: HTML string to convert
            url: Base URL for resolving relative links (optional)

        Returns:
            Converted Markdown text string
        """
        self.converter.baseurl = url
        text = self.converter.handle(html_str)
        self.converter.baseurl = ''
        return text


def html_to_text_func(html_str: str, url: str = '', format: str = 'MD') -> str:
    """
    Convert HTML string to plain text string.

    Supports two formats:
    - 'MD': Markdown format using html2text
    - Other: Plain text extraction using html_text

    Args:
        html_str: HTML string to convert
        url: Base URL for resolving relative links (optional, for MD format)
        format: Output format, either 'MD' for Markdown or other for plain text.
                Defaults to 'MD'.

    Returns:
        Converted text string in the specified format
    """
    if format == 'MD':
        instance = HTML2TextWrapper()
        return instance(html_str, url)
    else:
        return html_text.extract_text(html_str)


def itemify_id_html_to_item_label(map_html: str) -> dict[str, str]:
    """
    Extract item labels from mapped HTML with item IDs.

    Scans HTML elements with ITEM_ID_ATTR attributes and determines their labels
    (Main or Other) based on whether they or their descendants are selected.

    Args:
        map_html: HTML string with item ID attributes and selection labels

    Returns:
        Dictionary mapping item ID strings to label strings ('main' or 'other'),
        sorted by item ID as integers
    """

    def check_item_is_selected(item_element: html.HtmlElement) -> str:
        """
        Check if an item element or any of its descendants is selected.

        Args:
            item_element: HTML element to check

        Returns:
            'true' if selected, 'false' otherwise
        """
        # If any descendant has SELECT_ATTR=="true", return 'true'
        for element in item_element.iter():
            if element.get(SELECT_ATTR) == 'true':
                return 'true'
        return 'false'

    def convert_raw_label_to_tag_type(raw_label: str) -> TagType:
        """
        Convert raw label string to TagType enum.

        Args:
            raw_label: Raw label string ('true' or 'false')

        Returns:
            TagType.Main if 'true', TagType.Other otherwise
        """
        if raw_label == 'true':
            return TagType.Main
        else:
            return TagType.Other

    root = html_to_element(map_html)
    item_element_map = {}
    # Scan all elements for item IDs and determine their labels
    for element in root.iter():
        _item_id_str = element.get(ITEM_ID_ATTR)
        if _item_id_str is not None:
            raw_label = check_item_is_selected(element)
            item_element_map[_item_id_str] = convert_raw_label_to_tag_type(
                raw_label
            ).value

    # Sort item_element_map by item ID as integers
    item_label = {
        k: item_element_map[k]
        for k in sorted(item_element_map.keys(), key=lambda x: int(x))
    }
    return item_label


def pre_process(raw_html: str) -> tuple[str, str]:
    """
    Preprocess raw HTML to get simplified HTML and mapped HTML.

    Simplifies the HTML structure and adds item IDs for tracking.
    This is the first step in the evaluation pipeline.

    Args:
        raw_html: Raw HTML string to preprocess

    Returns:
        Tuple containing:
        - simplified_html: Simplified HTML structure
        - map_html: HTML with item ID attributes for mapping
    """
    simpled_html, map_html = simplify_html(raw_html)
    return simpled_html, map_html


def post_process(map_html: str, item_label: dict[str, str]) -> str:
    """
    Post-process mapped HTML using item labels to extract main content.

    Uses the item labels (Main/Other) to extract only the main content
    from the mapped HTML. This is the final step in the evaluation pipeline.

    Args:
        map_html: HTML string with item ID attributes
        item_label: Dictionary mapping item ID strings to label strings
                    ('main' or 'other')

    Returns:
        Extracted main content HTML string
    """
    main_html = extract_main_html(map_html, item_label)
    return main_html
