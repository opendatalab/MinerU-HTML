"""
HTML utility functions for element conversion.

This module provides helper functions to convert between HTML strings
and lxml HtmlElement objects.
"""

import html
from lxml import html as lxmlhtml


def html_to_element(html_str: str) -> lxmlhtml.HtmlElement:
    """
    Convert HTML string to lxml HtmlElement.

    Parses HTML string into an lxml HtmlElement tree with optimized parser
    settings for HTML processing.

    Args:
        html_str: HTML string to parse (can be str or bytes)

    Returns:
        Root HtmlElement of the parsed HTML tree
    """
    parser = lxmlhtml.HTMLParser(
        collect_ids=False,  # Don't collect ID attributes for performance
        encoding='utf-8',
        remove_blank_text=True,  # Remove blank text nodes
        remove_comments=True,  # Remove HTML comments
        remove_pis=True,  # Remove processing instructions
    )

    # Convert string to bytes if it contains an encoding declaration
    # This is needed for proper parsing when encoding is specified
    if isinstance(html_str, str) and (
        '<?xml' in html_str
        or '<meta charset' in html_str
        or 'encoding=' in html_str
    ):
        html_str = html_str.encode('utf-8')

    root = lxmlhtml.fromstring(html_str, parser=parser)
    return root


def element_to_html(root: lxmlhtml.HtmlElement, pretty_print=False) -> str:
    """
    Convert lxml HtmlElement to HTML string.

    Serializes an HtmlElement tree back to an HTML string.

    Args:
        root: Root HtmlElement to serialize
        pretty_print: Whether to format the output with indentation and line breaks

    Returns:
        HTML string representation of the element tree
    """
    html_str = lxmlhtml.tostring(
        root, pretty_print=pretty_print, encoding='utf-8'
    ).decode()
    return html_str


def element_to_html_unescaped(element: lxmlhtml.HtmlElement) -> str:
    """
    Convert lxml HtmlElement to HTML string without escaping.

    Serializes an lxml HtmlElement tree back to an HTML string without escaping.

    Args:
        element: lxml.html.HtmlElement

    Returns:
        HTML string representation of the element tree
    """
    s = element_to_html(element)
    return html.unescape(s)
