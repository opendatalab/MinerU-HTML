import html
import re

from lxml import html as lxml_html
from lxml.etree import ParserError


def html_to_element(html_str: str) -> lxml_html.HtmlElement:
    """
    Convert HTML string to element.

    Args:
        html_str: HTML string

    Returns:
        HTML element
    """

    parser = lxml_html.HTMLParser(
        collect_ids=False,
        encoding='utf-8',
        remove_blank_text=True,
        remove_comments=True,
        remove_pis=True,
    )
    if isinstance(html_str, str) and (
        '<?xml' in html_str or '<meta charset' in html_str or 'encoding=' in html_str
    ):
        html_input = html_str.encode('utf-8')
    else:
        html_input = html_str

    try:
        root = lxml_html.fromstring(html_input, parser=parser)
    except ParserError as e:
        if 'Document is empty' in str(e):
            return lxml_html.HtmlElement()
        raise e
    return root


def element_to_html(root: lxml_html.HtmlElement, pretty_print=False) -> str:
    """
    Convert HTML element to string.

    Args:
        root: HTML element

    Returns:
        HTML string
    """
    html_bytes = lxml_html.tostring(root, pretty_print=pretty_print, encoding='utf-8')
    html_str = (
        html_bytes.decode('utf-8') if isinstance(html_bytes, bytes) else html_bytes
    )
    return html_str


def element_to_html_unescaped(element: lxml_html.HtmlElement) -> str:
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


def decode_http_urls_only(html_str: str) -> str:
    """
    Decode (unescape) only the URLs in href or src attributes that start with http://, https://, ftp://, or //.

    This function searches the input HTML string for href or src attributes and, if the URL value starts
    with an allowed prefix, unescapes any HTML entities in that URL. Other URLs (not starting with those
    protocols) and all other text are left untouched.

    Args:
        html_str: The HTML content as a string.

    Returns:
        The HTML content with URLs in allowed href/src attributes decoded.
    """

    def decode_match(match):
        prefix = match.group(1)  # href=" or src="
        url = match.group(2)
        suffix = match.group(3)  # "

        if url.startswith(
            ('http://', 'https://', 'ftp://', 'HTTP://', 'HTTPS://', 'FTP://', '//')
        ):
            decoded_url = html.unescape(url)
            return f'{prefix}{decoded_url}{suffix}'
        return match.group(0)

    pattern = r'(href="|src=")(.*?)(")'
    return re.sub(pattern, decode_match, html_str, flags=re.IGNORECASE | re.DOTALL)
