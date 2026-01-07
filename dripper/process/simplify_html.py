"""
HTML simplification and processing utilities.

This module provides functions to simplify HTML structure, extract paragraphs,
and process HTML content for main content extraction tasks.
"""

import copy
import re
import uuid
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup
from lxml import etree, html
from selectolax.parser import HTMLParser

# Inline tags that should be treated as inline elements
inline_tags = {
    'map', 'optgroup', 'span', 'br', 'input', 'time', 'u', 'strong', 'textarea', 'small', 'sub',
    'samp', 'blink', 'b', 'code', 'nobr', 'strike', 'bdo', 'basefont', 'abbr', 'var', 'i', 'cccode-inline',
    'select', 's', 'pic', 'label', 'mark', 'object', 'dd', 'dt', 'ccmath-inline', 'svg', 'li',
    'button', 'a', 'font', 'dfn', 'sup', 'kbd', 'q', 'script', 'acronym', 'option', 'img', 'big', 'cite',
    'em', 'marked-tail', 'marked-text'
    # 'td', 'th'  # Commented out: table cells are handled specially
}

# Regarded as block level elements, there are no block level elements inside by default
no_block_tags = {
    "math"
}

# Labels that do not calculate text length
no_calc_text_tags = {
    "math", "table"
}

# Tags to remove from HTML (navigation, metadata, etc.)
tags_to_remove = {
    'head',
    'header',
    'footer',
    'nav',
    'aside',
    'style',
    'script',
    'noscript',
    'link',
    'meta',
    'iframe',
    'frame'
}

# Special tags to preserve even if they are inline tags
EXCLUDED_TAGS = {'img', 'br', 'li', 'dt', 'dd', 'td', 'th'}

# Attribute name patterns to remove (standalone words)
ATTR_PATTERNS_TO_REMOVE = {
    'nav', 'footer', 'header',  # Standalone words
}

# Attribute name patterns to remove (specific prefixes/suffixes)
ATTR_SUFFIX_TO_REMOVE = {
    # '-nav', '_nav',
    # '-footer', '_footer',  # Commented: special cases where dl lists may have custom footer attributes
    # '-header', '_header',  # Commented: special cases where custom headers may contain titles
}

# Custom tag for tail block elements
tail_block_tag = 'cc-alg-uc-text'


def add_data_uids(dom: html.HtmlElement) -> None:
    """
    Add data-uid attributes to all DOM nodes (recursively for all child nodes).

    Args:
        dom: HTML element to process
    """
    for node in dom.iter():
        try:
            node.set('data-uid', str(uuid.uuid4()))
        except TypeError:
            pass


def remove_all_uids(dom: html.HtmlElement) -> None:
    """
    Remove all data-uid attributes from DOM.

    Args:
        dom: HTML element to process
    """
    for node in dom.iter():
        if 'data-uid' in node.attrib:
            del node.attrib['data-uid']


def build_uid_map(dom: html.HtmlElement) -> Dict[str, html.HtmlElement]:
    """
    Build a mapping dictionary from data-uid to nodes.

    Args:
        dom: HTML element to process

    Returns:
        Dictionary mapping data-uid values to their corresponding nodes
    """
    return {node.get('data-uid'): node for node in dom.iter() if node.get('data-uid')}


def is_unique_attribute(tree, attr_name, attr_value):
    """
    Check if the given attribute name and value combination is unique in the document.

    Args:
        tree: XML/HTML tree to search
        attr_name: Attribute name to check
        attr_value: Attribute value to check

    Returns:
        True if the attribute-value combination appears exactly once, False otherwise
    """
    elements = tree.xpath(f"//*[@{attr_name}='{attr_value}']")
    return len(elements) == 1


def is_data_table(table_element: html.HtmlElement) -> bool:
    """
    Determine if a table is a data table rather than a layout table.

    Checks various indicators that suggest the table contains actual data
    rather than being used for page layout.

    Args:
        table_element: Table element to check

    Returns:
        True if the table is a data table, False if it's a layout table
    """
    # Check if table has caption tag
    if table_element.xpath('.//caption'):
        return True

    # Check if table has th tags
    if table_element.xpath('.//th'):
        return True

    # Check if table has thead or tfoot tags
    if table_element.xpath('.//thead') or table_element.xpath('.//tfoot'):
        return True

    # Check if table has colgroup or col tags
    if table_element.xpath('.//colgroup') or table_element.xpath('.//col'):
        return True

    # Check if table has summary attribute
    if table_element.get('summary'):
        return True

    # Check if table has role="table" or data-table attribute
    if table_element.get('role') == 'table' or table_element.get('data-table'):
        return True

    # Check if cells have headers attribute
    if table_element.xpath('.//*[@headers]'):
        return True

    return False


def extract_paragraphs(
    processing_dom: html.HtmlElement,
    uid_map: Dict[str, html.HtmlElement],
    include_parents: bool = True,
) -> List[Dict[str, str]]:
    """
    Extract paragraphs from HTML DOM.

    The content_type field is used to identify the type of paragraph content.
    Possible values include:

        'block_element': Standalone block-level element

        'inline_elements': Pure inline element combination

        'unwrapped_text': Unwrapped plain text content

        'mixed': Mixed content (contains both text and inline elements)

    Args:
        processing_dom: DOM element to process
        uid_map: Dictionary mapping data-uid to original elements
        include_parents: Whether to include parent elements in the structure

    Returns:
        List of paragraphs, each containing:
        - html: HTML string of the paragraph
        - content_type: Type of content in the paragraph
        - _original_element: Reference to the original element
    """

    # Create table type mapping to record whether each table is a data table or layout table
    table_types = {}

    # Analyze all table types first
    for table in processing_dom.xpath('.//table'):
        table_types[table.get('data-uid')] = is_data_table(table)

    def is_block_element(node) -> bool:
        """Determine if a node is a block-level element."""
        # Handle special case for table cells
        if node.tag in no_block_tags:
            return False
        if node.tag in ('td', 'th'):
            # Find the nearest ancestor table element
            table_ancestor = node
            while table_ancestor is not None and table_ancestor.tag != 'table':
                table_ancestor = table_ancestor.getparent()

            # For table cells, determine if block-level based on table type
            if table_ancestor is not None:
                table_uid = table_ancestor.get('data-uid')
                if table_types.get(table_uid, False):
                    # Data table td/th are not treated as block elements
                    return False
                else:
                    # Layout table td/th are treated as block elements
                    return True

        # Default handling for other elements
        if node.tag in inline_tags:
            return False
        return isinstance(node, html.HtmlElement)

    def has_block_children(node) -> bool:
        """Determine if a node has block-level children."""
        if node.tag in no_block_tags:
            return False
        return any(is_block_element(child) for child in node.iterchildren())

    def clone_structure(
        path: List[html.HtmlElement],
    ) -> Tuple[html.HtmlElement, html.HtmlElement]:
        """Clone node structure."""
        if not path:
            raise ValueError('Path cannot be empty')
        if not include_parents:
            last_node = html.Element(path[-1].tag, **path[-1].attrib)
            return last_node, last_node
        root = html.Element(path[0].tag, **path[0].attrib)
        current = root
        for node in path[1:-1]:
            new_node = html.Element(node.tag, **node.attrib)
            current.append(new_node)
            current = new_node
        last_node = html.Element(path[-1].tag, **path[-1].attrib)
        current.append(last_node)
        return root, last_node

    paragraphs = []

    def process_node(node: html.HtmlElement, path: List[html.HtmlElement]):
        """Recursively process nodes."""
        current_path = path + [node]
        inline_content = []
        content_sources = []

        # Process node text
        if node.text and node.text.strip():
            inline_content.append(('direct_text', node.text.strip()))
            content_sources.append('direct_text')

        # Process child nodes
        for child in node:
            if is_block_element(child):
                # Process accumulated inline content
                if inline_content:
                    try:
                        root, last_node = clone_structure(current_path)
                        merge_inline_content(last_node, inline_content)

                        content_type = 'mixed'
                        if all(t == 'direct_text' for t in content_sources):
                            content_type = 'unwrapped_text'
                        elif all(t == 'element' for t in content_sources):
                            content_type = 'inline_elements'

                        # Get original element
                        original_element = uid_map.get(node.get('data-uid'))
                        paragraphs.append({
                            'html': etree.tostring(root, encoding='unicode').strip(),
                            'content_type': content_type,
                            '_original_element': original_element,  # Add original element reference
                        })
                    except ValueError:
                        pass
                    inline_content = []
                    content_sources = []

                # Process block-level elements
                if not has_block_children(child):
                    try:
                        root, last_node = clone_structure(current_path + [child])
                        last_node.text = child.text if child.text else None
                        for grandchild in child:
                            last_node.append(copy.deepcopy(grandchild))

                        # Get original element
                        original_element = uid_map.get(child.get('data-uid'))
                        paragraphs.append({
                            'html': etree.tostring(root, encoding='unicode').strip(),
                            'content_type': 'block_element',
                            '_original_element': original_element,  # Add original element reference
                        })
                    except ValueError:
                        pass
                else:
                    process_node(child, current_path)

                # Process tail text
                if child.tail and child.tail.strip():
                    inline_content.append(('tail_text', child.tail.strip()))
                    content_sources.append('tail_text')
            else:
                inline_content.append(('element', child))
                content_sources.append('element')
                if child.tail and child.tail.strip():
                    inline_content.append(('tail_text', child.tail.strip()))
                    content_sources.append('tail_text')

        # Process remaining inline content
        if inline_content:
            try:
                root, last_node = clone_structure(current_path)
                merge_inline_content(last_node, inline_content)

                content_type = 'mixed'
                if all(t == 'direct_text' for t in content_sources):
                    content_type = 'unwrapped_text'
                elif all(t == 'element' for t in content_sources):
                    content_type = 'inline_elements'
                elif all(
                    t in ('direct_text', 'tail_text') for t in content_sources
                ):
                    content_type = 'unwrapped_text'

                # Get original element
                original_element = uid_map.get(node.get('data-uid'))
                paragraphs.append({
                    'html': etree.tostring(root, encoding='unicode').strip(),
                    'content_type': content_type,
                    '_original_element': original_element,  # Add original element reference
                })
            except ValueError:
                pass

    def merge_inline_content(
        parent: html.HtmlElement, content_list: List[Tuple[str, str]]
    ):
        """Merge inline content."""
        last_inserted = None
        for item_type, item in content_list:
            if item_type in ('direct_text', 'tail_text'):
                if last_inserted is None:
                    if not parent.text:
                        parent.text = item
                    else:
                        parent.text += ' ' + item
                else:
                    if last_inserted.tail is None:
                        last_inserted.tail = item
                    else:
                        last_inserted.tail += ' ' + item
            else:
                parent.append(copy.deepcopy(item))
                last_inserted = item

    # Start processing
    process_node(processing_dom, [])

    # Remove duplicates
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        if p['html'] not in seen:
            seen.add(p['html'])
            unique_paragraphs.append(p)

    return unique_paragraphs


def remove_xml_declaration(html_string):
    """
    Remove XML declaration and HTML comments from HTML string.

    Args:
        html_string: HTML string to process

    Returns:
        HTML string with XML declaration and comments removed
    """
    # Regular expression to match <?xml ...?> or <?xml ...> (without closing question mark)
    pattern = r'<\?xml\s+.*?\??>'
    html_content = re.sub(pattern, '', html_string, flags=re.DOTALL)
    # Remove HTML comments
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
    return html_content


def post_process_html(html_content: str) -> str:
    """
    Post-process simplified HTML.

    Removes HTML comments and normalizes whitespace outside tags
    while preserving line breaks within tag text.

    Args:
        html_content: Simplified HTML content to process

    Returns:
        Post-processed HTML string
    """
    if not html_content:
        return html_content

    # Remove HTML comments
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)

    # Process whitespace outside tags (preserve line breaks within tag text)
    def replace_outside_tag_space(match):
        """Replace only consecutive whitespace outside tags."""
        if match.group(1):  # If it's tag content
            return match.group(1)
        elif match.group(2):  # If it's non-tag content
            # Replace consecutive whitespace in non-tag content with single space
            return re.sub(r'\s+', ' ', match.group(2))
        return match.group(0)  # Default: return entire match

    # Use regex to match all tag content and non-tag content
    html_content = re.sub(
        r'(<[^>]+>)|([^<]+)', replace_outside_tag_space, html_content
    )

    return html_content.strip()


def remove_tags(dom):
    """
    Remove specific tags from DOM.

    Removes all tags specified in tags_to_remove from the DOM tree.

    Args:
        dom: HTML element to process
    """
    for tag in tags_to_remove:
        for node in dom.xpath(f'.//{tag}'):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)


def is_meaningful_content(element) -> bool:
    """
    Strictly determine if an element contains meaningful content.

    Checks if the element has text content, valid image src, or meaningful children.

    Args:
        element: HTML element to check

    Returns:
        True if element contains meaningful content, False otherwise
    """
    if element.text and element.text.strip():
        return True
    if element.tag == 'img':
        src = element.get('src', '')
        return bool(src and src.strip())
    for child in element:
        if is_meaningful_content(child):
            return True
    if element.tail and element.tail.strip():
        return True
    return False


def clean_attributes(element):
    """
    Clean element attributes.

    For images: preserves valid src (excluding base64), alt, class, and id.
    For other elements: preserves only class and id.

    Args:
        element: HTML element to clean
    """
    if element.tag == 'img':
        # Get image-related attributes
        src = element.get('src', '').strip()
        alt = element.get('alt', '').strip()
        class_attr = element.get('class', '').strip()
        id_attr = element.get('id', '').strip()

        element.attrib.clear()  # Clear all attributes

        # Preserve non-base64 src
        if src and not src.startswith('data:image/'):
            element.set('src', src)
        # Preserve alt if not empty
        if alt:
            element.set('alt', alt)
        # Preserve class and id if not empty
        if class_attr:
            element.set('class', class_attr)
        if id_attr:
            element.set('id', id_attr)
    else:
        # Non-image elements: only preserve class and id
        class_attr = element.get('class', '').strip()
        id_attr = element.get('id', '').strip()

        element.attrib.clear()  # Clear all attributes

        if class_attr:
            element.set('class', class_attr)
        if id_attr:
            element.set('id', id_attr)

    # Recursively process child elements
    for child in element:
        clean_attributes(child)


def remove_inline_tags(element):
    """
    Recursively remove all specified inline tags (including nested cases).

    Preserves img, br, and other EXCLUDED_TAGS tags.

    Args:
        element: HTML element to process
    """
    # Process child elements first (depth-first)
    for child in list(element.iterchildren()):
        remove_inline_tags(child)

    # If current element is an inline tag to remove
    if element.tag in inline_tags and element.tag not in EXCLUDED_TAGS:
        parent = element.getparent()
        if parent is None:
            return

        # Check if element contains tags to preserve (e.g., img, br)
        has_excluded_tags = any(
            child.tag in EXCLUDED_TAGS for child in element.iterdescendants()
        )

        # If it contains tags to preserve, don't remove current element
        if has_excluded_tags:
            return

        # Save parts of current element
        leading_text = element.text or ''  # Text before element
        trailing_text = element.tail or ''  # Text after element
        children = list(element)  # List of child elements

        # Get current element's position in parent
        element_index = parent.index(element)

        # 1. Process leading_text (text before element)
        if leading_text:
            if element_index == 0:  # If it's the first child
                parent.text = (parent.text or '') + leading_text
            else:
                prev_sibling = parent[element_index - 1]
                prev_sibling.tail = (prev_sibling.tail or '') + leading_text

        # 2. Move child elements to parent
        for child in reversed(children):
            parent.insert(element_index, child)

        # 3. Process trailing_text (text after element)
        if trailing_text:
            if len(children) > 0:  # If there are children, append to last child's tail
                last_child = children[-1]
                last_child.tail = (last_child.tail or '') + trailing_text
            elif element_index == 0:  # If no children and is first child
                parent.text = (parent.text or '') + trailing_text
            else:  # If no children and not first child
                prev_sibling = (
                    parent[element_index - 1] if element_index > 0 else None
                )
                if prev_sibling is not None:
                    prev_sibling.tail = (prev_sibling.tail or '') + trailing_text
                else:
                    parent.text = (parent.text or '') + trailing_text

        # 4. Remove current element
        parent.remove(element)


def simplify_list(element):
    """
    Simplify list elements, keeping only the first and last groups.

    For dl lists, preserves complete dt + all dd pairs.

    Args:
        element: List element to simplify (ul, ol, or dl)
    """
    if element.tag in ('ul', 'ol'):
        # Process regular lists (ul/ol)
        items = list(element.iterchildren())
        if len(items) > 2:
            # Keep first and last child elements
            for item in items[1:-1]:
                element.remove(item)

            # Add ellipsis between first and last
            ellipsis = etree.Element('span')
            ellipsis.text = '...'
            items[-1].addprevious(ellipsis)

    elif element.tag == 'dl':
        # Process definition lists (dl)
        items = list(element.iterchildren())
        if len(items) > 2:
            # Find all dt elements
            dts = [item for item in items if item.tag == 'dt']

            if len(dts) > 1:
                # Get first group: dt and all following dd
                first_dt_index = items.index(dts[0])
                next_dt_index = items.index(dts[1])
                first_group = items[first_dt_index:next_dt_index]

                # Get last group: dt and all following dd
                last_dt_index = items.index(dts[-1])
                last_group = items[last_dt_index:]

                # Clear dl element
                for child in list(element.iterchildren()):
                    element.remove(child)

                # Add first group complete content
                for item in first_group:
                    element.append(item)

                # Add ellipsis
                ellipsis = etree.Element('span')
                ellipsis.text = '...'
                element.append(ellipsis)

                # Add last group complete content
                for item in last_group:
                    element.append(item)

    # Recursively process child elements
    for child in element:
        simplify_list(child)


def should_remove_element(element) -> bool:
    """
    Determine if element's class or id attributes match patterns to remove.

    Args:
        element: HTML element to check

    Returns:
        True if element should be removed, False otherwise
    """
    # Check class attribute
    class_name = element.get('class', '')
    if class_name:
        class_parts = class_name.strip().split()
        for part in class_parts:
            # Check if it exactly matches a standalone word
            if part in ATTR_PATTERNS_TO_REMOVE:
                return True
            # Check if it contains specific prefix/suffix
            # for pattern in ATTR_SUFFIX_TO_REMOVE:
            #     if part.endswith(pattern):
            #         return True

    # Check id attribute
    id_name = element.get('id', '')
    if id_name:
        id_parts = id_name.strip().split('-')  # IDs are usually separated by hyphens
        for part in id_parts:
            # Check if it exactly matches a standalone word
            if part in ATTR_PATTERNS_TO_REMOVE:
                return True
            # Check if it contains specific prefix/suffix
            # for pattern in ATTR_SUFFIX_TO_REMOVE:
            #     if part.endswith(pattern):
            #         return True

    # Check style attribute
    style_attr = element.get('style', '')
    if style_attr:
        if 'display: none' in style_attr or 'display:none' in style_attr:
            return True

    return False


def remove_specific_elements(element):
    """
    Remove elements whose class or id names match specific patterns.

    Recursively processes children first, then removes the element if it matches.

    Args:
        element: HTML element to process
    """
    for child in list(element.iterchildren()):
        remove_specific_elements(child)

    if should_remove_element(element):
        parent = element.getparent()
        if parent is not None:
            # parent.remove(element)
            tail_text = element.tail or ""
            element.tail = None

            prev_sibling = element.getprevious()
            if prev_sibling is not None:
                if prev_sibling.tail is not None:
                    prev_sibling.tail += tail_text
                else:
                    if prev_sibling.text is not None:
                        prev_sibling.text += tail_text
                    else:
                        prev_sibling.text = tail_text
            else:
                if parent.text is not None:
                    parent.text += tail_text
                else:
                    parent.text = tail_text

            parent.remove(element)

def truncate_html_element_selective(element, max_length, ellipsis='...', exclude_tags=None):
    """
    Truncate the text content within the lxml element and exclude the text within the specified label from being included in the length limit,
    But the tail outside the specified label is included in the length limit, and an ellipsis is added after the text of the length limit,
    And ensure that ellipses are not inserted inside the excluded labels.

    params:
        element (lxml.etree.Element): The lxml elements to be processed
        max_length (int): Maximum allowed text length (excluding the length of specified labels)
        ellipsis (str): The omitted symbol added after truncation defaults to '...'
        exclude_tags (list): List of label names not included in length statistics, such as ['math ',' script ',' style ']

    return:
        lxml.etree.Element: Processed elements
    """
    if exclude_tags is None:
        exclude_tags = set()

    def _calculate_text_length(node):
        """Calculate the effective text length of the node and its descendants (excluding text within the specified label)"""
        total_length = 0

        if node.text and not _is_excluded(node):
            total_length += len(node.text)

        for child in node:
            total_length += _calculate_text_length(child)

        if node.tail:
            total_length += len(node.tail)
        return total_length

    def _is_excluded(node):
        """Check if the node or its ancestor node is in the exclusion list"""
        current = node
        while current is not None:
            if current.tag in exclude_tags:
                return True
            current = current.getparent()
        return False

    current_length = [0]
    ellipsis_added = [False]
    nodes_to_process = []

    def _collect_text_nodes(node):
        """
        Recursive collection of all text node information that needs to be processed (including text and tail)
        Simultaneously mark whether the node is allowed to be modified (not included in the exclusion label)
        """
        if node.text and not _is_excluded(node):
            nodes_to_process.append({
                'type': 'text',
                'node': node,
                'original_text': node.text,
                'can_modify': not _is_inside_excluded_tag(node)
            })

        for child in node:
            _collect_text_nodes(child)

        if node.tail:
            nodes_to_process.append({
                'type': 'tail',
                'node': node,
                'original_text': node.tail,
                'can_modify': not _is_inside_excluded_tag(node)
            })

    def _is_inside_excluded_tag(node):
        """Check if the node is located inside the excluded label"""
        return _is_excluded(node.getparent()) if node.getparent() is not None else False

    def _process_text_nodes():
        """Process the collected text nodes, perform truncation and ellipsis addition"""
        for node_info in nodes_to_process:
            if ellipsis_added[0]:
                if node_info['type'] == 'text':
                    node_info['node'].text = None
                else:
                    node_info['node'].tail = None
                continue

            text_len = len(node_info['original_text'])
            if current_length[0] + text_len <= max_length:
                current_length[0] += text_len
            else:
                if node_info['can_modify']:
                    remaining = max_length - current_length[0]
                    truncated_text = node_info['original_text'][:remaining] + ellipsis

                    if node_info['type'] == 'text':
                        node_info['node'].text = truncated_text
                    else:
                        node_info['node'].tail = truncated_text

                    current_length[0] = max_length
                    ellipsis_added[0] = True

                    _mark_truncation_point(node_info['node'])
                else:
                    current_length[0] += text_len

    def _mark_truncation_point(truncate_node):
        """Mark truncation points and clean up subsequent content"""
        parent = truncate_node.getparent()
        if parent is not None:
            children = list(parent)
            try:
                index = children.index(truncate_node)
                for sibling in children[index + 1:]:
                    parent.remove(sibling)
            except ValueError:
                pass

        _clean_ancestors_following_siblings(truncate_node)

    def _clean_ancestors_following_siblings(node):
        """Recursively clean up the subsequent sibling nodes of all ancestor nodes"""
        parent = node.getparent()
        if parent is None:
            return

        grandparent = parent.getparent()
        if grandparent is None:
            return

        children = list(grandparent)
        try:
            index = children.index(parent)
            for sibling in children[index + 1:]:
                grandparent.remove(sibling)
        except ValueError:
            pass

        _clean_ancestors_following_siblings(parent)

    # 1. First, calculate the total text length
    total_text_length = _calculate_text_length(element)

    # 2. If the total length does not exceed the limit, return directly
    if total_text_length <= max_length:
        return element

    # 3. Collect and process text nodes
    _collect_text_nodes(element)
    _process_text_nodes()

    return element

def process_paragraphs(
    paragraphs: List[Dict[str, str]], uid_map: Dict[str, html.HtmlElement], cutoff_length: int = 500
) -> Tuple[str, html.HtmlElement]:
    """
    Process paragraphs and add _item_id attributes.

    Adds _item_id to both simplified HTML and corresponding elements in original DOM.

    Args:
        paragraphs: List of paragraphs, each containing html, content_type, and _original_element
        uid_map: Dictionary mapping data-uid to original elements

    Returns:
        Tuple of (simplified HTML string, marked original DOM element)
    """
    result = []
    item_id = 1

    for para in paragraphs:
        try:
            html_content = re.sub(
                r'<!--.*?-->', '', para['html'], flags=re.DOTALL
            )
            # Parse paragraph HTML
            root = html.fromstring(html_content)
            root_for_xpath = copy.deepcopy(root)
            content_type = para.get('content_type', 'block_element')

            # Common processing steps
            clean_attributes(root)
            simplify_list(root)
            # remove_inline_tags(root)

            # Skip meaningless content
            if not is_meaningful_content(root):
                continue

            # Truncate overly long text content
            truncate_html_element_selective(root, max_length=cutoff_length, exclude_tags=no_calc_text_tags)

            # Add same _item_id to current paragraph and original element
            current_id = str(item_id)
            root.set('_item_id', current_id)

            # For non-block elements (inline_elements, unwrapped_text, mixed)
            original_parent = (
                para['_original_element']
            )  # Parent node of direct child elements in original webpage
            if content_type != 'block_element':
                if original_parent is not None:
                    # root_for_xpath has child elements
                    if len(root_for_xpath) > 0:
                        if (
                            root_for_xpath.tag in inline_tags
                            and uid_map.get(root_for_xpath.get('data-uid')).tag
                            != 'body'
                        ):
                            original_element = uid_map.get(
                                root_for_xpath.get('data-uid')
                            )
                            original_element.set('_item_id', current_id)
                        else:
                            # Collect child elements that need to be wrapped
                            children_to_wrap = []
                            for child in root_for_xpath.iterchildren():
                                child_uid = child.get('data-uid')
                                if child_uid and child_uid in uid_map:
                                    original_child = uid_map[child_uid]
                                    children_to_wrap.append(original_child)

                            if children_to_wrap:
                                # Determine wrapping range
                                first_child = children_to_wrap[0]
                                last_child = children_to_wrap[-1]

                                # Get positions in parent node
                                start_idx = original_parent.index(first_child)
                                end_idx = original_parent.index(last_child)

                                # Collect all nodes that need to be moved
                                nodes_to_wrap = []
                                for i in range(start_idx, end_idx + 1):
                                    nodes_to_wrap.append(original_parent[i])

                                # Process leading text
                                leading_text = (
                                    original_parent.text
                                    if start_idx == 0
                                    else original_parent[start_idx - 1].tail
                                )

                                # Process trailing text
                                # trailing_text = last_child.tail

                                # Create wrapper element
                                wrapper = etree.Element(tail_block_tag)
                                wrapper.set('_item_id', current_id)

                                # Set leading text
                                if leading_text:
                                    wrapper.text = leading_text
                                    if start_idx == 0:
                                        original_parent.text = None
                                    else:
                                        original_parent[start_idx - 1].tail = None

                                # Move nodes to wrapper
                                for node in nodes_to_wrap:
                                    original_parent.remove(node)
                                    wrapper.append(node)

                                # Insert wrapper
                                original_parent.insert(start_idx, wrapper)

                                # Set trailing text
                                # if trailing_text:
                                #     wrapper.tail = trailing_text
                                #     last_child.tail = None
                    else:
                        if content_type == 'inline_elements':
                            original_element = uid_map.get(
                                root_for_xpath.get('data-uid')
                            )
                            original_element.set('_item_id', current_id)
                        else:
                            # root_for_xpath only has text content
                            if root_for_xpath.text and root_for_xpath.text.strip():
                                # 1. Find matching text node in original DOM
                                found = False

                                # Check parent node's text
                                if (
                                    original_parent.text
                                    and original_parent.text.strip()
                                    == root_for_xpath.text.strip()
                                ):
                                    # Create wrapper
                                    wrapper = etree.Element(tail_block_tag)
                                    wrapper.set('_item_id', current_id)
                                    wrapper.text = original_parent.text

                                    # Replace parent node's text
                                    original_parent.text = None

                                    # Insert wrapper as first child
                                    if len(original_parent) > 0:
                                        original_parent.insert(0, wrapper)
                                    else:
                                        original_parent.append(wrapper)

                                    found = True

                                # Check child node's tail
                                if not found:
                                    for child in original_parent.iterchildren():
                                        if (
                                            child.tail
                                            and child.tail.strip()
                                            == root_for_xpath.text.strip()
                                        ):
                                            # Create wrapper
                                            wrapper = etree.Element(
                                                tail_block_tag
                                            )
                                            wrapper.set('_item_id', current_id)
                                            wrapper.text = child.tail

                                            # Replace tail
                                            child.tail = None

                                            # Insert wrapper after child node
                                            parent = child.getparent()
                                            index = parent.index(child)
                                            parent.insert(index + 1, wrapper)

                                            break

            else:
                # Block elements: set attribute directly
                original_parent.set('_item_id', current_id)
                for child in original_parent.iterdescendants():
                    if child.get('cc-select') is not None:
                        original_parent.set(
                            'cc-select', child.get('cc-select')
                        )
                        break

            item_id += 1

            # Save processing result
            cleaned_html = etree.tostring(
                root, method='html', encoding='unicode'
            ).strip()
            result.append(
                {
                    'html': cleaned_html,
                    '_item_id': current_id,
                    'content_type': content_type,
                }
            )

        except Exception:
            # import traceback
            # print(f'Error processing paragraph: {traceback.format_exc()}')
            continue

    # Assemble final HTML
    simplified_html = (
        '<html><head><meta charset="utf-8"></head><body>'
        + ''.join(p['html'] for p in result)
        + '</body></html>'
    )

    return post_process_html(simplified_html)


def simplify_html(html_str: str, cutoff_length: int = 500) -> Tuple[str, str]:
    """
    Simplify HTML structure and add item IDs.

    Processes HTML to create a simplified version for main content extraction,
    while preserving the original HTML with item ID markers.

    Args:
        html_str: Raw HTML string to process

    Returns:
        Tuple of:
        - simplified_html: Simplified HTML structure
        - original_html: Original HTML with _item_id attributes added
    """
    # Preprocessing
    preprocessed_html = remove_xml_declaration(html_str)

    # Fix unclosed tags using BeautifulSoup (lxml cannot fully fix them)
    # Prioritize using Selectolax for better performance
    try:
        soup = HTMLParser(preprocessed_html)
        fixed_html = soup.html
    except Exception:
        soup = BeautifulSoup(preprocessed_html, 'html.parser')
        fixed_html = str(soup)

    # Parse original DOM
    original_dom = html.fromstring(fixed_html)
    add_data_uids(original_dom)
    original_uid_map = build_uid_map(original_dom)

    # Create processing DOM (deep copy)
    processing_dom = copy.deepcopy(original_dom)
    # Clean DOM
    remove_tags(processing_dom)
    remove_specific_elements(processing_dom)

    # Extract paragraphs (will record original element references)
    paragraphs = extract_paragraphs(
        processing_dom, original_uid_map, include_parents=False
    )

    # Process paragraphs (synchronously add IDs)
    simplified_html = process_paragraphs(paragraphs, original_uid_map, cutoff_length)

    remove_all_uids(original_dom)
    original_html = etree.tostring(
        original_dom, pretty_print=True, method='html', encoding='unicode'
    )

    return simplified_html, original_html
