import copy
import re
import uuid
from typing import Dict, List, Tuple
from urllib.parse import quote

from bs4 import BeautifulSoup
from lxml import etree, html  # pyright: ignore[reportAttributeAccessIssue]
from selectolax.parser import HTMLParser

from mineru_html.base import MinerUHTMLCase, MinerUHTMLProcessData
from mineru_html.exceptions import MinerUHTMLError, MinerUHTMLPreprocessError

# Inline tags
inline_tags = {
    'map',
    'optgroup',
    'span',
    'input',
    'time',
    'u',
    'strong',
    'small',
    'sub',
    'samp',
    'blink',
    'b',
    'code',
    'nobr',
    'strike',
    'bdo',
    'basefont',
    'abbr',
    'var',
    'i',
    'cccode-inline',
    's',
    'pic',
    'label',
    'mark',
    'object',
    'ccmath-inline',
    'svg',
    'button',
    'a',
    'font',
    'dfn',
    'sup',
    'kbd',
    'q',
    'script',
    'acronym',
    'option',
    'img',
    'big',
    'cite',
    'em',
    'marked-tail',
    'marked-text',
    # 'td', 'th', 'dd', 'dt', 'li', 'br', 'textarea', 'select'
}

# Table-related tags that may be contained within tables
table_tags_set = {
    'caption',
    'colgroup',
    'col',
    'thead',
    'tbody',
    'tfoot',
    'tr',
    'td',
    'th',
    'br',
}

# Tags to remove
tags_to_remove = {
    'title',
    'head',
    'style',
    'script',
    'link',
    'meta',
    'iframe',
    'frame',
    'nav',
    # 'noscript',
}

# Regarded as block level elements, there are no block level elements inside by default
no_block_tags = {'math'}

no_calc_text_tags = {'math', 'table'}

# Special tags to preserve (even if they are inline tags)
EXCLUDED_TAGS = {'img', 'br', 'li', 'dt', 'dd', 'td', 'th'}

# Attribute name patterns to remove (standalone words)
ATTR_PATTERNS_TO_REMOVE = {
    'nav',  # 'footer', 'header',  # standalone words
}

# Attribute name patterns to remove (specific prefixes/suffixes)
ATTR_SUFFIX_TO_REMOVE = {
    # '-nav', '_nav',
    # '-footer', '_footer',  # Has exceptions, dl list may add custom footer attribute to last item, commented for now
    # '-header', '_header',  # Has exceptions, custom header may contain titles, commented for now
}

ATTR_INVISIBLE = {
    'display': 'none',
    'font-size': '0px',
    'color': 'transparent',
    'visibility': 'hidden',
    'opacity': '0',
}

# Custom tags
tail_block_tag = 'cc-alg-uc-text'


def add_data_uids(dom: html.HtmlElement) -> None:
    """Add data-uid attribute to all DOM nodes (recursively for all child nodes)."""
    for node in dom.iter():
        try:
            node.set('data-uid', str(uuid.uuid4()))
        except TypeError:
            pass


def remove_all_uids(dom: html.HtmlElement) -> None:
    """Remove all data-uid attributes from DOM."""
    for node in dom.iter():
        if 'data-uid' in node.attrib:
            del node.attrib['data-uid']


def build_uid_map(dom: html.HtmlElement) -> Dict[str, html.HtmlElement]:
    """Build a mapping dictionary from data-uid to nodes."""
    return {node.get('data-uid'): node for node in dom.iter() if node.get('data-uid')}


def judge_table_parent(table_element, node_list):
    """Check if any node in the list is a descendant of the table element.

    Args:
        table_element: The table element to check against.
        node_list: List of nodes to check.

    Returns:
        True if any node is a descendant of the table, False otherwise.
    """
    for node in node_list:
        ancestor = node.getparent()
        while ancestor is not None:
            if ancestor is table_element:
                return True
            elif ancestor.tag == 'table':
                break
            ancestor = ancestor.getparent()
    return False


def is_data_table(table_element: html.HtmlElement) -> bool:
    """Determine if a table is a data table rather than a layout table."""
    # Check if current table (excluding nested tables) has caption tag
    caption_nodes = table_element.xpath('.//caption')
    if judge_table_parent(table_element, caption_nodes):
        return True

    # Check if current table (excluding nested tables) has colgroup or col tags
    col_nodes = table_element.xpath('.//col')
    colgroup_nodes = table_element.xpath('.//colgroup')
    if judge_table_parent(table_element, col_nodes) or judge_table_parent(
        table_element, colgroup_nodes
    ):
        return True

    # Check if current table (excluding nested tables) cells have headers attribute
    cell_nodes = table_element.xpath('.//*[self::td or self::th][@headers]')
    if judge_table_parent(table_element, cell_nodes):
        return True

    # Check if there is role="table" or data-table attribute
    if table_element.get('role') == 'table' or table_element.get('data-table'):
        return True

    for node in table_element.iterdescendants():
        if node.tag in table_tags_set:
            continue
        if node.tag not in inline_tags:
            return False

    return True


def has_non_listitem_children(list_element):
    """Check if a list element contains direct child nodes that are not list items.

    :param list_element: lxml element object (ul, ol, dl)
    :return: True if non-list-item direct child nodes exist, False otherwise
    """

    # Determine allowed child element tags based on list type
    if list_element.tag in ['ul', 'ol']:
        allowed_tags = {'li'}
    elif list_element.tag == 'dl':
        allowed_tags = {'dt', 'dd'}

    # Use XPath to directly find if there are disallowed direct child elements
    # For example, for <ul>, find all direct child elements that are not <li>
    # For <dl>, find all direct child elements that are not <dt> or <dd>
    exclude_conditions = ' and '.join([f"name()!='{tag}'" for tag in allowed_tags])
    disallowed_children_xpath = f'./*[{exclude_conditions}]'

    if list_element.xpath(disallowed_children_xpath):
        return True

    # Check if there are non-whitespace text nodes
    text_children = list_element.xpath('./text()')
    non_whitespace_text = any(text.strip() for text in text_children)

    return non_whitespace_text


def extract_paragraphs(
    processing_dom: html.HtmlElement,
    uid_map: Dict[str, html.HtmlElement],
    include_parents: bool = True,
) -> List[Dict[str, str]]:
    """Extract paragraphs from DOM structure.

    Args:
        processing_dom: DOM tree to extract paragraphs from.
        uid_map: Mapping from data-uid to original DOM elements.
        include_parents: Whether to include parent elements in extraction.

    Returns:
        List of paragraph dictionaries, each containing:
            - html: HTML string of the paragraph
            - content_type: Type of content ('block_element', 'inline_elements',
              'unwrapped_text', or 'mixed')
            - _original_element: Reference to original element in DOM
    """

    # Create table type mapping, recording whether each table is a data table or layout table
    table_types = {}

    # First analyze the type of all tables
    for table in processing_dom.xpath('.//table'):
        table_types[table.get('data-uid')] = is_data_table(table)

    # Create list type mapping, recording whether each list is a content list or layout list
    list_types = {}

    def is_block_element(node) -> bool:
        """Determine if it is a block-level element."""

        def judge_special_case(node, expected_tags, types_map):
            ancestor = node
            while ancestor is not None and ancestor.tag not in expected_tags:
                ancestor = ancestor.getparent()

            if ancestor is not None:
                ancestor_uid = ancestor.get('data-uid')
                if types_map.get(ancestor_uid, False):
                    return False
                else:
                    return True

        if node.tag in ('td', 'th'):
            return judge_special_case(node, ['table'], table_types)

        if node.tag == 'li':
            return judge_special_case(node, ['ul', 'ol'], list_types)

        if node.tag == 'dt' or node.tag == 'dd':
            return judge_special_case(node, ['dl'], list_types)

        if node.tag in no_block_tags or node.tag in inline_tags:
            return False
        return isinstance(node, html.HtmlElement)

    def has_block_descendants(node):
        if node.tag in no_block_tags:
            return False
        for child in node.iterdescendants():
            parent = child.getparent()
            if parent is not None and (
                parent.tag in no_block_tags or parent.get('cc-no-block') == 'true'
            ):
                child.set('cc-no-block', 'true')
            if child.get('cc-no-block') != 'true' and is_block_element(child):
                if node.tag in inline_tags:
                    original_element = uid_map.get(node.get('data-uid'))
                    original_element.set('cc-block-type', 'true')
                return True
        return False

    def is_content_list(list_element):
        items = list_element.xpath('li | dt | dd')

        if len(items) == 0:
            return False
        if has_non_listitem_children(list_element):
            return False

        for item in items:
            if has_block_descendants(item):
                return False

        return True
    for list_element in processing_dom.xpath('.//ul | .//ol | .//dl'):
        list_types[list_element.get('data-uid')] = is_content_list(list_element)

    def clone_structure(
        path: List[html.HtmlElement],
    ) -> Tuple[html.HtmlElement, html.HtmlElement]:
        """Clone node structure."""
        if not path:
            raise ValueError('Path cannot be empty')
        if not include_parents:
            last_node = html.Element(path[-1].tag)
            last_node.attrib.update(path[-1].attrib)
            return last_node, last_node

        root = html.Element(path[0].tag)
        root.attrib.update(path[0].attrib)
        current = root
        for node in path[1:-1]:
            new_node = html.Element(node.tag)
            new_node.attrib.update(node.attrib)
            current.append(new_node)
            current = new_node

        last_node = html.Element(path[-1].tag)
        last_node.attrib.update(path[-1].attrib)
        current.append(last_node)
        return root, last_node

    paragraphs = []

    def process_node(node: html.HtmlElement, path: List[html.HtmlElement]):
        """Recursively process nodes."""
        current_path = path + [node]
        inline_content = []
        content_sources = []

        if node.text and node.text.strip():
            inline_content.append(('direct_text', node.text.strip()))
            content_sources.append('direct_text')

        for child in node:
            if is_block_element(child) or has_block_descendants(child):
                if child.tag == 'br':
                    inline_content.append(('element', child))
                    content_sources.append('element')
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
                        paragraphs.append(
                            {
                                'html': etree.tostring(
                                    root, encoding='unicode'
                                ).strip(),
                                'content_type': content_type,
                                '_original_element': original_element,  # Add original element reference
                            }
                        )
                    except ValueError:
                        pass
                    inline_content = []
                    content_sources = []
                if child.tag != 'br':
                    # Process block elements
                    if table_types.get(child.get('data-uid')) or (
                        not has_block_descendants(child)
                    ):
                        try:
                            root, last_node = clone_structure(current_path + [child])
                            last_node.text = child.text if child.text else None
                            for grandchild in child:
                                last_node.append(copy.deepcopy(grandchild))

                            # Get original element
                            original_element = uid_map.get(child.get('data-uid'))
                            paragraphs.append(
                                {
                                    'html': etree.tostring(
                                        root, encoding='unicode'
                                    ).strip(),
                                    'content_type': 'block_element',
                                    '_original_element': original_element,  # Add original element reference
                                }
                            )
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

        if inline_content:
            try:
                root, last_node = clone_structure(current_path)
                merge_inline_content(last_node, inline_content)

                content_type = 'mixed'
                if all(t == 'direct_text' for t in content_sources):
                    content_type = 'unwrapped_text'
                elif all(t == 'element' for t in content_sources):
                    content_type = 'inline_elements'
                elif all(t in ('direct_text', 'tail_text') for t in content_sources):
                    content_type = 'unwrapped_text'

                # Get original element
                original_element = uid_map.get(node.get('data-uid'))
                paragraphs.append(
                    {
                        'html': etree.tostring(root, encoding='unicode').strip(),
                        'content_type': content_type,
                                '_original_element': original_element,
                    }
                )
            except ValueError:
                pass

    def merge_inline_content(
        parent: html.HtmlElement, content_list: List[Tuple[str, str]]
    ):
        """Merge inline content."""
        last_inserted = None
        for idx, (item_type, item) in enumerate(content_list):
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
                item_copy = copy.deepcopy(item)
                if idx == len(content_list) - 1 and item_copy.tag == 'br':
                    item_copy.tail = None
                parent.append(item_copy)
                last_inserted = item

    process_node(processing_dom, [])

    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        if p['html'] not in seen:
            seen.add(p['html'])
            unique_paragraphs.append(p)

    return unique_paragraphs


def remove_xml_declaration(html_string):
    # Regex match <?xml ...?> or <?xml ...> (case without question mark ending)
    pattern = r'<\?xml\s+.*?\??>'
    html_content = re.sub(pattern, '', html_string, flags=re.DOTALL)

    return html_content


def post_process_html(html_content: str) -> str:
    """Post-process simplified HTML."""
    if not html_content:
        return html_content

    # Process whitespace outside tags (preserve line breaks in tag text)
    def replace_outside_tag_space(match):
        """Replace only consecutive whitespace outside tags."""
        if match.group(1):
            return match.group(1)
        elif match.group(2):
            return re.sub(r'\s+', ' ', match.group(2))
        return match.group(0)
    html_content = re.sub(r'(<[^>]+>)|([^<]+)', replace_outside_tag_space, html_content)

    return html_content.strip()


def remove_tags(dom):
    """Remove specific tags from DOM.

    Args:
        dom: DOM tree to process.
    """
    for tag in tags_to_remove:
        for node in dom.xpath(f'.//{tag}'):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)


def is_meaningful_content(element) -> bool:
    """Strictly determine if an element contains meaningful content."""
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


def clean_attributes(element, short_src=False):
    """Clean element attributes, preserving valid src (excluding base64) and alt for images, and class and id for all elements."""
    if element.tag == 'img':
        src = element.get('src', '').strip()
        alt = element.get('alt', '').strip()
        class_attr = element.get('class', '').strip()
        id_attr = element.get('id', '').strip()

        element.attrib.clear()

        if src and not src.startswith('data:image/'):
            if short_src:
                if len(quote(src)) <= 10:
                    element.set('src', src)
                else:
                    element.set('src', src[:5] + '...' + src[-5:])
            else:
                element.set('src', src)
        if alt:
            element.set('alt', alt)
        if class_attr:
            element.set('class', class_attr)
        if id_attr:
            element.set('id', id_attr)
    else:
        class_attr = element.get('class', '').strip()
        id_attr = element.get('id', '').strip()

        element.attrib.clear()

        if class_attr:
            element.set('class', class_attr)
        if id_attr:
            element.set('id', id_attr)

    for child in element:
        clean_attributes(child)


def remove_inline_tags(element):
    """Recursively remove all specified inline tags (including nested cases), preserving EXCLUDED_TAGS like img and br."""
    for child in list(element.iterchildren()):
        remove_inline_tags(child)

    if element.tag in inline_tags and element.tag not in EXCLUDED_TAGS:
        parent = element.getparent()
        if parent is None:
            return

        has_excluded_tags = any(
            child.tag in EXCLUDED_TAGS for child in element.iterdescendants()
        )

        if has_excluded_tags:
            return

        leading_text = element.text or ''
        trailing_text = element.tail or ''
        children = list(element)

        element_index = parent.index(element)

        if leading_text:
            if element_index == 0:
                parent.text = (parent.text or '') + leading_text
            else:
                prev_sibling = parent[element_index - 1]
                prev_sibling.tail = (prev_sibling.tail or '') + leading_text

        for child in reversed(children):
            parent.insert(element_index, child)

        if trailing_text:
            if len(children) > 0:
                last_child = children[-1]
                last_child.tail = (last_child.tail or '') + trailing_text
            elif element_index == 0:
                parent.text = (parent.text or '') + trailing_text
            else:
                prev_sibling = parent[element_index - 1] if element_index > 0 else None
                if prev_sibling is not None:
                    prev_sibling.tail = (prev_sibling.tail or '') + trailing_text
                else:
                    parent.text = (parent.text or '') + trailing_text

        parent.remove(element)


def simplify_list(element):
    """Simplify list elements, keeping only the first and last groups (for dl lists, keep complete dt+all dd)."""
    if element.tag in ('ul', 'ol'):
        items = list(element.iterchildren())
        if len(items) > 2:
            for item in items[1:-1]:
                element.remove(item)

            ellipsis = etree.Element('span')
            ellipsis.text = '...'
            items[-1].addprevious(ellipsis)

    elif element.tag == 'dl':
        items = list(element.iterchildren())
        if len(items) > 2:
            dts = [item for item in items if item.tag == 'dt']

            if len(dts) > 1:
                first_dt_index = items.index(dts[0])
                next_dt_index = items.index(dts[1])
                first_group = items[first_dt_index:next_dt_index]

                last_dt_index = items.index(dts[-1])
                last_group = items[last_dt_index:]

                for child in list(element.iterchildren()):
                    element.remove(child)

                for item in first_group:
                    element.append(item)

                ellipsis = etree.Element('span')
                ellipsis.text = '...'
                element.append(ellipsis)

                for item in last_group:
                    element.append(item)

    for child in element:
        simplify_list(child)


def should_remove_element(element) -> bool:
    """Determine if an element should be removed based on its attributes.

    Checks class/id patterns, style attributes for invisibility, and details
    element visibility.

    Args:
        element: Element to check.

    Returns:
        True if element should be removed, False otherwise.
    """
    class_name = element.get('class', '')
    id_name = element.get('id', '')

    if class_name in ATTR_PATTERNS_TO_REMOVE or id_name in ATTR_PATTERNS_TO_REMOVE:
        parent = element.getparent()
        if parent is not None and parent.tag == 'body':
            return True

    style_attr = element.get('style', '')
    if style_attr:
        style_list = style_attr.split(';')
        for attr in style_list:
            if ':' not in attr:
                continue

            split_items = attr.split(':')
            key = split_items[0]
            value = ':'.join(split_items[1:])

            if ATTR_INVISIBLE.get(key.strip()) == value.strip():
                return True

    parent = element.getparent()
    if parent is not None and parent.tag == 'details':
        if element.tag == 'summary':
            return False
        else:
            if parent.get('open') is None:
                return True
            else:
                return False

    return False


def remove_specific_elements(element):
    """Recursively remove elements that match removal conditions.

    Args:
        element: Root element to process.
    """
    for child in list(element.iterchildren()):
        remove_specific_elements(child)

    if should_remove_element(element):
        parent = element.getparent()
        if parent is not None:
            tail_text = element.tail or ''
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


def truncate_html_element_selective(
    element, max_length, ellipsis='...', exclude_tags=None
):
    """Truncate text content within an element, excluding specified tags from length calculation.

    The text within excluded tags is not counted toward the length limit, but tail
    text outside excluded tags is included. Ellipsis is added after truncation,
    ensuring ellipses are not inserted inside excluded tags.

    Args:
        element: The lxml element to be processed.
        max_length: Maximum allowed text length (excluding text in excluded tags).
        ellipsis: The ellipsis symbol added after truncation (default: '...').
        exclude_tags: Set of tag names excluded from length statistics
            (e.g., {'math', 'script', 'style'}).

    Returns:
        Processed element (modified in place).
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
            nodes_to_process.append(
                {
                    'type': 'text',
                    'node': node,
                    'original_text': node.text,
                    'can_modify': not _is_inside_excluded_tag(node),
                }
            )

        for child in node:
            _collect_text_nodes(child)

        if node.tail:
            nodes_to_process.append(
                {
                    'type': 'tail',
                    'node': node,
                    'original_text': node.tail,
                    'can_modify': not _is_inside_excluded_tag(node),
                }
            )

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
                for sibling in children[index + 1 :]:
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
            for sibling in children[index + 1 :]:
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
    paragraphs: List[Dict[str, str]],
    uid_map: Dict[str, html.HtmlElement],
    cutoff_length: int = 500,
) -> Tuple[str, html.HtmlElement]:
    """Process paragraphs and add _item_id, while adding the same ID to corresponding elements in the original DOM.

    Args:
        paragraphs: List of paragraphs, each containing html, content_type, and _original_element
        original_dom: Original DOM tree

    Returns:
        Tuple[simplified HTML, marked original DOM]
    """
    result = []
    item_id = 1

    for para in paragraphs:
        try:
            root = html.fragment_fromstring(para['html'], create_parent=False)
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
            truncate_html_element_selective(
                root, max_length=cutoff_length, exclude_tags=no_calc_text_tags
            )

            # Add the same _item_id to current paragraph and original element
            current_id = str(item_id)
            root.set('_item_id', current_id)

            # For non-block elements (inline_elements, unwrapped_text, mixed)
            original_parent = para['_original_element']
            if content_type != 'block_element':
                if original_parent is not None:
                    original_element = uid_map.get(root_for_xpath.get('data-uid'))
                    if len(root_for_xpath) > 0:
                        if (
                            root_for_xpath.tag in inline_tags
                            and original_element.tag != 'body'
                            and original_element.get('cc-block-type') != 'true'
                        ):
                            original_element.set('_item_id', current_id)
                        else:
                            children_to_wrap = []
                            for child in root_for_xpath.iterchildren():
                                child_uid = child.get('data-uid')
                                if child_uid and child_uid in uid_map:
                                    original_child = uid_map[child_uid]
                                    children_to_wrap.append(original_child)

                            if children_to_wrap:
                                first_child = children_to_wrap[0]
                                last_child = children_to_wrap[-1]

                                start_idx = original_parent.index(first_child)
                                end_idx = original_parent.index(last_child)

                                # Collect all nodes that need to be moved
                                nodes_to_wrap = []
                                for i in range(start_idx, end_idx + 1):
                                    nodes_to_wrap.append(original_parent[i])

                                # Process preceding text
                                leading_text = (
                                    original_parent.text
                                    if start_idx == 0
                                    else original_parent[start_idx - 1].tail
                                )

                                # Process following text
                                trailing_text = last_child.tail

                                wrapper = etree.Element(tail_block_tag)
                                wrapper.set('_item_id', current_id)
                                if original_parent.get('cc-select') is not None:
                                    wrapper.set(
                                        'cc-select', original_parent.get('cc-select')
                                    )

                                if leading_text:
                                    wrapper.text = leading_text
                                    if start_idx == 0:
                                        original_parent.text = None
                                    else:
                                        original_parent[start_idx - 1].tail = None

                                for node in nodes_to_wrap:
                                    original_parent.remove(node)
                                    wrapper.append(node)

                                original_parent.insert(start_idx, wrapper)
                                if last_child.tag == 'br' and trailing_text:
                                    wrapper.tail = trailing_text
                                    last_child.tail = None
                    else:
                        if content_type == 'inline_elements':
                            original_element.set('_item_id', current_id)
                        else:
                            if root_for_xpath.text and root_for_xpath.text.strip():
                                found = False

                                if (
                                    original_parent.text
                                    and original_parent.text.strip()
                                    == root_for_xpath.text.strip()
                                ):
                                    wrapper = etree.Element(tail_block_tag)
                                    wrapper.set('_item_id', current_id)
                                    wrapper.text = original_parent.text
                                    if original_parent.get('cc-select') is not None:
                                        wrapper.set(
                                            'cc-select',
                                            original_parent.get('cc-select'),
                                        )
                                    original_parent.text = None

                                    if len(original_parent) > 0:
                                        original_parent.insert(0, wrapper)
                                    else:
                                        original_parent.append(wrapper)

                                    found = True

                                if not found:
                                    for child in original_parent.iterchildren():
                                        if (
                                            child.tail
                                            and child.tail.strip()
                                            == root_for_xpath.text.strip()
                                        ):
                                            wrapper = etree.Element(tail_block_tag)
                                            wrapper.set('_item_id', current_id)
                                            wrapper.text = child.tail
                                            if (
                                                original_parent.get('cc-select')
                                                is not None
                                            ):
                                                wrapper.set(
                                                    'cc-select',
                                                    original_parent.get('cc-select'),
                                                )
                                            child.tail = None

                                            parent = child.getparent()
                                            index = parent.index(child)
                                            parent.insert(index + 1, wrapper)

                                            break

            else:
                original_parent.set('_item_id', current_id)
                for child in original_parent.iterdescendants():
                    if child.get('cc-select') is not None:
                        original_parent.set('cc-select', child.get('cc-select'))
                        break

            item_id += 1

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
            continue

    simplified_html = (
        '<html><head><meta charset="utf-8"></head><body>'
        + ''.join(p['html'] for p in result)
        + '</body></html>'
    )

    return post_process_html(simplified_html)


def simplify_html(html_str, cutoff_length: int = 500) -> etree.Element:
    """Simplify HTML structure for model processing.

    Args:
        html_str: Raw HTML string to simplify.
        cutoff_length: Maximum length for text content truncation.

    Returns:
        Tuple of (simplified_html, original_html_with_item_ids).
    """
    try:
        soup = HTMLParser(html_str)
        fixed_html = soup.html
    except Exception:
        soup = BeautifulSoup(html_str, 'html.parser')
        fixed_html = str(soup)

    preprocessed_html = remove_xml_declaration(fixed_html)
    parser = html.HTMLParser(remove_comments=True)
    original_dom = html.fromstring(preprocessed_html, parser=parser)
    add_data_uids(original_dom)
    original_uid_map = build_uid_map(original_dom)

    processing_dom = copy.deepcopy(original_dom)
    remove_tags(processing_dom)
    remove_specific_elements(processing_dom)

    paragraphs = extract_paragraphs(
        processing_dom, original_uid_map, include_parents=False
    )
    simplified_html = process_paragraphs(
        paragraphs, original_uid_map, cutoff_length=cutoff_length
    )

    remove_all_uids(original_dom)
    original_html = etree.tostring(
        original_dom, pretty_print=False, method='html', encoding='unicode'
    )

    return simplified_html, original_html


def simplify_single_input(input_case: MinerUHTMLCase) -> MinerUHTMLCase:
    """Preprocess raw input and simplify HTML.

    Simplifies raw HTML to a format suitable for model processing.

    Args:
        input_case: Case containing raw HTML input.

    Returns:
        Case with process_data set to simplified HTML.

    Raises:
        MinerUHTMLPreprocessError: If preprocessing fails.
    """
    try:
        simplified_html, map_html = simplify_html(input_case.input_data.raw_html)
        process_data = MinerUHTMLProcessData(
            simpled_html=simplified_html,
            map_html=map_html,
        )
        input_case.process_data = process_data

        return input_case

    except Exception as e:
        if isinstance(e, MinerUHTMLError):
            e.set_case_id(input_case.case_id)
            raise e
        else:
            raise MinerUHTMLPreprocessError(
                f'Preprocess failed: {str(e)}', case_id=input_case.case_id
            ) from e
