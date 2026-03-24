from enum import Enum

# HTML attribute constants
ITEM_ID_ATTR = '_item_id'
TAIL_BLOCK_TAG = 'cc-alg-uc-text'
SELECT_ATTR = 'cc-select'
CLASS_ATTR = 'mark-selected'


class TagType(Enum):
    """Enumeration for HTML tag types in the extraction process."""

    Main = 'main'  # Main content tag
    Other = 'other'  # Other/non-main content tag
