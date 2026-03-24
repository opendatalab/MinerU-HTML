from .base import MainHTMLExtractor


class MagicHTML_Extractor(MainHTMLExtractor):
    """
    HTML extractor using magic-html library (ArticleExtractor).

    Extracts main HTML content from articles using magic-html's ArticleExtractor.
    """

    name_str = 'magichtml'

    def __init__(self):
        """
        Initialize MagicHTML_Extractor.
        """
        super().__init__()
        from magic_html import ArticleExtractor

        self.extractor = ArticleExtractor()

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using magic-html ArticleExtractor.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML
        """
        data = self.extractor.extract(input_html, base_url=url)
        return data['html']


class MagicHTML_MD_Extractor(MagicHTML_Extractor):
    """Magic-html extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MagicHTML_Text_Extractor(MagicHTML_Extractor):
    """Magic-html extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class MagicForumHTML_Extractor(MainHTMLExtractor):
    """
    HTML extractor using magic-html library (ForumExtractor).

    Extracts main HTML content from forum pages using magic-html's ForumExtractor.
    """

    name_str = 'magichtmlforum'

    def __init__(self):
        super().__init__()
        from magic_html import ForumExtractor

        self.extractor = ForumExtractor()

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using magic-html ForumExtractor.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML
        """
        data = self.extractor.extract(input_html, base_url=url)
        return data['html']


class MagicForumHTML_MD_Extractor(MagicForumHTML_Extractor):
    """Magic-html forum extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MagicForumHTML_Text_Extractor(MagicForumHTML_Extractor):
    """Magic-html forum extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'
