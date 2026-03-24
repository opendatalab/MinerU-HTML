from .base import MainContentExtractor, MainHTMLExtractor


class BoilerPy3HTMLExtractor(MainHTMLExtractor):
    """
    HTML extractor using boilerpy3 library.

    Extracts main HTML content using boilerpy3's ArticleExtractor.
    """

    name_str = 'boilerpy3'

    def __init__(self):
        """
        Initialize BoilerPy3HTMLExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__()
        from boilerpy3 import extractors

        self.extractor = extractors.ArticleExtractor(raise_on_failure=False)

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using boilerpy3.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML with boilerplate marked
        """
        return self.extractor.get_marked_html(input_html)


class BoilerPy3_HTML_MD_Extractor(BoilerPy3HTMLExtractor):
    """BoilerPy3 extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class BoilerPy3_HTML_Text_Extractor(BoilerPy3HTMLExtractor):
    """BoilerPy3 extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class BoilerPy3TextExtractor(MainContentExtractor):
    """
    Text extractor using boilerpy3 library.

    Extracts plain text content directly using boilerpy3's ArticleExtractor.
    """

    name_str = 'boilerpy3'
    format_str = 'text'

    def __init__(self):
        super().__init__()
        from boilerpy3 import extractors

        self.extractor = extractors.ArticleExtractor(raise_on_failure=False)

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text content using boilerpy3.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted text content)
        """
        content = self.extractor.get_content(input_html)
        return '', content
