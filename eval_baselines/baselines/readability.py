from .base import MainHTMLExtractor


class ReadabilityExtractor(MainHTMLExtractor):
    """
    HTML extractor using readability library.

    Extracts main HTML content using readability's Document class.
    """

    name_str = 'readability'

    def __init__(self):
        super().__init__()
        from readability import Document

        self.extractor = Document

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using readability.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML summary
        """
        doc = self.extractor(input_html)
        return doc.summary()


class Readability_HTML_MD_Extractor(ReadabilityExtractor):
    """Readability extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class Readability_HTML_Text_Extractor(ReadabilityExtractor):
    """Readability extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'
