from .base import MainContentExtractor, MainHTMLExtractor


class GNE_Text_Extractor(MainContentExtractor):
    """
    Text extractor using GNE (General News Extractor) library.

    Extracts plain text content directly using GNE's GeneralNewsExtractor.
    """

    name_str = 'gne'
    format_str = 'text'

    def __init__(self):
        from gne import GeneralNewsExtractor

        self.extractor = GeneralNewsExtractor()

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text using GNE.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted content)
        """
        content = self.extractor.extract(input_html)['content']
        return '', content


class GNE_HTML_Extractor(MainHTMLExtractor):
    """
    HTML extractor using GNE (General News Extractor) library.

    Extracts main HTML content using GNE's GeneralNewsExtractor with body HTML.
    """

    name_str = 'gne'

    def __init__(self):
        super().__init__()
        from gne import GeneralNewsExtractor

        self.extractor = GeneralNewsExtractor()

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using GNE.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted body HTML
        """
        main_html = self.extractor.extract(input_html, with_body_html=True)['body_html']
        return main_html


class GNE_HTML_MD_Extractor(GNE_HTML_Extractor):
    """GNE HTML extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class GNE_HTML_Text_Extractor(GNE_HTML_Extractor):
    """GNE HTML extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'
