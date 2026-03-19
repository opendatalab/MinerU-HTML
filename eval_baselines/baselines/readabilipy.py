from .base import MainHTMLExtractor


class ReadabiliPyExtractor(MainHTMLExtractor):
    """
    HTML extractor using readabilipy library.

    Extracts main HTML content using readabilipy's simple_tree_from_html_string.
    """

    name_str = 'readabilipy'

    def __init__(self):
        super().__init__()

    def extract_main_html(self, input_html, url):
        """
        Extract main HTML using readabilipy.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML as string
        """
        from readabilipy.simple_tree import simple_tree_from_html_string

        soup = simple_tree_from_html_string(input_html)
        return str(soup)


class ReadabiliPy_HTML_MD_Extractor(ReadabiliPyExtractor):
    """Readabilipy extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class ReadabiliPy_HTML_Text_Extractor(ReadabiliPyExtractor):
    """Readabilipy extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'
