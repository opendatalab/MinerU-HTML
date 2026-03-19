from .base import MainHTMLExtractor


class FullPageExtractor(MainHTMLExtractor):
    """
    HTML extractor that passes through HTML unchanged.

    This extractor returns the input HTML as-is, then converts it to text
    using the specified format (MD or TEXT).
    """

    name_str = 'fullpage'

    def extract_main_html(self, input_html, url):
        """
        Return input HTML unchanged.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Input HTML string (unchanged)
        """
        return input_html


class FullPage_MD_Extractor(FullPageExtractor):
    """FullPage extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class FullPage_Text_Extractor(FullPageExtractor):
    """FullPage extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'
