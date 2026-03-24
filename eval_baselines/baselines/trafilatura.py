from .base import MainContentExtractor, MainHTMLExtractor


class TrafilaturaExtractor(MainHTMLExtractor):
    """
    HTML extractor using trafilatura library.

    Extracts main HTML content using trafilatura with HTML output format.
    """

    name_str = 'trafilatura'

    def __init__(self):
        super().__init__()
        from trafilatura.settings import Extractor

        self.options = Extractor(output_format='html')

    def extract_main_html(self, input_html, url) -> str:
        """
        Extract main HTML using trafilatura.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML (or None if extraction fails)
        """
        from trafilatura import extract

        output_html = extract(input_html, url=url, options=self.options)
        return output_html


class Trafilatura_HTML_MD_Extractor(TrafilaturaExtractor):
    """Trafilatura HTML extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class Trafilatura_HTML_Text_Extractor(TrafilaturaExtractor):
    """Trafilatura HTML extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class Trafilatura_Text_Extractor(MainContentExtractor):
    """
    Text extractor using trafilatura library.

    Extracts plain text content directly using trafilatura with text output format.
    """

    name_str = 'trafilatura'
    format_str = 'text'

    def __init__(self):
        """
        Initialize Trafilatura_Text_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        from trafilatura.settings import Extractor

        self.options = Extractor(output_format='txt')

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text using trafilatura.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted text content)
        """
        from trafilatura import extract

        output_markdown = extract(input_html, url=url, options=self.options)
        if not output_markdown:
            output_markdown = ''
        return '', output_markdown


class Trafilatura_MD_Extractor(MainContentExtractor):
    """
    Markdown extractor using trafilatura library.

    Extracts content as Markdown directly using trafilatura with markdown output format.
    """

    name_str = 'trafilatura'
    format_str = 'md'

    def __init__(self):
        """
        Initialize Trafilatura_MD_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        from trafilatura.settings import Extractor

        self.options = Extractor(output_format='markdown')

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract Markdown content using trafilatura.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted Markdown content)
        """
        from trafilatura import extract

        output_markdown = extract(input_html, url=url, options=self.options)
        if not output_markdown:
            output_markdown = ''
        return '', output_markdown
