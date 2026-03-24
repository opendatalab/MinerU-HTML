from .base import MainHTMLExtractor


class MinerUHTMLExtractor(MainHTMLExtractor):
    """
    HTML extractor using MinerU-HTML library.

    Extracts main HTML content using the custom MinerU-HTML extraction system.
    """

    name_str = 'mineru_html'

    def __init__(self, config: dict):
        super().__init__()
        from mineru_html import MinerUHTML, MinerUHTMLConfig

        model_path = config.pop('model_path', None)
        if model_path is None:
            raise ValueError('model_path is required')
        self.extractor = MinerUHTML(
            model_path=model_path, config=MinerUHTMLConfig(**config)
        )

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using MinerU-HTML.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML (empty string if extraction fails)
        """
        mineru_html_output = self.extractor.process(input_html)[0]
        main_html = mineru_html_output.main_html
        if main_html is None:
            main_html = ''
        return main_html

    def extract_main_html_batch(self, input_list: list[tuple[str, str]]) -> list[str]:
        """
        Extract main HTML for a batch of inputs using MinerU-HTML.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of extracted main HTML strings (empty string on error)
        """
        mineru_html_output_list = self.extractor.process(
            [input_html for input_html, _ in input_list]
        )
        result_list = []
        for mineru_html_output in mineru_html_output_list:
            if mineru_html_output.main_html is not None:
                result_list.append(mineru_html_output.main_html)
            else:
                result_list.append('')
        return result_list


class MinerU_HTML_MD_Extractor(MinerUHTMLExtractor):
    """MinerU_HTML HTML extractor with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MinerU_HTML_Text_Extractor(MinerUHTMLExtractor):
    """MinerU_HTML HTML extractor with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class MinerUHTMLFallbackExtractor(MinerUHTMLExtractor):
    """
    HTML extractor using MinerU-HTML library with fallback enabled.

    Extracts main HTML content using MinerU-HTML with fallback mechanism enabled.
    """

    name_str = 'mineru_html_fallback'

    def __init__(self, config: dict):
        config['use_fall_back'] = 'trafilatura'
        super().__init__(config)


class MinerUHTMLFallback_MD_Extractor(MinerUHTMLFallbackExtractor):
    """MinerU-HTML extractor (with fallback) with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MinerUHTMLFallback_Text_Extractor(MinerUHTMLFallbackExtractor):
    """MinerU-HTML extractor (with fallback) with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class MinerUHTMLBypassFallbackExtractor(MinerUHTMLExtractor):
    """
    HTML extractor using MinerU-HTML library with bypass fallback enabled.

    Extracts main HTML content using MinerU-HTML with bypass fallback mechanism enabled.
    """
    name_str = 'mineru_html_bypass_fallback'

    def __init__(self, config: dict):
        config['use_fall_back'] = 'bypass'
        super().__init__(config)


class MinerUHTMLBypassFallback_MD_Extractor(MinerUHTMLBypassFallbackExtractor):
    """MinerU-HTML extractor (with bypass fallback) with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MinerUHTMLBypassFallback_Text_Extractor(MinerUHTMLBypassFallbackExtractor):
    """MinerU-HTML extractor (with bypass fallback) with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class MinerUHTMLEmptyFallbackExtractor(MinerUHTMLExtractor):
    """
    HTML extractor using MinerU-HTML library with empty fallback enabled.

    Extracts main HTML content using MinerU-HTML with empty fallback mechanism enabled.
    """

    name_str = 'mineru_html_emptyfallback'

    def __init__(self, config: dict):
        config['use_fall_back'] = 'empty'
        super().__init__(config)


class MinerUHTMLEmptyFallback_MD_Extractor(MinerUHTMLEmptyFallbackExtractor):
    """MinerU-HTML extractor (with empty fallback) with Markdown output format."""

    format_str = 'html-md'

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MinerUHTMLEmptyFallback_Text_Extractor(MinerUHTMLEmptyFallbackExtractor):
    """MinerU-HTML extractor (with empty fallback) with plain text output format."""

    format_str = 'html-text'

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'
