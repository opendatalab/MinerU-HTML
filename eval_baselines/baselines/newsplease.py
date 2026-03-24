from .base import MainContentExtractor


class NewsPleaseExtractor(MainContentExtractor):
    """
    Text extractor using newsplease library.

    Extracts main text content from news articles using newsplease.
    """

    name_str = 'newsplease'
    format_str = 'text'

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract main text using newsplease.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted main text)
        """
        from newsplease import NewsPlease

        try:
            result = NewsPlease.from_html(input_html, url, fetch_images=False).maintext
            if result is None:
                result = ''
            return '', result
        except Exception:
            return '', ''
