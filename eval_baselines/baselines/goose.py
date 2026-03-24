from .base import MainContentExtractor


class Goose3Extractor(MainContentExtractor):
    """
    Text extractor using goose3 library.

    Extracts plain text content using goose3's article extractor.
    """

    name_str = 'goose3'
    format_str = 'text'

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text using goose3.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted cleaned text)
        """
        from goose3 import Goose

        g = Goose()
        return '', g.extract(raw_html=input_html).cleaned_text
