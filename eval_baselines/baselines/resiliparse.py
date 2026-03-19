from .base import MainContentExtractor


class ResiliparseTextExtractor(MainContentExtractor):
    """
    Text extractor using resiliparse library.

    Extracts plain text content directly using resiliparse's html2text extractor.
    """

    name_str = 'resiliparse'
    format_str = 'text'

    def __init__(self):
        from resiliparse.extract.html2text import extract_plain_text

        self._extract = extract_plain_text

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text using resiliparse.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted text content)
        """
        return '', self._extract(
            input_html,
            main_content=True,
            alt_texts=False,
            links=False,
            comments=False,
        )
