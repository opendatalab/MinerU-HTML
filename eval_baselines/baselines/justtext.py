from .base import MainContentExtractor


class JusttextExtractor(MainContentExtractor):
    """
    Text extractor using justext library.

    Extracts plain text content by removing boilerplate using justext.
    """

    name_str = 'justtext'
    format_str = 'text'

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text using justext.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted text content)
        """
        import justext

        paragraphs = justext.justext(input_html, justext.get_stoplist('English'))
        valid = [
            paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate
        ]

        return '', ' '.join(valid)
