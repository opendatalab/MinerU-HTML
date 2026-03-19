from abc import ABC, abstractmethod
from typing import Optional

import html2text
import html_text


class HTML2TextWrapper:
    def __init__(self):
        self.converter = html2text.HTML2Text(bodywidth=0)
        self.converter.ignore_links = True
        self.converter.ignore_images = True

    def __call__(self, html_str: str, url: str = '') -> str:
        self.converter.baseurl = url
        text = self.converter.handle(html_str)
        self.converter.baseurl = ''
        return text


def html_to_text_func(html_str: str, url: str, format: str) -> str:
    """Convert a html string to a text string

    Args:
        html_str (str): the html string
        url (str, optional): the url of the html string. Defaults to "".
        format (str, optional): the format of the text string. Defaults to "MD".
    Returns:
        content (str): the text string
    """
    if format == 'MD':
        instance = HTML2TextWrapper()
        return instance(html_str, url)
    else:
        return html_text.extract_text(html_str)


class BaseExtractor(ABC):
    """
    Base class for HTML content extractors.

    Defines the interface that all extractors must implement for extracting
    main HTML and main content from HTML pages.
    """

    name_str: Optional[str] = None
    format_str: Optional[str] = None

    @abstractmethod
    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML content from input HTML.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML string
        """
        pass

    @abstractmethod
    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract both main HTML and main content.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (main_html, main_content)
        """
        pass

    def extract_main_html_batch(self, input_list: list[tuple[str, str]]) -> list[str]:
        """
        Extract main HTML for a batch of inputs.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of extracted main HTML strings (empty string on error)
        """
        result_list = []
        for input_html, url in input_list:
            try:
                result_list.append(self.extract_main_html(input_html, url))
            except Exception:
                result_list.append('')
        return result_list

    def extract_batch(self, input_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Extract main HTML and content for a batch of inputs.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of (main_html, main_content) tuples ((empty, empty) on error)
        """
        result_list = []
        for input_html, url in input_list:
            try:
                result_list.append(self.extract(input_html, url))
            except Exception:
                result_list.append(('', ''))
        return result_list

    @classmethod
    def full_name(cls) -> str:
        assert cls.name_str is not None
        assert cls.format_str is not None
        return f'{cls.name_str}-{cls.format_str}'


class MainHTMLExtractor(BaseExtractor):
    """
    Base class for extractors that extract main HTML first.

    These extractors first extract main HTML, then convert it to text content
    using a specified format (MD or TEXT).
    """

    def __init__(self):
        """
        Initialize MainHTMLExtractor.
        """
        self.format = None
        self.set_format()

    def set_format(self):
        """
        Set the output format for text conversion.

        Must be implemented by subclasses to set self.format to 'MD' or 'TEXT'.

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract main HTML and convert to text content.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (main_html, main_content)
        """
        main_html = self.extract_main_html(input_html, url)
        try:
            main_content = html_to_text_func(main_html, url, self.format)
        except Exception:
            main_content = ''
        return main_html, main_content

    def extract_batch(self, input_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Extract main HTML and content for a batch of inputs.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of (main_html, main_content) tuples
        """
        main_html_list = self.extract_main_html_batch(input_list)
        result_list = []
        for (input_html, url), main_html in zip(input_list, main_html_list):
            if main_html == '':
                result_list.append(('', ''))
                continue
            try:
                main_content = html_to_text_func(main_html, url, self.format)
            except Exception:
                main_content = ''
            result_list.append((main_html, main_content))
        return result_list


class MainContentExtractor(BaseExtractor):
    """
    Base class for extractors that extract main content directly.

    These extractors extract text content directly without intermediate HTML.
    """

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML (returns empty string for content-only extractors).

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Empty string (content-only extractors don't produce HTML)
        """
        return ''

    def extract_main_html_batch(self, input_list: list[tuple[str, str]]) -> list[str]:
        """
        Extract main HTML for a batch (returns empty strings).

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of empty strings
        """
        return ['' for _ in input_list]
