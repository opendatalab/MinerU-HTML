"""
Baseline extractor implementations for HTML content extraction.

This module provides various extractor implementations for benchmarking,
including implementations based on popular libraries like trafilatura,
readability, magic-html, and custom Dripper extractors.
"""
import asyncio
import re
from abc import ABC, abstractmethod

import html_text


class HTML2TextWrapper:
    def __init__(self):
        import html2text
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

    def __init__(self, name: str):
        """
        Initialize BaseExtractor.

        Args:
            name: Name identifier for this extractor
        """
        self.name = name

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

    def extract_main_html_batch(
        self, input_list: list[tuple[str, str]]
    ) -> list[str]:
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

    def extract_batch(
        self, input_list: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
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


class MainHTMLExtractor(BaseExtractor):
    """
    Base class for extractors that extract main HTML first.

    These extractors first extract main HTML, then convert it to text content
    using a specified format (MD or TEXT).
    """

    def __init__(self, name: str):
        """
        Initialize MainHTMLExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
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

    def extract_batch(
        self, input_list: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
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

    def extract_main_html_batch(
        self, input_list: list[tuple[str, str]]
    ) -> list[str]:
        """
        Extract main HTML for a batch (returns empty strings).

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of empty strings
        """
        return ['' for _ in input_list]


class BoilerPy3HTMLExtractor(MainHTMLExtractor):
    """
    HTML extractor using boilerpy3 library.

    Extracts main HTML content using boilerpy3's ArticleExtractor.
    """

    def __init__(self, name: str):
        """
        Initialize BoilerPy3HTMLExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from boilerpy3 import extractors

        self.extractor = extractors.ArticleExtractor(raise_on_failure=False)

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using boilerpy3.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML with boilerplate marked
        """
        return self.extractor.get_marked_html(input_html)


class BoilerPy3_HTML_MD_Extractor(BoilerPy3HTMLExtractor):
    """BoilerPy3 extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class BoilerPy3_HTML_Text_Extractor(BoilerPy3HTMLExtractor):
    """BoilerPy3 extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class BoilerPy3TextExtractor(MainContentExtractor):
    """
    Text extractor using boilerpy3 library.

    Extracts plain text content directly using boilerpy3's ArticleExtractor.
    """

    def __init__(self, name: str):
        """
        Initialize BoilerPy3TextExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from boilerpy3 import extractors

        self.extractor = extractors.ArticleExtractor(raise_on_failure=False)

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract plain text content using boilerpy3.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted text content)
        """
        content = self.extractor.get_content(input_html)
        return '', content


class NewsPleaseExtractor(MainContentExtractor):
    """
    Text extractor using newsplease library.

    Extracts main text content from news articles using newsplease.
    """

    def __init__(self, name: str):
        """
        Initialize NewsPleaseExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)

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
            result = NewsPlease.from_html(
                input_html, url, fetch_images=False
            ).maintext
            if result is None:
                result = ''
            return '', result
        except Exception:
            return '', ''


class MagicHTML_Extractor(MainHTMLExtractor):
    """
    HTML extractor using magic-html library (ArticleExtractor).

    Extracts main HTML content from articles using magic-html's ArticleExtractor.
    """

    def __init__(self, name: str):
        """
        Initialize MagicHTML_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from magic_html import ArticleExtractor

        self.extractor = ArticleExtractor()

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using magic-html ArticleExtractor.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML
        """
        data = self.extractor.extract(input_html, base_url=url)
        return data['html']


class MagicHTML_MD_Extractor(MagicHTML_Extractor):
    """Magic-html extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MagicHTML_Text_Extractor(MagicHTML_Extractor):
    """Magic-html extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class MagicForumHTML_Extractor(MainHTMLExtractor):
    """
    HTML extractor using magic-html library (ForumExtractor).

    Extracts main HTML content from forum pages using magic-html's ForumExtractor.
    """

    def __init__(self, name: str):
        """
        Initialize MagicForumHTML_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from magic_html import ForumExtractor

        self.extractor = ForumExtractor()

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using magic-html ForumExtractor.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML
        """
        data = self.extractor.extract(input_html, base_url=url)
        return data['html']


class MagicForumHTML_MD_Extractor(MagicForumHTML_Extractor):
    """Magic-html forum extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class MagicForumHTML_Text_Extractor(MagicForumHTML_Extractor):
    """Magic-html forum extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class TrafilaturaExtractor(MainHTMLExtractor):
    """
    HTML extractor using trafilatura library.

    Extracts main HTML content using trafilatura with HTML output format.
    """

    def __init__(self, name: str):
        """
        Initialize TrafilaturaExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
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

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class Trafilatura_HTML_Text_Extractor(TrafilaturaExtractor):
    """Trafilatura HTML extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class Trafilatura_Text_Extractor(MainContentExtractor):
    """
    Text extractor using trafilatura library.

    Extracts plain text content directly using trafilatura with text output format.
    """

    def __init__(self, name: str):
        """
        Initialize Trafilatura_Text_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        self.name = name
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

    def __init__(self, name: str):
        """
        Initialize Trafilatura_MD_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        self.name = name
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


class ResiliparseTextExtractor(MainContentExtractor):
    """
    Text extractor using resiliparse library.

    Extracts plain text content directly using resiliparse's html2text extractor.
    """

    def __init__(self, name: str):
        """
        Initialize ResiliparseTextExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
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


class ReadabilityExtractor(MainHTMLExtractor):
    """
    HTML extractor using readability library.

    Extracts main HTML content using readability's Document class.
    """

    def __init__(self, name: str):
        """
        Initialize ReadabilityExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from readability import Document

        self.extractor = Document

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using readability.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML summary
        """
        doc = self.extractor(input_html)
        return doc.summary()


class Readability_HTML_MD_Extractor(ReadabilityExtractor):
    """Readability extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class Readability_HTML_Text_Extractor(ReadabilityExtractor):
    """Readability extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class ReadabiliPyExtractor(MainHTMLExtractor):
    """
    HTML extractor using readabilipy library.

    Extracts main HTML content using readabilipy's simple_tree_from_html_string.
    """

    def __init__(self, name: str):
        """
        Initialize ReadabiliPyExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)

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

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class ReadabiliPy_HTML_Text_Extractor(ReadabiliPyExtractor):
    """Readabilipy extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class HTML2TextExtractor(MainHTMLExtractor):
    """
    HTML extractor that passes through HTML unchanged.

    This extractor returns the input HTML as-is, then converts it to text
    using the specified format (MD or TEXT).
    """

    def __init__(self, name: str):
        """
        Initialize HTML2TextExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)

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


class HTML2Text_MD_Extractor(HTML2TextExtractor):
    """HTML2Text extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class HTML2Text_Text_Extractor(HTML2TextExtractor):
    """HTML2Text extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class JusttextExtractor(MainContentExtractor):
    """
    Text extractor using justext library.

    Extracts plain text content by removing boilerplate using justext.
    """

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

        paragraphs = justext.justext(
            input_html, justext.get_stoplist('English')
        )
        valid = [
            paragraph.text
            for paragraph in paragraphs
            if not paragraph.is_boilerplate
        ]

        return '', ' '.join(valid)


class Goose3Extractor(MainContentExtractor):
    """
    Text extractor using goose3 library.

    Extracts plain text content using goose3's article extractor.
    """

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


class GNE_Text_Extractor(MainContentExtractor):
    """
    Text extractor using GNE (General News Extractor) library.

    Extracts plain text content directly using GNE's GeneralNewsExtractor.
    """

    def __init__(self, name: str):
        """
        Initialize GNE_Text_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        self.name = name
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

    def __init__(self, name: str):
        """
        Initialize GNE_HTML_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
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
        main_html = self.extractor.extract(
            input_html, with_body_html=True
        )['body_html']
        return main_html


class GNE_HTML_MD_Extractor(GNE_HTML_Extractor):
    """GNE HTML extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class GNE_HTML_Text_Extractor(GNE_HTML_Extractor):
    """GNE HTML extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class Crawl4aiHTMLExtractor(MainHTMLExtractor):
    """
    HTML extractor using crawl4ai library.

    Extracts main HTML content using crawl4ai's AsyncWebCrawler.
    """

    def __init__(self, name: str):
        """
        Initialize Crawl4aiHTMLExtractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

        self.crawler = AsyncWebCrawler()
        self.config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async def _extract_main_html(self, input_html: str, url: str) -> str:
        """
        Async method to extract main HTML using crawl4ai.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted HTML string
        """
        async with self.crawler as crawler:
            result = await crawler.arun(
                url='raw:' + input_html, config=self.config
            )
            if isinstance(result.html, str):
                return result.html
            else:
                return ''

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using crawl4ai (synchronous wrapper).

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted HTML string
        """
        # Get event loop and wait for the async result
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self._extract_main_html(input_html, url)
        )
        return result


class Crawl4ai_HTML_MD_Extractor(Crawl4aiHTMLExtractor):
    """Crawl4ai HTML extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class Crawl4ai_HTML_Text_Extractor(Crawl4aiHTMLExtractor):
    """Crawl4ai HTML extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class Crawl4ai_Text_Extractor(MainContentExtractor):
    """
    Text extractor using crawl4ai library.

    Extracts Markdown content directly using crawl4ai's AsyncWebCrawler.
    """

    def __init__(self, name: str):
        """
        Initialize Crawl4ai_Text_Extractor.

        Args:
            name: Name identifier for this extractor
        """
        super().__init__(name)
        from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

        self.crawler = AsyncWebCrawler()
        self.config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async def _extract_text(self, input_html: str, url: str) -> str:
        """
        Async method to extract text using crawl4ai.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted Markdown content
        """
        async with self.crawler as crawler:
            result = await crawler.arun(
                url='raw:' + input_html, config=self.config
            )
            return result.markdown

    def extract(self, input_html: str, url: str) -> tuple[str, str]:
        """
        Extract text using crawl4ai (synchronous wrapper).

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Tuple of (empty string, extracted Markdown content)
        """
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._extract_text(input_html, url))
        return '', result


class DripperHTMLExtractor(MainHTMLExtractor):
    """
    HTML extractor using Dripper library.

    Extracts main HTML content using the custom Dripper extraction system.
    """

    def __init__(self, name: str, config: dict):
        """
        Initialize DripperHTMLExtractor.

        Args:
            name: Name identifier for this extractor
            config: Configuration dictionary for Dripper (must contain 'model_path')
        """
        super().__init__(name)
        from dripper.api import Dripper

        self.extractor = Dripper(config)

    def extract_main_html(self, input_html: str, url: str) -> str:
        """
        Extract main HTML using Dripper.

        Args:
            input_html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main HTML (empty string if extraction fails)
        """
        dripper_output = self.extractor.process(input_html)[0]
        main_html = dripper_output.main_html
        if main_html is None:
            main_html = ''
        return main_html

    def extract_main_html_batch(
        self, input_list: list[tuple[str, str]]
    ) -> list[str]:
        """
        Extract main HTML for a batch of inputs using Dripper.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of extracted main HTML strings (empty string on error)
        """
        dripper_output_list = self.extractor.process(
            [input_html for input_html, _ in input_list]
        )
        result_list = []
        for dripper_output in dripper_output_list:
            if dripper_output.main_html is not None:
                result_list.append(dripper_output.main_html)
            else:
                result_list.append('')
        return result_list


class Dripper_HTML_MD_Extractor(DripperHTMLExtractor):
    """Dripper HTML extractor with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class Dripper_HTML_Text_Extractor(DripperHTMLExtractor):
    """Dripper HTML extractor with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


class DripperHTMLFallbackExtractor(DripperHTMLExtractor):
    """
    HTML extractor using Dripper library with fallback enabled.

    Extracts main HTML content using Dripper with fallback mechanism enabled.
    """

    def __init__(self, name: str, config: dict):
        """
        Initialize DripperHTMLFallbackExtractor.

        Args:
            name: Name identifier for this extractor
            config: Configuration dictionary for Dripper (must contain 'model_path')
        """
        config['use_fall_back'] = True
        super().__init__(name, config)


class DripperHTMLFallback_MD_Extractor(DripperHTMLFallbackExtractor):
    """Dripper HTML extractor (with fallback) with Markdown output format."""

    def set_format(self):
        """Set output format to Markdown."""
        self.format = 'MD'


class DripperHTMLFallback_Text_Extractor(DripperHTMLFallbackExtractor):
    """Dripper HTML extractor (with fallback) with plain text output format."""

    def set_format(self):
        """Set output format to plain text."""
        self.format = 'TEXT'


# ReaderLM implementation
# Regular expression patterns for HTML cleaning
SCRIPT_PATTERN = r'<[ ]*script.*?\/[ ]*script[ ]*>'
STYLE_PATTERN = r'<[ ]*style.*?\/[ ]*style[ ]*>'
META_PATTERN = r'<[ ]*meta.*?>'
COMMENT_PATTERN = r'<[ ]*!--.*?--[ ]*>'
LINK_PATTERN = r'<[ ]*link.*?>'
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r'(<svg[^>]*>)(.*?)(<\/svg>)'


def replace_svg(html: str, new_content: str = 'this is a placeholder') -> str:
    """
    Replace SVG content with a placeholder.

    Args:
        html: HTML string containing SVG elements
        new_content: Replacement content for SVG body (default: 'this is a placeholder')

    Returns:
        HTML string with SVG content replaced
    """
    return re.sub(
        SVG_PATTERN,
        lambda match: f'{match.group(1)}{new_content}{match.group(3)}',
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = '#') -> str:
    """
    Replace base64-encoded images with a placeholder src.

    Args:
        html: HTML string containing base64 images
        new_image_src: Replacement image src (default: '#')

    Returns:
        HTML string with base64 images replaced
    """
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(
    html: str, clean_svg: bool = False, clean_base64: bool = False
) -> str:
    """
    Clean HTML by removing scripts, styles, meta tags, comments, and links.

    Optionally can also clean SVG content and base64 images.

    Args:
        html: Raw HTML string to clean
        clean_svg: Whether to replace SVG content with placeholder (default: False)
        clean_base64: Whether to replace base64 images with placeholder (default: False)

    Returns:
        Cleaned HTML string
    """
    html = re.sub(
        SCRIPT_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        STYLE_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        META_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        COMMENT_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    html = re.sub(
        LINK_PATTERN, '', html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html


def create_prompt(
    text: str, tokenizer=None, instruction: str = None, schema: str = None
) -> str:
    """
    Create a prompt for the LLM with optional instruction and JSON schema.

    Args:
        text: HTML text to include in the prompt
        tokenizer: Tokenizer to apply chat template (required)
        instruction: Custom instruction text (default: extract main content to Markdown)
        schema: Optional JSON schema string for structured extraction

    Returns:
        Formatted prompt string with chat template applied
    """
    if not instruction:
        instruction = (
            'Extract the main content from the given HTML and convert it to '
            'Markdown format.'
        )
    if schema:
        instruction = (
            'Extract the specified information from a list of news threads and '
            'present it in a structured JSON format.'
        )
        prompt = (
            f'{instruction}\n```html\n{text}\n```\n'
            f'The JSON schema is as follows:```json\n{schema}\n```'
        )
    else:
        prompt = f'{instruction}\n```html\n{text}\n```'

    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class ReaderLMExtractor(MainContentExtractor):
    """
    Text extractor using ReaderLM (LLM-based extraction).

    Uses a language model to extract main content from HTML by generating
    Markdown-formatted content. Includes HTML cleaning and prompt generation.
    """

    def __init__(self, name: str, config: dict):
        """
        Initialize ReaderLMExtractor.

        Args:
            name: Name identifier for this extractor
            config: Configuration dictionary (must contain 'model_path')
        """
        self.name = name
        self.model_path = config.get('model_path')
        from vllm import SamplingParams

        self.sampling_params = SamplingParams(
            temperature=0,
            top_k=1,
            presence_penalty=1.13,
            repetition_penalty=0.25,
            max_tokens=8192,
            frequency_penalty=0.25,
        )
        self.max_model_len = 256000
        self.llm = None
        self.tokenizer = None

    def get_llm(self):
        """
        Get or initialize the LLM instance (lazy loading).

        Returns:
            Initialized vLLM LLM instance
        """
        if self.llm is None:
            from vllm import LLM

            self.llm = LLM(
                model=self.model_path,
                max_model_len=self.max_model_len,
                dtype='float16',
            )
        return self.llm

    def get_tokenizer(self):
        """
        Get or initialize the tokenizer instance (lazy loading).

        Returns:
            Initialized AutoTokenizer instance
        """
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True
            )
        return self.tokenizer

    def preprocess(self, html: str) -> str:
        """
        Preprocess HTML and create LLM prompt.

        Cleans HTML (removes scripts, styles, SVG, base64 images) and creates
        a formatted prompt for the LLM.

        Args:
            html: Raw HTML string

        Returns:
            Formatted prompt string ready for LLM inference
        """
        html = clean_html(html, clean_svg=True, clean_base64=True)

        tokenizer = self.get_tokenizer()
        prompt = create_prompt(html, tokenizer)
        return prompt

    def postprocess(self, response: str) -> str:
        """
        Postprocess LLM response.

        Args:
            response: Raw response string from LLM

        Returns:
            Stripped response string
        """
        return response.strip()

    def extract(self, html: str, url: str) -> str:
        """
        Extract main content using ReaderLM.

        Args:
            html: Raw HTML string
            url: URL where the HTML was obtained from

        Returns:
            Extracted main content as text (Markdown format)
        """
        prompt = self.preprocess(html)
        result = (
            self.get_llm()
            .generate(prompt, sampling_params=self.sampling_params)[0]
            .outputs[0]
            .text
        )
        return self.postprocess(result)

    def check_valid(self, prompt: str) -> bool:
        """
        Check if prompt length is within model limits.

        Args:
            prompt: Prompt string to validate

        Returns:
            True if prompt is within max_model_len, False otherwise
        """
        tokenizer = self.get_tokenizer()
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        return len(tokens) < self.max_model_len

    def extract_batch(
        self, input_list: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """
        Extract main content for a batch of inputs using ReaderLM.

        Filters out prompts that exceed model length limits before processing.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of (empty string, extracted content) tuples
        """
        prompts = [
            (idx, self.preprocess(html))
            for idx, (html, url) in enumerate(input_list)
        ]

        # Filter out prompts that exceed model length
        valid_prompts = [item for item in prompts if self.check_valid(item[1])]

        results = self.get_llm().generate(
            [p[1] for p in valid_prompts], sampling_params=self.sampling_params
        )
        # Map results back to original indices
        result_map = {
            item[0]: self.postprocess(result.outputs[0].text)
            for item, result in zip(valid_prompts, results)
        }

        return [('', result_map.get(i, '')) for i in range(len(prompts))]


class ExtractorFactory:
    """
    Factory class for creating extractor instances.

    Provides a centralized way to create extractor instances by name,
    handling configuration for extractors that require it.
    """

    @staticmethod
    def create_extractor(name: str, config: dict = None) -> BaseExtractor:
        """
        Create an extractor instance by name.

        Args:
            name: Name of the extractor to create (e.g., 'dripper-md', 'trafilatura-html-text')
            config: Optional configuration dictionary (required for some extractors like
                    ReaderLMExtractor and Dripper extractors)

        Returns:
            BaseExtractor instance of the requested type

        Raises:
            ValueError: If the extractor name is not recognized
        """
        mapping = {
            'magic-html-md': MagicHTML_MD_Extractor,
            'magic-html-text': MagicHTML_Text_Extractor,
            'magic-forum-html-md': MagicForumHTML_MD_Extractor,
            'magic-forum-html-text': MagicForumHTML_Text_Extractor,
            'trafilatura-html-md': Trafilatura_HTML_MD_Extractor,
            'trafilatura-html-text': Trafilatura_HTML_Text_Extractor,
            'trafilatura-text': Trafilatura_Text_Extractor,
            'trafilatura-md': Trafilatura_MD_Extractor,
            'resiliparse': ResiliparseTextExtractor,
            'readability-html-md': Readability_HTML_MD_Extractor,
            'readability-html-text': Readability_HTML_Text_Extractor,
            'readabilipy-html-md': ReadabiliPy_HTML_MD_Extractor,
            'readabilipy-html-text': ReadabiliPy_HTML_Text_Extractor,
            'html2text-md': HTML2Text_MD_Extractor,
            'html2text-text': HTML2Text_Text_Extractor,
            'justtext': JusttextExtractor,
            'goose3': Goose3Extractor,
            'readerlm': ReaderLMExtractor,
            'dripper-md': Dripper_HTML_MD_Extractor,
            'dripper-text': Dripper_HTML_Text_Extractor,
            'dripper-fallback-md': DripperHTMLFallback_MD_Extractor,
            'dripper-fallback-text': DripperHTMLFallback_Text_Extractor,
            'gne-content': GNE_Text_Extractor,
            'gne-html-text': GNE_HTML_Text_Extractor,
            'gne-html-md': GNE_HTML_MD_Extractor,
            'boilerpy3-text': BoilerPy3TextExtractor,
            'boilerpy3-html-md': BoilerPy3_HTML_MD_Extractor,
            'boilerpy3-html-text': BoilerPy3_HTML_Text_Extractor,
            'newsplease': NewsPleaseExtractor,
            'crawl4ai-html-md': Crawl4ai_HTML_MD_Extractor,
            'crawl4ai-html-text': Crawl4ai_HTML_Text_Extractor,
            'crawl4ai-text': Crawl4ai_Text_Extractor,
        }
        if name not in mapping:
            raise ValueError(f'Unknown extractor name: {name}')
        cls = mapping[name]
        # Extractors that require config: pass config, others ignore it
        if cls in {
            ReaderLMExtractor,
            Dripper_HTML_MD_Extractor,
            Dripper_HTML_Text_Extractor,
            DripperHTMLFallback_MD_Extractor,
            DripperHTMLFallback_Text_Extractor,
        }:
            return cls(name, config)
        return cls(name)
