import re

from .base import MainContentExtractor

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


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False) -> str:
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

    name_str = 'readerlm'
    format_str = 'text'

    def __init__(self, config: dict):
        """
        Initialize ReaderLMExtractor.

        Args:
            config: Configuration dictionary (must contain 'model_path')
        """
        from vllm import SamplingParams

        self.model_path = config.get('model_path')

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

    def extract_batch(self, input_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Extract main content for a batch of inputs using ReaderLM.

        Filters out prompts that exceed model length limits before processing.

        Args:
            input_list: List of (input_html, url) tuples

        Returns:
            List of (empty string, extracted content) tuples
        """
        prompts = [
            (idx, self.preprocess(html)) for idx, (html, url) in enumerate(input_list)
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
