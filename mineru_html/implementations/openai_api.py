from mineru_html.api import MinerUHTMLConfig, MinerUHTMLGeneric
from mineru_html.inference.factory import create_openai_backend


class MinerUHTML_OpenAI(MinerUHTMLGeneric):
    """MinerUHTML implementation using OpenAI API backend."""

    def __init__(
        self,
        base_url: str,
        sk: str,
        model: str,
        config: MinerUHTMLConfig = None,
        retry_times: int = 3,
    ):
        if config is None:
            config = MinerUHTMLConfig(
                prompt_version='v2', response_format='json'
            )

        # Create OpenAI backend
        llm = create_openai_backend(
            base_url=base_url,
            sk=sk,
            model=model,
            max_context_window=128 * 1024,
            retry_times=retry_times,
            response_format=config.response_format,
        )

        super().__init__(llm, config)
