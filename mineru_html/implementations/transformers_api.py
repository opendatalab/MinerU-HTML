from typing import Any, Dict, Optional

from transformers import AutoTokenizer

from mineru_html.base import DEFALUT_MODEL
from mineru_html.api import MinerUHTMLConfig, MinerUHTMLGeneric
from mineru_html.inference.factory import create_transformers_backend


class MinerUHTML_Transformers(MinerUHTMLGeneric):
    """MinerUHTML implementation using Transformers backend."""

    def __init__(
        self,
        model_path: str = None,
        config: MinerUHTMLConfig = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        model_gen_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MinerUHTML with Transformers backend.

        Args:
            model_path: Path to the model directory or model identifier.
            config: Optional configuration. If None, uses default config.
            tokenizer: Optional tokenizer. If None, will be loaded from model_path.
            model_init_kwargs: Optional kwargs for model initialization.
            model_gen_kwargs: Optional kwargs for model generation.
        """

        if model_path is None:
            model_path = DEFALUT_MODEL

        if config is None:
            config = MinerUHTMLConfig(
                prompt_version='short_compact', response_format='compact'
            )

        if model_init_kwargs is None:
            model_init_kwargs = {
                'device_map': 'auto',
                'dtype': 'auto',
            }

        llm = create_transformers_backend(
            model_path=model_path,
            max_context_window=256 * 1024,
            tokenizer=tokenizer,
            model_init_kwargs=model_init_kwargs,
            model_gen_kwargs=model_gen_kwargs,
        )
        llm.response_format = config.response_format

        super().__init__(llm, config)
