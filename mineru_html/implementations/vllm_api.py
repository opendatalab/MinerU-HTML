from mineru_html.base import DEFALUT_MODEL
from mineru_html.api import MinerUHTMLConfig, MinerUHTMLGeneric
from mineru_html.inference.factory import create_vllm_backend


class MinerUHTML(MinerUHTMLGeneric):
    """MinerUHTML implementation using VLLM backend."""

    def __init__(self, model_path: str = None, config: MinerUHTMLConfig = None):
        """Initialize MinerUHTML with VLLM backend.

        Args:
            model_path: Path to the model directory or model identifier.
            config: Optional configuration. If None, uses default config.
        """

        if model_path is None:
            model_path = DEFALUT_MODEL

        if config is None:
            config = MinerUHTMLConfig(
                prompt_version='short_compact', response_format='compact'
            )

        model_init_kwargs = {
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.8,
        }

        llm = create_vllm_backend(
            model_path=model_path,
            response_format=config.response_format,
            max_context_window=256 * 1024,
            model_init_kwargs=model_init_kwargs,
        )

        super().__init__(llm, config)
