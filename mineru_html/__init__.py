# mineru_html/__init__.py
from .api import MinerUHTMLConfig, MinerUHTMLGeneric
from .implementations import (MinerUHTML, MinerUHTML_OpenAI,
                              MinerUHTML_Transformers)
from .inference.factory import (create_openai_backend,
                                create_transformers_backend,
                                create_vllm_backend)

__all__ = [
    'MinerUHTMLGeneric',
    'MinerUHTMLConfig',
    'MinerUHTML',  # VLLM implementation
    'MinerUHTML_OpenAI',
    'MinerUHTML_Transformers',
    'create_vllm_backend',
    'create_transformers_backend',
    'create_openai_backend',
]
