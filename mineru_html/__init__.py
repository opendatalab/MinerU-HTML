# mineru_html/__init__.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_LAZY_IMPORTS = {
    'MinerUHTMLGeneric': ('.api', 'MinerUHTMLGeneric'),
    'MinerUHTMLConfig': ('.api', 'MinerUHTMLConfig'),
    'MinerUHTML': ('.implementations', 'MinerUHTML'),
    'MinerUHTML_OpenAI': ('.implementations', 'MinerUHTML_OpenAI'),
    'MinerUHTML_Transformers': ('.implementations', 'MinerUHTML_Transformers'),
    'create_vllm_backend': ('.inference.factory', 'create_vllm_backend'),
    'create_transformers_backend': (
        '.inference.factory',
        'create_transformers_backend',
    ),
    'create_openai_backend': ('.inference.factory', 'create_openai_backend'),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    import importlib

    module_name, attr = _LAZY_IMPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
