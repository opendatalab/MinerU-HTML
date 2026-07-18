from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .openai_api import MinerUHTML_OpenAI
    from .transformers_api import MinerUHTML_Transformers
    from .vllm_api import MinerUHTML

__all__ = ['MinerUHTML', 'MinerUHTML_OpenAI', 'MinerUHTML_Transformers']

_LAZY_IMPORTS = {
    'MinerUHTML': ('.vllm_api', 'MinerUHTML'),
    'MinerUHTML_OpenAI': ('.openai_api', 'MinerUHTML_OpenAI'),
    'MinerUHTML_Transformers': ('.transformers_api', 'MinerUHTML_Transformers'),
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
