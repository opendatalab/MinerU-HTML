# inference/__init__.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_backend import InferenceBackend, ModelResponse
    from .factory import (create_openai_backend, create_transformers_backend,
                          create_vllm_backend)

__all__ = [
    'InferenceBackend',
    'ModelResponse',
    'create_vllm_backend',
    'create_transformers_backend',
    'create_openai_backend',
]

_LAZY_IMPORTS = {
    'InferenceBackend': ('.base_backend', 'InferenceBackend'),
    'ModelResponse': ('.base_backend', 'ModelResponse'),
    'create_vllm_backend': ('.factory', 'create_vllm_backend'),
    'create_transformers_backend': ('.factory', 'create_transformers_backend'),
    'create_openai_backend': ('.factory', 'create_openai_backend'),
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
