from typing import Any, Dict, Optional

from mineru_html.inference.base_backend import InferenceBackend


def create_vllm_backend(
    model_path: str,
    response_format: str,
    max_context_window: int = 256 * 1024,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    model_gen_kwargs: Optional[Dict[str, Any]] = None,
) -> InferenceBackend:
    """Create a VLLM inference backend.

    Args:
        model_path: Path to the model directory.
        response_format: Response format for the model.
        max_context_window: Maximum context window size in tokens.
        model_init_kwargs: Optional kwargs for model initialization.
        model_gen_kwargs: Optional kwargs for model generation.

    Returns:
        VLLM inference backend instance.
    """
    try:
        from mineru_html.inference.vllm_backend import VLLMInferenceBackend
    except ImportError:
        raise ImportError(
            "VLLM backend requires 'vllm' package. "
            'Please install it with: pip install vllm'
        )

    return VLLMInferenceBackend(
        model_path=model_path,
        max_context_window=max_context_window,
        response_format=response_format,
        model_init_kwargs=model_init_kwargs or {},
        model_gen_kwargs=model_gen_kwargs or {},
    )


def create_transformers_backend(
    model_path: str,
    max_context_window: int = 256 * 1024,
    tokenizer=None,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    model_gen_kwargs: Optional[Dict[str, Any]] = None,
    response_format: str = 'compact',
) -> InferenceBackend:
    """Create a Transformers inference backend.

    Args:
        model_path: Path to the model directory.
        max_context_window: Maximum context window size in tokens.
        tokenizer: Optional tokenizer. If None, will be loaded from model_path.
        model_init_kwargs: Optional kwargs for model initialization.
        model_gen_kwargs: Optional kwargs for model generation.
        response_format: Response format for the model.
    Returns:
        Transformers inference backend instance.
    """
    try:
        from mineru_html.inference.transformers_backend import \
            TransformersInferenceBackend
    except ImportError:
        raise ImportError(
            "Transformers backend requires 'transformers' package. "
            'Please install it with: pip install transformers'
        )

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return TransformersInferenceBackend(
        model_path=model_path,
        max_context_window=max_context_window,
        tokenizer=tokenizer,
        model_init_kwargs=model_init_kwargs or {},
        model_gen_kwargs=model_gen_kwargs or {},
        response_format=response_format,
    )


def create_openai_backend(
    base_url: str,
    sk: str,
    model: str,
    max_context_window: int = 128 * 1024,
    retry_times: int = 3,
    response_format: str = 'json',
) -> InferenceBackend:
    """Create an OpenAI-compatible API inference backend.

    Args:
        base_url: Base URL for the OpenAI-compatible API.
        sk: API key/secret key for authentication.
        model: Model name to use.
        max_context_window: Maximum context window size in tokens.
        retry_times: Number of retry attempts for API calls.
        response_format: Response format for the model ("json" or "compact").

    Returns:
        OpenAI API inference backend instance.
    """
    try:
        from mineru_html.inference.openai_backend import \
            OpenaiAPIInferenceBackend
    except ImportError:
        raise ImportError(
            "OpenAI backend requires 'openai' package. "
            'Please install it with: pip install openai'
        )

    return OpenaiAPIInferenceBackend(
        base_url=base_url,
        sk=sk,
        max_context_window=max_context_window,
        model=model,
        retry_times=retry_times,
        response_format=response_format,
    )
