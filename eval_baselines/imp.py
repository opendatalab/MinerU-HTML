import importlib
import inspect
import pkgutil
from typing import Dict, Type

from eval_baselines.baselines.base import BaseExtractor


def discover_extractors() -> Dict[str, Type[BaseExtractor]]:
    """
    Automatically discover all classes that inherit from BaseExtractor.

    Returns:
        A dictionary mapping extractor names to their class types.
    """
    extractors = {}

    import eval_baselines.baselines

    package = eval_baselines.baselines

    for finder, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        try:
            # Import the module
            full_module_name = f'{package.__name__}.{module_name}'
            module = importlib.import_module(full_module_name)

            # Iterate through all attributes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's a subclass of BaseExtractor (excluding BaseExtractor itself)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseExtractor)
                    and attr != BaseExtractor
                ):
                    # Ensure the class is fully defined (has name_str and format_str)
                    if hasattr(attr, 'name_str') and hasattr(attr, 'format_str'):
                        if attr.name_str and attr.format_str:
                            extractors[attr.full_name()] = attr

        except ImportError as e:
            print(f'Warning: Failed to import module {module_name}: {e}')
            continue

    return extractors


class ExtractorFactory:
    """
    Factory class for creating extractor instances.

    Provides a centralized way to create extractor instances by name,
    handling configuration for extractors that require it.
    """

    @staticmethod
    def _class_requires_config(class_type: Type[BaseExtractor]) -> bool:
        """Check if the class requires a config parameter.

        Args:
            class_type: The extractor class to check.

        Returns:
            True if the class requires a config parameter, False otherwise.
        """
        # Get the __init__ method signature
        try:
            sig = inspect.signature(class_type.__init__)
            # Check if 'config' parameter exists (excluding self)
            params = list(sig.parameters.keys())
            has_config = (
                'config' in params[1:] if params else False
            )  # params[0] is usually 'self'
        except (ValueError, TypeError):
            # If signature cannot be obtained (may be a built-in method), assume no config needed
            has_config = False
        return has_config

    @staticmethod
    def create_extractor(name: str, config: dict = None) -> BaseExtractor:
        """
        Create an extractor instance by name.

        Args:
            name: Name of the extractor to create (e.g., 'mineru_html-html-md', 'trafilatura-html-text')
            config: Optional configuration dictionary (required for some extractors like
                    ReaderLMExtractor and MinerU_HTML extractors)

        Returns:
            BaseExtractor instance of the requested type

        Raises:
            ValueError: If the extractor name is not recognized
        """
        mapping = discover_extractors()
        if name not in mapping:
            raise ValueError(f'Unknown extractor name: {name}')
        class_type = mapping[name]

        if ExtractorFactory._class_requires_config(class_type):
            return class_type(config)
        else:
            return class_type()
