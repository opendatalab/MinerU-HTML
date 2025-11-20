"""
Custom exception classes for the Dripper HTML extraction system.

This module defines a hierarchy of exceptions used throughout the Dripper
system to provide more specific error handling and debugging information.
"""


class DripperError(Exception):
    """
    Base exception class for all Dripper-related errors.

    All custom exceptions in the Dripper system inherit from this class,
    allowing for generic exception handling when needed.
    """
    pass


class DripperResponseParseError(DripperError):
    """
    Exception raised when LLM response parsing fails.

    This exception is raised when the system cannot parse the response
    from the language model into the expected format.
    """
    pass


class DripperLoadModelError(DripperError):
    """
    Exception raised when model loading fails.

    This exception is raised when there are errors loading the LLM model,
    such as missing model files, incompatible model formats, or insufficient resources.
    """
    pass


class DripperEnvError(DripperError):
    """
    Exception raised for environment-related errors.

    This exception is raised when there are issues with the runtime environment,
    such as missing dependencies, incorrect environment variables, or system
    configuration problems.
    """
    pass


class DripperConfigError(DripperError):
    """
    Exception raised for configuration-related errors.

    This exception is raised when there are issues with the system configuration,
    such as invalid parameter values, missing required settings, or conflicting
    configuration options.
    """
    pass


class DripperPreprocessError(DripperError):
    """
    Exception raised during HTML preprocessing.

    This exception is raised when errors occur during the preprocessing stage
    of HTML extraction, such as invalid HTML structure or preprocessing failures.
    """
    pass


class DripperPostprocessError(DripperError):
    """
    Exception raised during HTML postprocessing.

    This exception is raised when errors occur during the postprocessing stage
    of HTML extraction, such as issues with result formatting or validation.
    """
    pass


class DripperTypeError(DripperError):
    """
    Exception raised for type-related errors.

    This exception is raised when there are type mismatches or invalid type
    conversions in the code, such as passing incorrect data types to functions.
    """
    pass


class DripperLogitsError(DripperError):
    """
    Exception raised for Logits Processor related errors.

    This exception is raised when there are errors in the logits processing
    pipeline, such as issues with state machine processing or logits extraction.
    """
    pass


class DripperPromptError(DripperError):
    """
    Exception raised for prompt-related errors.

    This exception is raised when there are errors in prompt generation or
    processing, such as invalid prompt templates or prompt formatting issues.
    """
    pass
