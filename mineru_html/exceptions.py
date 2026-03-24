class MinerUHTMLError(Exception):
    """MinerUHTML base Exception"""

    def __init__(self, message: str, case_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.case_id = case_id

    def set_case_id(self, case_id: str | None):
        self.case_id = case_id

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}(message={self.message}, case_id={self.case_id})'
        )


class MinerUHTMLResponseParseError(MinerUHTMLError):
    """MinerUHTML response parse error"""

    pass


class MinerUHTMLLoadModelError(MinerUHTMLError):
    """MinerUHTML model load error"""

    pass


class MinerUHTMLEnvError(MinerUHTMLError):
    """MinerUHTML environment error"""

    pass


class MinerUHTMLConfigError(MinerUHTMLError):
    """MinerUHTML config error"""

    pass


class MinerUHTMLPreprocessError(MinerUHTMLError):
    """MinerUHTML preprocess error"""

    pass


class MinerUHTMLInputTooLongError(MinerUHTMLError):
    """MinerUHTML input too long error"""

    pass


class MinerUHTMLPostprocessError(MinerUHTMLError):
    """MinerUHTML postprocess error"""

    pass


class MinerUHTMLDetectNoMainError(MinerUHTMLError):
    """MinerUHTML detect no main error"""

    pass


class MinerUHTMLTypeError(MinerUHTMLError):
    """MinerUHTML type error"""

    pass


class MinerUHTMLLogitsError(MinerUHTMLError):
    """MinerUHTML logits error"""

    pass


class MinerUHTMLPromptError(MinerUHTMLError):
    """MinerUHTML prompt error"""

    pass


class MinerUHTMLMapToMainError(MinerUHTMLError):
    """MinerUHTML map to main error"""

    pass


class MinerUHTMLFallbackError(MinerUHTMLError):
    """MinerUHTML fallback error"""

    pass


class MinerUHTMLConvert2ContentError(MinerUHTMLError):
    """MinerUHTML convert2content error"""

    pass
