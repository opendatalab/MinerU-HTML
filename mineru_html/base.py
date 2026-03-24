from typing import Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from mineru_html.exceptions import MinerUHTMLError


class MinerUHTMLBase(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid4()))


class MinerUHTMLInput(MinerUHTMLBase):
    raw_html: str
    url: Optional[str] = None


class MinerUHTMLOutput(MinerUHTMLBase):
    main_html: str
    main_content: Optional[str] = None


class MinerUHTMLGenerateInput(MinerUHTMLBase):
    full_prompt: str


class MinerUHTMLGenerateOutput(MinerUHTMLBase):
    response: str


class MinerUHTMLProcessData(MinerUHTMLBase):
    simpled_html: str
    map_html: str


class MinerUHTMLParseResult(MinerUHTMLBase):
    item_label: dict[str, str]


class MinerUHTMLCase:
    def __init__(self, input_data: MinerUHTMLInput):
        self.case_id: str = str(uuid4())
        self.input_data: MinerUHTMLInput = input_data
        self.process_data: Union[MinerUHTMLProcessData, None] = None
        self.generate_input: Union[MinerUHTMLGenerateInput, None] = None
        self.generate_output: Union[MinerUHTMLGenerateOutput, None] = None
        self.parse_result: Union[MinerUHTMLParseResult, None] = None
        self.output_data: Union[MinerUHTMLOutput, None] = None
        self.error: Union[MinerUHTMLError, None] = None

    def set_error(self, error: MinerUHTMLError):
        self.error = error

    @property
    def main_html(self) -> Union[str, None]:
        if self.output_data is None:
            return None
        return self.output_data.main_html
