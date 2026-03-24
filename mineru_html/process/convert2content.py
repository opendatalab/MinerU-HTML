from webpage_converter.convert import convert_html_to_structured_data

from mineru_html.base import MinerUHTMLCase
from mineru_html.exceptions import (MinerUHTMLConvert2ContentError,
                                    MinerUHTMLError)


def convert2content(
    input_case: MinerUHTMLCase,
    output_format: str = 'mm_md',
) -> MinerUHTMLCase:
    """
    Convert HTML to structured data.
    """
    try:
        if output_format == 'none':
            return input_case
        elif not input_case.output_data:
            raise MinerUHTMLConvert2ContentError(
                'Output data is not set for case',
                case_id=input_case.case_id
            )
        else:
            main_content = convert_html_to_structured_data(
                main_html=input_case.output_data.main_html,
                url=input_case.input_data.url,
                output_format=output_format,
            )
            input_case.output_data.main_content = main_content
        return input_case
    except Exception as e:
        if isinstance(e, MinerUHTMLError):
            e.set_case_id(input_case.case_id)
            raise e
        else:
            raise MinerUHTMLConvert2ContentError(
                f'Convert main HTML to structured data failed: {str(e)}',
                case_id=input_case.case_id,
            ) from e
