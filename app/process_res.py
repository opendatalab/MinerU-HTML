"""
Process LLM response files and extract prediction labels.

This script reads LLM response data from a JSONL file, parses each response
to extract prediction labels, and writes the results to output files.
Errors during parsing are collected and written to a separate error file.
"""

import logging

from dripper.exceptions import DripperResponseParseError
from dripper.inference.logits import parse_llm_response

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for processing LLM responses.

    Reads response data from a JSONL file, parses each response to extract
    prediction labels, and writes results to output files. Handles parsing
    errors gracefully by collecting error cases separately.
    """
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process LLM response files and extract prediction labels'
    )
    parser.add_argument(
        '--response',
        type=str,
        required=True,
        help='Path to input JSONL file containing LLM responses with case_id and response fields'
    )
    parser.add_argument(
        '--answer',
        type=str,
        required=True,
        help='Path to output JSONL file for successfully parsed answers'
    )
    parser.add_argument(
        '--error',
        type=str,
        required=True,
        help='Path to output JSONL file for cases that failed to parse'
    )
    args = parser.parse_args()

    # Read all response data into a dictionary keyed by case_id
    data_map = {}
    with open(args.response, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            case_id = data['case_id']
            data_map[case_id] = data

    # Process each response and extract prediction labels
    output_data_list = []
    error_id_list = []
    for data in data_map.values():
        try:
            # Parse LLM response to extract prediction labels
            output_data = {
                'track_id': data['case_id'],
                'predict_label': parse_llm_response(data['response']),
            }
        except DripperResponseParseError as e:
            # Handle parsing errors gracefully
            print(f"DripperResponseParseError for case_id {data['case_id']}: {e}")
            error_id_list.append(data['case_id'])
            continue
        except Exception as e:
            # Re-raise unexpected errors
            raise e

        output_data_list.append(output_data)

    # Write successfully parsed answers to output file
    with open(args.answer, 'w', encoding='utf-8') as f:
        for output_data in output_data_list:
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    # Write error cases to error file
    with open(args.error, 'w', encoding='utf-8') as f:
        for error_id in error_id_list:
            f.write(json.dumps({'case_id': error_id}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
