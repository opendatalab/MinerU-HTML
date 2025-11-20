"""
Run inference on benchmark cases using vLLM and Ray.

This script performs distributed inference on HTML processing tasks using
vLLM for efficient LLM inference and Ray for parallel batch processing.
It reads simplified HTML files from a task directory and generates predictions.
"""

import json
from pathlib import Path

import ray
from vllm import LLM

from dripper.base import DripperGenerateInput, DripperGenerateOutput
from dripper.inference.inference import generate
from dripper.inference.prompt import get_full_prompt

# Initialize Ray for distributed processing
ray.init()


def gather_data_from_task_dir(task_dir: str) -> list[DripperGenerateInput]:
    """
    Gather input data from task directory.

    Reads simplified HTML files from each case subdirectory and creates
    DripperGenerateInput objects for inference.

    Args:
        task_dir: Root directory containing case subdirectories

    Returns:
        List of DripperGenerateInput objects, one per case
    """
    data_list = []
    task_dir = Path(task_dir) / 'cases'

    # Iterate through each case directory
    for case in task_dir.iterdir():
        # Read simplified HTML file for this case
        simpled_html = open(case / 'simpled_html.html', 'r').read()
        data_list.append(
            DripperGenerateInput(
                alg_html=simpled_html,
                prompt=get_full_prompt,
                case_id=case.name,
            )
        )
    return data_list


@ray.remote
def inference_batch(
    data_list: list[DripperGenerateInput],
    model_path: str,
    tp: int,
    use_state_machine: str,
):
    """
    Perform batch inference on a list of input data.

    This is a Ray remote function that runs on a worker node. It initializes
    a vLLM model and generates predictions for the input batch.

    Args:
        data_list: List of input data items to process
        model_path: Path to the LLM model
        tp: Tensor parallel size (number of GPUs to use per worker)
        use_state_machine: State machine version to use ('v2' or None)

    Returns:
        List of DripperGenerateOutput objects containing inference results
    """
    # Initialize vLLM model with tensor parallelism
    llm = LLM(model=model_path, tensor_parallel_size=tp)

    # Generate predictions for the batch
    result_list = generate(llm, data_list, use_state_machine=use_state_machine)
    return result_list


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run distributed inference on benchmark cases using vLLM and Ray'
    )
    parser.add_argument(
        '--task_dir',
        type=str,
        required=True,
        help='Directory containing case subdirectories with simpled_html.html files'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the LLM model for inference'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output JSONL file for inference results'
    )
    parser.add_argument(
        '--tp',
        type=int,
        default=1,
        help='Tensor parallel size (number of GPUs per worker)'
    )
    parser.add_argument(
        '--no_logits',
        action='store_true',
        default=False,
        help='Disable state machine processing (do not use logits)'
    )
    parser.add_argument(
        '--machine_version',
        type=str,
        default='v2',
        help='State machine version to use (e.g., "v2"). Ignored if --no_logits is set'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Number of cases to process in each batch'
    )
    args = parser.parse_args()

    # Gather input data from task directory
    data_list = gather_data_from_task_dir(args.task_dir)

    # Split data into batches for parallel processing
    batch_list = [
        data_list[i : i + args.batch_size]
        for i in range(0, len(data_list), args.batch_size)
    ]

    # Determine state machine configuration
    if not args.no_logits:
        use_state_machine = args.machine_version
    else:
        use_state_machine = None

    # Submit all batches to Ray workers for parallel processing
    # Each worker uses args.tp GPUs and 8 CPUs
    unfinished_batch_result_list = [
        inference_batch.options(num_gpus=args.tp, num_cpus=8).remote(
            batch, args.model_path, args.tp, use_state_machine
        )
        for batch in batch_list
    ]

    # Collect results as batches complete
    batch_result_list: list[DripperGenerateOutput] = []
    while len(unfinished_batch_result_list) > 0:
        print(
            f'waiting for {len(unfinished_batch_result_list)}/{len(batch_list)} batches'
        )
        # Wait for at least one batch to complete (timeout: 5 seconds)
        ready_list, unfinished_batch_result_list = ray.wait(
            unfinished_batch_result_list, timeout=5
        )
        # Retrieve and extend results from completed batches
        for ready_task in ready_list:
            batch_result_list.extend(ray.get(ready_task))

    # Write all results to output file in JSONL format
    with open(args.output_path, 'w') as f:
        for result in batch_result_list:
            f.write(json.dumps(result.to_dict()) + '\n')
