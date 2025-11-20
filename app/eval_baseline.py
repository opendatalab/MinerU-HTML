"""
Evaluation baseline script for benchmarking extractors.

This script runs baseline evaluations on a dataset using Ray for distributed processing.
It supports resuming from previous results and can be configured for CPU or GPU execution.
"""

import argparse
import json
from pathlib import Path

from dripper.eval_baselines.base import (RayBatchProcessMaper, build_dataset,
                                         export_results, reduce_results)


def main():
    """Main entry point for the evaluation baseline script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run baseline evaluation on a benchmark dataset'
    )
    parser.add_argument(
        '--bench',
        type=str,
        required=True,
        help='Benchmark dataset path to evaluate'
    )
    parser.add_argument(
        '--task_dir',
        type=str,
        required=True,
        help='Directory to store task results and intermediate files'
    )
    parser.add_argument(
        '--extractor_name',
        type=str,
        required=True,
        help='Name of the extractor to use for evaluation, details in dripper/eval_baselines/baselines/imp.py'
    )
    parser.add_argument(
        '--default_config',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help='Default configuration preset: "cpu" or "gpu". "GPU" is for Dripper and ReaderLM, "CPU" is for other extractors'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the model file (if required by extractor)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for processing (overrides default_config if provided)'
    )
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=None,
        help='Number of GPUs to use (overrides default_config if provided)'
    )
    parser.add_argument(
        '--cpu_num',
        type=int,
        default=None,
        help='Number of CPUs to use (overrides default_config if provided)'
    )
    parser.add_argument(
        '--key',
        type=str,
        default=None,
        help='Process only a specific case by its key (for debugging)'
    )
    parser.add_argument(
        '--force_update',
        action='store_true',
        help='Force re-evaluation of all cases, ignoring existing results'
    )
    args = parser.parse_args()

    # Build the baseline dataset from the benchmark
    baseline_dataset = build_dataset(args.bench)

    # Check for existing results and skip already completed cases
    finished_results = []
    if not args.force_update:
        cases_dir = Path(args.task_dir) / 'cases'
        cases_dir.mkdir(parents=True, exist_ok=True)

        # Scan for completed cases with existing score files
        for case_dir in cases_dir.iterdir():
            if case_dir.is_dir():
                score_file = case_dir / 'rouge_score.json'
                if score_file.exists():
                    # Load existing score and create result dictionary
                    score = json.loads(score_file.read_text(encoding='utf-8'))
                    result_dict = {
                        'track_id': case_dir.name,
                        **score,
                        'meta.level': baseline_dataset[case_dir.name].level
                    }
                    finished_results.append(result_dict)
                    # Remove from dataset to skip re-processing
                    baseline_dataset.pop(case_dir.name)

        print(f'Found {len(finished_results)} finished results, skipping them')

    # Filter dataset to a specific key if requested (for debugging)
    if args.key is not None:
        baseline_dataset = {args.key: baseline_dataset[args.key]}

    # Apply default configuration presets if parameters are not explicitly set
    if args.default_config == 'cpu':
        args.batch_size = 1 if args.batch_size is None else args.batch_size
        args.gpu_num = 0 if args.gpu_num is None else args.gpu_num
        args.cpu_num = 1 if args.cpu_num is None else args.cpu_num
    elif args.default_config == 'gpu':
        args.batch_size = 512 if args.batch_size is None else args.batch_size
        args.gpu_num = 1 if args.gpu_num is None else args.gpu_num
        args.cpu_num = 4 if args.cpu_num is None else args.cpu_num

    # Initialize and run the Ray batch processor
    mapper = RayBatchProcessMaper(
        args.task_dir,
        baseline_dataset,
        args.extractor_name,
        {'model_path': args.model_path, 'state_machine': 'v2'},
        args.batch_size,
        args.gpu_num,
        args.cpu_num,
    )
    results = mapper.run()

    # Collect and validate results from the mapper
    result_benchmark_datas = []
    for result in results:
        if isinstance(result, tuple) or isinstance(result, dict):
            result_benchmark_datas.append(result)
        else:
            raise ValueError(f'Unknown result type: {type(result)}')

    # Merge with previously finished results
    result_benchmark_datas.extend(finished_results)

    # Export results to files and generate summary statistics
    flat_eval_df = export_results(result_benchmark_datas, args.task_dir)
    reduce_results(flat_eval_df, args.task_dir)


if __name__ == '__main__':
    main()
