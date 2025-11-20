"""
Evaluation script with answer data support.

This script evaluates benchmark datasets using answer data, supporting two-step evaluation:
- Step 1: Extract and process intermediate data (HTML, content, labels)
- Step 2: Evaluate predictions against ground truth labels

Supports both single-process and multi-process execution modes.
"""

import argparse
import json
import os
from typing import Union

import pandas as pd

from dripper.eval.benckmark import (AnswerData, AnswerDataset, BenchmarkData,
                                    BenchmarkDataset, EvalResult,
                                    IntermediateData)
from dripper.eval.eval import eval_step_1, eval_step_2


def check_step1_complete(benchmark_data: BenchmarkData) -> bool:
    """
    Check if Step 1 intermediate data is complete.

    Step 1 requires the following intermediate data items:
    - gt_main_html: Ground truth main HTML
    - gt_main_content: Ground truth main content
    - simpled_html: Simplified HTML
    - map_html: Mapped HTML
    - gt_item_label: Ground truth item labels

    Args:
        benchmark_data: Benchmark data to check

    Returns:
        True if all required Step 1 data is present, False otherwise
    """
    imd = benchmark_data.intermediate_data
    item_name_list = [
        'gt_main_html',
        'gt_main_content',
        'simpled_html',
        'map_html',
        'gt_item_label',
    ]
    for item_name in item_name_list:
        item = getattr(imd, item_name)
        if item is None:
            print(f'Step 1 intermediate data is not complete, {item_name} is {item}')
            return False
    return True


def check_step2_complete(benchmark_data: BenchmarkData) -> bool:
    """
    Check if Step 2 evaluation is complete.

    Step 2 requires all intermediate data items and all evaluation result items
    to be present and non-None.

    Args:
        benchmark_data: Benchmark data to check

    Returns:
        True if all required Step 2 data is present, False otherwise
    """
    imd = benchmark_data.intermediate_data

    # Check all intermediate data items
    for item_name in IntermediateData.__annotations__.keys():
        item = getattr(imd, item_name)
        if item is None:
            print(f'Step 2 intermediate data is not complete, {item_name} is {item}')
            return False

    # Check all evaluation result items
    for item_name in EvalResult.__annotations__.keys():
        item = getattr(benchmark_data.eval_result, item_name)
        if item is None:
            print(f'Step 2 eval result is not complete, {item_name} is {item}')
            return False

    return True


def eval_one_benchmark_data(
    benchmark_data: BenchmarkData,
    answer_data: Union[AnswerData, str],
    target_dir: str,
    config: dict = {},
):
    """
    Evaluate a single benchmark data item.

    This function performs two-step evaluation:
    - Step 1: Extract and process intermediate data (HTML, content, labels)
    - Step 2: Evaluate predictions against ground truth (requires Step 1 to be complete)

    The function supports:
    - 'only_step1': Only run Step 1 evaluation
    - 'only_step2': Only run Step 2 evaluation (requires Step 1 results to exist)
    - AnswerData: Run both steps using provided answer data

    Args:
        benchmark_data: Benchmark data to evaluate
        answer_data: Answer data (AnswerData object) or mode string ('only_step1'/'only_step2')
        target_dir: Directory to save evaluation results
        config: Configuration dictionary with options:
            - force_update: If True, re-run even if results exist
            - dry_run: If True, use ground truth labels as predictions when predict_label is None

    Returns:
        BenchmarkData with evaluation results, or None/error dict if skipped/failed
    """
    # Load existing results from target directory if available
    target_dir_benchmark_data = BenchmarkData.load_from_dir(target_dir)

    force_update = config.get('force_update', False)

    # Step 1: Extract and process intermediate data
    if answer_data == 'only_step1' or isinstance(answer_data, AnswerData):
        if not force_update and check_step1_complete(target_dir_benchmark_data):
            # Use existing Step 1 results
            step1_res = target_dir_benchmark_data
        else:
            # Run Step 1 evaluation
            step1_res = eval_step_1(benchmark_data)
            step1_res.dump_to_dir(target_dir)

    # Early return if only Step 1 is requested
    if answer_data == 'only_step1':
        return

    # Step 2: Evaluate predictions against ground truth
    if answer_data == 'only_step2' or isinstance(answer_data, AnswerData):
        # Verify Step 1 is complete before proceeding
        if not check_step1_complete(target_dir_benchmark_data):
            raise ValueError('Step 1 is not complete')

        # Check if Step 2 results already exist
        if not force_update and check_step2_complete(target_dir_benchmark_data):
            step2_res = target_dir_benchmark_data
        else:
            # Determine prediction labels to use
            if isinstance(answer_data, AnswerData):
                # Use labels from provided answer data
                predict_label = answer_data.predict_label
            else:
                # Use labels from intermediate data
                predict_label = (
                    target_dir_benchmark_data.intermediate_data.pred_item_label
                )

                # In dry_run mode, use ground truth labels if predictions are missing
                if config.get('dry_run', False) and predict_label is None:
                    predict_label = (
                        target_dir_benchmark_data.intermediate_data.gt_item_label.copy()
                    )

                # Skip if no prediction labels available
                if predict_label is None:
                    return {
                        'track_id': target_dir_benchmark_data.track_id,
                        'info': 'skip'
                    }

            # Run Step 2 evaluation
            step2_res = eval_step_2(target_dir_benchmark_data, predict_label)
            step2_res.dump_to_dir(target_dir)

        return step2_res


def read_answer_file(answer_file: str) -> dict[str, dict]:
    """
    Read answer data from a JSONL file.

    Each line in the file should be a JSON object with at least a 'track_id' field.

    Args:
        answer_file: Path to the JSONL file containing answer data

    Returns:
        Dictionary mapping track_id to answer data
    """
    answer_dict = {}
    with open(answer_file, 'r') as f:
        for line in f:
            answer = json.loads(line)
            answer_dict[answer['track_id']] = answer
    return answer_dict


def build_dataset(
    bench_path: str, answer: str, step: str
) -> tuple[BenchmarkDataset, Union[AnswerDataset, str]]:
    """
    Build benchmark and answer datasets.

    Args:
        bench_path: Path to the benchmark dataset
        answer: Path to answer file (JSONL format), or None to use step mode
        step: Step mode ('1' for only_step1, '2' for only_step2) when answer is None

    Returns:
        Tuple of (benchmark_dataset, answer_dataset)
        answer_dataset can be AnswerDataset object or mode string ('only_step1'/'only_step2')

    Raises:
        ValueError: If step is invalid when answer is None
    """
    benchmark_dataset = BenchmarkDataset(bench_path, config={'patch_files': 'auto'})

    if answer is None:
        # Determine step mode based on step parameter
        if step == '1':
            answer_dataset = 'only_step1'
        elif step == '2':
            answer_dataset = 'only_step2'
        else:
            raise ValueError(
                f'Invalid step: {step}, only support 1: only_step1, 2: only_step2'
            )
    else:
        # Load answer dataset from file
        answer_dataset = AnswerDataset(answer)

    return benchmark_dataset, answer_dataset


def build_task_map(
    benchmark_dataset: BenchmarkDataset,
    answer_dataset: Union[AnswerDataset, str],
    args: argparse.Namespace,
) -> dict[str, dict]:
    """
    Build a task map for evaluation.

    Creates a mapping from track_id to task configuration, including:
    - benchmark_data: The benchmark data to evaluate
    - answer_data: Answer data or mode string
    - target_dir: Directory to save results
    - config: Evaluation configuration (force_update, dry_run)

    Args:
        benchmark_dataset: Dataset containing benchmark data
        answer_dataset: Answer dataset or mode string ('only_step1'/'only_step2')
        args: Command line arguments namespace

    Returns:
        Dictionary mapping track_id to task configuration dictionary
    """
    task_map: dict[str, dict] = {}

    for key in benchmark_dataset.data_map.keys():
        # Filter by key if specified (for debugging single cases)
        if args.key and args.key != key:
            continue

        benchmark_data = benchmark_dataset.data_map[key]

        # Determine answer data for this task
        if answer_dataset == 'only_step1':
            answer_data = 'only_step1'
        elif answer_dataset == 'only_step2':
            answer_data = 'only_step2'
        else:
            # Use answer from dataset if available, otherwise create blank answer
            # Only consider answers with IDs in benchmark_dataset, ignore others
            if key not in answer_dataset.data_map.keys():
                blank_answer = AnswerData(track_id=key, predict_label={})
                answer_data = blank_answer
            else:
                answer_data = answer_dataset.data_map[key]

        # Set up target directory for this task's results
        target_dir = os.path.join(args.task_dir, 'cases', key)

        task_map[key] = {
            'benchmark_data': benchmark_data,
            'answer_data': answer_data,
            'target_dir': target_dir,
            'config': {
                'force_update': args.force_update,
                'dry_run': args.dry_run,
            },
        }

    return task_map


class MultiProcessAsyncMaper:
    """
    Multi-process asynchronous mapper for parallel evaluation.

    Uses multiprocessing to evaluate multiple benchmark data items in parallel,
    with progress tracking and status reporting.
    """

    def __init__(self, arg_list: list, cpu_num: int):
        """
        Initialize the multi-process mapper.

        Args:
            arg_list: List of task configuration dictionaries
            cpu_num: Number of CPU processes to use
        """
        import multiprocessing as mp

        self.arg_list = arg_list
        self.pool = mp.Pool(processes=cpu_num)
        self.manager = mp.Manager()
        self.done_task_dict = self.manager.dict()

        # Initialize task status tracking
        for task_args in arg_list:
            self.done_task_dict[task_args['benchmark_data'].track_id] = 'not_started'
            task_args['done_task_dict'] = self.done_task_dict

    @staticmethod
    def eval_one_benchmark_data_wrap(task_args: dict):
        """
        Wrapper function for evaluating a single benchmark data item.

        This is a static method designed to be called by multiprocessing workers.
        It updates the shared task status dictionary and handles errors.

        Args:
            task_args: Task configuration dictionary containing:
                - benchmark_data: Benchmark data to evaluate
                - answer_data: Answer data or mode string
                - target_dir: Directory to save results
                - config: Evaluation configuration
                - done_task_dict: Shared dictionary for tracking task status

        Returns:
            Evaluation result (BenchmarkData) or error dictionary
        """
        done_task_dict = task_args['done_task_dict']
        print('Evaluating task:', task_args['benchmark_data'].track_id)
        benchmark_data = task_args['benchmark_data']
        answer_data = task_args['answer_data']

        # Update status to running
        done_task_dict[benchmark_data.track_id] = 'running'

        try:
            res = eval_one_benchmark_data(
                benchmark_data,
                answer_data,
                task_args['target_dir'],
                task_args['config'],
            )
        except Exception as e:
            print(f'Error when evaluating {benchmark_data.track_id}: {e}')
            done_task_dict[benchmark_data.track_id] = 'error'
            return {'track_id': benchmark_data.track_id, 'info': str(e)}

        # Update status to done
        done_task_dict[benchmark_data.track_id] = 'done'
        return res

    def run(self):
        """
        Run evaluation for all tasks in parallel.

        Uses async map to process tasks and periodically reports progress
        by counting tasks in different states (not_started, running, done, error).

        Returns:
            List of evaluation results
        """
        total_tasks = len(self.arg_list)
        map_result = self.pool.map_async(
            MultiProcessAsyncMaper.eval_one_benchmark_data_wrap,
            self.arg_list,
            chunksize=1,
        )

        # Monitor progress while tasks are running
        while not map_result.ready():
            map_result.wait(1)

            # Count tasks by status
            count_dict = {}
            not_finished_task_ids = []
            for task_id, status in self.done_task_dict.items():
                if status not in count_dict:
                    count_dict[status] = 0
                count_dict[status] += 1

                if not status == 'done':
                    not_finished_task_ids.append(task_id)

            # Print progress information
            info_str = ''
            for status, count in count_dict.items():
                info_str += f'{status}: {count}, '
            print(f'{info_str} Total tasks: {total_tasks}')

            # Print list of unfinished tasks if there are few enough
            if len(not_finished_task_ids) > 0 and len(not_finished_task_ids) < 20:
                print(f'Not finished tasks: {not_finished_task_ids}')

        results = map_result.get()
        return results


class SingleProcessMaper:
    """
    Single-process mapper for sequential evaluation.

    Evaluates benchmark data items one at a time in the current process.
    Useful for debugging or when processing a single case.
    """

    def __init__(self, arg_list: list):
        """
        Initialize the single-process mapper.

        Args:
            arg_list: List of task configuration dictionaries
        """
        self.arg_list = arg_list

    def run(self):
        """
        Run evaluation for all tasks sequentially.

        Returns:
            List of evaluation results
        """
        results = []
        for task_args in self.arg_list:
            result = eval_one_benchmark_data(
                task_args['benchmark_data'],
                task_args['answer_data'],
                task_args['target_dir'],
                task_args['config'],
            )
            results.append(result)
        return results


def export_results(results: list, task_dir: str) -> pd.DataFrame:
    """
    Export evaluation results to files.

    Separates successful results from errors, saves errors to JSONL file,
    and exports flat evaluation results to CSV.

    Args:
        results: List of evaluation results (BenchmarkData objects or error dicts)
        task_dir: Directory to save exported files

    Returns:
        DataFrame containing flat evaluation results

    Raises:
        ValueError: If result type is unknown
    """
    result_benchmark_datas = []
    error_list = []

    # Separate successful results from errors
    for result in results:
        if isinstance(result, dict):
            error_list.append(result)
        elif isinstance(result, BenchmarkData):
            result_benchmark_datas.append(result)
        else:
            raise ValueError(f'Unknown result type: {type(result)}')

    # Report and save errors
    print(f'Error tasks: {len(error_list)}')
    for error in error_list:
        print(error)
    with open(os.path.join(task_dir, 'error.jsonl'), 'w') as f:
        for error in error_list:
            f.write(json.dumps(error, ensure_ascii=False) + '\n')

    # Export successful results to CSV
    res_benchmark_dataset = BenchmarkDataset.from_results(result_benchmark_datas)
    flat_eval_df = res_benchmark_dataset.export_flat_eval_result()
    flat_csv_path = os.path.join(task_dir, 'flat_eval_result.csv')
    flat_eval_df.to_csv(flat_csv_path, index=False)
    return flat_eval_df


def reduce_results(flat_eval_df: pd.DataFrame, task_dir: str):
    """
    Calculate and save mean evaluation results.

    Computes the mean value for each numeric metric in the evaluation results
    and saves the summary to a JSON file.

    Args:
        flat_eval_df: DataFrame containing flat evaluation results
        task_dir: Directory to save the mean results file
    """
    mean_dict = {}
    all_mean_dict = {}

    # Calculate mean for each metric column
    for metric in flat_eval_df.columns:
        try:
            all_mean_dict[metric] = flat_eval_df[metric].mean()
        except TypeError:
            # Skip non-numeric columns
            pass

    mean_dict['all'] = all_mean_dict

    # Save mean results to JSON file
    with open(os.path.join(task_dir, 'mean_eval_result.json'), 'w') as f:
        json.dump(mean_dict, f, ensure_ascii=False, indent=2)


def main():
    """
    Main entry point for the evaluation script.

    Parses command line arguments, sets up evaluation tasks, runs evaluation
    (either single-process or multi-process), and exports results.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate benchmark dataset with answer data support'
    )
    parser.add_argument(
        '--bench',
        type=str,
        required=True,
        help='Path to the benchmark dataset'
    )
    parser.add_argument(
        '--task_dir',
        type=str,
        required=True,
        help='Directory to store task results and intermediate files'
    )
    parser.add_argument(
        '--step',
        type=str,
        default=None,
        help='Step mode: "1" for only_step1, "2" for only_step2 (requires --answer=None)'
    )
    parser.add_argument(
        '--answer',
        type=str,
        default=None,
        help='Path to answer file (JSONL format). If None, use step mode'
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=None,
        help='Number of CPU processes to use (default: cpu_count - 1)'
    )
    parser.add_argument(
        '--key',
        type=str,
        default=None,
        help='Process only a specific case by its track_id (for debugging)'
    )
    parser.add_argument(
        '--force_update',
        action='store_true',
        help='Force re-evaluation of all cases, ignoring existing results'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Use ground truth labels as predictions when predict_label is None'
    )
    args = parser.parse_args()

    # Prepare task directory structure
    os.makedirs(args.task_dir, exist_ok=True)
    target_cases_dir = os.path.join(args.task_dir, 'cases')
    os.makedirs(target_cases_dir, exist_ok=True)

    # Build benchmark and answer datasets
    benchmark_dataset, answer_dataset = build_dataset(
        args.bench, args.answer, args.step
    )

    # Build task map from datasets
    task_map = build_task_map(benchmark_dataset, answer_dataset, args)

    # Run evaluation using appropriate mapper
    if args.key:
        # Use single-process mapper for debugging single cases
        mapper = SingleProcessMaper(task_map.values())
        results = mapper.run()
    else:
        # Use multi-process mapper for parallel evaluation
        cpu_num = args.cpus if args.cpus else os.cpu_count() - 1
        mapper = MultiProcessAsyncMaper(task_map.values(), cpu_num)
        results = mapper.run()

    # Skip result export if only Step 1 was requested
    if args.step == '1':
        return

    # Collect and validate results
    result_benchmark_datas = []
    for result in results:
        if isinstance(result, dict) or isinstance(result, BenchmarkData):
            result_benchmark_datas.append(result)
        else:
            raise ValueError(f'Unknown result type: {type(result)}')

    # Export results to files and generate summary statistics
    flat_eval_df = export_results(result_benchmark_datas, args.task_dir)
    reduce_results(flat_eval_df, args.task_dir)


if __name__ == '__main__':
    main()
