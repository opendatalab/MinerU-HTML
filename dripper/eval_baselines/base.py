"""
Baseline evaluation framework for HTML content extraction.

This module provides data structures, evaluation functions, and processing
mappers for running baseline extractor evaluations on benchmark datasets.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import jieba
import pandas as pd
import ray
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

from dripper.eval_baselines.baselines.imp import (BaseExtractor,
                                                  ExtractorFactory)

jieba.setLogLevel(jieba.logging.INFO)


def calc_rouge_n_score(target_input: str, prediction_input: str, n: int = 5) -> dict:
    """
    Calculate the ROUGE-N score between the target and prediction inputs.

    Args:
        target_input (str): The ground truth text.
        prediction_input (str): The predicted text.
        n (int, optional): The n-gram size. Defaults to 5.

    Returns:
        dict: A dictionary containing the precision, recall, and F1 score.
    """
    target = target_input.strip()
    prediction = prediction_input.strip()

    # When both target and prediction are empty
    # we consider the prediction to be perfect
    if len(target) == 0 and len(prediction) == 0:
        return {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}

    target_tokens_list = [x for x in jieba.lcut(target_input)]
    target_ngrams = _create_ngrams(target_tokens_list, n)

    prediction_tokens_list = [x for x in jieba.lcut(prediction_input)]
    prediction_ngrams = _create_ngrams(prediction_tokens_list, n)

    score = _score_ngrams(target_ngrams, prediction_ngrams)

    # 将scoress转换为rouge-L的precision, recall, f1-score
    result = {'prec': score.precision, 'rec': score.recall, 'f1': score.fmeasure}
    return result


@dataclass
class BaselineData:
    """
    Data structure for a single baseline evaluation case.

    Contains the raw HTML, ground truth content, difficulty level, and URL
    for a benchmark case.
    """

    track_id: str
    html: str
    convert_main_content: str
    level: str
    url: str

    @classmethod
    def from_dict(cls, data: dict) -> 'BaselineData':
        """
        Create BaselineData instance from dictionary.

        Args:
            data: Dictionary containing benchmark data with keys:
                  - 'track_id': Unique identifier for the case
                  - 'html': Raw HTML string
                  - 'convert_main_content': Ground truth main content
                  - 'meta': Dictionary containing 'level' key
                  - 'url': Optional URL string

        Returns:
            BaselineData instance
        """
        return cls(
            track_id=data['track_id'],
            html=data['html'],
            convert_main_content=data['convert_main_content'],
            level=data['meta']['level'],
            url=data.get('url', ''),
        )


def eval_batch_cases(
    baseline_data_list: list[BaselineData],
    cases_dir_str: str,
    extractor: BaseExtractor,
) -> list[dict]:
    """
    Evaluate a batch of baseline cases using an extractor.

    Extracts main HTML and content for each case, calculates ROUGE scores
    against ground truth, and saves results to individual case directories.

    Args:
        baseline_data_list: List of BaselineData instances to evaluate
        cases_dir_str: Root directory path for storing case results
        extractor: BaseExtractor instance to use for extraction

    Returns:
        List of dictionaries containing evaluation results with keys:
        - 'track_id': Case identifier
        - ROUGE score metrics (rouge-1, rouge-2, rouge-l, etc.)
        - 'meta.level': Difficulty level
    """
    result_list = []
    # Prepare input list for batch extraction
    input_list = [
        (baseline_data.html, baseline_data.url)
        for baseline_data in baseline_data_list
    ]
    extract_result_list = extractor.extract_batch(input_list)

    for baseline_data, result in zip(baseline_data_list, extract_result_list):
        track_id = baseline_data.track_id
        input_html = baseline_data.html
        convert_main_content = baseline_data.convert_main_content
        level = baseline_data.level
        main_html, main_content = result

        # Calculate ROUGE scores comparing predicted vs ground truth
        rouge_score = calc_rouge_n_score(convert_main_content, main_content)

        # Create case directory and save results
        case_dir_path = Path(cases_dir_str) / track_id
        os.makedirs(case_dir_path, exist_ok=True)
        (case_dir_path / 'input.html').write_text(
            input_html, encoding='utf-8'
        )
        (case_dir_path / 'gt_main.txt').write_text(
            convert_main_content, encoding='utf-8'
        )
        (case_dir_path / 'pred_main.txt').write_text(
            main_content, encoding='utf-8'
        )
        (case_dir_path / 'rouge_score.json').write_text(
            json.dumps(rouge_score, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        # Save predicted HTML if available
        if main_html:
            (case_dir_path / 'pred_main.html').write_text(
                main_html, encoding='utf-8'
            )
        result_list.append(
            {'track_id': track_id, **rouge_score, 'meta.level': level}
        )
    return result_list


def build_dataset(bench_path_str: str) -> dict[str, BaselineData]:
    """
    Build benchmark dataset from JSONL file.

    Reads a JSONL file where each line is a JSON object representing a
    benchmark case, and creates a dictionary mapping track_id to BaselineData.

    Args:
        bench_path_str: Path to JSONL file containing benchmark data

    Returns:
        Dictionary mapping track_id to BaselineData instances

    Raises:
        ValueError: If the benchmark path is not a file
    """
    bench_path = Path(bench_path_str)
    bench_data_map = {}
    if bench_path.is_file():
        with open(bench_path, 'r') as f:
            for line in f:
                data_dict = json.loads(line)
                data = BaselineData.from_dict(data_dict)
                bench_data_map[data.track_id] = data
    else:
        raise ValueError(f'Benchmark path {bench_path} is not a file')

    return bench_data_map


def export_results(results: list, task_dir: str) -> pd.DataFrame:
    """
    Export evaluation results to files.

    Separates successful results from errors, saves errors to JSONL file,
    and exports successful results to CSV file.

    Args:
        results: List of result dictionaries or error tuples
        task_dir: Directory path to save output files

    Returns:
        DataFrame containing successful evaluation results

    Raises:
        ValueError: If result type is not recognized
    """
    result_benchmark_datas = []
    error_list = []
    for result in results:
        if isinstance(result, tuple):
            # Error case: tuple format
            error_list.append(result)
        elif isinstance(result, dict):
            # Success case: dictionary format
            result_benchmark_datas.append(result)
        else:
            raise ValueError(f'Unknown result type: {type(result)}')

    # Print and save errors
    print(f'Error tasks: {len(error_list)}')
    for error in error_list:
        print(error)
    with open(os.path.join(task_dir, 'error.jsonl'), 'w') as f:
        for error in error_list:
            f.write(json.dumps(error, ensure_ascii=False) + '\n')

    # Export successful results to CSV
    flat_eval_df = pd.DataFrame(result_benchmark_datas)
    flat_csv_path = os.path.join(task_dir, 'flat_eval_result.csv')
    flat_eval_df.to_csv(flat_csv_path, index=False)
    return flat_eval_df


def reduce_results(flat_eval_df: pd.DataFrame, task_dir: str) -> None:
    """
    Calculate and save mean evaluation metrics.

    Computes mean values for all numeric metrics across all cases and
    separately for each difficulty level, then saves to JSON file.

    Args:
        flat_eval_df: DataFrame containing evaluation results
        task_dir: Directory path to save mean results JSON file
    """
    mean_dict = {}
    all_mean_dict = {}

    # Calculate mean for all cases
    for metric in flat_eval_df.columns:
        try:
            all_mean_dict[metric] = flat_eval_df[metric].mean()
        except TypeError:
            # Skip non-numeric columns
            pass
    mean_dict['all'] = all_mean_dict

    # Calculate mean for each difficulty level
    for level in flat_eval_df['meta.level'].unique():
        level_mean_dict = {}
        for metric in flat_eval_df.columns:
            try:
                level_mean_dict[metric] = flat_eval_df[
                    flat_eval_df['meta.level'] == level
                ][metric].mean()
            except TypeError:
                # Skip non-numeric columns
                pass
        mean_dict[level] = level_mean_dict

    # Save mean results to JSON file
    with open(os.path.join(task_dir, 'mean_eval_result.json'), 'w') as f:
        json.dump(mean_dict, f, ensure_ascii=False, indent=2)


class SingleProcessMaper:
    """
    Single-process mapper for baseline evaluation.

    Processes benchmark cases sequentially in a single process without
    parallelization. Suitable for small datasets or debugging.
    """

    def __init__(
        self,
        target_dir: str,
        benchmark_dataset: dict[str, BaselineData],
        extractor_name: str,
        config: dict,
    ):
        """
        Initialize SingleProcessMaper.

        Args:
            target_dir: Directory path to save evaluation results
            benchmark_dataset: Dictionary mapping track_id to BaselineData
            extractor_name: Name of the extractor to use (e.g., 'dripper-md')
            config: Configuration dictionary for the extractor
        """
        self.target_dir = target_dir
        self.benchmark_dataset = benchmark_dataset
        self.extractor_name = extractor_name
        self.config = config

    def run(self) -> list[list[dict]]:
        """
        Run single-process evaluation on all benchmark cases.

        Creates extractor instance and processes each case sequentially,
        saving results to case directories.

        Returns:
            List of result lists (one list per case, each containing dict results)
        """
        extractor = ExtractorFactory.create_extractor(
            self.extractor_name, self.config
        )
        cases_dir_str = os.path.join(self.target_dir, 'cases')
        os.makedirs(cases_dir_str, exist_ok=True)
        results = []
        # Process each case individually
        for baseline_data in self.benchmark_dataset.values():
            result = eval_batch_cases(
                [baseline_data], cases_dir_str, extractor
            )
            results.append(result)
        return results


class RayBatchProcessMaper:
    """
    Ray-based batch process mapper for parallel baseline evaluation.

    Processes benchmark cases in parallel batches using Ray for distributed
    computing. Supports GPU and CPU resource allocation per batch.
    """

    def __init__(
        self,
        target_dir: str,
        benchmark_dataset: dict[str, BaselineData],
        extractor_name: str,
        extractor_config: dict,
        batch_size: int,
        gpu_num: int,
        cpu_num: int,
    ):
        """
        Initialize RayBatchProcessMaper.

        Args:
            target_dir: Directory path to save evaluation results
            benchmark_dataset: Dictionary mapping track_id to BaselineData
            extractor_name: Name of the extractor to use (e.g., 'dripper-md')
            extractor_config: Configuration dictionary for the extractor
            batch_size: Number of cases to process in each batch
            gpu_num: Number of GPUs to allocate per batch task
            cpu_num: Number of CPUs to allocate per batch task
        """
        self.target_dir = target_dir
        self.benchmark_dataset = benchmark_dataset
        self.extractor_name = extractor_name
        self.extractor_config = extractor_config
        self.gpu_num = gpu_num
        self.cpu_num = cpu_num
        self.batch_size = batch_size

    @staticmethod
    @ray.remote
    def eval_batch(
        batch: list[BaselineData],
        cases_dir_str: str,
        extractor_name: str,
        extractor_config: dict,
    ) -> list[dict]:
        """
        Remote function to evaluate a batch of cases (Ray task).

        This method is executed remotely by Ray workers. It creates an extractor
        instance and processes a batch of cases.

        Args:
            batch: List of BaselineData instances to evaluate
            cases_dir_str: Root directory path for storing case results
            extractor_name: Name of the extractor to use
            extractor_config: Configuration dictionary for the extractor

        Returns:
            List of evaluation result dictionaries
        """
        extractor = ExtractorFactory.create_extractor(
            extractor_name, extractor_config
        )
        # Process batch using batch extraction
        results = eval_batch_cases(batch, cases_dir_str, extractor)
        return results

    def run(self) -> list[dict]:
        """
        Run parallel batch evaluation using Ray.

        Splits dataset into batches, submits Ray tasks with resource allocation,
        and collects results as tasks complete.

        Returns:
            List of evaluation result dictionaries from all batches
        """
        cases_dir_str = os.path.join(self.target_dir, 'cases')
        ray.init()

        # Split dataset into batches
        batch_list = [
            list(self.benchmark_dataset.values())[i : i + self.batch_size]
            for i in range(
                0, len(self.benchmark_dataset.values()), self.batch_size
            )
        ]

        # Submit Ray tasks with resource allocation
        tasks = [
            RayBatchProcessMaper.eval_batch.options(
                num_gpus=self.gpu_num, num_cpus=self.cpu_num
            ).remote(
                batch, cases_dir_str, self.extractor_name, self.extractor_config
            )
            for batch in batch_list
        ]

        # Wait for tasks to complete
        unfinished_tasks = tasks
        finished_tasks = []
        print(f'start to process {len(tasks)} batches')
        while len(unfinished_tasks) > 0:
            ready_tasks, unfinished_tasks = ray.wait(
                unfinished_tasks, timeout=5
            )
            finished_tasks.extend(ready_tasks)
            print(
                f'waiting for {len(unfinished_tasks)}/{len(tasks)} batches'
            )

        # Collect results from all finished tasks
        results = []
        for task in finished_tasks:
            results.extend(ray.get(task))
        return results
