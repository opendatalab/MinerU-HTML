"""
Benchmark data structures and utilities for evaluation.

This module defines data classes and dataset loaders for benchmark evaluation,
including intermediate data, evaluation results, and answer data structures.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dripper.base import CLASS_ATTR, SELECT_ATTR
from dripper.process.html_utils import element_to_html, html_to_element


def clean_select_attr(html_str: str) -> str:
    """
    Remove SELECT_ATTR from all elements in HTML.

    Args:
        html_str: HTML string to clean

    Returns:
        HTML string with SELECT_ATTR removed from all elements
    """
    root = html_to_element(html_str)
    for element in root.iter():
        # Remove SELECT_ATTR attribute
        element.attrib.pop(SELECT_ATTR, None)
    return element_to_html(root)


def clean_class_attr(html_str: str) -> str:
    """
    Remove CLASS_ATTR from class attributes in HTML.

    Removes the CLASS_ATTR value from class attributes, handling cases where
    it appears as the only class or as part of a space-separated class list.

    Args:
        html_str: HTML string to clean

    Returns:
        HTML string with CLASS_ATTR removed from class attributes
    """
    root = html_to_element(html_str)
    for element in root.iter():
        # Remove CLASS_ATTR if it is in "class" attribute
        class_str = element.attrib.get('class', '')
        if class_str.strip() == CLASS_ATTR:
            # Remove entire class attribute if it only contains CLASS_ATTR
            element.attrib.pop('class', None)
        elif f' {CLASS_ATTR}' in class_str:
            # Remove CLASS_ATTR from space-separated class list
            element.attrib['class'] = class_str.replace(
                f' {CLASS_ATTR}', ''
            ).strip()
        else:
            pass
    return element_to_html(root)


@dataclass
class IntermediateData:
    """
    Intermediate data structure for benchmark evaluation.

    Contains ground truth data, processed HTML, predictions, and semi-ground
    truth data used during the evaluation pipeline.
    """

    gt_main_html: str = None  # Ground truth main HTML
    gt_main_content: str = None  # Ground truth main content (text)
    simpled_html: str = None  # Simplified HTML
    map_html: str = None  # Mapped HTML with item identifiers
    gt_item_label: dict = None  # Ground truth item labels
    pred_item_label: dict = None  # Predicted item labels
    pred_main_html: str = None  # Predicted main HTML
    pred_main_content: str = None  # Predicted main content (text)
    semi_gt_main_html: str = None  # Semi-ground truth main HTML
    semi_gt_main_content: str = None  # Semi-ground truth main content (text)

    @classmethod
    def from_dict(cls, data: dict) -> 'IntermediateData':
        """
        Create IntermediateData from a dictionary.

        Args:
            data: Dictionary containing intermediate data fields

        Returns:
            IntermediateData instance
        """
        return cls(
            gt_main_html=data.get('gt_main_html', None),
            gt_main_content=data.get('gt_main_content', None),
            simpled_html=data.get('simpled_html', None),
            map_html=data.get('map_html', None),
            gt_item_label=data.get('gt_item_label', None),
            pred_item_label=data.get('pred_item_label', None),
            pred_main_html=data.get('pred_main_html', None),
            pred_main_content=data.get('pred_main_content', None),
            semi_gt_main_html=data.get('semi_gt_main_html', None),
            semi_gt_main_content=data.get('semi_gt_main_content', None),
        )

    @staticmethod
    def dump_file_map() -> dict[str, str]:
        """
        Get mapping of field names to file names for serialization.

        Returns:
            Dictionary mapping field names to file names with extensions
        """
        return {
            'gt_main_html': 'gt_main_html.html',
            'gt_main_content': 'gt_main_content.txt',
            'simpled_html': 'simpled_html.html',
            'map_html': 'map_html.html',
            'gt_item_label': 'gt_item_label.json',
            'pred_item_label': 'pred_item_label.json',
            'pred_main_html': 'pred_main_html.html',
            'pred_main_content': 'pred_main_content.txt',
            'semi_gt_main_html': 'semi_gt_main_html.html',
            'semi_gt_main_content': 'semi_gt_main_content.txt',
        }

    def to_dict(self) -> dict:
        """
        Convert IntermediateData to a dictionary.

        Returns:
            Dictionary representation of the intermediate data
        """
        return {
            'gt_main_html': self.gt_main_html,
            'gt_main_content': self.gt_main_content,
            'simpled_html': self.simpled_html,
            'map_html': self.map_html,
            'gt_item_label': self.gt_item_label,
            'pred_item_label': self.pred_item_label,
            'pred_main_html': self.pred_main_html,
            'pred_main_content': self.pred_main_content,
            'semi_gt_main_html': self.semi_gt_main_html,
            'semi_gt_main_content': self.semi_gt_main_content,
        }


@dataclass
class EvalResult:
    """
    Evaluation result data structure.

    Contains various evaluation metrics including item-level scores,
    ROUGE scores, and LLM-based ROUGE scores.
    """

    item_score: dict = None  # Item-level evaluation scores
    semi_rouge_score: dict = None  # ROUGE scores against semi-ground truth
    rouge_score: dict = None  # ROUGE scores against ground truth
    LLM_rouge_score: dict = None  # LLM-based ROUGE scores

    @classmethod
    def from_dict(cls, data: dict) -> 'EvalResult':
        """
        Create EvalResult from a dictionary.

        Args:
            data: Dictionary containing evaluation result fields

        Returns:
            EvalResult instance
        """
        return cls(
            item_score=data.get('item_score', None),
            semi_rouge_score=data.get('semi_rouge_score', None),
            rouge_score=data.get('rouge_score', None),
            LLM_rouge_score=data.get('LLM_rouge_score', None),
        )

    def to_dict(self) -> dict:
        """
        Convert EvalResult to a dictionary.

        Returns:
            Dictionary representation of the evaluation result
        """
        return {
            'item_score': self.item_score,
            'semi_rouge_score': self.semi_rouge_score,
            'rouge_score': self.rouge_score,
            'LLM_rouge_score': self.LLM_rouge_score,
        }

    def to_flat_dict(self) -> dict:
        """
        Convert nested dictionary to flat dictionary with dot-separated keys.

        Recursively flattens nested dictionaries by joining keys with dots.
        For example, {'a': {'b': 1}} becomes {'a.b': 1}.

        Returns:
            Flat dictionary with dot-separated keys
        """
        raw_dict = self.to_dict()
        flat_dict = {}

        def walk_dict_recursive(dict_to_walk: dict, key_list: list[str] = []):
            """Recursively walk dictionary and flatten nested structures."""
            for key, value in dict_to_walk.items():
                if isinstance(value, dict):
                    walk_dict_recursive(value, key_list + [key])
                else:
                    flat_dict['.'.join(key_list + [key])] = value

        walk_dict_recursive(raw_dict)
        return flat_dict


@dataclass
class BenchmarkData:
    """
    Complete benchmark data structure for a single case.

    Contains all information needed for evaluation including raw HTML,
    intermediate processing data, and evaluation results.
    """

    track_id: str  # Unique identifier for this benchmark case
    html: str  # Raw HTML content
    url: str  # URL where the HTML was obtained from
    meta: dict  # Metadata dictionary
    intermediate_data: IntermediateData  # Intermediate processing data
    eval_result: EvalResult  # Evaluation results

    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkData':
        """
        Create BenchmarkData from a dictionary.

        Args:
            data: Dictionary containing benchmark data fields

        Returns:
            BenchmarkData instance
        """
        return cls(
            track_id=data['track_id'],
            html=data['html'],
            url=data['url'],
            meta=data.get('meta', {}),
            intermediate_data=IntermediateData.from_dict(
                data.get('intermediate_data', {})
            ),
            eval_result=EvalResult.from_dict(data.get('eval_result', {})),
        )

    def to_dict(self) -> dict:
        """
        Convert BenchmarkData to a dictionary.

        Returns:
            Dictionary representation of the benchmark data
        """
        return {
            'track_id': self.track_id,
            'html': self.html,
            'url': self.url,
            'meta': self.meta,
            'intermediate_data': self.intermediate_data.to_dict(),
            'eval_result': self.eval_result.to_dict(),
        }

    def dump_to_dir(self, dir_path: str):
        """
        Dump benchmark data to a directory.

        Saves all data fields to files in the specified directory,
        using appropriate file extensions based on data type.

        Args:
            dir_path: Directory path to save files to
        """
        os.makedirs(dir_path, exist_ok=True)

        def dump_auto(item_to_export, target_name: str):
            """
            Automatically serialize data to file based on type.

            Args:
                item_to_export: Data to export
                target_name: Target file name

            Raises:
                ValueError: If data type is not supported
            """
            if target_name.endswith('json'):
                str_to_export = json.dumps(
                    item_to_export, ensure_ascii=False, indent=2
                )
            elif isinstance(item_to_export, str):
                str_to_export = item_to_export
            else:
                raise ValueError(
                    f'Unsupported type: {type(item_to_export)}, '
                    f'when dump to {target_name}'
                )

            with open(os.path.join(dir_path, target_name), 'w') as f:
                f.write(str_to_export)

        # Dump basic fields
        dump_auto(self.track_id, 'track_id.txt')
        dump_auto(self.html, 'html.html')
        dump_auto(self.url, 'url.txt')
        dump_auto(self.meta, 'meta.json')

        # Dump intermediate data fields
        for name, target_file_name in IntermediateData.dump_file_map().items():
            item_to_dump = getattr(self.intermediate_data, name)
            if item_to_dump is not None:
                dump_auto(item_to_dump, target_file_name)

        # Dump evaluation results
        dump_auto(self.eval_result.to_dict(), 'eval_result.json')

        os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def load_from_dir(cls, dir_path: str):
        """
        Load benchmark data from a directory.

        Reads all data files from the specified directory and reconstructs
        the BenchmarkData object.

        Args:
            dir_path: Directory path containing the data files

        Returns:
            BenchmarkData instance loaded from directory
        """
        def load_auto(target_name: str):
            """
            Automatically load data from file based on extension.

            Args:
                target_name: File name to load

            Returns:
                Loaded data (dict for JSON, str for text), or None if file doesn't exist
            """
            file_path = os.path.join(dir_path, target_name)
            if not os.path.exists(file_path):
                return None
            with open(file_path, 'r') as f:
                file_content = f.read()
                if target_name.endswith('json'):
                    return json.loads(file_content.strip()) if file_content else {}
                else:
                    return file_content

        # Load basic fields
        track_id = load_auto('track_id.txt')
        html = load_auto('html.html')
        url = load_auto('url.txt')
        meta = load_auto('meta.json')

        # Load evaluation results
        eval_result_dict = load_auto('eval_result.json')
        if eval_result_dict is None:
            eval_result = EvalResult()
        else:
            eval_result = EvalResult.from_dict(eval_result_dict)

        # Load intermediate data
        intermediate_data_dict = {
            key: load_auto(target_file_name)
            for key, target_file_name in IntermediateData.dump_file_map().items()
        }
        intermediate_data = IntermediateData.from_dict(intermediate_data_dict)

        return cls(
            track_id=track_id,
            html=html,
            url=url,
            meta=meta,
            intermediate_data=intermediate_data,
            eval_result=eval_result,
        )


class BenchmarkDataset:
    """
    Dataset container for benchmark evaluation data.

    Loads and manages multiple BenchmarkData instances from a JSONL file
    or from a list of BenchmarkData objects.
    """

    def __init__(self, data_path: Optional[str], config: dict = {}):
        """
        Initialize BenchmarkDataset.

        Args:
            data_path: Path to JSONL file containing benchmark data, or None for empty dataset
            config: Configuration dictionary (e.g., {'patch_files': 'auto'})

        Raises:
            FileNotFoundError: If data_path does not exist
            ValueError: If data_path is not a file or has unsupported format
        """
        if data_path is None:
            self.data_map = {}
            return

        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Data path {data_path} does not exist')
        self.config = config
        self.data_map: dict[str, BenchmarkData] = {}
        self.load_data()

    def load_data(self):
        """
        Load benchmark data from file.

        Reads JSONL file and creates BenchmarkData instances for each line.
        Cleans class attributes from HTML during loading.

        Raises:
            ValueError: If file type is unsupported or path is not a file
        """
        if os.path.isfile(self.data_path):
            if self.data_path.endswith('.jsonl'):
                with open(self.data_path, 'r') as f:
                    for line in f:
                        data_dict = json.loads(line)
                        track_id = data_dict['track_id']
                        benchmark_data = BenchmarkData.from_dict(data_dict)
                        # Clean class attributes from HTML
                        benchmark_data.html = clean_class_attr(
                            benchmark_data.html
                        )
                        self.data_map[track_id] = benchmark_data
            else:
                raise ValueError(
                    f'Unsupported file type: {self.data_path}'
                )
        else:
            raise ValueError(
                f'Data path {self.data_path} is not a file'
            )

    @classmethod
    def from_results(
        cls, benchmark_datas: list[BenchmarkData]
    ) -> 'BenchmarkDataset':
        """
        Create BenchmarkDataset from a list of BenchmarkData objects.

        Args:
            benchmark_datas: List of BenchmarkData instances

        Returns:
            BenchmarkDataset instance containing the provided data
        """
        dataset = cls(None, config={})
        for benchmark_data in benchmark_datas:
            dataset.data_map[benchmark_data.track_id] = benchmark_data
        return dataset

    def export_flat_eval_result(self):
        """
        Export evaluation results as a flat DataFrame.

        Converts nested evaluation results to a flat structure and creates
        a pandas DataFrame with one row per benchmark case.

        Returns:
            DataFrame with flattened evaluation results, indexed by track_id
        """
        flat_eval_result_dict = {}
        for bench_data in self.data_map.values():
            eval_result = {'track_id': bench_data.track_id}

            # Add flattened evaluation metrics
            eval_result.update(bench_data.eval_result.to_flat_dict())
            flat_eval_result_dict[bench_data.track_id] = eval_result

        # Convert to DataFrame (transpose so track_id becomes index)
        flat_eval_df = pd.DataFrame(flat_eval_result_dict).T

        return flat_eval_df


@dataclass
class AnswerData:
    """
    Answer data structure for evaluation.

    Contains predicted labels for a benchmark case, used for evaluation
    against ground truth labels.
    """

    track_id: str  # Unique identifier for this benchmark case
    predict_label: dict  # Dictionary mapping item IDs to predicted labels

    @classmethod
    def from_dict(cls, data: dict) -> 'AnswerData':
        """
        Create AnswerData from a dictionary.

        Args:
            data: Dictionary containing 'track_id' and 'predict_label'

        Returns:
            AnswerData instance
        """
        return cls(
            track_id=data['track_id'],
            predict_label=data['predict_label'],
        )

    def to_dict(self) -> dict:
        """
        Convert AnswerData to a dictionary.

        Returns:
            Dictionary representation of the answer data
        """
        return {
            'track_id': self.track_id,
            'predict_label': self.predict_label,
        }


class AnswerDataset:
    """
    Dataset container for answer data.

    Loads and manages multiple AnswerData instances from a JSONL file.
    """

    def __init__(self, data_path: str):
        """
        Initialize AnswerDataset.

        Args:
            data_path: Path to JSONL file containing answer data

        Raises:
            FileNotFoundError: If data_path does not exist
            ValueError: If data_path is not a file
        """
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Data path {data_path} does not exist')
        self.data_map: dict[str, AnswerData] = {}
        self.load_data()

    def load_data(self):
        """
        Load answer data from file.

        Reads JSONL file and creates AnswerData instances for each line.

        Raises:
            ValueError: If data_path is not a file
        """
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r') as f:
                for line in f:
                    data_dict = json.loads(line)
                    track_id = data_dict['track_id']
                    self.data_map[track_id] = AnswerData.from_dict(data_dict)
        else:
            raise ValueError(f'Data path {self.data_path} is not a file')
