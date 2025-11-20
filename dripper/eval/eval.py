"""
Evaluation functions for benchmark data processing.

This module provides two-step evaluation functions:
- Step 1: Preprocess HTML and extract ground truth data
- Step 2: Evaluate predictions against ground truth and calculate metrics
"""

from dripper.eval.benckmark import BenchmarkData
from dripper.eval.metric import calc_item_score, calc_rouge_n_score
from dripper.eval.process import (html_to_text_func,
                                  itemify_id_html_to_item_label, post_process,
                                  pre_process, prune_labeled_html)


def eval_step_1(
    benchmark_data: BenchmarkData, config: dict = {}
) -> BenchmarkData:
    """
    Step 1: Preprocess HTML and extract ground truth data.

    This function performs the first step of evaluation:
    1. Preprocesses the labeled HTML to create simplified and mapped HTML
    2. Extracts ground truth main HTML by pruning labeled elements
    3. Converts ground truth HTML to text content
    4. Extracts ground truth item labels from mapped HTML

    Args:
        benchmark_data: Benchmark data containing raw HTML
        config: Optional configuration dictionary (currently unused)

    Returns:
        BenchmarkData with intermediate_data populated with:
        - simpled_html: Simplified HTML structure
        - map_html: Mapped HTML with item identifiers
        - gt_main_html: Ground truth main HTML
        - gt_main_content: Ground truth main content (text)
        - gt_item_label: Ground truth item labels
    """
    labeled_html = benchmark_data.html

    # Preprocess HTML to create simplified and mapped versions
    simpled_html, map_html = pre_process(labeled_html)

    # Extract ground truth main HTML by removing labeled elements
    gt_main_html = prune_labeled_html(labeled_html)
    # Convert HTML to plain text
    gt_main_content = html_to_text_func(gt_main_html)

    # Extract ground truth item labels from mapped HTML
    gt_item_label = itemify_id_html_to_item_label(map_html)

    # Store all intermediate data
    benchmark_data.intermediate_data.gt_main_html = gt_main_html
    benchmark_data.intermediate_data.gt_main_content = gt_main_content
    benchmark_data.intermediate_data.simpled_html = simpled_html
    benchmark_data.intermediate_data.map_html = map_html
    benchmark_data.intermediate_data.gt_item_label = gt_item_label

    return benchmark_data


def eval_step_2(
    benchmark_data: BenchmarkData, pred_item_label: dict, config: dict = {}
) -> BenchmarkData:
    """
    Step 2: Evaluate predictions and calculate metrics.

    This function performs the second step of evaluation:
    1. Extracts predicted main HTML and content from predicted labels
    2. Extracts semi-ground truth main HTML and content from ground truth labels
    3. Calculates item-level scores comparing predicted vs ground truth labels
    4. Calculates ROUGE scores comparing predicted vs ground truth content
    5. Calculates semi-ROUGE scores comparing semi-ground truth vs ground truth
    6. Calculates LLM ROUGE scores comparing predicted vs semi-ground truth

    Args:
        benchmark_data: Benchmark data with Step 1 intermediate data populated
        pred_item_label: Dictionary mapping item IDs to predicted labels
        config: Optional configuration dictionary (currently unused)

    Returns:
        BenchmarkData with intermediate_data and eval_result populated:
        - pred_item_label: Predicted item labels
        - pred_main_html: Predicted main HTML
        - pred_main_content: Predicted main content (text)
        - semi_gt_main_html: Semi-ground truth main HTML
        - semi_gt_main_content: Semi-ground truth main content (text)
        - item_score: Item-level evaluation scores
        - rouge_score: ROUGE scores (predicted vs ground truth)
        - semi_rouge_score: ROUGE scores (semi-ground truth vs ground truth)
        - LLM_rouge_score: ROUGE scores (predicted vs semi-ground truth)
    """
    # Store predicted item labels
    benchmark_data.intermediate_data.pred_item_label = pred_item_label

    # Extract predicted main HTML from mapped HTML using predicted labels
    pred_main_html = post_process(
        benchmark_data.intermediate_data.map_html,
        benchmark_data.intermediate_data.pred_item_label,
    )
    # Convert predicted HTML to plain text
    pred_main_content = html_to_text_func(pred_main_html)
    benchmark_data.intermediate_data.pred_main_html = pred_main_html
    benchmark_data.intermediate_data.pred_main_content = pred_main_content

    # Extract semi-ground truth main HTML from mapped HTML using ground truth labels
    # This represents what the model should extract if it perfectly follows the labels
    semi_gt_main_html = post_process(
        benchmark_data.intermediate_data.map_html,
        benchmark_data.intermediate_data.gt_item_label,
    )
    # Convert semi-ground truth HTML to plain text
    semi_gt_main_content = html_to_text_func(semi_gt_main_html)
    benchmark_data.intermediate_data.semi_gt_main_html = semi_gt_main_html
    benchmark_data.intermediate_data.semi_gt_main_content = semi_gt_main_content

    # Calculate item-level scores (comparing predicted vs ground truth labels)
    item_score = calc_item_score(
        benchmark_data.intermediate_data.gt_item_label,
        benchmark_data.intermediate_data.pred_item_label,
    )
    benchmark_data.eval_result.item_score = item_score

    # Calculate ROUGE scores (predicted content vs ground truth content)
    rouge_score = calc_rouge_n_score(
        benchmark_data.intermediate_data.gt_main_content,
        benchmark_data.intermediate_data.pred_main_content,
    )
    benchmark_data.eval_result.rouge_score = rouge_score

    # Calculate semi-ROUGE scores (semi-ground truth vs ground truth)
    # This measures how well the label-based extraction performs
    semi_rouge_score = calc_rouge_n_score(
        benchmark_data.intermediate_data.gt_main_content,
        benchmark_data.intermediate_data.semi_gt_main_content,
    )
    benchmark_data.eval_result.semi_rouge_score = semi_rouge_score

    # Calculate LLM ROUGE scores (predicted vs semi-ground truth)
    # This measures how well the model's predictions match the label-based extraction
    LLM_rouge_score = calc_rouge_n_score(
        benchmark_data.intermediate_data.semi_gt_main_content,
        benchmark_data.intermediate_data.pred_main_content,
    )
    benchmark_data.eval_result.LLM_rouge_score = LLM_rouge_score

    return benchmark_data
