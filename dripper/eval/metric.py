"""
Evaluation metrics for HTML content extraction.

This module provides functions to calculate ROUGE scores and item-level
classification metrics for evaluating HTML content extraction performance.
"""

from typing import Callable

import jieba
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

from dripper.base import TagType

# Set jieba logging level to INFO to reduce verbosity
jieba.setLogLevel(jieba.logging.INFO)


def calc_rouge_n_score(
    target_input: str, prediction_input: str, n: int = 5
) -> dict:
    """
    Calculate the ROUGE-N score between the target and prediction inputs.

    Uses jieba for Chinese tokenization and computes n-gram based ROUGE scores.
    When both inputs are empty, returns perfect scores (1.0).

    Args:
        target_input: The ground truth text
        prediction_input: The predicted text
        n: The n-gram size. Defaults to 5

    Returns:
        Dictionary containing:
        - 'prec': Precision score
        - 'rec': Recall score
        - 'f1': F1 score
    """
    target = target_input.strip()
    prediction = prediction_input.strip()

    # When both target and prediction are empty,
    # we consider the prediction to be perfect
    if len(target) == 0 and len(prediction) == 0:
        return {'prec': 1.0, 'rec': 1.0, 'f1': 1.0}

    # Tokenize using jieba and create n-grams for target
    target_tokens_list = [x for x in jieba.lcut(target_input)]
    target_ngrams = _create_ngrams(target_tokens_list, n)

    # Tokenize using jieba and create n-grams for prediction
    prediction_tokens_list = [x for x in jieba.lcut(prediction_input)]
    prediction_ngrams = _create_ngrams(prediction_tokens_list, n)

    # Calculate n-gram scores
    score = _score_ngrams(target_ngrams, prediction_ngrams)

    # Convert score to ROUGE-L style precision, recall, and F1-score
    result = {
        'prec': score.precision,
        'rec': score.recall,
        'f1': score.fmeasure,
    }
    return result


def calc_item_score(target: dict[str, str], predict: dict[str, str]) -> dict:
    """
    Calculate item-level classification scores between target and prediction.

    Computes accuracy and per-tag-type (main/other) precision, recall, and F1
    scores for item-level classification evaluation.

    Args:
        target: Ground truth item-level labels (item_id -> label)
        predict: Predicted item-level labels (item_id -> label)

    Returns:
        Dictionary containing:
        - 'acc': Overall accuracy
        - 'main': Dictionary with 'prec', 'rec', 'f1' for main tag type
        - 'other': Dictionary with 'prec', 'rec', 'f1' for other tag type
    """
    # If target is empty, return perfect scores for all tag types
    if len(target) == 0:
        return {
            k.value: {'prec': 1.0, 'rec': 1.0, 'f1': 1.0} for k in TagType
        }

    def count_by_condition(
        target_condition: Callable[[str], bool],
        predict_condition: Callable[[str], bool],
    ) -> int:
        """
        Count items that satisfy both target and prediction conditions.

        Args:
            target_condition: Function to filter target items
            predict_condition: Function to filter predicted items

        Returns:
            Number of items satisfying both conditions (intersection)
        """
        target_keys = [k for k, v in target.items() if target_condition(v)]
        predict_keys = [k for k, v in predict.items() if predict_condition(v)]
        return len(set(target_keys) & set(predict_keys))

    result = {}

    # Calculate overall accuracy
    acc = 0
    total = 0
    for target_key in target.keys():
        if (
            target_key in predict.keys()
            and target[target_key] == predict[target_key]
        ):
            acc += 1
        total += 1
    result['acc'] = acc / total

    # Calculate precision, recall, and F1 for each tag type
    for tag_type in TagType:
        tag_str = tag_type.value

        # Count True Positives, False Positives, and False Negatives
        tag_TP = count_by_condition(
            lambda x: x == tag_str, lambda x: x == tag_str
        )
        tag_FP = count_by_condition(
            lambda x: x != tag_str, lambda x: x == tag_str
        )
        tag_FN = count_by_condition(
            lambda x: x == tag_str, lambda x: x != tag_str
        )

        # Calculate precision
        if tag_TP + tag_FP > 0:
            # If tag_TP + tag_FP > 0, calculate precision
            tag_precision = tag_TP / (tag_TP + tag_FP)
        else:
            # If tag_TP + tag_FP == 0, set precision to 1.0 (no false positives)
            tag_precision = 1.0

        # Calculate recall
        if tag_TP + tag_FN > 0:
            # If tag_TP + tag_FN > 0, calculate recall
            tag_recall = tag_TP / (tag_TP + tag_FN)
        else:
            # If tag_TP + tag_FN == 0, set recall to 1.0 (no false negatives)
            tag_recall = 1.0

        # Calculate F1 score
        if tag_precision + tag_recall > 0:
            tag_f1 = (
                2 * tag_precision * tag_recall / (tag_precision + tag_recall)
            )
        else:
            tag_f1 = 0.0

        tag_score = {
            'prec': tag_precision,
            'rec': tag_recall,
            'f1': tag_f1,
        }
        result[tag_type.value] = tag_score
    return result
