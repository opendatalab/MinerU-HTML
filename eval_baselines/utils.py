import jieba
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

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

    # Convert scores to ROUGE-L precision, recall, and F1-score
    result = {'prec': score.precision, 'rec': score.recall, 'f1': score.fmeasure}
    return result
