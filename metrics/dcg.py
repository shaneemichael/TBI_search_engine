import math

def dcg(ranking, p=None):
    """
    Discounted Cumulative Gain (DCG).

    DCG@p = sum_{i=1}^{p} rel_i / log2(i + 1)

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.
    p : int or None
        Cutoff rank. If None, use the full ranking.

    Returns
    -------
    float
        DCG score.
    """
    if p is not None:
        ranking = ranking[:p]
    score = 0.0
    for i, rel in enumerate(ranking, start=1):
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score
