from .dcg import dcg

def ndcg(ranking, p=None):
    """
    Normalized Discounted Cumulative Gain (NDCG).

    NDCG@p = DCG@p / IDCG@p
    where IDCG@p is the ideal DCG (best possible ordering).

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.
    p : int or None
        Cutoff rank. If None, use the full ranking.

    Returns
    -------
    float
        NDCG score in [0, 1].
    """
    if p is not None:
        ranking = ranking[:p]
    actual_dcg = dcg(ranking)
    # Ideal ranking: all relevant docs first
    ideal_ranking = sorted(ranking, reverse=True)
    ideal_dcg = dcg(ideal_ranking)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg
