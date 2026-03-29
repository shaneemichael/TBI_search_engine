def ap(ranking):
    """
    Average Precision (AP).

    AP = (1 / R) * sum_{k: rel_k=1} Precision@k
    where R is the total number of relevant documents in the ranking,
    and Precision@k = (# relevant docs in top k) / k.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector.

    Returns
    -------
    float
        AP score in [0, 1].
    """
    R = sum(ranking)
    if R == 0:
        return 0.0
    score = 0.0
    num_relevant = 0
    for k, rel in enumerate(ranking, start=1):
        if rel == 1:
            num_relevant += 1
            score += num_relevant / k
    return score / R
