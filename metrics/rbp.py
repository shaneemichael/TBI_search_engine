def rbp(ranking, p=0.8):
    """
    Rank Biased Precision (RBP) with patience parameter p.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector, e.g. [1, 0, 1, 1, 0].
        ranking[i] = 1 means the document at rank (i+1) is relevant.
    p : float
        Patience parameter (default 0.8).

    Returns
    -------
    float
        RBP score in [0, 1].
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] * (p ** (i - 1))
    return (1 - p) * score
