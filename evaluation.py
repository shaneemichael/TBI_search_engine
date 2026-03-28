import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> IR Metrics

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


######## >>>>> Load qrels

def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """
    Loads query relevance judgments (qrels) in a dictionary of dictionary format.
    qrels[query_id][document_id] = 1 if relevant, 0 if not.
    """
    qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


######## >>>>> Evaluation

def eval(qrels, query_file="queries.txt", k=1000):
    """
    Evaluate the search engine with all metrics (RBP, DCG, NDCG, AP)
    against 30 queries, for TF-IDF and BM25 methods.
    """
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    with open(query_file) as file:
        queries = [line.strip().split() for line in file]

    def _get_ranking(retrieve_fn, query, qid):
        ranking = []
        for (score, doc) in retrieve_fn(query, k=k):
            # Extract numeric doc id from paths like ./collection/7/625.txt
            m = re.search(r'[\\/](\d+)\.txt$', doc)
            if m:
                did = int(m.group(1))
                ranking.append(qrels[qid].get(did, 0))
        return ranking

    methods = [
        ("TF-IDF",    BSBI_instance.retrieve_tfidf),
        ("BM25",      BSBI_instance.retrieve_bm25),
        ("BM25-WAND", BSBI_instance.retrieve_bm25_wand),
    ]

    for method_name, retrieve_fn in methods:
        rbp_scores, dcg_scores, ndcg_scores, ap_scores = [], [], [], []
        for parts in queries:
            qid = parts[0]
            query = " ".join(parts[1:])
            ranking = _get_ranking(retrieve_fn, query, qid)
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(ap(ranking))

        n = len(queries)
        print(f"\nHasil evaluasi [{method_name}] terhadap {n} queries")
        print(f"  Mean RBP  = {sum(rbp_scores)  / n:.4f}")
        print(f"  Mean DCG  = {sum(dcg_scores)  / n:.4f}")
        print(f"  Mean NDCG = {sum(ndcg_scores) / n:.4f}")
        print(f"  Mean AP   = {sum(ap_scores)   / n:.4f}")


if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels is incorrect"
    assert qrels["Q1"][300] == 0, "qrels is incorrect"

    # Verify metric implementations with simple examples
    sample = [1, 0, 1, 1, 0]
    print("Sample ranking:", sample)
    print(f"  RBP  = {rbp(sample):.4f}")
    print(f"  DCG  = {dcg(sample):.4f}")
    print(f"  NDCG = {ndcg(sample):.4f}")
    print(f"  AP   = {ap(sample):.4f}")
    print()

    eval(qrels)