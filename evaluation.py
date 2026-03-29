import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> IR Metrics

from metrics import rbp, dcg, ndcg, ap


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