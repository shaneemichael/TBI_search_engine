import re
import math
import os
import time

from indexers import BSBIIndex, SPIMIIndex
from compression import VBEPostings, EliasGammaPostings
from search import AdaptiveRetriever
from metrics import rbp, dcg, ndcg, ap

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

def eval(qrels, query_file="queries.txt", k=1000):
    """
    Evaluate the search engine with all metrics (RBP, DCG, NDCG, AP)
    against 30 queries, evaluating ALL possible major architecture combinations:
    1. Base Configs
    2. Advanced Lexical Configs
    3. Dense Semantic FAISS Configs
    """
    with open(query_file) as file:
        queries = [line.strip().split() for line in file]

    methods = []

    # 1. Base Configuration (Mandatory Assignment)
    print("\nLoading Base Index (BSBI + VBEPostings)...")
    bsbi_instance = BSBIIndex(data_dir='collection', postings_encoding=VBEPostings, output_dir='index')
    try:
        bsbi_instance.load()
        if len(bsbi_instance.term_id_map) > 0:
            _ = bsbi_instance.retrieve_tfidf("virus", k=1)
    except Exception:
        print("Warning: Base Index is either missing or encoded with a different algorithm on disk (e.g. Elias vs VBE). Rebuilding base index to baseline automatically...")
        bsbi_instance.index()
        bsbi_instance.load()

    methods.append(("TF-IDF (Base BSBI)", bsbi_instance.retrieve_tfidf))
    methods.append(("BM25 (Base BSBI)", bsbi_instance.retrieve_bm25))
    methods.append(("BM25-WAND (Base BSBI)", bsbi_instance.retrieve_bm25_wand))

    # 2. Advanced Lexical Configuration (SPIMI + Trie + EliasGamma)
    if os.path.exists('index_bonus/spimi_main.index'):
        print("Loading Advanced Lexical Index (SPIMI + Trie + EliasGamma)...")
        spimi_index = SPIMIIndex(data_dir='collection', postings_encoding=EliasGammaPostings, output_dir='index_bonus', index_name='spimi_main')
        try:
            spimi_index.load()
            if len(spimi_index.term_id_map) > 0:
                _ = spimi_index.retrieve_tfidf("virus", k=1)
        except Exception:
            print("Warning: Lexical Bonus Index format mismatch on disk (e.g. Elias vs VBE). Rebuilding spimi_main to baseline automatically...")
            spimi_index.index()
            spimi_index.load()
            
        methods.append(("Lexical WAND (SPIMI Bonus)", lambda q, k: spimi_index.retrieve_bm25_wand(q, k=k)))
        
        print("Loading Dense Semantic FAISS (Adaptive Retriever)...")
        try:
            from indexers import SPIMIPatriciaIndex
            lexical_index = SPIMIPatriciaIndex(data_dir='collection', postings_encoding=EliasGammaPostings, output_dir='index_bonus', index_name='spimi_patricia_index')
            try:
                lexical_index.load()
                if len(lexical_index.term_id_map) > 0:
                    _ = lexical_index.retrieve_tfidf("virus", k=1)
            except Exception:
                print("Warning: LSI base index format mismatch on disk! Rebuilding SPIMI+Patricia+Elias to baseline automatically...")
                lexical_index.index()
                lexical_index.load()
                
            adaptive_retriever = AdaptiveRetriever(lexical_index, output_dir='index_bonus')
            methods.append(("Adaptive Hybrid (FAISS + Patricia)", lambda q, top_k: adaptive_retriever.retrieve_adaptive(q, k=top_k)))
        except Exception as e:
            print("Could not load FAISS module for combination testing:", e)

    print("\n" + "="*100)
    print(f"🚀 {'EXHAUSTIVE COMBINED EVALUATION STARTED':^96} 🚀")
    print("="*100)
    print(f"{'Retrieval Architecture':<38} | {'Time (s)':<10} | {'Mean RBP':<10} | {'Mean DCG':<10} | {'Mean NDCG':<10} | {'Mean AP':<10}")
    print("-" * 100)

    def _get_ranking(retrieve_fn, query, qid):
        ranking = []
        for (score, doc) in retrieve_fn(query, k):
            m = re.search(r'[\\/](\d+)\.txt$', doc)
            if m:
                did = int(m.group(1))
                ranking.append(qrels[qid].get(did, 0))
        return ranking

    for method_name, retrieve_fn in methods:
        rbp_scores, dcg_scores, ndcg_scores, ap_scores = [], [], [], []
        
        start_time = time.time()
        for parts in queries:
            qid = parts[0]
            query = " ".join(parts[1:])
            ranking = _get_ranking(retrieve_fn, query, qid)
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(ap(ranking))
        end_time = time.time()
        
        num_q = len(queries)
        speed = end_time - start_time
        m_rbp = sum(rbp_scores) / num_q
        m_dcg = sum(dcg_scores) / num_q
        m_ndcg = sum(ndcg_scores) / num_q
        m_ap = sum(ap_scores) / num_q
        
        print(f"{method_name:<38} | {speed:<10.3f} | {m_rbp:<10.4f} | {m_dcg:<10.4f} | {m_ndcg:<10.4f} | {m_ap:<10.4f}")
    
    print("="*100 + "\n")

if __name__ == '__main__':
    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels is incorrect"
    assert qrels["Q1"][300] == 0, "qrels is incorrect"

    eval(qrels)