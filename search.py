import os
import re
import math
import pickle
import argparse
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from bsbi import tokenize
from indexers import BSBIIndex, SPIMIIndex, SPIMIPatriciaIndex
from compression import VBEPostings, EliasGammaPostings, StandardPostings

def generate_snippet(query, doc_path, window_size=30):
    query_terms = set(tokenize(query))
    if not query_terms:
        return ""
        
    try:
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except:
        return ""
        
    words = text.split()
    if not words:
        return ""
        
    best_window = []
    max_score = -1
    
    for i in range(max(1, len(words) - window_size + 1)):
        window = words[i:i + window_size]
        score = sum(1 for w in window if any(qt in w.lower() for qt in query_terms))
        if score > max_score:
            max_score = score
            best_window = window
            
    snippet = " ".join(best_window)
    
    for qt in query_terms:
        snippet = re.sub(f"(?i)({qt}[a-z]*)", r"\033[1;33m\1\033[0m", snippet)
        
    return f"... {snippet} ..."

class AdaptiveRetriever:
    def __init__(self, lexical_index, output_dir='index_bonus'):
        if not HAS_FAISS:
            raise Exception("FAISS is not installed! Cannot initialize AdaptiveRetriever.")

        self.lexical_index = lexical_index

        try:
            self.faiss_idx = faiss.read_index(f"{output_dir}/lsi.faiss")        
            with open(f"{output_dir}/lsi_params.pkl", 'rb') as f:
                params = pickle.load(f)
                self.term_vectors = params['term_vectors']
        except Exception as e:
            raise Exception(f"Failed loading FAISS from {output_dir}! Did you run build_lsi_faiss.py? {e}")

        self.num_docs = len(self.lexical_index.doc_id_map)
        
    def query_to_lsi_vector(self, query):
        q_tokens = set(tokenize(query))
        q_vec = np.zeros(self.term_vectors.shape[1], dtype=np.float32)
        valid_terms = 0
        
        for token in q_tokens:
            if token in self.lexical_index.term_id_map:
                tid = self.lexical_index.term_id_map[token]
                q_vec += self.term_vectors[tid]
                valid_terms += 1
                
        if valid_terms > 0:
            q_vec /= np.linalg.norm(q_vec) + 1e-10
            
        return np.expand_dims(q_vec, 0)
        
    def retrieve_adaptive(self, query, k=10, faiss_weight=0.6):
        wand_results = self.lexical_index.retrieve_bm25_wand(query, k=100)
        
        q_vec = self.query_to_lsi_vector(query)
        LSI_topk = min(self.num_docs, 100)
        D, I = self.faiss_idx.search(q_vec, LSI_topk)
        
        lex_scores = {doc: score for score, doc in wand_results}
        sem_scores = {}
        for score, doc_id in zip(D[0], I[0]):
            if doc_id >= 0 and doc_id < self.num_docs:
                docname = self.lexical_index.doc_id_map[int(doc_id)]
                sem_scores[docname] = score
                
        max_lex = max(lex_scores.values()) if lex_scores else 0
        min_lex = min(lex_scores.values()) if lex_scores else 0
        lex_range = max(max_lex - min_lex, 1e-5)
        
        max_sem = max(sem_scores.values()) if sem_scores else 0
        min_sem = min(sem_scores.values()) if sem_scores else 0
        sem_range = max(max_sem - min_sem, 1e-5)
            
        q_tokens = tokenize(query)
        if len(q_tokens) <= 1:
            faiss_weight = 0.2
            
        combined = []
        candidates = set(lex_scores.keys()).union(set(sem_scores.keys()))
        for doc in candidates:
            norm_lex = (lex_scores.get(doc, min_lex) - min_lex) / lex_range if lex_range > 0 and lex_scores else 0
            norm_sem = (sem_scores.get(doc, min_sem) - min_sem) / sem_range if sem_range > 0 and sem_scores else 0
            
            final_score = (1 - faiss_weight) * norm_lex + faiss_weight * norm_sem
            combined.append((final_score, doc))
            
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined[:k]

def main():
    parser = argparse.ArgumentParser(description="Search Engine CLI")
    parser.add_argument("query", type=str, nargs='?', default=None, help="The query string to search for")
    parser.add_argument("--index", type=str, choices=["bsbi", "spimi"], default="bsbi", help="Indexing algorithm to use")
    parser.add_argument("--dictionary", type=str, choices=["standard", "trie", "patricia"], default="standard", help="Dictionary structure (trie/patricia only for spimi)")
    parser.add_argument("--compression", type=str, choices=["standard", "vbe", "elias"], default="vbe", help="Compression model")
    parser.add_argument("--metric", type=str, choices=["tfidf", "bm25", "wand", "adaptive", "all"], default="all", help="Retrieval metric to use")
    parser.add_argument("--build", action="store_true", help="Build the index before searching")
    parser.add_argument("--snippets", action="store_true", help="Show Google-like context snippets")
    
    args = parser.parse_args()
    
    # Map compression
    comp_map = {
        'standard': StandardPostings,
        'vbe': VBEPostings,
        'elias': EliasGammaPostings
    }
    compression_class = comp_map[args.compression]
    
    # Map indexing and dictionary
    if args.index == 'bsbi':
        if args.dictionary != 'standard':
            print("Warning: BSBI only uses standard dictionary. Ignoring dictionary arg.")
        index_instance = BSBIIndex(data_dir='collection', postings_encoding=compression_class, output_dir='index')
    else: # spimi
        if args.dictionary == 'patricia':
            index_instance = SPIMIPatriciaIndex(data_dir='collection', postings_encoding=compression_class, output_dir='index_bonus', index_name='spimi_patricia_index')
        else: # trie or standard
            if args.dictionary == 'standard':
                print("Warning: SPIMI defaults to TrieIdMap. Using SPIMIIndex.")
            index_instance = SPIMIIndex(data_dir='collection', postings_encoding=compression_class, output_dir='index_bonus', index_name='spimi_main')

    if args.build:
        print(f"Building {args.index.upper()} index with {args.dictionary} dict and {args.compression} compression...")
        index_instance.index()
        print("Index successfully built.\n")
    else:
        try:
            index_instance.load()
            # Perform a quick dummy access and fetch a posting to verify binary integrity vs currently passed codec
            _ = index_instance.term_id_map
            if len(index_instance.term_id_map) > 0:
                # Issue a dummy retrieval to test if the physical codec decompresses correctly 
                dummy_query = "virus" # common word in corpus
                _ = index_instance.retrieve_tfidf(dummy_query, k=1)
        except Exception:
            print(f"Warning: Index format '{args.index}+{args.compression}' mismatch on disk. Rebuilding automatically...")
            index_instance.index()
            index_instance.load()
            print("Index successfully auto-built.\n")

    # Initialize Adaptive Retriever if selected
    adaptive_retriever = None
    if args.metric in ['adaptive', 'all']:
        out_dir = 'index' if args.index == 'bsbi' else 'index_bonus'
        try:
            adaptive_retriever = AdaptiveRetriever(index_instance, output_dir=out_dir)
        except Exception as e:
            if args.metric == 'adaptive':
                print(e)
                return

    queries = [args.query] if args.query else [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)

        if args.metric in ['tfidf', 'all'] and hasattr(index_instance, 'retrieve_tfidf'):
            print("\n[TF-IDF]")
            for (score, doc) in index_instance.retrieve_tfidf(q, k=5):
                print(f"  {doc:45s} {score:>8.4f}")
                if args.snippets:
                    print(f"    {generate_snippet(q, doc)}")

        if args.metric in ['bm25', 'all'] and hasattr(index_instance, 'retrieve_bm25'):
            print("\n[BM25]")
            for (score, doc) in index_instance.retrieve_bm25(q, k=5):
                print(f"  {doc:45s} {score:>8.4f}")
                if args.snippets:
                    print(f"    {generate_snippet(q, doc)}")

        if args.metric in ['wand', 'all'] and hasattr(index_instance, 'retrieve_bm25_wand'):
            print("\n[BM25 + WAND Top-K]")
            for (score, doc) in index_instance.retrieve_bm25_wand(q, k=5):
                print(f"  {doc:45s} {score:>8.4f}")
                if args.snippets:
                    print(f"    {generate_snippet(q, doc)}")

        if args.metric in ['adaptive', 'all'] and adaptive_retriever:
            print("\n[Adaptive Hybrid FAISS]")
            for (score, doc) in adaptive_retriever.retrieve_adaptive(q, k=5):
                print(f"  {doc:45s} {score:>8.4f}")
                if args.snippets:
                    print(f"    {generate_snippet(q, doc)}")

if __name__ == '__main__':
    main()