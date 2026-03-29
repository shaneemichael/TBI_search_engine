import os
import re
import math
import argparse
import pickle
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from bsbi import tokenize
from indexers import SPIMIIndex, SPIMIPatriciaIndex
from compression import EliasGammaPostings

def generate_snippet(query, doc_path, window_size=30):
    """
    Dynamically generates a Google-like search snippet by scanning the original document text.

    This function locates the densest cluster of query terms within the document to
    provide the user with maximum contextual relevance. It mimics commercial search engines
    (like Google or Bing) by highlighting the matched terms directly using ANSI color codes.

    Algorithm:
      1. Tokenize the input query into unique terms (stems/lemmas).
      2. Ingest the raw document text and tokenize it into a linear array of words.
      3. Use a sliding window protocol (default width: 30 words) to scan the document.
      4. At each step, score the window by counting how many words contain target stem prefixes.
      5. Extract the window with the absolute highest density score.
      6. Use Regular Expressions to surgically inject terminal ANSI formatting codes around matched prefixes.
      
    Returns:
        str: A formatted excerpt combining the raw text context + highlighted keywords.
    """
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
    def __init__(self, data_dir='collection', output_dir='index_bonus', index_name="spimi_patricia_index"):
        if not HAS_FAISS:
            print("FAISS is not installed! Cannot initialize AdaptiveRetriever.")
            exit(1)

        print("Loading SPIMIPatriciaIndex for Lexical Base...")
        self.lexical_index = SPIMIPatriciaIndex(data_dir, output_dir, EliasGammaPostings, index_name=index_name)
        self.lexical_index.load()
        
        print("Loading FAISS Semantic Index & SVD LSI Projections...")
        try:
            self.faiss_idx = faiss.read_index(f"{output_dir}/lsi.faiss")
            with open(f"{output_dir}/lsi_params.pkl", 'rb') as f:
                params = pickle.load(f)
                self.term_vectors = params['term_vectors']
        except Exception as e:
            print("Failed loading FAISS! Did you run build_lsi_faiss.py?", e)
            exit(1)
            
        self.num_docs = len(self.lexical_index.doc_id_map)
        
    def query_to_lsi_vector(self, query):
        """
        Maps a freetext string query directly into the K-dimensional semantic latent space.

        Since our FAISS index is built on a Truncated Singular Value Decomposition (TruncatedSVD)
        of the Term-Document Matrix, queries must be geometrically projected into this exactly
        same compressed LSI mathematical dimension.

        Algorithm:
          1. Lexically tokenize the query into its base stems.
          2. Fetch the pre-computed LSI mathematical vector for each term from `term_vectors`.
          3. Spherically aggregate (Sum) the component term vectors to form a master Query Vector.
          4. L2-Normalize the final vector so it safely targets pure cosine-similarity within FAISS.
          
        Returns:
            np.ndarray: A 1xK dense dimensional embedding representing the semantic intent.
        """
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
        
    def retrieve_adaptive(self, query, top_k=10, faiss_weight=0.6):
        """
        PyTerrier-style Cascade Adaptive Hybrid Retrieval Algorithm.

        This is the most mathematically advanced function in the engine. It combines two
        completely distinct search modalities: Exact Lexical matching and Dense Semantic similarity.
        It then utilizes Min-Max Normalization to harmonize their widely differing scales.

        The Cascading Architecture:
          1. **Base Ranker (Lexical WAND Stage):**
             Instantly retrieves a highly competent candidate pool (Top 100) using the strict WAND
             early-termination logic via BM25 calculation. This provides strong keyword guarantees.
          2. **Reranker (Dense FAISS Stage):**
             Executes a semantic similarity scan converting the query to dense LSI projections,
             retrieving documents that are conceptually similar even if keyword matching utterly fails.
          3. **MinMax Combinatorics Route:**
             Normalizes raw BM25 scores (range ~[0, 20]) and raw FAISS Cosine L2 scores (range ~[-1, +1]) 
             independently into absolute [0, 1] thresholds.
          4. **Adaptive Weighting Check:**
             If the query is extremely short (e.g. 1 exact word), lexical intent implies exact-match.
             The function drops the FAISS weight to 0.2 to prioritize deterministic matching.
             If the query is long, FAISS retains its powerful ~0.6 weighting scale to capture semantics.
             
        Returns:
            list(tuple): A list containing the (Hybrid_Score, Doc_ID) pairs sorted descendingly.
        """
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
            norm_lex = (lex_scores.get(doc, min_lex) - min_lex) / lex_range if lex_scores else 0
            norm_sem = (sem_scores.get(doc, min_sem) - min_sem) / sem_range if sem_scores else 0
            
            final_score = (1 - faiss_weight) * norm_lex + faiss_weight * norm_sem
            combined.append((final_score, doc))
            
        combined.sort(key=lambda x: x[0], reverse=True)
        return combined[:top_k]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Search Engine (Bonus)")
    parser.add_argument("query", type=str, nargs='?', default=None, help="The query string to search for")
    parser.add_argument("--mode", type=str, default="adaptive", choices=["lexical", "adaptive"], 
                        help="Retrieval mode: 'lexical' (WAND/SPIMI) or 'adaptive' (WAND + FAISS SVD)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()

    # Pre-defined queries for demo if standard query is empty
    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy"
    ]

    print("\n" + "="*80)
    print(f"🚀 ADVANCED SEARCH ENGINE DEMO (BONUS) | MODE: \033[1;35m{args.mode.upper()}\033[0m")
    if args.mode == 'lexical':
        print("Features: SPIMI Indexing, Trie Dictionary, Elias-Gamma, WAND, Snippets")
    else:
        print("Features: SPIMI Patricia Tree, WAND Lexical, LSI FAISS Semantics, Adaptive Routing, Snippets")
    print("="*80)

    if args.mode == 'lexical':
        # Pure Lexical Mode via WAND
        print("Initializing Lexical SPIMI Indexer with TrieIdMap and EliasGammaPostings...")
        index_instance = SPIMIIndex(data_dir='collection', postings_encoding=EliasGammaPostings,
                                    output_dir='index_bonus', index_name='spimi_main')
                                    
        if not os.path.exists('index_bonus/spimi_main.index'):
            print("Building SPIMI Index (Single-Pass In-Memory) from scratch...")
            if not os.path.exists('index_bonus'):
                os.makedirs('index_bonus')
            index_instance.index()
        else:
            print("Loading existing Lexical SPIMI index...")
            index_instance.load()

        search_function = lambda q, k: index_instance.retrieve_bm25_wand(q, k=k)

    else:
        # Adaptive Multi-stage Hybrid via WAND + FAISS SVD
        retriever = AdaptiveRetriever(data_dir='collection', output_dir='index_bonus', index_name='spimi_patricia_index')
        search_function = lambda q, k: retriever.retrieve_adaptive(q, top_k=k)

    # Execute Searches
    targets = [args.query] if args.query else queries

    for q in targets:
        print(f"\n\033[1;36mQuery:\033[0m {q}")
        print('-'*80)

        results = search_function(q, args.top_k)
        
        if not results:
            print("  No documents found.")
            continue
            
        for rank, (score, doc) in enumerate(results, 1):
            if args.mode == 'adaptive':
                print(f"  \033[1m{rank}. {doc}\033[0m  (Adaptive Score: {score:.4f})")
            else:
                print(f"  \033[1m{rank}. {doc}\033[0m  (WAND Lexical Score: {score:.4f})")
                
            snippet = generate_snippet(q, doc)
            print(f"     {snippet}\n")
