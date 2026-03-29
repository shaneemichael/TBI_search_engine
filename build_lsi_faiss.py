import os
import faiss
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from indexers import SPIMIPatriciaIndex
from index import InvertedIndexReader
from compression import EliasGammaPostings
def build():
    """
    Constructs a Dense Semantic Vector Engine over the Indexed Document Corpus.

    This builder mathematically crushes the sparse, high-dimensional exact-match keyword space
    down into a condensed 'Latent Semantic' continuous vector space using SVD.
    
    Algorithm Pipeline:
      1. Bootstraps the `SPIMIPatriciaIndex` to read the entirely constructed inverted indices.
      2. Materializes the Term-Document Matrix efficiently using SciPy Sparse Matrices (CSC format)
         incorporating the BM25/TF-IDF baseline frequencies to respect standard scaling.
      3. Executes `scipy.sparse.linalg.svds`: Truncated Singular Value Decomposition,
         retaining exactly `k=100` geometric dimensions. This factorizes the matrix finding
         "hidden" correlations between terms (e.g., words like `car` and `automobile` cluster together).
      4. Projects Document Vectors and Term Vectors.
      5. Loads Document Vectors directly into the `faiss.IndexFlatIP` (Facebook AI Similarity Search)
         to achieve lightning-speed L2/Cosine retrieval in production.
      6. Serializes the geometry bounds (`lsi_params.pkl`) allowing queries to map into
         the specific semantic space on the fly.
    """
    print("="*60)
    print("🚀 LATENT SEMANTIC INDEXING (FAISS) GENERATOR")
    print("="*60)
    
    print("Loading SPIMIPatriciaIndex...")
    if not os.path.exists('index_bonus'):
        os.makedirs('index_bonus')
    index = SPIMIPatriciaIndex('collection', 'index_bonus', EliasGammaPostings, index_name="spimi_patricia_index")
    if not os.path.exists('index_bonus/spimi_patricia_index.index'):
        print("Must run SPIMIPatriciaIndex first. Building Index...")
        index.index()
    else:
        index.load()

    num_terms = len(index.term_id_map)
    num_docs = len(index.doc_id_map)
    print(f"Index loaded. Terms: {num_terms}, Docs: {num_docs}")

    if num_docs == 0 or num_terms == 0:
        print("Empty index!")
        return

    # Build Sparse Matrix (Rows: Terms, Cols: Docs)
    print("\nBuilding Sparse Term-Document Matrix...")
    rows, cols, data = [], [], []
    
    with InvertedIndexReader(index.index_name, index.postings_encoding, directory=index.output_dir) as reader:
        for term_id, (pos, df, _, _) in reader.postings_dict.items():
            postings, tf_list = reader.get_postings_list(term_id)
            idf = np.log((num_docs) / (df + 1))
            for doc_id, tf in zip(postings, tf_list):
                weight = (1 + np.log(tf)) * idf
                rows.append(term_id)
                cols.append(doc_id)
                data.append(weight)

    sparse_matrix = sp.csc_matrix((data, (rows, cols)), shape=(num_terms, num_docs))
    
    print("\nComputing Truncated SVD (k=100)...")
    k = min(100, num_terms - 1, num_docs - 1)
    if k <= 0:
        print("Corpus too small for SVD!")
        return
        
    U, Sigma, VT = svds(sparse_matrix, k=k)
    
    # Project vectors. Query mapping requires q^T * U * S^-1
    term_vectors = U.astype(np.float32) * (1.0 / Sigma).astype(np.float32)
    
    # Document vectors mapped to latent space
    doc_vectors = VT.T.astype(np.float32) * Sigma.astype(np.float32)
    
    # L2 normalize doc vectors so Cosine Similarity == Inner Product (FlatIP)
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    doc_vectors_norm = doc_vectors / norms
    
    print("\nBuilding FAISS Index...")
    faiss_index = faiss.IndexFlatIP(k)
    faiss_index.add(doc_vectors_norm)
    
    print("Saving Models to disk...")
    with open('index_bonus/lsi_params.pkl', 'wb') as f:
        pickle.dump({'term_vectors': term_vectors}, f)
        
    faiss.write_index(faiss_index, "index_bonus/lsi.faiss")
    print("Done!")

if __name__ == '__main__':
    build()
