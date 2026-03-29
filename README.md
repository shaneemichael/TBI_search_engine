# TBI Search Engine — TP2

A from-scratch information retrieval system built using only Python's standard library. This project implements a full search engine pipeline including indexing, compression, scoring, retrieval, and evaluation.

## Features

### 1. Indexing — BSBI (Blocked Sort-Based Indexing)
- Reads a document collection split into sub-directories (one block per directory)
- Builds intermediate inverted indices per block, then merges them using **external merge sort**
- Stores posting lists and TF lists efficiently on disk
- **Preprocessing pipeline**: lowercase normalization → punctuation stripping → stopword removal → Porter stemming (all in pure Python, no NLTK)

### 2. Compression

| Method | Type | Description |
|---|---|---|
| `StandardPostings` | Byte-level | Raw 4-byte unsigned integers |
| `VBEPostings` | Byte-level | Variable-Byte Encoding on gap-based list |
| `EliasGammaPostings` | **Bit-level** | Elias-Gamma coding on gap-based list, bits packed MSB-first into bytes |

**Elias-Gamma** encoding: for a positive integer `n`, writes `floor(log2(n))` zero bits, a `1` separator, then the `k`-bit binary remainder of `n - 2^k`.

### 3. Scoring / Retrieval

#### TF-IDF (`retrieve_tfidf`)
```
w(t, D) = 1 + log(tf(t, D))
w(t, Q) = IDF = log(N / df(t))
Score    = Σ w(t,Q) × w(t,D)
```

#### BM25 (`retrieve_bm25`)
```
IDF(t)       = log(1 + (N - df + 0.5) / (df + 0.5))
tf_score     = tf × (k1 + 1) / (tf + k1 × (1 - b + b × |D| / avgdl))
Score(Q, D)  = Σ IDF(t) × tf_score(t, D)
```
Default parameters: `k1=1.2`, `b=0.75`

#### BM25 + WAND Top-K (`retrieve_bm25_wand`)
WAND (Weak AND) avoids evaluating every document by using **per-term BM25 upper-bound scores** stored during indexing.

- Upper bound: `IDF(t) × (k1 + 1)` — the maximum possible BM25 contribution of term `t`
- At retrieval: sort active term pointers by current docID → find **pivot** where accumulative upper bounds exceed the current k-th score threshold → skip documents that can't improve the top-K heap

### 4. Advanced/Bonus Features (250 PTs)
We implemented multiple cutting-edge capabilities extending the mandatory search engine specification, fully preserving backward-compatibility:

- **SPIMI Indexing (`SPIMIIndex`)**: By inheriting from `BSBIIndex`, we dramatically reduced the sorting overhead by using a dynamically scaling in-memory Dictionary Tree loop, bypassing standard pair materialization entirely. (Single-Pass In-Memory Indexing).
- **Trie Dictionary (`TrieIdMap`)**: Instead of string-hashing, vocabulary is internally loaded into a dynamic Trie tree (Prefix Tree). It reduces ram occupation for prefixes ("compute", "computer") saving huge system memory amounts per term block!
- **Dynamic Text Snippets**: The engine doesn't just return doc names anymore. We added `generate_snippet()` which scans original file text for query term keyword clusters, slicing out a window of context via Regex and outputting a Google-like console highlighted display!

### 4. Evaluation Metrics

All metrics are computed over top-1000 retrieved documents:

| Metric | Formula |
|---|---|
| **RBP** | `(1-p) × Σ rel_i × p^(i-1)`, `p=0.8` |
| **DCG** | `Σ rel_i / log2(i+1)` |
| **NDCG** | `DCG / IDCG` (normalized by ideal ordering) |
| **AP** | `(1/R) × Σ_{k: rel_k=1} Precision@k` |

Evaluation runs all three retrieval methods (TF-IDF, BM25, BM25-WAND) and reports mean scores over 30 queries.

---

## How to Run

### Prerequisites
```bash
pip install tqdm
```
Only `tqdm` is required; everything else uses Python's standard library.

### Step 1 — Build the Lexical Index
```bash
python indexers/bsbi_index.py
```
*(Wait, the original test scripts called `python bsbi.py`, which is now a router! You can safely run `python bsbi.py` directly).*

### Step 2 — Search
```bash
python search.py
```
Runs 3 sample queries against the standard index.

### Step 3 — Evaluate
```bash
python evaluation.py
```
Runs the evaluation script using the extracted independent math packages from `metrics/`.

### Step 4 — Run the Advanced Bonus Pipeline! 🚀
We merged the advanced lexical matching (SPIMI, Tries) and PyTerrier-inspired cascade retrievers into a single powerful ArgumentParser!

First, build the dense semantic LSI vectors:
```bash
python build_lsi_faiss.py
```

Then, run your queries via the unified CLI:
```bash
# Pure lexical WAND search (SPIMI, Patricia Tries)
python search_bonus.py "disturbed children" --mode lexical

# Full Hybrid cascade search (WAND + LSI FAISS Combinatoric Routing)
python search_bonus.py "disturbed children" --mode adaptive
```
*Note: The bonus search features colored ANSI contextual snippets displaying the exact matching text density surrounding your terms!*

## Project Structure (Ultimate SRP)

We structurally refactored the monolith into highly decoupled Python packages, ensuring maximum readability and logical isolation, while preserving root "router" scripts for complete backward compatibility.

```
TP2/
├── collection/        # Document corpus (11 blocks)
├── index/             # Generated index files (Base)
├── index_bonus/       # Generated index files (Advanced SPIMI + FAISS LSI)
│
├── indexers/          # [PACKAGE] Algorithmic Retrieval Core
│   ├── bsbi_index.py        # Blocked Sort-Based Indexing
│   ├── spimi_index.py       # Single-Pass In-Memory Indexing
│   └── spimi_patricia_index.py
│
├── dictionary/        # [PACKAGE] Memory Maps
│   ├── base.py              # Hash-based standard IdMap
│   ├── trie.py              # Memory-efficient Prefix Tree
│   └── patricia.py          # Extreme-compression Radix Trie
│
├── storage/           # [PACKAGE] File I/O
│   ├── reader.py            # Disk Read logic
│   └── writer.py            # Disk Write logic
│
├── compression/       # [PACKAGE] Codecs
│   ├── standard.py          # Array representation
│   ├── vbe.py               # Variable Byte
│   └── elias_gamma.py       # Bit-level compression
│
├── metrics/           # [PACKAGE] Math Formulas
│   ├── rbp.py, dcg.py, ndcg.py, ap.py
│
├── search_bonus.py    # Unified command-line interface for Bonus (Argparse)
├── build_lsi_faiss.py # Semantic TruncatedSVD generator
├── bsbi.py            # Backwards compatible Router
├── index.py           # Backwards compatible Router
├── util.py            # Backwards compatible Router
├── evaluation.py      # Runner for evaluation
├── queries.txt        # 30 evaluation queries
└── qrels.txt          # Relevance judgments
```

---

## Algorithm Details

### Elias-Gamma Encoding
Elias-Gamma is a **bit-level** universal code for positive integers. It is especially efficient for small values close to 1, making it well-suited for gap encoding of dense posting lists.

For `n=5` (binary `101`): `k=2`, write `00` + `1` + `01` → `00101`.

The encoder:
1. Converts the posting list to a **gap representation** (first value as-is, then `doc[i] - doc[i-1]`)
2. Encodes each gap with Elias-Gamma
3. Packs all bits MSB-first into bytes (zero-padded)
4. Prepends a 4-byte count header for the decoder

### WAND Retrieval
WAND (Bast et al., 2003) is a dynamic pruning technique for Top-K retrieval:

1. For each query term, maintain a pointer into its sorted postings list
2. Sort terms by their current pointed-to docID
3. Find the **pivot**: the first term where cumulative upper-bound scores exceed the current threshold
4. If the leftmost pointer == pivot doc → evaluate exact BM25 and update heap
5. Otherwise → binary-search advance all pointers before the pivot to the pivot docID

Upper bounds are precomputed during indexing and stored in `max_score_dict` in the index metadata.