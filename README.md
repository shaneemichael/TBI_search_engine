# Information Retrieval System

An enterprise-grade, from-scratch search engine built using Python's standard library. The pipeline implements efficient indexing, bit-level compression, heuristic scoring, cascade retrieval arrays, and exhaustive IR evaluation methodologies.

## Core Capabilities

1. **BSBI Indexing**: Blocked Sort-Based Indexing with external merge sort for massive corpus ingestion.
2. **Text Normalization**: Custom tokenizer pipeline (lowercase → punctuation strip → stopword removal → Porter stemming) implemented entirely without external NLP libraries.
3. **Advanced Compression Codecs**: Memory optimization utilizing raw bytes, Variable-Byte Encoding (VBE), and Elias-Gamma bit-level dense packing.
4. **Ranking Models**: Implements `TF-IDF` and parameterized `BM25` (with `k1=1.2`, `b=0.75`).
5. **Dynamic Pruning**: Employs WAND (Weak AND) algorithms leveraging precomputed IDF bounds to dramatically prune Top-K retrieval execution trees.

## Advanced Implementations

The system natively implements the following advanced retrieval structures:

- **SPIMI (Single-Pass In-Memory Indexing)**: Replaces sequential pair materialization with a dynamic hash map, exponentially reducing indexing RAM requirements.
- **Patricia Radix Tries (`PatriciaTreeIdMap`)**: Replaces linear dictionary structures with a deeply compressed Prefix Tree array to minimize vocabulary mapping overhead.
- **Latent Semantic Indexing (LSI + FAISS)**: Utilizes Truncated SVD across sparse CSC Term-Document Matrices to project terms into a continuous semantic factor space, optimized via FAISS L2 similarity.
- **Dynamic Context Snippets**: Features a dynamic regex engine to map terms back to original documents, outputting highlighted context windows (Google-style) directly to the console.

---

## Usage Guide

*Requirement: `pip install tqdm`*

### 1. Build the Indices
```bash
python bsbi.py                   # Builds Base BSBI Index
python build_lsi_faiss.py        # Computes LSI Dense Semantic Vectors
```

### 2. Search Execution
Execute base model tests, or seamlessly pivot to the advanced semantic framework:
```bash
python search.py                 # Evaluates baseline TF-IDF vs BM25 WAND
python search_bonus.py "tumor" --mode lexical   # Uses SPIMI, Patricia Trie, Elias-Gamma
python search_bonus.py "tumor" --mode adaptive  # Triggers Hybrid FAISS routing 
```

### 3. Quantitative Evaluation
```bash
python evaluation.py             # Executes metrics suite (RBP, DCG, NDCG, AP)
```

---

## Architectural Benchmarks

### 1. Evaluation Results (`python evaluation.py`)

Evaluating 30 medical queries across permutations demonstrates massive relevance improvements using the Hybrid Semantic topology (+12% Mean RBP).

```text
Retrieval Architecture                 | Time (s)   | Mean RBP   | Mean DCG   | Mean NDCG  | Mean AP   
----------------------------------------------------------------------------------------------------
TF-IDF (Base BSBI)                     | 0.467      | 0.6467     | 5.7871     | 0.8211     | 0.5637    
BM25 (Base BSBI)                       | 0.443      | 0.6722     | 5.8999     | 0.8307     | 0.5849    
BM25-WAND (Base BSBI)                  | 0.428      | 0.6903     | 5.9715     | 0.8372     | 0.5945    
Lexical WAND (SPIMI Bonus)             | 0.491      | 0.6903     | 5.9715     | 0.8372     | 0.5945    
Adaptive Hybrid (FAISS + Patricia)     | 0.534      | 0.7722     | 6.5673     | 0.8930     | 0.7058    
```

### 2. Live Retrieval Comparisons

**Lexical Engine Output** (`python search_bonus.py "alkylated with radioactive iodoacetate" --mode lexical`)
```text
Features: SPIMI Indexing, Trie Dictionary, Elias-Gamma, WAND, Snippets
Query: alkylated with radioactive iodoacetate
--------------------------------------------------------------------------------
  1. .\collection\6\507.txt  (WAND Lexical Score: 13.1882)
     ... the thiol group has been **alkylated** with **radioactive** iodoacetate in the presence of urea. partial acid hydroly- sis of the **alkylated** protein gives, according to the conditions, mainly 3 **radioactive** ...

  2. .\collection\11\1003.txt  (WAND Lexical Score: 7.9258)
     ... of chlorimine (short-acting **alkylating** agent). experience with five cases. a method is discussed for providing palliative treatment for patients with nonresectable lung cancer by means of administering a short-acting **alkylating** ...

  3. .\collection\6\554.txt  (WAND Lexical Score: 7.4333)
     ... distribution and excretion of **radioactivity** after parenteral administration of **radioactive** polydiethylstilbestrol phosphate to rats and a cow. polydiethylstilbestrol phosphate (psp), a water soluble polyester of phos- phoric acid and diethylstilbestrol, ...
```

**Dense Semantic Engine Output** (`python search_bonus.py "alkylated with radioactive iodoacetate" --mode adaptive`)
```text
Features: SPIMI Patricia Tree, WAND Lexical, LSI FAISS Semantics, Adaptive Routing, Snippets
Query: alkylated with radioactive iodoacetate
--------------------------------------------------------------------------------
  1. .\collection\6\507.txt  (Adaptive Score: 0.9476)
     ... the thiol group has been **alkylated** with **radioactive** iodoacetate in the presence of urea. partial acid hydroly- sis of the **alkylated** protein gives, according to the conditions, mainly 3 **radioactive** ...

  2. .\collection\8\745.txt  (Adaptive Score: 0.6558)
     ... metabolism and excretion of c14-labeled bilirubin in children with biliary atresia and judson g. randolph **radioactive** bilirubin was injected intravenously into 3 children with biliary atresia. the isotope over a ...

  3. .\collection\1\92.txt  (Adaptive Score: 0.6400)
     ... did not enter the area, suggesting a thrombus within the aneurysm . in 5, the aneurysm was detectable on the film and was also seen to be filled with **radioactivity** ...
```

**Base Engine Console Matrix** (`python search.py`)
```text
============================================================
Query: alkylated with radioactive iodoacetate
============================================================

[TF-IDF]
  .\collection\6\507.txt                         23.2421
  .\collection\6\554.txt                         11.6201
  .\collection\11\1003.txt                        8.7171
  .\collection\4\387.txt                          8.2780
  .\collection\4\388.txt                          8.2780

[BM25]
  .\collection\6\507.txt                         19.2556
  .\collection\11\1003.txt                        7.9258
  .\collection\6\554.txt                          7.4333
  .\collection\6\506.txt                          6.8043
  .\collection\8\793.txt                          6.3867

[BM25 + WAND Top-K]
  .\collection\6\507.txt                         19.2556
  .\collection\11\1003.txt                        7.9258
  .\collection\6\554.txt                          7.4333
  .\collection\6\506.txt                          6.8043
  .\collection\8\793.txt                          6.3867
```