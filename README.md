# TBI - Search Engine From Scratch

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

*Requirement: `pip install -r requirements.txt`* or *`pip install tqdm faiss-cpu numpy`*

### 1. Build the Indices
You can build the base index using the default script or customize the architecture via the unified CLI. It is **required** to build the indices before running your first search.

```bash
# Build the default base index (BSBI + VBE + Standard Dictionary)
python bsbi.py

# Build a custom advanced index (e.g., SPIMI + Patricia Tree + Elias-Gamma)
# Note: Use --build flag to construct the index before a search can be run if you haven't built it.
python search.py --build --index spimi --dictionary patricia --compression elias

# For Bonus: Build Semantic Vectors for Adaptive/FAISS Retrieval
python build_lsi_faiss.py
```

### 2. Search Execution (Unified CLI)
The search interface is highly parameterized. You can specify the exact architecture you want to test.

**CLI Arguments:**
- `query`: The query string to search for. If omitted, runs default queries.
- `--index`: Indexing algorithm (`bsbi` or `spimi`). Default: `bsbi`.
- `--dictionary`: Dictionary structure (`standard`, `trie`, `patricia`). Default: `standard`.
- `--compression`: Compression model (`standard`, `vbe`, `elias`). Default: `vbe`.
- `--metric`: Retrieval metric (`tfidf`, `bm25`, `wand`, `adaptive`, `all`). Default: `all`.
- `--build`: Build the index before searching (Boolean flag).
- `--snippets`: Show Google-like context snippets (Boolean flag).

**Examples to evaluate Specific Features:**

* **Evaluate Base System (TF-IDF vs BM25 vs WAND Top-K)**
  ```bash
  python search.py
  ```

* **Evaluate Task 1 (Elias-Gamma Bit-Level Compression)**
  ```bash
  # Ensure index is built with elias compression
  python search.py "cancer" --build --compression elias
  ```

* **Evaluate Task 2 & 4 (BM25 & WAND Optimization)**
  ```bash
  python search.py "tumor treatment" --metric wand --snippets
  ```

* **Evaluate Bonus Modules (SPIMI, Tries, Adaptive LSI retrieval)**
  ```bash
  # Test SPIMI and Patricia Tree Dictionary
  python search.py "blood pressure" --build --index spimi --dictionary patricia --metric wand --snippets

  # Test FAISS-powered Adaptive LSI Search
  python search.py "viral infection" --index spimi --dictionary patricia --metric adaptive --snippets
  ```

### 3. Quantitative Evaluation (Metrics)
Run the automated evaluation script to grade the engine's performance against the provided `qrels.txt` using **RBP, DCG, NDCG, and AP** metrics (Task 3).

> ⚠️ **Important:** Because `evaluation.py` explicitly loads multiple pre-built architectures into memory simultaneously (Base VBE vs Advanced Elias), you must ensure the disk binaries match its expectations before running it. If you have been playing around with custom `--build` commands, please run the following rebuild sequence to align the bin files first:
>
> ```bash
> # 1. Align Base Index to VBE
> python search.py --build --index bsbi --compression vbe
> # 2. Align Lexical Index to Trie + Elias
> python search.py --build --index spimi --dictionary trie --compression elias
> # 3. Align Semantic Index to Patricia + Elias
> python search.py --build --index spimi --dictionary patricia --compression elias
> ```

```bash
python evaluation.py
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

**Lexical Engine Output** (`python search.py "alkylated with radioactive iodoacetate" --index spimi --dictionary trie --compression elias --metric wand --snippets`)
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

**Dense Semantic Engine Output** (`python search.py "alkylated with radioactive iodoacetate" --index spimi --dictionary patricia --compression elias --metric adaptive --snippets`) 
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

**Base Engine Console Matrix** (`python search.py "alkylated with radioactive iodoacetate"`)
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