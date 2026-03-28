from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

# indexing has been completed previously
# BSBIIndex is just an abstraction for the index
BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = [
    "alkylated with radioactive iodoacetate",
    "psychodrama for disturbed children",
    "lipid metabolism in toxemia and normal pregnancy",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    print("\n[TF-IDF]")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=5):
        print(f"  {doc:45s} {score:>8.4f}")

    print("\n[BM25]")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=5):
        print(f"  {doc:45s} {score:>8.4f}")

    print("\n[BM25 + WAND Top-K]")
    for (score, doc) in BSBI_instance.retrieve_bm25_wand(query, k=5):
        print(f"  {doc:45s} {score:>8.4f}")