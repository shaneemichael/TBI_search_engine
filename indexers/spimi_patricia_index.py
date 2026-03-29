from .spimi_index import SPIMIIndex
from dictionary import PatriciaTreeIdMap

class SPIMIPatriciaIndex(SPIMIIndex):
    """
    SPIMI implementation utilizing a Patricia Tree (Radix Trie) for memory compression.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="spimi_patricia_index"):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        # Override the TrieIdMap with the PatriciaTreeIdMap
        self.term_id_map = PatriciaTreeIdMap()
        self.doc_id_map = PatriciaTreeIdMap()
