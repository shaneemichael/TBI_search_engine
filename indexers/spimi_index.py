import os
import contextlib
from tqdm import tqdm
from .bsbi_index import BSBIIndex, tokenize
from dictionary import TrieIdMap
from storage import InvertedIndexWriter, InvertedIndexReader

class SPIMIIndex(BSBIIndex):
    """
    Single-Pass In-Memory Indexing (SPIMI) implementation scaling on BSBIIndex.
    Uses Dictionary Trie to dramatically reduce term storage footprint in memory.
    Constructs index sequentially without needing memory-heavy arrays of (term, doc) pairs.
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name="spimi_index"):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        # Override the standard IdMap with the Trie-based dictionary structure
        self.term_id_map = TrieIdMap()
        self.doc_id_map = TrieIdMap()

    def spimi_invert(self, block_dir_relative, index_id):
        """
        Parses exactly one document block directly building `term_dict` and `term_tf` in memory.
        Writes to disk sequentially using an InvertedIndexWriter.

        Mathematical Mechanics (Single-Pass In-Memory Indexing):
          The standard BSBI materializes raw (Term_ID, Doc_ID) pairs into a massive Python list,
          consuming massive amounts of RAM before finally initiating an expensive quicksort.

          SPIMI sidesteps sorting arrays entirely:
          1. Reads the block sequentially token-by-token.
          2. Directly aggregates Document IDs into a Python `set()` mapped by `term_id`. 
             Sets are automatically unique and inherently fast to build.
          3. When the block finishes, the memory dictionary is converted into an alphabetically
             sorted Postings Sequence (O(N log N) where N is just the unique vocab size, NOT
             the total length of the corpus array).
          4. Flushes the highly compressed block sequentially to disk.

        This mechanism drastically increases ingestion speed and exponentially lowers system RAM 
        bottlenecks compared to standard textbook BSBI.
        """
        term_dict = {}
        term_tf = {}
        
        dir = os.path.join(".", self.data_dir, block_dir_relative)
        for filename in next(os.walk(dir))[2]:
            docname = os.path.join(dir, filename)
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                # Add doc_id parsing directly over token iter to skip materializing td_pairs
                doc_id = self.doc_id_map[docname]
                for token in tokenize(f.read()):
                    term_id = self.term_id_map[token]
                    if term_id not in term_dict:
                        term_dict[term_id] = set()
                        term_tf[term_id] = {}
                    term_dict[term_id].add(doc_id)
                    
                    if doc_id not in term_tf[term_id]:
                        term_tf[term_id][doc_id] = 0
                    term_tf[term_id][doc_id] += 1
                    
        # Flush the in-memory block to disk via InvertedIndexWriter
        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
            for term_id in sorted(term_dict.keys()):
                sorted_doc_id = sorted(list(term_dict[term_id]))
                assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
                index.append(term_id, sorted_doc_id, assoc_tf)

    def index(self):
        """
        Orchestrates the entire SPIMI index construction pipeline.
        
        Algorithm Pipeline:
          1. Iterates through all available document blocks identically to standard BSBI.
          2. Generates an intermediate `index_id` for tracking.
          3. Offloads the heavy lifting to `spimi_invert` which builds the compressed hash map natively.
          4. After all intermediate memory indices are safely flushed to disk,
             triggers the standard Multi-way Merge phase to finalize the engine.
          5. Computes BM25 upper-bounds for O(1) WAND dynamic pruning execution.
        """
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            self.spimi_invert(block_dir_relative, index_id)
            
        self.save()
        
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)
                
        self._compute_wand_upper_bounds()
