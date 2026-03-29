import os
import re
import pickle
import contextlib
import heapq
import time
import math

from storage import InvertedIndexReader, InvertedIndexWriter
from dictionary import IdMap
from util import sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Minimal English stopword list (no external libraries)
# ---------------------------------------------------------------------------
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'as', 'if', 'not', 'no', 'nor',
    'so', 'yet', 'both', 'either', 'neither', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just',
    'about', 'above', 'after', 'all', 'also', 'am', 'any', 'because',
    'before', 'between', 'during', 'he', 'her', 'him', 'his', 'how',
    'i', 'into', 'me', 'my', 'our', 'out', 'own', 'same', 'she',
    'their', 'them', 'then', 'there', 'they', 'through', 'under', 'until',
    'up', 'us', 'we', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'you', 'your',
}

# ---------------------------------------------------------------------------
# Minimal Porter Stemmer (rule-based, standard library only)
# ---------------------------------------------------------------------------
_VOWELS = set('aeiou')

def _has_vowel(stem):
    return any(c in _VOWELS for c in stem)

def _ends_cvc(word):
    """True if word ends with consonant-vowel-consonant, and last consonant is not w/x/y."""
    if len(word) < 3:
        return False
    c1, v, c2 = word[-3], word[-2], word[-1]
    return (c1 not in _VOWELS and v in _VOWELS and
            c2 not in _VOWELS and c2 not in 'wxy')

def porter_stem(word):
    """
    Simplified Porter stemmer covering the most impactful rules.
    Operates on lowercase words.
    """
    if len(word) <= 2:
        return word

    # Step 1a
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif word.endswith('ss'):
        pass
    elif word.endswith('s'):
        word = word[:-1]

    # Step 1b
    if word.endswith('eed'):
        if len(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith('ed'):
        stem = word[:-2]
        if _has_vowel(stem):
            word = stem
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                word += 'e'
            elif (len(word) > 1 and word[-1] == word[-2] and word[-1] not in 'lsz'):
                word = word[:-1]
    elif word.endswith('ing'):
        stem = word[:-3]
        if _has_vowel(stem):
            word = stem
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                word += 'e'
            elif (len(word) > 1 and word[-1] == word[-2] and word[-1] not in 'lsz'):
                word = word[:-1]

    # Step 1c
    if word.endswith('y') and _has_vowel(word[:-1]):
        word = word[:-1] + 'i'

    # Step 2 (most common suffixes)
    step2_map = {
        'ational': 'ate', 'tional': 'tion', 'enci': 'ence',
        'anci': 'ance', 'izer': 'ize', 'abli': 'able',
        'alli': 'al',   'entli': 'ent', 'eli': 'e',
        'ousli': 'ous', 'ization': 'ize', 'ation': 'ate',
        'ator': 'ate',  'alism': 'al',   'iveness': 'ive',
        'fulness': 'ful', 'ousness': 'ous', 'aliti': 'al',
        'iviti': 'ive', 'biliti': 'ble',
    }
    for suffix, replacement in step2_map.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) > 0:
                word = stem + replacement
            break

    # Step 3
    step3_map = {
        'icate': 'ic', 'ative': '', 'alize': 'al',
        'iciti': 'ic', 'ical': 'ic', 'ful': '', 'ness': '',
    }
    for suffix, replacement in step3_map.items():
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) > 0:
                word = stem + replacement
            break

    # Step 4 (remove derivational suffixes)
    step4_list = [
        'ement', 'ment', 'ance', 'ence', 'able', 'ible',
        'ant',   'ent',  'ion',  'ism',  'ate',  'iti',
        'ous',   'ive',  'ize',  'al',   'er',   'ic',
    ]
    for suffix in step4_list:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) > 1:
                word = stem
            break

    # Step 5a
    if word.endswith('e'):
        stem = word[:-1]
        if len(stem) > 1:
            word = stem
        elif len(stem) == 1 and not _ends_cvc(stem):
            word = stem

    # Step 5b
    if len(word) > 1 and word.endswith('ll') and len(word) > 2:
        word = word[:-1]

    return word


def tokenize(text):
    """
    Lowercase, strip punctuation/numbers, remove stopwords, apply stemming.
    Returns a list of processed tokens.
    """
    # lowercase & strip non-alpha characters
    tokens = re.findall(r'[a-z]+', text.lower())
    result = []
    for t in tokens:
        if len(t) > 1 and t not in STOPWORDS:
            stemmed = porter_stem(t)
            if len(stemmed) > 1:
                result.append(stemmed)
    return result


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): To map terms to termIDs
    doc_id_map(IdMap): To map relative paths from documents (e.g.,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output index files
    postings_encoding: See compression.py, candidates are StandardPostings,
                    VBEPostings, EliasGammaPostings, dsb.
    index_name(str): Name of the file containing the inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # To store the filenames of all intermediate inverted indices
        self.intermediate_indices = []

    def save(self):
        """Save doc_id_map and term_id_map to output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Load doc_id_map and term_id_map from output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Parse the text file so it becomes a sequence of
        <termID, docID> pairs.

        Preprocessing pipeline:
          1. Lowercase
          2. Strip non-alphabetic characters (tokenize with regex)
          3. Remove stopwords
          4. Apply Porter stemming

        Parameters
        ----------
        block_dir_relative : str
            Relative Path to the directory containing text files for a block.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
        """
        dir = os.path.join(".", self.data_dir, block_dir_relative)
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = os.path.join(dir, filename)
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in tokenize(f.read()):
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Performs inversion of td_pairs (list of <termID, docID> pairs) and
        saves them to the index. BSBI concept with SPIMI strategy
        (hashtable per block).

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
        index: InvertedIndexWriter
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Merge all intermediate inverted indices into
        a single index (External Merge Sort).

        Parameters
        ----------
        indices: List[InvertedIndexReader]
        merged_index: InvertedIndexWriter
        """
        # the following code assumes there is at least 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    # ------------------------------------------------------------------
    # BM25 helper: compute upper-bound score for WAND
    # upper bound = IDF * (k1 + 1)  (achieved when tf -> inf, |D| -> 0)
    # ------------------------------------------------------------------
    @staticmethod
    def _bm25_idf(df, N):
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    @staticmethod
    def _bm25_tf_component(tf, doc_len, avgdl, k1=1.2, b=0.75):
        return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))

    def _compute_wand_upper_bounds(self, k1=1.2):
        """
        After indexing, compute BM25 upper-bound scores per term and store
        them in the merged index's max_score_dict.
        Upper bound = IDF(t) * (k1 + 1)
        """
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return
            for term_id, (pos, df, _, _) in merged_index.postings_dict.items():
                idf = self._bm25_idf(df, N)
                upper_bound = idf * (k1 + 1)
                merged_index.max_score_dict[term_id] = upper_bound

    def retrieve_tfidf(self, query, k = 10):
        """
        Performs Ranked Retrieval with TaaT (Term-at-a-Time) scheme.
        Method will return top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       if tf(t, D) > 0
                = 0                        otherwise

        w(t, Q) = IDF = log (N / df(t))

        Score = for each term in query, accumulate w(t, Q) * w(t, D).
                (no need to normalize with document length)

        Parameters
        ----------
        query: str
            Query tokens separated by space

        Result
        ------
        List[(int, str)]
            List of tuple: first element is similarity score, and the
            second is the document name.
            List of Top-K documents sorted in descending order BASED ON SCORE.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = []
        for word in tokenize(query):
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map[word])

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Ranked Retrieval with TaaT scheme using BM25 scoring.

        BM25 score for a query Q and document D:
            Score(Q, D) = sum_{t in Q} IDF(t) * tf(t,D)*(k1+1) /
                          (tf(t,D) + k1*(1 - b + b*|D|/avgdl))

            IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

        Parameters
        ----------
        query : str
            Query tokens separated by space
        k : int
            Number of top documents returned (default: 10)
        k1 : float
            BM25 term saturation parameter (default: 1.2)
        b : float
            BM25 length normalization parameter (default: 0.75)

        Returns
        -------
        List[(float, str)]
            Top-K documents sorted in descending order based on BM25 score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = []
        for word in tokenize(query):
            if word in self.term_id_map.str_to_id:
                terms.append(self.term_id_map[word])

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = self._bm25_idf(df, N)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for doc_id, tf in zip(postings, tf_list):
                        doc_len = merged_index.doc_length.get(doc_id, avgdl)
                        tf_score = self._bm25_tf_component(tf, doc_len, avgdl, k1, b)
                        scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_score

            docs = [(score, self.doc_id_map[doc_id]) for doc_id, score in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        WAND (Weak AND) Top-K Retrieval with BM25 scoring.

        WAND uses upper-bound scores of each term to do
        early termination and avoids full evaluation of all documents.

        WAND Algorithm:
          1. Initialize posting pointer for each query term.
          2. Sort term based on currently pointed docID (ascending).
          3. Find pivot: accumulate upper bounds from left until passing
             threshold (minimum score in top-K heap).
          4. If all pointers point to the same docID as pivot
             -> calculate exact BM25, update heap.
          5. Advance pointers below pivot to pivot docID, then repeat.

        Parameters
        ----------
        query : str
        k : int
        k1 : float
        b : float

        Returns
        -------
        List[(float, str)]
            Top-K documents sorted in descending order based on BM25 score.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = []
        for word in tokenize(query):
            if word in self.term_id_map.str_to_id:
                term_id = self.term_id_map[word]
                if term_id not in terms:   # deduplicate
                    terms.append(term_id)

        if not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            # Filter to terms that are in the index
            valid_terms = [t for t in terms if t in merged_index.postings_dict]
            if not valid_terms:
                return []

            # Load postings lists and upper bounds into memory
            postings_lists = {}
            tf_lists = {}
            upper_bounds = {}
            for term in valid_terms:
                pl, tfl = merged_index.get_postings_list(term)
                postings_lists[term] = pl
                tf_lists[term] = tfl
                df = merged_index.postings_dict[term][1]
                idf = self._bm25_idf(df, N)
                # Upper bound: IDF * (k1+1) — the max possible tf-component is (k1+1)
                upper_bounds[term] = idf * (k1 + 1)

            # Pointer into each postings list
            pointers = {term: 0 for term in valid_terms}
            top_k_heap = []  # min-heap of (score, doc_id); size <= k
            threshold = 0.0  # minimum score to enter top-K

            def advance_to(term, doc_id):
                """Binary search to advance pointer to first entry >= doc_id."""
                pl = postings_lists[term]
                lo, hi = pointers[term], len(pl)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if pl[mid] < doc_id:
                        lo = mid + 1
                    else:
                        hi = mid
                pointers[term] = lo

            while True:
                # Sort active terms by their current docID
                active = [(postings_lists[t][pointers[t]], t)
                          for t in valid_terms if pointers[t] < len(postings_lists[t])]
                if not active:
                    break
                active.sort()

                # Find pivot: accumulate upper bounds until sum > threshold
                pivot_idx = -1
                acc = 0.0
                for i, (doc_id, term) in enumerate(active):
                    acc += upper_bounds[term]
                    if acc > threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    # Even the full upper bound can't beat threshold; done
                    break

                pivot_doc = active[pivot_idx][0]

                # Check if all pointers up to pivot_idx point to pivot_doc
                if active[0][0] == pivot_doc:
                    # Full evaluation: compute exact BM25 for pivot_doc
                    doc_len = merged_index.doc_length.get(pivot_doc, avgdl)
                    score = 0.0
                    for _, term in active:
                        ptr = pointers[term]
                        if ptr < len(postings_lists[term]) and postings_lists[term][ptr] == pivot_doc:
                            tf = tf_lists[term][ptr]
                            df = merged_index.postings_dict[term][1]
                            idf = self._bm25_idf(df, N)
                            score += idf * self._bm25_tf_component(tf, doc_len, avgdl, k1, b)

                    if len(top_k_heap) < k:
                        heapq.heappush(top_k_heap, (score, pivot_doc))
                    elif score > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (score, pivot_doc))

                    threshold = top_k_heap[0][0] if len(top_k_heap) == k else 0.0

                    # Advance all pointers that were at pivot_doc
                    for _, term in active:
                        ptr = pointers[term]
                        if ptr < len(postings_lists[term]) and postings_lists[term][ptr] == pivot_doc:
                            pointers[term] = ptr + 1
                else:
                    # Advance all pointers before pivot_idx to pivot_doc
                    for i in range(pivot_idx):
                        term = active[i][1]
                        advance_to(term, pivot_doc)

        results = [(score, self.doc_id_map[doc_id]) for score, doc_id in top_k_heap]
        return sorted(results, key=lambda x: x[0], reverse=True)

    def index(self):
        """
        Base indexing code — BSBI (Blocked Sort-Based Indexing).

        Scan all data in collection, parse each block, invert, save
        to intermediate index, then merge all into one final index.
        After merge, calculate BM25 upper-bound scores for WAND.
        """
        # loop for each sub-directory in the collection folder (each block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

        # Compute BM25 upper-bound scores for WAND after merge is complete
        self._compute_wand_upper_bounds()


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() 
