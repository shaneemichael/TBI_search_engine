import pickle
import os

class InvertedIndex:
    """
    Class that implements how to efficiently scan or read an
    Inverted Index stored in a file; and also provides a
    mechanism to write the Inverted Index to a file (storage) during indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict is the "Dictionary" concept which is part of the
        Inverted Index. This postings_dict is assumed to fit entirely in memory.

        As the name suggests, "Dictionary" is implemented as a Python dictionary
        that maps an (integer) term ID to a 4-tuple:
           1. start_position_in_index_file : (in bytes) position where the
              corresponding postings reside in the file. We can use the "seek"
              operation to reach it.
           2. number_of_postings_in_list : number of docIDs in the postings
              (Document Frequency)
           3. length_in_bytes_of_postings_list : length of the postings list in bytes.
           4. length_in_bytes_of_tf_list : length of the term frequencies list
              from the associated postings list in bytes.

    terms: List[int]
        List of term IDs, to keep track of the order of terms inserted into
        the Inverted Index.

    """
    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Name used to store the files containing the index
        postings_encoding : See compression.py, candidates are StandardPostings,
                        GapBasedPostings, etc.
        directory (str): directory where the index file is located
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # To keep track of the term order inserted into the index
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
                                # This will be useful later for Score normalization by document length
                                # when calculating score with TF-IDF or BM25
        self.max_score_dict = {}  # key: term ID (int), value: BM25 upper-bound score per term
                                  # Used by the WAND algorithm for early termination

    def __enter__(self):
        """
        Load all metadata when entering context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator for the List containing the order of terms entering the
                index during construction. ---> term_iter
            3. doc_length, a Python dictionary containing key = doc id, and
                value = the number of tokens in that document (document length).
                Useful for length normalization when using TF-IDF or BM25
                scoring regime; useful for finding N when calculating IDF,
                where N is the number of documents in the collection.

        Metadata is saved to a file using the "pickle" library.

        You also need to understand the special __enter__(..) method in Python and also the
        Context Manager concept in Python. Please study the following link:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Open index file
        self.index_file = open(self.index_file_path, 'rb+')

        # We load postings dict and terms iterator from metadata file
        with open(self.metadata_file_path, 'rb') as f:
            metadata = pickle.load(f)
            if len(metadata) == 4:
                self.postings_dict, self.terms, self.doc_length, self.max_score_dict = metadata
            else:
                # Backward compatibility with older index files
                self.postings_dict, self.terms, self.doc_length = metadata
                self.max_score_dict = {}
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Close index_file and save postings_dict and terms when exiting context"""
        # Close index file
        self.index_file.close()

        # Save metadata (postings dict and terms) to metadata file using pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length, self.max_score_dict], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class that implements how to efficiently scan or read an
    Inverted Index stored in a file.
    """
    def __iter__(self):
        return self

    def reset(self):
        """
        Reset file pointer to the beginning, and reset term iterator pointer
        to the beginning.
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__() # reset term iterator

    def __next__(self):
        """
        Class InvertedIndexReader is also iterable (has an iterator).
        Please study:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        When an instance of the InvertedIndexReader class is used as an iterator
        in a loop scheme, the special method __next__(...) is responsible for
        returning the next (term, postings_list, tf_list) pair in the inverted index.

        ATTENTION! This method must return a small chunk of data from the
        large index file. Why only a small part? So it fits correctly in memory.
        DO NOT LOAD THE ENTIRE INDEX TO MEMORY!
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[curr_term]
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Return a postings list (list of docIDs) and its associated list
        of term frequencies for a term (stored as a tuple (postings_list, tf_list)).

        ATTENTION! This method must not iterate over the entire index from
        start to finish. This method must jump directly to a specific byte
        position in the file (index file) where the postings list (and TF list)
        of a term is stored.
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class that implements how to efficiently write an
    Inverted Index stored in a file.
    """
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list, max_score=None):
        """
        Appends a term, postings_list, and the associated TF list 
        to the end of the index file.

        This method does 4 things:
        1. Encode postings_list using self.postings_encoding (encode method),
        2. Encode tf_list using self.postings_encoding (encode_tf method),
        3. Store metadata in self.terms, self.postings_dict, and self.doc_length.
           Remember that self.postings_dict maps a termID to a
           4-tuple: - start_position_in_index_file
                    - number_of_postings_in_list
                    - length_in_bytes_of_postings_list
                    - length_in_bytes_of_tf_list
        4. Append the bytestream of the encoded postings_list and the encoded
           tf_list to the end position of the index file on disk.

        Don't forget to update self.terms and self.doc_length too!

        SEARCH ON YOUR FAVORITE SEARCH ENGINE:
        - You might want to read about Python I/O
          https://docs.python.org/3/tutorial/inputoutput.html
          In this link we can also learn how to append information to the end of a file.
        - Several file object methods that might be useful such as seek(...) and tell()

        Parameters
        ----------
        term:
            term or termID which is the unique identifier of a term
        postings_list: List[Int]
            List of docIDs where the term appears
        tf_list: List[Int]
            List of term frequencies
        """
        self.terms.append(term) # update self.terms

        # update self.doc_length
        for i in range(len(postings_list)):
            doc_id, freq = postings_list[i], tf_list[i]
            if doc_id not in self.doc_length:
                self.doc_length[doc_id] = 0
            self.doc_length[doc_id] += freq

        # update max_score_dict for WAND
        if max_score is not None:
            self.max_score_dict[term] = max_score

        self.index_file.seek(0, os.SEEK_END)
        curr_position_in_byte = self.index_file.tell()
        compressed_postings = self.postings_encoding.encode(postings_list)
        compressed_tf_list = self.postings_encoding.encode_tf(tf_list)
        self.index_file.write(compressed_postings)
        self.index_file.write(compressed_tf_list)
        self.postings_dict[term] = (curr_position_in_byte, len(postings_list), \
                                    len(compressed_postings), len(compressed_tf_list))


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1,2], "terms array is incorrect"
        assert index.doc_length == {2:2, 3:38, 4:25, 5:56, 8:3, 10:30}, "doc_length is incorrect"
        assert index.postings_dict == {1: (0, \
                                           5, \
                                           len(VBEPostings.encode([2,3,4,8,10])), \
                                           len(VBEPostings.encode_tf([2,4,2,3,30]))),
                                       2: (len(VBEPostings.encode([2,3,4,8,10])) + len(VBEPostings.encode_tf([2,4,2,3,30])), \
                                           3, \
                                           len(VBEPostings.encode([3,4,5])), \
                                           len(VBEPostings.encode_tf([34,23,56])))}, "postings dictionary is incorrect"
        
        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(len(VBEPostings.encode([3,4,5])))) == [3,4,5], "there is an error"
        assert VBEPostings.decode_tf(index.index_file.read(len(VBEPostings.encode_tf([34,23,56])))) == [34,23,56], "there is an error"
