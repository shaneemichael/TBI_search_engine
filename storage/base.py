import pickle
import os

class InvertedIndex:
    """
    Class that implements how to efficiently scan or read an
    Inverted Index stored in a file; and also provides a
    mechanism to write the Inverted Index to a file (storage) during indexing.
    """
    def __init__(self, index_name, postings_encoding, directory=''):
        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # To keep track of the term order inserted into the index
        self.doc_length = {}    # key: doc ID (int), value: document length (number of tokens)
        self.max_score_dict = {}  # key: term ID (int), value: BM25 upper-bound score per term

    def __enter__(self):
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
        # Close index file
        self.index_file.close()

        # Save metadata (postings dict and terms) to metadata file using pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length, self.max_score_dict], f)
