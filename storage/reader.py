from .base import InvertedIndex

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
        When an instance of the InvertedIndexReader class is used as an iterator
        in a loop scheme, the special method __next__(...) is responsible for
        returning the next (term, postings_list, tf_list) pair in the inverted index.
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
        """
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[term]
        self.index_file.seek(pos)
        postings_list = self.postings_encoding.decode(self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(self.index_file.read(len_in_bytes_of_tf))
        return (postings_list, tf_list)
