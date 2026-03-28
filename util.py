class IdMap:
    """
    Recall from the lecture that practically, a document and a term will be represented 
    as an integer. Therefore, we need to maintain a mapping between string terms (or
    documents) to their corresponding integers, and vice versa. This IdMap class will handle that.
    """

    def __init__(self):
        """
        Mapping from string (term or document name) to id is stored in a 
        Python dictionary; quite efficient. Reverse mapping is stored in a Python list.

        example:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Returns the number of terms (or documents) stored in the IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Returns the string associated with index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Returns the integer id i corresponding to a string s.
        If s is not in the IdMap, it assigns a new integer id and returns that new id.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        __getitem__(...) is a special method in Python that allows a 
        collection class (like this IdMap) to have an access or modification 
        mechanism with the syntax [..] like in Python lists and dictionaries.

        Please search for this information on your favorite Web search engine. I found the following link:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        If the key is an integer, use __get_str;
        if the key is a string, use __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Merges two lists of tuples (doc id, tf) and returns the merged result 
    (TF needs to be accumulated for all tuples with the same doc id), with the following rules:

    example: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Two sorted lists of tuples to be merged.

    Returns
    -------
    List[(Comparablem, int)]
        The sorted merged list
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id is wrong"
    assert term_id_map[1] == "semua", "term_id is wrong"
    assert term_id_map[0] == "halo", "term_id is wrong"
    assert term_id_map["selamat"] == 2, "term_id is wrong"
    assert term_id_map["pagi"] == 3, "term_id is wrong"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id is wrong"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs is wrong"
