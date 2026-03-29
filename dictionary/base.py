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

        If the key is an integer, use __get_str;
        if the key is a string, use __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError
