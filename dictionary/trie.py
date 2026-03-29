from .base import IdMap

class TrieNode:
    def __init__(self):
        self.children = {}
        self.id = -1

class TrieIdMap(IdMap):
    """
    A dictionary mapping using a Trie to save space with common prefixes.
    Functions similarly to IdMap but uses a Trie internally for string -> id lookup.
    """
    def __init__(self):
        super().__init__()
        self.root = TrieNode()
        # Bind str_to_id to self so that checks like `word in map.str_to_id` hit `__contains__`
        self.str_to_id = self

    def __get_id(self, s):
        node = self.root
        for char in s:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
        if node.id == -1:
            self.id_to_str.append(s)
            node.id = len(self.id_to_str) - 1
            
        return node.id

    def __contains__(self, s):
        if not isinstance(s, str):
            return False
        node = self.root
        for char in s:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.id != -1

    def __getitem__(self, key):
        if type(key) is int:
            return self._IdMap__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError
