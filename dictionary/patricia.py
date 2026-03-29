from .base import IdMap

class PatriciaNode:
    def __init__(self, text=""):
        self.text = text
        self.children = {}
        self.id = -1

class PatriciaTreeIdMap(IdMap):
    """
    A dictionary mapping using a Patricia Tree (Radix Trie) to save space.
    Compresses edges to represent fully intact sub-strings rather than characters.
    """
    def __init__(self):
        super().__init__()
        self.root = PatriciaNode()
        self.str_to_id = self

    def __get_id(self, s):
        """
        Dynamically routes a string through the Radix Trie to resolve its persistent global ID.
        If the string has never been indexed before, the Radix splits tree nodes automatically.

        Mathematical Mechanics (The Patricia Tree Compression Algorithm):
          Unlike standard Prefix Tries (where every character consumes 1 full node object), a Patricia Tree
          collapses unbreakable sequences of characters into a single combined node. 
          When string `s` is traversed:

          1. Traverses down existing shared character sequences.
          2. If it encounters a branching fork (e.g. tree holds "computer" and user inserts "compute"):
             - It forcefully SPLITS the existing node precisely at the common prefix.
             - "compute" becomes the parent, passing the remaining 'r' down to a new independent child constraint.
          3. Reduces overall object allocations by up to ~60-80% compared to standard Tries depending
             on vocabulary overlap density (in English, exceptionally high).
        """
        node = self.root
        remains = s
        
        while True:
            if not remains:
                if node.id == -1:
                    self.id_to_str.append(s)
                    node.id = len(self.id_to_str) - 1
                return node.id
                
            first_char = remains[0]
            if first_char not in node.children:
                new_node = PatriciaNode(text=remains)
                self.id_to_str.append(s)
                new_node.id = len(self.id_to_str) - 1
                node.children[first_char] = new_node
                return new_node.id
                
            child = node.children[first_char]
            
            # Find common prefix length
            common_len = 0
            while common_len < len(remains) and common_len < len(child.text) and remains[common_len] == child.text[common_len]:
                common_len += 1
                
            if common_len == len(child.text):
                # Consume whole child text, move down
                node = child
                remains = remains[common_len:]
            else:
                # Split the child
                split_node = PatriciaNode(text=child.text[common_len:])
                split_node.id = child.id
                split_node.children = child.children
                
                # Turn original child into the common prefix
                child.text = child.text[:common_len]
                child.id = -1
                child.children = {split_node.text[0]: split_node}
                
                if common_len < len(remains):
                    new_leaf = PatriciaNode(text=remains[common_len:])
                    self.id_to_str.append(s)
                    new_leaf.id = len(self.id_to_str) - 1
                    child.children[new_leaf.text[0]] = new_leaf
                    return new_leaf.id
                else:
                    self.id_to_str.append(s)
                    child.id = len(self.id_to_str) - 1
                    return child.id

    def __contains__(self, s):
        if not isinstance(s, type("")):
            return False
            
        node = self.root
        remains = s
        
        while remains:
            first_char = remains[0]
            if first_char not in node.children:
                return False
                
            child = node.children[first_char]
            common_len = 0
            while common_len < len(remains) and common_len < len(child.text) and remains[common_len] == child.text[common_len]:
                common_len += 1
                
            if common_len < len(child.text):
                return False
                
            node = child
            remains = remains[common_len:]
            
        return node.id != -1

    def __getitem__(self, key):
        if type(key) is int:
            return self._IdMap__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError
