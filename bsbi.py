from indexers import BSBIIndex, tokenize

# Expose SPIMI classes through bsbi for backward compatibility
# in case the user's scripts expected them to be here initially
from indexers import SPIMIIndex, SPIMIPatriciaIndex

__all__ = ['BSBIIndex', 'tokenize', 'SPIMIIndex', 'SPIMIPatriciaIndex']
