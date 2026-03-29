from dictionary import IdMap, TrieNode, TrieIdMap, PatriciaNode, PatriciaTreeIdMap

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Merges two lists of tuples (doc id, tf) and returns the merged result 
    (TF needs to be accumulated for all tuples with the same doc id)
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

__all__ = ['IdMap', 'TrieNode', 'TrieIdMap', 'PatriciaNode', 'PatriciaTreeIdMap', 'sorted_merge_posts_and_tfs']
