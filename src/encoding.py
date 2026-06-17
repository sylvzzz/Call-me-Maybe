def build_trie(model, values: list[str]) -> dict:
    """
    Builds a prefix tree (trie) from a list of strings.
    Each level in the tree is a token ID.
    "END" marks the end of a complete valid string.

    Example: ["fn_add", "fn_sub"]
       { 8822: { 2891: {"END": True}, OTHER_TOKEN: {"END": True} } }

                 root
                  |
                [8822] -- fn
                /   |      |
            [2891] [1889] [43277] ...
            _add    _g    _reverse
            |       |
        [32964]   [3744]
        _numbers   reet
    
    Usage: at each generation step, traverse the trie.
    Valid tokens are the keys of the current node.
    When you hit "END", the value is complete.
    """

    trie = {}
    for val in values:
        # Encode the string into token IDs
        tokens = model.encode(val).tolist()[0]

        # Walk through the trie, creating nodes as needed
        current = trie
        for token in tokens:
            if token not in current:
                current[token] = {}
            current = current[token]

        # Mark this path as a complete valid value
        current["END"] = True

    return trie
