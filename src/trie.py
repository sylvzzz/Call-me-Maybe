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


def select_from_trie(model, trie: dict, quote_token_id: int, input_ids: list[int]) -> list[int]:
    """Walk the trie, masking logits at each step, until the quote token wins.

    Args:
        model: the LLM SDK wrapper.
        trie: a trie built by build_trie, mapping token_id -> subtrie,
            with "END" marking complete strings.
        quote_token_id: the token id for the closing '"' character
            used as the real, scoreable stand-in for "stop here".
        input_ids: the token ids generated so far (prompt + any prior
            generation), used as context for the next logits call.

    Returns:
        The list of token ids chosen for this string (not including
        the closing quote itself).
    """
    chosen_tokens: list[int] = []
    current_node = trie

    while True:
        legal_ids = []
        for key in current_node.keys():
            if key == "END":
                legal_ids.append(quote_token_id)
            else:
                legal_ids.append(key)

        logits = model.get_logits_from_input_ids(input_ids + chosen_tokens)

        best_token = max(legal_ids, key=lambda t: logits[t])

        if best_token == quote_token_id:
            break  # model chose to stop -> string is complete

        chosen_tokens.append(best_token)
        current_node = current_node[best_token]

    return chosen_tokens