def is_allowed_char(c: str) -> bool:
    # ord converts char to ascii/unicode, return true if the char is allowed
    return ord(c) < 128 or c == 'Ġ' or c == 'Ċ'


def is_allowed_token(s: str) -> bool:
    # a iterator that returns true if all chars all allowed
    return all(is_allowed_char(char) for char in s)


def polish_token(token: str) -> str:
    # this llm uses these two as spaces and newline
    # it doesnt use native space and newline
    return token.replace("Ġ", "").replace("Ċ", "")


def load_llm_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """Build an id-to-text lookup from the raw vocabulary mapping.

    Inverts the {token_string: id} vocab and decodes byte-level BPE
    markers (Ġ -> space, Ċ -> newline) into real characters. Tokens
    that decode to non-ASCII byte fragments are excluded.

    Args:
        vocab: Raw vocabulary as loaded from vocab.json.

    Returns:
        Mapping from token id to its clean text representation,
        containing only directly usable tokens.
    """
    result: dict[int, str] = {}
    for token, token_id in vocab.items():
        if is_allowed_token(token):
            result[token_id] = polish_token(token)
        else:
            continue
    return result