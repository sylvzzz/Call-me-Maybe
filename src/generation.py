"""Constrained generation for function calling.

Provides the core loop that enforces valid output:
- force_text:  generate exact text, token by token
- generate_choice: let model pick from valid options using a trie
"""

from llm_sdk import Small_LLM_Model


def force_text(model: Small_LLM_Model, input_ids: list[int], text: str) -> list[int]:
    """Generate an exact text string by forcing each token.

    For every token in the encoded text, block every other token
    so the model has no choice but to output what we want.

    input_ids = the conversation so far, as a list of token IDs.
               Starts as the encoded prompt. Every time we generate
               a new token we append it to input_ids.

    forced_id = a single token ID from the encoded target text.
                e.g. if text is '"name"', encode gives [1, 606, 1],
                so forced_id is 1 on first loop, 606 on second, etc.

    constrained = a list of 150k+ -inf values (one per vocab entry).
                  We restore the original score only for the forced_id
                  position. The model can only pick that token.

    Example: for text = '{"name": "', we know the exact keys/braces/
             quotes that must appear. The model never chooses these --
             we force them. Only the values (function name, args)
             are chosen by the model via generate_choice.

    Parameters
    ----------
    model : Small_LLM_Model
    input_ids : list[int]
        Context so far (prompt + previously generated tokens).
    text : str
        The exact string to generate.

    Returns
    -------
    list[int]
        Updated input_ids with the new tokens appended.
    """
    # Turn the target text into token IDs (e.g. "name" -> [1, 606, 1])
    target_tokens = model.encode(text).tolist()[0]
    for forced_id in target_tokens:
        # Ask the model for scores of every possible next token
        logits = model.get_logits_from_input_ids(input_ids)
        # Create a list of 150k+ scores, all set to -inf (lowest possible)
        constrained = [float("-inf")] * len(logits)
        # Restore the original score only for the token we want
        constrained[forced_id] = logits[forced_id]
        # Pick the highest score (which can only be our forced token)
        next_id = constrained.index(max(constrained))
        # Append it to the growing context
        input_ids.append(next_id)
    return input_ids


def generate_choice(
    model: Small_LLM_Model,
    input_ids: list[int],
    trie: dict,
    max_tokens: int = 50,
) -> tuple[list[int], str]:
    """Let the model choose among valid completions defined by a trie.

    At each step, only the token IDs that are keys of the current trie
    node are allowed. The model picks among them. Stops at "END".

    Parameters
    ----------
    model : Small_LLM_Model
    input_ids : list[int]
        Context so far.
    trie : dict
        Nested dict of valid token sequences (from build_trie).
    max_tokens : int
        Safety limit (default 50).

    Returns
    -------
    tuple[list[int], str]
        Updated input_ids and the generated text.
    """
    # Start at the root of the trie. Each node is a dict whose
    # keys are valid next token IDs. "END" marks a complete value.
    current_node = trie
    # Track generated token IDs so we can decode them to text at the end
    generated_ids = []

    for _ in range(max_tokens):
        # Get all keys from the current node except "END"
        # These are the only token IDs the model is allowed to pick
        valid_ids = [k for k in current_node if k != "END"]
        if not valid_ids:
            # No valid choices means we reached a dead end (shouldn't happen)
            break

        # Ask the model for scores of every possible next token
        logits = model.get_logits_from_input_ids(input_ids)
        # Create a list of 150k+ scores, all set to -inf
        constrained = [float("-inf")] * len(logits)
        # Restore the original score only for the valid token IDs
        for ids in valid_ids:
            constrained[ids] = logits[ids]

        # Pick the highest score among the valid tokens
        # The model chooses which function/parameter it wants
        next_id = constrained.index(max(constrained))
        # Append the chosen token to the growing context
        input_ids.append(next_id)
        generated_ids.append(next_id)

        # Move deeper into the trie along the chosen path
        current_node = current_node[next_id]
        # If this node has "END", the value is completely generated
        if "END" in current_node:
            break

    return input_ids, model.decode(generated_ids)
