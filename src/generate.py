def generate_string_value(model, quote_token_id, input_ids_so_far, vocab: dict[int, str]):
    # Build the "safe" list once this never changes during the loop,
    # so no point recomputing it every step.
    """
    Built a list of "safe" tokens once every token that, written as plain text, doesn't contain a " character. This matters because a stray " showing up in the middle of the value would prematurely close the JSON string and corrupt the output.
    Manually added the quote token back in even though it technically "contains a quote" (it IS the quote), you deliberately allow it, because it's the one special token you want the model to be able to choose, specifically as the signal "I'm done writing the value."
    Looped, one token at a time:

    Asked the model for logits (its raw opinion on what token comes next, given everything written so far prompt + whatever's been generated in this loop already).
    Looked only at the scores for the "safe" tokens from step 1/2 (ignoring all other tokens' scores entirely).
    Picked whichever safe token scored highest i.e., what the model actually thinks is the best next character/word-piece, restricted to the safe set.
    If that pick happened to be the quote token stop, the value is finished.
    Otherwise remember that token as part of the answer, and loop again, now with one more token added to the context.


    Returned the list of tokens it built up which, when turned back into text, spelled shrek.
    """
    legal_ids = [
        token_id
        for token_id, token_text in vocab.items()
        if '"' not in token_text
    ]

    legal_ids.append(quote_token_id)

    generated_tokens = []

    while True:
        logits = model.get_logits_from_input_ids(input_ids_so_far + generated_tokens)
        
        best_token = max(legal_ids, key=lambda token_id: logits[token_id])

        if best_token == quote_token_id:
            break  # model chose to stop, string is done

        generated_tokens.append(best_token)

    return generated_tokens

def generate_number_value(model, input_ids_so_far, is_last_parameter, prompt, vocab: dict[int, str]):

    digit_ids = [model.encode(c).tolist()[0][0] for c in ["0","1","2","3","4","5","6","7","8","9"]]
    dot_id, comma_id, close_brace_id = [model.encode(c).tolist()[0][0] for c in [".",",","}"]]

    generated_tokens = []
    used_dot = False
    result_text = ""  # tracks decoded digits/dot so far for checks

    while True:
        structural_ids = digit_ids[:]
        if not used_dot:
            structural_ids.append(dot_id)

        # keep only structural candidates that are actually grounded in the prompt
        legal_ids = [
            token_id for token_id in structural_ids
            if (result_text + vocab[token_id]) in prompt
        ]

        if is_last_parameter:
            legal_ids.append(close_brace_id)
        else:
            legal_ids.append(comma_id)

        # NEW: nothing groundable left and no stop chosen yet -> bail out
        if not legal_ids:
            break

        logits = model.get_logits_from_input_ids(input_ids_so_far + generated_tokens)

        best_token = max(legal_ids, key=lambda token_id: logits[token_id])

        if best_token == comma_id or best_token == close_brace_id:
            break

        if best_token == dot_id:
            used_dot = True

        result_text += vocab[best_token]  # NEW
        generated_tokens.append(best_token)

    return generated_tokens
