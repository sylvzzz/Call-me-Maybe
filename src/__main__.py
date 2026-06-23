import json
from llm_sdk import Small_LLM_Model
from src import load_llm_vocab, is_allowed_char, is_allowed_token, build_trie, select_from_trie

def generate_string_value(model, quote_token_id, input_ids_so_far, id_to_str):
    # Build the "safe" list once — this never changes during the loop,
    # so no point recomputing it every step.
    """
    Built a list of "safe" tokens once — every token that, written as plain text, doesn't contain a " character. This matters because a stray " showing up in the middle of the value would prematurely close the JSON string and corrupt the output.
    Manually added the quote token back in — even though it technically "contains a quote" (it IS the quote), you deliberately allow it, because it's the one special token you want the model to be able to choose, specifically as the signal "I'm done writing the value."
    Looped, one token at a time:

    Asked the model for logits (its raw opinion on what token comes next, given everything written so far — prompt + whatever's been generated in this loop already).
    Looked only at the scores for the "safe" tokens from step 1/2 (ignoring all other tokens' scores entirely).
    Picked whichever safe token scored highest — i.e., what the model actually thinks is the best next character/word-piece, restricted to the safe set.
    If that pick happened to be the quote token → stop, the value is finished.
    Otherwise → remember that token as part of the answer, and loop again, now with one more token added to the context.


    Returned the list of tokens it built up — which, when turned back into text, spelled shrek.
    """
    legal_ids = [
        token_id
        for token_id, token_text in id_to_str.items()
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


def main(model: Small_LLM_Model) -> None:
    prompt = "What is the sum of 2 and 3?"
    input_ids = model.encode(prompt)
    print(type(input_ids), input_ids)

    ids_list = input_ids.tolist()[0]
    print(ids_list)

    logits = model.get_logits_from_input_ids(ids_list)
    print(type(logits))
    print(len(logits))
    print(logits[:10])
    qid = model.encode('"').tolist()[0]
    print(qid)

    # build a reverse lookup: token string -> token id, like our vocab dict already is
    # but we want id -> string, so invert it
    id_to_token = {v: k for k, v in vocab.items()}

    for token_id in qid:
        print(token_id, repr(id_to_token.get(token_id)))


if __name__ == "__main__":
    with open("data/input/functions_definition.json", "r") as file:
        functions_data = json.load(file)
    
    with open("/home/dbotelho/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json") as f:
        vocab = json.load(f)

    load_llm_vocab(vocab)

    functions = [f.get("name") for f in functions_data]
    model = Small_LLM_Model()
    id_to_str = load_llm_vocab(vocab)

    """
    testing string value generation

    prompt = "Greet shrek"
    prefix_text = f'{prompt}\n{{"name": "fn_greet", "parameters": {{"name": "'
    input_ids_so_far = model.encode(prefix_text).tolist()[0]

    generated = generate_string_value(model, quote_token_id=1, input_ids_so_far=input_ids_so_far, id_to_str=id_to_str)

    print(generated)
    print([id_to_str.get(t) for t in generated])
    print("".join(id_to_str.get(t, "") for t in generated))"""

    """trie = build_trie(model, functions)

    prompt = "What is the sum of 2 and 3?"
    input_ids = model.encode(prompt).tolist()[0]

    chosen_tokens = select_from_trie(model, trie, quote_token_id=1, input_ids=input_ids)
    print(chosen_tokens)
    print([id_to_str.get(t) for t in chosen_tokens])

    prompt2 = "Greet shrek"
    input_ids2 = model.encode(prompt2).tolist()[0]
    chosen2 = select_from_trie(model, trie, quote_token_id=1, input_ids=input_ids2)
    print([id_to_str.get(t) for t in chosen2])"""