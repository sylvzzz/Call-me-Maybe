import json
from llm_sdk import Small_LLM_Model
from src import load_llm_vocab, is_allowed_char, is_allowed_token, build_trie, select_from_trie

with open("/home/dbotelho/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json") as f:
    vocab = json.load(f)

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

    # build a reverse lookup: token string -> token id, like your vocab dict already is
    # but you want id -> string, so invert it
    id_to_token = {v: k for k, v in vocab.items()}

    for token_id in qid:
        print(token_id, repr(id_to_token.get(token_id)))


if __name__ == "__main__":
    with open("data/input/functions_definition.json", "r") as file:
        functions_data = json.load(file)

    functions = [f.get("name") for f in functions_data]
    model = Small_LLM_Model()
    id_to_str = load_llm_vocab(vocab)

    trie = build_trie(model, functions)

    prompt = "What is the sum of 2 and 3?"
    input_ids = model.encode(prompt).tolist()[0]

    chosen_tokens = select_from_trie(model, trie, quote_token_id=1, input_ids=input_ids)
    print(chosen_tokens)
    print([id_to_str.get(t) for t in chosen_tokens])

    prompt2 = "Greet shrek"
    input_ids2 = model.encode(prompt2).tolist()[0]
    chosen2 = select_from_trie(model, trie, quote_token_id=1, input_ids=input_ids2)
    print([id_to_str.get(t) for t in chosen2])