import json
from llm_sdk import Small_LLM_Model
from src import load_llm_vocab, generate_function_call, build_trie


def main(model: Small_LLM_Model, prompt: str, trie: dict, functions, vocab: dict[int, str]) -> None:
    for function in functions:
        print(generate_function_call(model=model, prompt_text=prompt,
                                     function_definition=function, trie=trie, vocab=vocab))


if __name__ == "__main__":

    model = Small_LLM_Model()
    vocab_file = model.get_path_to_vocab_file()
    functions_file = "data/input/functions_definition.json"
    tests_file = "data/input/function_calling_tests.json"

    with open(functions_file, "r") as file:
        functions_data = json.load(file)
    
    with open(vocab_file) as file:
        vocab = json.load(file)
    
    with open(tests_file) as file:
        tests = json.load(file)

    functions = [function for function in functions_data]
    functions_names = [f.get("name") for f in functions_data]
    trie = build_trie(model=model, values=functions_names)
    vocab = load_llm_vocab(vocab)

    prompts = [prompt.get("prompt") for prompt in tests]
    
    for prompt in prompts:
        main(model=model, prompt=prompt, trie=trie, functions=functions, vocab=vocab)

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