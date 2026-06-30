from src import generate_number_value, generate_string_value
from src import select_from_trie, TrieNode
from llm_sdk import Small_LLM_Model  # type: ignore


def generate_function_call(model: Small_LLM_Model, prompt_text: str,
                           functions: list,
                           trie: TrieNode, vocab: dict[int, str],
                           user_prompt: str) -> tuple[dict, bool]:

    input_ids = model.encode(prompt_text).tolist()[0]
    quote_token_id = model.encode('"').tolist()[0][0]
    valid_result = True

    # open object, open name key
    input_ids.extend(model.encode('{"name": "').tolist()[0])

    # decision: which function
    name_tokens = select_from_trie(model=model,
                                   trie=trie,
                                   quote_token_id=quote_token_id,
                                   input_ids=input_ids)
    input_ids.extend(name_tokens)
    function_name = model.decode(name_tokens)

    # close name value, open parameters object
    input_ids.extend(model.encode('", "parameters": {').tolist()[0])

    # find the function index on the json list
    function_id = None
    for i, function in enumerate(functions):
        if function.get("name") == function_name:
            function_id = i
            break

    if function_id is None:
        raise ValueError(f"Model produced unknown function: {function_name}")

    # dict: {"a": {"type": "number"}, "b": {...}}
    params = functions[function_id].get("parameters")
    param_items = list(params.items())

    parsed_params = {}
    for i, (param_name, param_info) in enumerate(param_items):
        is_last = (i == len(param_items) - 1)

        # this param's key + colon
        input_ids.extend(model.encode(f'"{param_name}": ').tolist()[0])

        if param_info["type"] == "number":
            value_tokens = generate_number_value(model,
                                                 input_ids,
                                                 is_last,
                                                 user_prompt,
                                                 vocab)

            input_ids.extend(value_tokens)
            raw_value = model.decode(value_tokens)
            try:
                parsed_params[param_name] = float(raw_value)
            except ValueError:
                print("\033[33m" +
                      "Warning: a numeric value for parameter "
                      f"'{param_name}' could not be resolved..." +
                      "\033[0m")

                parsed_params[param_name] = 0.0
                valid_result = False

        elif param_info["type"] == "string":
            input_ids.extend(quote_token_id)
            value_tokens = generate_string_value(model=model,
                                                 input_ids_so_far=input_ids,
                                                 quote_token_id=quote_token_id,
                                                 vocab=vocab)
            input_ids.extend(value_tokens)
            # closing quote, since generator only stopped ON it
            input_ids.append(quote_token_id)
            parsed_params[param_name] = model.decode(value_tokens)

        # separator between params or close everything
        if not is_last:
            input_ids.extend(model.encode(", ").tolist()[0])
        else:
            # close parameters object + close outer object
            input_ids.extend(model.encode("}}").tolist()[0])

    return {
        "name": function_name,
        "parameters": parsed_params,
    }, valid_result
