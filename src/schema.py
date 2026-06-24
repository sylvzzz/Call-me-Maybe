from src import select_from_trie, generate_number_value, generate_string_value


def generate_function_call(model, prompt_text, functions, trie, vocab):

    input_ids = model.encode(prompt_text).tolist()[0]
    quote_token_id = model.encode('"').tolist()[0][0]

    # --- skeleton: open object, open name key ---
    input_ids.extend(model.encode('{"name": "').tolist()[0])

    # --- decision: which function ---
    name_tokens = select_from_trie(model=model, trie=trie, quote_token_id=quote_token_id, input_ids=input_ids)
    input_ids.extend(name_tokens)
    function_name = model.decode(name_tokens)

    # --- skeleton: close name value, open parameters object ---
    input_ids.extend(model.encode('", "parameters": {').tolist()[0])

    # find the function index on the json list
    # find the function index on the json list
    function_id = None
    for i, function in enumerate(functions):
        if function.get("name") == function_name:
            function_id = i
            break

    if function_id is None:
        raise ValueError(f"Model produced unknown function name: {function_name!r}")

    params = functions[function_id].get("parameters")  # dict: {"a": {"type": "number"}, "b": {...}}
    param_items = list(params.items())

    for i, (param_name, param_info) in enumerate(param_items):
        is_last = (i == len(param_items) - 1)

        # --- skeleton: this param's key + colon ---
        input_ids.extend(model.encode(f'"{param_name}": ').tolist()[0])

        if param_info["type"] == "number":
            value_tokens = generate_number_value(model, input_ids, is_last)
            input_ids.extend(value_tokens)
            # generator stopped on , or } but didn't emit it — skeleton must

        elif param_info["type"] == "string":
            input_ids.extend(model.encode('"').tolist()[0])
            # opening quote — generator doesn't add this either
            value_tokens = generate_string_value(model=model, input_ids_so_far=input_ids,
                                                 quote_token_id=quote_token_id, vocab=vocab)
            input_ids.extend(value_tokens)
            input_ids.append(quote_token_id)   # closing quote, since generator only stopped ON it

        # --- skeleton: separator between params, or close everything ---
        if not is_last:
            input_ids.extend(model.encode(", ").tolist()[0])
        else:
            input_ids.extend(model.encode("}}").tolist()[0])   # close parameters object + close outer object

    return model.decode(input_ids)