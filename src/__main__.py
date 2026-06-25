import json
from llm_sdk import Small_LLM_Model
from src import load_llm_vocab, generate_function_call, build_trie
import os


def valid_brackets(s: str) -> bool:
    stack = []
    pairs = {
        ")": "(",
        "}": "{",
        "]": "["
    }

    for ch in s:
        if ch in "{[(":
            stack.append(ch)

        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()

    return len(stack) == 0


def render_functions_block(functions_data):
    lines = ["Available functions:"]
    for fn in functions_data:
        params = fn.get("parameters", {})
        param_str = ", ".join(f'{name}: {info.get("type")}' for name, info in params.items())
        lines.append(f'- {fn["name"]}({param_str})')
    return "\n".join(lines)


def main(model: Small_LLM_Model, prompt: str, trie: dict, functions, vocab: dict[int, str]) -> None:
    print(generate_function_call(model=model, prompt_text=prompt,
                                 functions=functions, trie=trie, vocab=vocab))


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
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
    
    functions_block = render_functions_block(functions_data)

    results = []
    validation_results = []

    print()
    print("============== Results ==============")

    for prompt in prompts:
        full_prompt = f"{functions_block}\n\nUser request: {prompt}\n"
        call_result = generate_function_call(model=model, prompt_text=full_prompt,
                                            functions=functions, trie=trie, vocab=vocab)
        function_call_result = {
            "prompt": prompt,
            "name": call_result["name"],
            "parameters": call_result["parameters"],
        }
        results.append(function_call_result)

        json_validation = json.dumps(function_call_result)
        is_valid = valid_brackets(json_validation)

        if is_valid is not True:
            print("\033[31m"
                f"Result for prompt: '{prompt}'"
                + " - [INVALID]\033[0m")
            validation_results.append(False)
            print(json_validation)
        else:
            print("\033[92m"
                  f"Result for prompt: '{prompt}'"
                  + " - [OK]\033[0m")
            validation_results.append(True)
        
    tests_passed = sum(1 for res in validation_results if res is True)

    success_rate = tests_passed / len(validation_results) * 100

    print()
    print("=========== Tests Results ===========")
    if success_rate >= 90:
        print("\033[92m" + f"[OK] - {tests_passed}/{len(validation_results)} tests passed!" + "\033[0m")
        print("\033[92m" + f"[OK] - {success_rate}% valid JSON generated ready for function calling!" + "\033[0m")
    elif success_rate > 50 and success_rate < 90:
        print("\033[33m" + f"{tests_passed}/{len(validation_results)} tests passed!" + "\033[0m")
        print("\033[33m" + f"{success_rate}% valid JSON generated!" + "\033[0m")
    else:
        print("\033[33" + f"[KO] - {tests_passed}/{len(validation_results)} tests passed!" + "\033[0m")
        print("\033[33" + f"[KO] - {success_rate}% valid JSON generated!" + "\033[0m")

    dir_path = "data/output"
    output_file = "data/output/function_calls.json"

    os.makedirs(dir_path, exist_ok=True)

    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)