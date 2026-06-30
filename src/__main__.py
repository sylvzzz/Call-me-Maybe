from llm_sdk import Small_LLM_Model  # type: ignore
from src import load_llm_vocab, generate_function_call, build_trie
import os
import json
import argparse
import sys


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


def parse_available_functions(functions_data: list[dict]) -> str:
    """Returns all the functions that the llm can use and infuse
    it into the starting prompt"""
    lines = ["Available functions:"]
    for fn in functions_data:
        params = fn.get("parameters", {})
        param_str = ", ".join(f'{name}: {info.get("type")}'
                              for name, info in params.items())

        lines.append(f'- {fn["name"]}({param_str})')
    return "\n".join(lines)


def show_final_results(success_rate: float, total_tests: int,
                       tests_passed: int,
                       invalid_prompts_handled: int) -> None:

    """Final report on the prompts tests"""
    print()
    print("=========== Tests Results ===========")

    if success_rate >= 90:
        print("\033[92m" + f"[OK] - {tests_passed}/"
              f"{total_tests} tests passed!")
        print(f"[OK] - {success_rate:.2f}% valid JSON generated"
              "ready for function calling!" + "\033[0m")

    elif success_rate > 50 and success_rate < 90:
        print("\033[33m" + f"{tests_passed}/{total_tests} tests passed!")
        print(f"{success_rate:.2f}% valid JSON generated!" + "\033[0m")

    else:
        print("\033[31m" + f"[KO] - {tests_passed}/"
              "{total_tests} tests passed!")
        print(f"[KO] - {success_rate:.2f}% valid JSON generated!" + "\033[0m")

    print("\033[92m" + f"[OK] - {invalid_prompts_handled}"
          " invalid prompts handled!" + "\033[0m")


def main(model: Small_LLM_Model, functions_file: str,
         tests_file: str, output_file: str) -> None:
    """
    Main function where we build the whole pipeline for the core of the project

    Loads the functions, prompt tests and save the results in the output file

    At last we provide a small report of the results
    """

    # clear the terminal
    os.system("cls" if os.name == "nt" else "clear")

    # vocab file
    vocab_file = model.get_path_to_vocab_file()

    # create the input folder if somehow it disapeared
    if "data/input/" in output_file:
        dir_path = "data/input"

        os.makedirs(dir_path, exist_ok=True)

    # read and load function data as json (python dict)
    with open(functions_file, "r") as file:
        functions_data = json.load(file)

    # read and vocab as json (python dict)
    with open(vocab_file) as file:
        vocab = json.load(file)

    # read and load test prompts as json (python dict)
    with open(tests_file) as file:
        tests = json.load(file)

    # list of functions
    functions = [function for function in functions_data]
    if len(functions) == 0:
        print("\033[31m" +
              f"No functions provided, check {functions_file}" +
              "\033[0m")
        return

    # getting function names to build a prefix trie so the model easily choses
    functions_names = [f.get("name") for f in functions_data]
    trie = build_trie(model=model, values=functions_names)
    vocab = load_llm_vocab(vocab)

    prompts = [prompt.get("prompt") for prompt in tests]
    if len(prompts) == 0:
        print("\033[31m" +
              f"No test prompts provided, check {tests_file}" +
              "\033[0m")
        return

    available_functions = parse_available_functions(functions_data)

    results = []  # results actualy saved
    validation_results = []  # results for validation

    print()
    print("============== Results ==============")

    # test our project for every user prompt
    for prompt in prompts:
        full_prompt = f"{available_functions}\n\nUser request: {prompt}\n"
        data = generate_function_call(model=model,
                                      prompt_text=full_prompt,
                                      functions=functions,
                                      trie=trie,
                                      vocab=vocab,
                                      user_prompt=prompt)

        call_result = data[0]
        is_valid_output = data[1]

        function_call_result = {
            "prompt": prompt,
            "name": call_result["name"],
            "parameters": call_result["parameters"],
        }

        # loading the result as a python dict and validating json
        json_validation = json.dumps(function_call_result)
        is_valid_json = valid_brackets(json_validation)
        invalid_json_generated = 0

        # not valid json (CRITICAL)
        if is_valid_json is not True:
            print("\033[31m" +
                  f"PROMPT: {prompt}"
                  + " - [INVALID]\033[0m")
            invalid_json_generated += 1

        # not valid output (asking a sum between strings) (Discarted)
        elif is_valid_output is not True:
            print("\033[33m" +
                  f"PROMPT: {prompt}" +
                  " - [DISCARTED]\033[0m")
            validation_results.append(False)

        # saving ok json
        else:
            print("\033[92m"
                  f"PROMPT: '{prompt}'"
                  + " - [OK]\033[0m")
            validation_results.append(True)
            results.append(function_call_result)

    # fully ok jsons
    tests_passed = sum(1 for res in validation_results if res is True)

    # discarted jsons
    invalid_prompts_handled = sum(1 for res in validation_results
                                  if res is not True)

    # discarted + ok json's
    handled_tests = invalid_prompts_handled + tests_passed

    # success rate
    total_tests = handled_tests + invalid_json_generated
    success_rate = ((handled_tests) / total_tests) * 100

    # printing final report
    show_final_results(success_rate=success_rate,
                       total_tests=total_tests,
                       tests_passed=handled_tests,
                       invalid_prompts_handled=invalid_prompts_handled)

    # creating output dir if doesnt exist
    if "data/output/" in output_file:
        dir_path = "data/output"

        os.makedirs(dir_path, exist_ok=True)

    print()
    print("=========== Saving Results ==========")
    print()
    print(f"Writing results to {output_file} ...")

    # saving results in json file
    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--functions_definition', default="data/input/functions_definition.json")
        parser.add_argument('--input', default="data/input/function_calling_tests.json")
        parser.add_argument('--output', "data/output/function_calls.json")
        parser.add_argument("--model", default="Qwen/Qwen3-0.6B")

        # convert args to a object
        args = parser.parse_args()

        # if not defined use default
        functions_file = args.functions_definition
        tests_file = args.input
        output_file = args.output

        if functions_file == tests_file:
            print("Functions definition and input file cannot be the same.")
            sys.exit(1)

        if functions_file == output_file:
            print("Functions definition and output file cannot be the same.")
            sys.exit(1)

        if output_file == tests_file:
            print("Output file and tests file cannot be the same file")
            sys.exit(1)

        model = Small_LLM_Model(args.model)

        main(model=model, functions_file=functions_file,
             tests_file=tests_file, output_file=output_file)
    except FileNotFoundError as error:
        print()
        print("\033[31m" + f"File {error.filename} not found" + "\033[0m")
    except IsADirectoryError as error:
        print()
        print("\033[31m" +
              f"Argument file {error.filename} cannot be a directory" +
              "\033[0m")
    finally:
        print()
        print("\033[92m" + f"Results successfully saved in {output_file}!"
              + "\033[0m")
        print()
        print("Made by dbotelho at 42 Lisbon.")
        print()
