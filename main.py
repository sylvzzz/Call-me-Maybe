from llm_sdk.llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()

def build_prompt(user_request: str, functions: list) -> str:
    functions_text = ""
    for fn in functions:
        params = ", ".join(
            f"{name}: {info['type']}" 
            for name, info in fn["parameters"].items()
        )
        functions_text += f"- {fn['name']}({params})\n"

    return f"""You are a function calling assistant.
Given a user request, output ONLY a valid JSON object with exactly these keys:
- "name": the function to call
- "parameters": the arguments with correct types

Available functions:
{functions_text}
User request: {user_request}
Output:"""


"""
def generate_debugging(model: Small_LLM_Model, prompt: str) -> str:
    input_ids = model.encode(prompt).tolist()[0]  # full context
    generated_ids = []                             # only generated tokens

    for _ in range(200):
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = logits.index(max(logits))

        input_ids.append(next_id)      # grow the context
        generated_ids.append(next_id)  # track what was generated

        generated_text = model.decode(generated_ids)
        print(generated_text)  # watch it generate

        if generated_text.strip().endswith("}"):
            break

    return generated_text
"""

def generate(model: Small_LLM_Model, prompt: str) -> str:
    input_ids = model.encode(prompt).tolist()[0]
    generated_ids = []

    for _ in range(200):
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = logits.index(max(logits))

        input_ids.append(next_id)
        generated_ids.append(next_id)

        generated_text = model.decode(generated_ids)

        if generated_text.strip().endswith("}"):
            break

    return generated_text


def main() -> None:
    with open("data/input/functions_definition.json") as f:
        functions = json.load(f)

    with open("data/input/function_calling_tests.json") as f:
        tests = json.load(f)

    results = []
    for test in tests:
        prompt = test["prompt"]
        llm_prompt = build_prompt(prompt, functions)
        
        output_json = json.loads(generate(model, llm_prompt))  # parse string → dict
        
        results.append({
            "prompt": prompt,
            "name": output_json["name"],
            "parameters": output_json["parameters"]
        })

    with open("data/output/function_calling_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()