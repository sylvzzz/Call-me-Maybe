*This project has been created as part of the 42 curriculum by <a href ="https://github.comsylvzzz">dbotelho</a>.*

# Call-me-Maybe
Introduction to Constrained Decoding, Function calling and parsing with LLM's

# Description

This project introduces `Function Calling` in Large Language Models by building a system that translates natural human language prompts into structured function calls with typed arguments. The main goal and skill larned with this project was the process of `Constrained Decoding` to guarantee valid JSON output, achieving perfect reliability with a small 0.5 Bilion parameter model (for reference a popular LLM as Chat GPT has ~3 Trilion), optimizing generation of valid function calling output in form of JSON on weak models.

# Instructions

## Installation

#### Installing in a 42 School Computer
If you are running this in a 42 school computer, please run this in the `sgoinfre` directory, since in 42 we dont have a lot of space in home directory

```bash
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/sgoinfre/$(USER)/.local/bin sh

# run this when you're in the project directory

export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"

make
```

#### Installing in your personal computer
```bash
curl -LsSf https://astral.sh/uv/install.sh

make
```

## Usage

### Default run (uses built-in test data)

```bash
make run
```

This reads functions available from `data/input/functions_definition.json` and prompt tests from `data/input/function_calling_tests.json`, then writes results to `data/output/function_calls.json`.

### Custom files

```bash
uv run python -m src \
  --functions_definition my_fns.json \
  --input my_tests.json \
  --output my_results.json
```

### Input format | function definitions

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "a": { "type": "number" },
            "b": { "type": "number" }
        }
    },
    {
        "name": "fn_greet",
        "description": "Greet a person by name.",
        "parameters": {
            "name": { "type": "string" }
        }
    }
]
```

### Input format | test prompts

```json
[
    { "prompt": "What is the sum of 265 and 345?" },
    { "prompt": "Greet shrek" },
    { "prompt": "Reverse the string 'hello'" }
]
```

### Output format

```json
[
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": { "a": 265.0, "b": 345.0 }
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": { "name": "shrek" }
    }
]
```

Invalid or discarded prompts are omitted from the output file, and their count is shown in the terminal report.

# Resources
- [Constrained Generation](https://medium.com/%40docherty/controlling-your-llm-deep-dive-into-constrained-generation-1e561c736a20)

- [Building Chat GPT from Scratch](https://youtu.be/kCc8FmEb1nY?si=pPc9orlMT8W5Djro)

- [argparse Docs](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args)

- LLM SDK code

- Claude

### AI Usage

In this project AI was used for:

- README polishing
- Understanding concepts as code
- Research about constrained decoding

<br>


# What is an LLM?
An LLM is basically just a next-word guesser. A user sends it a text and it guesses what comes next.
But computers don't understand text, so everything gets converted to numbers first.

### A simple example of a llm loop
<pre>
The full journey, simply
"What is 2 + 3?"
       |
   TOKENIZATION
       |
["What", "is", "2", "+", "3", "?"]
       |
   INPUT IDs (vocab lookup)
       |
[892, 318, 17, 10, 18, 30]
       |
   LLM (neural network)
       |
   LOGITS (scores for every possible next token)
[token_0: -2.1, token_1: 0.3, token_42: 8.7, token_99: -0.1, ...]
       |
   PICK HIGHEST SCORE → token_42 = "5"
       |
   APPEND "5" TO INPUT, REPEAT...</pre>

## What are logits?
Logits are like a list of scores, one per token in the vocabulary. If the vocab has 100,000 tokens, you get 100,000 scores. The highest score = the token the model thinks comes next.
```python

vocab size = 100,000 tokens
logits     = [0.1, -2.3, 8.7, 0.02, ...]
                          |
                      # this one wins → thats the next token
```

The model repeats this loop, one token at a time, until it generates an EOS token (end of sentence) the llm sending the signal that he's done.

On this project, our workflow differs just a bit from this, because just picking the highest score for the next token is a technique called greedy sampling, and with that we would be relying on the model to return always valid JSON and thats neither a good practice neither whats wanted from this project. What we do instead is what's expected from us, the <b>Constrained Decoding</b>.

## Constrained Decoding

Constrained Decoding is a technique that restricts Large Language Model (LLM) outputs to comply to strict syntactic, structural, or semantic rules by masking invalid tokens during the generation process. Instead of relying on post-processing or prompting, it ensures that every generated token maintains the sequence within a valid language defined by constraints.

Normally the model can pick any token freely. But to generate valid JSON for function calling, we want the output to always be valid JSON.
Before picking the next token, you look at the logits and set invalid tokens to -infinity so they can never be picked. On this project, the masking follows a per-stage strategy, documented below.

## Algorithm Explanation

The constrained decoding pipeline is structured as a state machine with four distinct generation phases, each with its own masking strategy:

### Phase 1: Function Name Selection (Trie Walk)

Given a set of available function, a prefix trie is built where each node maps a token ID to child nodes, with `"END"` marking complete names. At each generation step:

1. The current trie node's children define the **legal token set**
2. `"END"` is mapped to the `"` token ID (the JSON closing quote)
3. Logits for all tokens outside the legal set are ignored
4. The highest-scoring legal token is selected
5. If the quote token wins, the function name is complete; otherwise, descend into the chosen child node and repeat

This guarantees the generated function name is always one of the available functions.

### Phase 2: String Parameter Values

Only tokens whose decoded text does **not** contain a `"` character are allowed. The quote token itself is added to the legal set as the termination signal. This prevents premature JSON string closure and ensures output validity.

### Phase 3: Numeric Parameter Values

Tokens are restricted to the set `[0-9]` plus `.` (once) and structural terminators (`,` or `}`). Additionally, the concatenated digit string **must appear as a substring of the user's original prompt** , this grounds numeric values in the user's request and prevents hallucination.

### Phase 4: JSON Structure Injection

The JSON structure (`{"name": "`, `", "parameters": {`, `"param_name": `, `}}`) is injected directly as encoded token IDs rather than generated by the model. This eliminates structural JSON errors entirely while the model only fills in values.

## Design Decisions

| Decision | Explanation |
|---|---|
| **Trie over regex/grammar** | A trie maps naturally to the token-by-token generation loop. Each step is a simple hash lookup `O(1)` per child node check  vs. re-parsing a full grammar at every position. |
| **JSON skeleton injected** | A 0.5B model cannot reliably produce valid JSON structure from scratch. By hardcoding the braces, quotes, and keys, we eliminate ~90% of potential syntax errors. |
| **Prompt grounding for numbers** | The model might invent plausible-looking digits. Requiring the digit string to exist in the prompt text acts as a factuality check. |
| **`"` filtering for string values** | A single stray `"` token would close the JSON string early and corrupt everything downstream. Pre-filtering all non-quote tokens that contain `"` is a simple safeguard. |
| **Vocabulary filtering** | Tokens containing non-ASCII byte fragments are stripped at load time. This prevents decoding artifacts from reaching the output. |
| **Per-parameter comma/close logic** | The model decides per-parameter whether to emit a separator (`,`) or close the object (`}`) based on whether it's the last parameter. This lets the model "learn" the structure without generating arbitrary tokens. |

## Performance Analysis

### Accuracy

Because JSON structure is injected directly and token masks prevent invalid continuations, **the system produces 100% structurally valid JSON output** in practice. The only failure modes are:

- **Discarded prompts** (marked yellow): The model generates a valid function call but the parameter values don't semantically match the user's request (e.g., asking for a string sum and getting numbers). These are detected and reported separately.
- **Unknown function names**: Prevented by the trie, impossible by construction.
- **JSON syntax errors**: Prevented by phase-specific masking, impossible by construction.

### Reliability

The system handles degraded prompts gracefully:
- Invalid prompts (e.g., asking for a function that doesn't exist) are detected and discarded with a yellow warning
- Unparseable numeric values are catched in a try block and discarted
- Robust error handling are handled with clear messages

<br>

# Challenges Faced

The main challenges i faced were the actual finding of resources and debugging part, since i programmed before and was already used to it no mather the language, but realted to AI and token direct interaction was something really new to i struglled a bit, and since AI and LLM's are a recent topic i found some ok resources in the theoritical part, the hardest part was actual coding, i found few coding examples in regards to this subject, to overcome this challenges i digged deep into the concepts and theoretical part and asked AI for small help, for example asked how processes and concepts looked like in coded, asked for small pieces of pseudocode, once i understood everything the code seemed much simpler.

## Made at 42 Lisboa by <a href ="https://github.comsylvzzz">dbotelho</a>.