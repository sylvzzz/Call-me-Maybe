# What is an LLM doing, really?
An LLM is basically a next-word guesser. That's it. You give it some text, it guesses what comes next.
But computers don't understand text, so everything gets converted to numbers first.

<pre>
The full journey, simply
"What is 2 + 3?"
       ↓
   TOKENIZATION
       ↓
["What", "is", "2", "+", "3", "?"]
       ↓
   INPUT IDs (vocab lookup)
       ↓
[892, 318, 17, 10, 18, 30]
       ↓
   LLM (neural network)
       ↓
   LOGITS (scores for every possible next token)
[token_0: -2.1, token_1: 0.3, token_42: 8.7, token_99: -0.1, ...]
       ↓
   PICK HIGHEST SCORE → token_42 = "5"
       ↓
   APPEND "5" TO INPUT, REPEAT...
  </pre>

The model repeats this loop, one token at a time, until it generates an EOS token (end of sentence — its way of saying "I'm done").

# What are logits?
Just a list of scores, one per token in the vocabulary. If the vocab has 100,000 tokens, you get 100,000 scores. The highest score = the token the model thinks comes next.
vocab size = 100,000 tokens
logits     = [0.1, -2.3, 8.7, 0.02, ...] ← 100,000 numbers
                              ↑
                         this one wins → that's the next token

# What is constrained decoding?
Normally the model can pick any token freely. But for function calling, you want the output to always be valid JSON — so you cheat.
Before picking the next token, you look at the logits and set invalid tokens to -infinity so they can never be picked:
You're generating JSON and just wrote: {"function_name": "
                                                          ↑
                               only string tokens are valid here!

logits before constraint:
  token "{" : 5.2   ← INVALID here, set to -inf
  token "add": 3.1  ← valid, keep it
  token "123": 2.8  ← INVALID here, set to -inf
  token "sub": 1.9  ← valid, keep it

logits after constraint:
  token "{" : -inf
  token "add": 3.1  ← wins
  token "123": -inf
  token "sub": 1.9
This guarantees the output is always parseable JSON, no hallucinated structure, no broken brackets.

How your SDK fits in
pythonmodel = Small_LLM_Model()

# Step 1: convert your prompt to token IDs
input_ids = model.encode("What is 2 + 3?").tolist()[0]
→ [892, 318, 17, 10, 18, 30]

 Step 2: get scores for every possible next token
logits = model.get_logits_from_input_ids(input_ids)
 → [0.1, -2.3, 8.7, ...] (one score per vocab token)

Step 3: pick the best token
next_id = logits.index(max(logits))

Step 4: add it to input and repeat
input_ids.append(next_id)
For constrained decoding, you'd modify the logits between steps 2 and 3 — zeroing out anything that would break your JSON schema.

Summary in one sentence

The LLM scores every possible next token → you block the ones that would break your JSON → you pick the best remaining one → repeat until done.


Want me to now look at your functions_definition.json to understand what schema you need to enforce?

User question: "what is 2 + 3?"
        ↓
Build a prompt that includes the function definitions
        ↓
LLM generates JSON token by token
        ↓
At each step, constrained decoding blocks invalid tokens
  - wrong function name → blocked
  - string where number expected → blocked
  - broken JSON structure → blocked
        ↓
Output is guaranteed valid JSON → parse it → call the function

# Export bc of space on 42 pc

# Point HuggingFace cache to sgoinfre instead of home
export HF_HOME=/sgoinfre/dbotelho/hf_cache

# Then run again
python3 main.py


What is a prompt made of?
A good prompt for function calling has 4 parts:
1. ROLE        → tell the model what it is
2. TASK        → tell it exactly what to output
3. CONTEXT     → give it the available functions
4. REQUEST     → the actual user question
In our code that maps to:
pythonreturn f"""You are a function calling assistant.       ← ROLE

Given a user request, output ONLY a valid JSON object  ← TASK
with exactly these keys: ...

Available functions:                                   ← CONTEXT
- fn_add_numbers(a: number, b: number)
...

User request: What is the sum of 2 and 3?             ← REQUEST
Output:"""                                            ← triggers generation


generation loop — this is the core of the project.

What it needs to do
1. Take the prompt
2. Encode it to token IDs
3. Get logits from the model
4. (constrained decoding) block invalid tokens
5. Pick the best remaining token
6. Add it to the input
7. Repeat until we have complete valid JSON

Normal generation: the LLM scores every token in its vocab (100k+), picks the highest score, adds it to the input, repeats.
Constrained decoding: you block tokens that would break your desired structure before picking one. You're generating JSON and just wrote { — only keys/strings are valid next, so you set every other token's score to -infinity. The model can only pick from the valid ones.
This guarantees 100% valid JSON output regardless of model size — no broken brackets, wrong types, or hallucinated keys. The Qwen3-0.6B might only produce valid JSON 30% of the time from prompting alone; constrained decoding pushes that to ~100%.


# Trie

A Trie, also known as a prefix tree, is a tree-like data structure used for storing a dynamic set of strings. It is commonly used for efficient retrieval and storage of keys in a large dataset. The structure supports operations such as insertion, search, and deletion of keys, making it a valuable tool in fields like computer science and information retrieval.

<pre>
                root
                  |
                [8822] -- fn
                /   |      |
            [2891] [1889] [43277]
            _add    _g    _reverse
            |       |
        [32964]   [3744]
        _numbers   reet
</pre>

# Constrained Decoding

1. Fixed text (like {"name": ") — force the exact tokens
2. A choice from a trie (like the function name) — traverse the trie, only allow its children
The loop:
1. Encode the prompt → input_ids
2. For each segment:
   - If fixed text: encode it, force each token one by one
   - If a choice (trie): start at root, at each step:
       a. Get logits from model
       b. Find current node in trie
       c. Zero out all logits except keys of current node
       d. Pick the best remaining token
       e. Move to that child in trie
       f. Repeat until hitting "END"
3. Return the generated text
After the function name, you'd have another fixed segment ", "parameters": { then another choice segment for the parameters, then }.

The generation loop is just two modes:
Mode A — "Force this exact string"
You pick each token. No choice. Just shove it in.
Mode B — "Choose from this list"
Model picks. You just filter.
The whole JSON is just a sequence of Mode A and Mode B:
{"name": "  →  Mode A (force exact tokens)
fn_...          →  Mode B (model chooses function)
", "parameters": {  →  Mode A
param values       →  Mode B (model chooses args)
}                 →  Mode A
The loop walks through this plan top to bottom. That's it.